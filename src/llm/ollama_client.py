"""Ollama LLM client for answer generation."""

import requests
from typing import Optional, List
from src.config import get_config_loader


class OllamaClient:
    """Client for Ollama LLM (uses config defaults if not specified)."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            model: Model name (uses config default if not provided)
            base_url: Base URL for Ollama server (uses config default if not provided)
        """
        # Load from config/env if not provided
        if model is None or base_url is None:
            try:
                import os
                # Try config first
                config_loader = get_config_loader()
                llm_config = config_loader.get_llm_config()
                model = model or llm_config.get("model")
                base_url = base_url or llm_config.get("base_url")
            except Exception:
                # Fallback to env defaults
                import os
                model = model or os.getenv("DEFAULT_OLLAMA_LLM_MODEL", "llama2")
                base_url = base_url or os.getenv("DEFAULT_OLLAMA_BASE_URL", "http://localhost:11434")

        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.list_url = f"{base_url}/api/tags"

        # Verify model is available
        self._verify_model()

    def _verify_model(self):
        """Check if the specified model is available in Ollama."""
        try:
            import os
            timeout = int(os.getenv("OLLAMA_TIMEOUT", os.getenv("DEFAULT_OLLAMA_TIMEOUT", "5")))
            response = requests.get(self.list_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            available_models = [m["name"].split(":")[0] for m in data.get("models", [])]

            model_base = self.model.split(":")[0]
            if not available_models:
                raise ValueError(
                    f"❌ No models found in Ollama. "
                    f"Pull a model first: ollama pull mistral"
                )

            if model_base not in available_models:
                models_list = ", ".join(available_models)
                raise ValueError(
                    f"❌ Model '{self.model}' not found in Ollama.\n"
                    f"Available models: {models_list}\n"
                    f"Pull a model: ollama pull mistral"
                )
        except requests.ConnectionError:
            raise ConnectionError(
                f"❌ Could not connect to Ollama at {self.base_url}. "
                f"Make sure it's running: ollama serve"
            )
        except Exception as e:
            if "Model" in str(e):
                raise
            raise ConnectionError(f"Error checking Ollama models: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            import os
            timeout = int(os.getenv("OLLAMA_TIMEOUT", os.getenv("DEFAULT_OLLAMA_TIMEOUT", "5")))
            response = requests.get(self.list_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return [m["name"].split(":")[0] for m in data.get("models", [])]
        except Exception as e:
            raise Exception(f"Error fetching models: {e}")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> str:
        """
        Generate an answer using Ollama.

        Args:
            prompt: User prompt/question
            context: Optional context (retrieved chunks)
            temperature: Sampling temperature (0-1)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated answer text
        """
        # Build prompt with context if provided
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"

        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "stream": False
                },
                timeout=300  # 5 minute timeout for LLM
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

        except requests.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama serve"
            )
        except requests.Timeout:
            raise TimeoutError(f"Ollama request timed out. Model: {self.model}")
        except Exception as e:
            raise Exception(f"Error generating response: {e}")

    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer from a question and context.

        Args:
            question: User question
            context: Retrieved context chunks
            temperature: Sampling temperature

        Returns:
            Generated answer
        """
        return self.generate(question, context=context, temperature=temperature)
