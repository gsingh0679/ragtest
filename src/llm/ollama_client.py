"""Ollama LLM client for answer generation."""

import requests
from typing import Optional


class OllamaClient:
    """Client for Ollama LLM."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "llama2", "mistral", "neural-chat")
            base_url: Base URL for Ollama server
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

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
