"""
Embedding implementations for different providers.

Supports:
- Ollama (local)
- HuggingFace (local)
- OpenAI (cloud)
"""

import requests
from typing import List, Optional

from src.embeddings.base import EmbeddingsBase
from src.config import get_config_loader


class OllamaEmbeddings(EmbeddingsBase):
    """Generate embeddings using Ollama."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Ollama embeddings.

        Args:
            model: Model name (uses config default if not provided)
            base_url: Ollama server URL (uses config default if not provided)
        """
        # Load from config/env if not provided
        if model is None or base_url is None:
            try:
                import os
                # Try config first
                config_loader = get_config_loader()
                embeddings_config = config_loader.get_embeddings_config()
                model = model or embeddings_config.get("model")
                base_url = base_url or embeddings_config.get("base_url")
            except Exception:
                # Fallback to env defaults
                import os
                model = model or os.getenv("DEFAULT_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
                base_url = base_url or os.getenv("DEFAULT_OLLAMA_BASE_URL", "http://localhost:11434")

        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/embeddings"
        self._embedding_dim = None
        self.verify_connection()

    def verify_connection(self) -> bool:
        """Verify Ollama is running and model is available."""
        try:
            import os
            timeout = int(os.getenv("OLLAMA_TIMEOUT", os.getenv("DEFAULT_OLLAMA_TIMEOUT", "5")))
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status {response.status_code}")

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if self.model not in model_names:
                print(f"⚠️  Model '{self.model}' not found in Ollama")
                print(f"Available models: {model_names}")
                print(f"\nTo pull the model, run:")
                print(f"  ollama pull {self.model}")
                raise ValueError(f"Model '{self.model}' not available in Ollama")

            print(f"✓ Connected to Ollama at {self.base_url}")
            print(f"✓ Using model: {self.model}")
            return True

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}\n"
                "Make sure Ollama is running: ollama serve"
            )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            import os
            timeout = int(os.getenv("EMBEDDINGS_TIMEOUT", os.getenv("DEFAULT_EMBEDDINGS_TIMEOUT", "30")))
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": text},
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out for text: {text[:100]}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama embedding request failed: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(texts)} embeddings...")
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self._embedding_dim is not None:
            return self._embedding_dim

        try:
            test_text = "test"
            embedding = self.embed_text(test_text)
            self._embedding_dim = len(embedding)
            return self._embedding_dim
        except Exception as e:
            print(f"Warning: Could not determine embedding dimension: {e}")
            return 768  # Default for nomic-embed-text


class HuggingFaceEmbeddings(EmbeddingsBase):
    """Generate embeddings using HuggingFace sentence-transformers (local)."""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize HuggingFace embeddings.

        Args:
            model: Model name from sentence-transformers (uses config default if not provided)
        """
        if model is None:
            import os
            model = os.getenv("HUGGINGFACE_MODEL") or os.getenv("DEFAULT_HUGGINGFACE_MODEL", "all-MiniLM-L6-v2")
        self.model = model
        self._model_instance = None
        self._embedding_dim = None
        self.verify_connection()

    def verify_connection(self) -> bool:
        """Verify sentence-transformers is installed and model can be loaded."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"✓ Loading HuggingFace model: {self.model}")
            self._model_instance = SentenceTransformer(self.model)
            print(f"✓ Model loaded successfully")
            return True
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for HuggingFace embeddings\n"
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            embedding = self._model_instance.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            raise RuntimeError(f"HuggingFace embedding failed: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)."""
        try:
            embeddings = self._model_instance.encode(texts, convert_to_tensor=False)
            return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        except Exception as e:
            raise RuntimeError(f"HuggingFace batch embedding failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self._embedding_dim is not None:
            return self._embedding_dim

        try:
            embedding = self.embed_text("test")
            self._embedding_dim = len(embedding)
            return self._embedding_dim
        except Exception as e:
            print(f"Warning: Could not determine embedding dimension: {e}")
            return 384  # Default for all-MiniLM-L6-v2


class OpenAIEmbeddings(EmbeddingsBase):
    """Generate embeddings using OpenAI API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize OpenAI embeddings.

        Args:
            model: OpenAI model name (uses config default if not provided)
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        if model is None:
            import os
            model = os.getenv("OPENAI_MODEL") or os.getenv("DEFAULT_OPENAI_MODEL", "text-embedding-3-small")
        self.model = model
        self.api_key = api_key
        self._embedding_dim = None
        self.verify_connection()

    def verify_connection(self) -> bool:
        """Verify OpenAI API key and connection."""
        try:
            import openai
            from openai import OpenAI

            if self.api_key:
                openai.api_key = self.api_key

            client = OpenAI(api_key=self.api_key)
            print(f"✓ Connected to OpenAI")
            print(f"✓ Using model: {self.model}")
            self._client = client
            return True

        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings\n"
                "Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI connection failed: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings_dict = {item.index: item.embedding for item in response.data}
            return [embeddings_dict[i] for i in range(len(texts))]
        except Exception as e:
            raise RuntimeError(f"OpenAI batch embedding failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self._embedding_dim is not None:
            return self._embedding_dim

        try:
            embedding = self.embed_text("test")
            self._embedding_dim = len(embedding)
            return self._embedding_dim
        except Exception as e:
            print(f"Warning: Could not determine embedding dimension: {e}")
            return 1536  # Default for text-embedding-3-small
