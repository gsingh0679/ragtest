"""
Abstract base class for embedding generators.

Defines the interface that all embedding implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingsBase(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    def verify_connection(self) -> bool:
        """
        Verify the embedding model is accessible.

        Returns:
            True if connection successful, raises exception otherwise
        """
        pass
