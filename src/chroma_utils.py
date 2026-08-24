"""
ChromaDB utilities for proper initialization and consistency.

Prevents dimension mismatch issues by ensuring consistent embedding models.
"""

import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from src.embeddings.base import EmbeddingsBase


class OllamaEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function wrapper for Ollama embeddings."""

    def __init__(self, embeddings_model: EmbeddingsBase):
        """
        Initialize embedding function.

        Args:
            embeddings_model: EmbeddingsBase instance to use
        """
        self.embeddings = embeddings_model

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for documents.

        Args:
            input: List of documents to embed

        Returns:
            List of numpy arrays (required by ChromaDB Embeddings type)
        """
        # Handle both single strings and lists
        if isinstance(input, str):
            embedding = self.embeddings.embed_text(input)
            # Convert to numpy array (required by ChromaDB)
            if isinstance(embedding, np.ndarray):
                return [embedding]
            else:
                return [np.array(embedding, dtype=np.float32)]

        # For lists, use batch embedding
        embeddings = self.embeddings.embed_texts(input)

        # Convert all to numpy arrays
        result = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                result.append(emb.astype(np.float32))
            else:
                result.append(np.array(emb, dtype=np.float32))
        return result


def init_chroma_collection(
    db_path: str,
    collection_name: str,
    embeddings_model: EmbeddingsBase
):
    """
    Initialize Chroma collection with proper embedding function.

    Args:
        db_path: Path to Chroma database
        collection_name: Name of collection
        embeddings_model: Embeddings model to use

    Returns:
        Chroma collection instance and client
    """
    # Create persistent client
    client = chromadb.PersistentClient(path=db_path)

    # Create embedding function wrapper
    embedding_function = OllamaEmbeddingFunction(embeddings_model)

    # Get or create collection with embedding function
    # This tells ChromaDB to use our embeddings function
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function
    )

    return collection, client


def verify_embedding_consistency(
    collection,
    embeddings_model: EmbeddingsBase,
    test_text: str = "test embedding consistency"
):
    """
    Verify that stored embeddings match the embedding model dimensions.

    Args:
        collection: Chroma collection
        embeddings_model: Embeddings model to verify against
        test_text: Test text to generate embedding

    Raises:
        ValueError: If dimensions don't match
    """
    # Get embedding dimension from model
    model_dim = embeddings_model.get_embedding_dimension()

    # Test collection has data
    collection_count = collection.count()
    if collection_count > 0:
        # Query with test embedding
        test_embedding = embeddings_model.embed_text(test_text)
        test_dim = len(test_embedding)

        if model_dim != test_dim:
            raise ValueError(
                f"Embedding dimension mismatch! "
                f"Model produces {test_dim}-dim embeddings but expected {model_dim}-dim"
            )

        # Try a query to verify
        try:
            results = collection.query(
                query_embeddings=[test_embedding],
                n_results=1
            )
        except Exception as e:
            raise ValueError(
                f"ChromaDB dimension mismatch error: {e}\n"
                f"Solution: Delete ./chroma_db and rebuild with consistent embedding model"
            )

    return True
