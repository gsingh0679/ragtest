"""
ChromaDB utilities for proper initialization and consistency.

Prevents dimension mismatch issues by ensuring consistent embedding models.
"""

import chromadb
from src.embeddings.base import EmbeddingsBase


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
        Chroma collection instance
    """
    # Create persistent client
    client = chromadb.PersistentClient(path=db_path)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
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
