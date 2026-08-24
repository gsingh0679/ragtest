#!/usr/bin/env python
"""
Diagnose ChromaDB collection dimensions and embedding mismatches.
"""

import chromadb
from src.embeddings.factory import EmbeddingsFactory
from src.config import get_config_loader
import json


def diagnose_collection():
    """Diagnose collection dimensions and embedding model."""
    print("\n🔍 ChromaDB Collection Diagnostic\n")

    config_loader = get_config_loader()
    kb_config = config_loader.get_kb_config()
    embeddings_config = config_loader.get_embeddings_config()

    kb_name = kb_config["name"]
    db_path = kb_config["db_path"]

    print(f"Knowledge Base: {kb_name}")
    print(f"Database Path: {db_path}\n")

    # Connect to Chroma
    client = chromadb.PersistentClient(path=db_path)

    # Try to get collection
    try:
        collection = client.get_collection(name=kb_name)
        print(f"✓ Collection '{kb_name}' found\n")

        # Get collection metadata
        metadata = collection.metadata
        print(f"Collection Metadata:")
        print(f"  {json.dumps(metadata, indent=2)}\n")

        # Count items
        count = collection.count()
        print(f"Stored Items: {count}\n")

        if count > 0:
            # Get a sample item to check embedding dimension
            print("Retrieving sample item to check dimensions...")
            sample = collection.get(limit=1)

            if sample["embeddings"] and sample["embeddings"][0]:
                stored_dim = len(sample["embeddings"][0])
                print(f"✓ Stored embedding dimension: {stored_dim}\n")

                # Get current embeddings model dimension
                print("Checking configured embedding model...")
                embeddings = EmbeddingsFactory.create_from_config(embeddings_config)
                model_dim = embeddings.get_embedding_dimension()
                print(f"✓ Model embedding dimension: {model_dim}\n")

                # Compare
                if stored_dim == model_dim:
                    print(f"✅ MATCH - Both are {stored_dim}-dimensional")
                else:
                    print(f"❌ MISMATCH - Stored: {stored_dim}D, Model: {model_dim}D")
                    print(f"\nThis collection was built with a different embedding model!")
                    print(f"\nPOSSIBLE CAUSES:")
                    print(f"1. Collection was built with 768D model (e.g., OpenAI, other)")
                    print(f"2. Then tried to query with 384D model (nomic-embed-text)")
                    print(f"\nSOLUTION OPTIONS:")
                    print(f"Option A (SAFEST): Delete and rebuild with consistent model")
                    print(f"Option B: Use original embedding model that created collection")

                    return {
                        "status": "MISMATCH",
                        "stored_dimension": stored_dim,
                        "model_dimension": model_dim,
                        "collection_name": kb_name,
                        "db_path": db_path,
                    }
            else:
                print("⚠️  CRITICAL ISSUE: Collection has 272 items but NO embeddings!")
                print(f"\nThis means the collection was created/populated WITHOUT embeddings.")
                print(f"Possible causes:")
                print(f"  1. Items were added with 'include_embeddings=False'")
                print(f"  2. Embeddings parameter was None when calling collection.add()")
                print(f"  3. Collection was created expecting an embedding function")
                print(f"\nThe ONLY fix is to DELETE and REBUILD:")
                print(f"  rm -rf {db_path}")
                print(f"  python main.py build\n")

                return {
                    "status": "NO_EMBEDDINGS",
                    "item_count": count,
                    "collection_name": kb_name,
                    "db_path": db_path,
                }
        else:
            print("ℹ️  Collection is empty - no items to check")

    except ValueError as e:
        print(f"❌ Collection '{kb_name}' not found: {e}")
        print(f"Collections in this database: {client.list_collections()}")
        return None

    return {"status": "OK", "dimension": None}


def show_fix_options(diagnosis):
    """Show fix options based on diagnosis."""
    if not diagnosis or diagnosis.get("status") == "OK":
        print("\n✅ No issues found!\n")
        return

    print(f"\n{'='*80}")
    print("FIX OPTIONS")
    print(f"{'='*80}\n")

    if diagnosis.get("status") == "NO_EMBEDDINGS":
        print("STATUS: Collection exists but has NO embeddings stored")
        print(f"Items in collection: {diagnosis['item_count']}")
        print(f"Database: {diagnosis['db_path']}\n")
        print("⚠️  This is why you're getting dimension mismatch errors!")
        print("Chroma can't query without embeddings.\n")
        print("SOLUTION - Delete and rebuild (only option):")
        print(f"  rm -rf {diagnosis['db_path']}")
        print(f"  python main.py build\n")
        return

    stored_dim = diagnosis.get("stored_dimension")
    model_dim = diagnosis.get("model_dimension")

    print(f"OPTION A - SAFEST: Delete and Rebuild")
    print(f"{'─'*80}")
    print(f"# Delete old collection")
    print(f"rm -rf {diagnosis['db_path']}")
    print(f"\n# Rebuild with consistent model")
    print(f"python main.py build --kb-name {diagnosis['collection_name']}\n")

    print(f"\nOPTION B - INVESTIGATE: Find original embedding model")
    print(f"{'─'*80}")
    print(f"The {stored_dim}D collection suggests one of these models:")
    if stored_dim == 768:
        print(f"  • text-embedding-3-small (OpenAI)")
        print(f"  • text-embedding-ada-002 (OpenAI)")
        print(f"  • all-MiniLM-L12-v2 (HuggingFace)")
        print(f"  • Other 768D embedding models")
    elif stored_dim == 1536:
        print(f"  • text-embedding-3-large (OpenAI)")
        print(f"  • text-davinci-003 embeddings")
    elif stored_dim == 384:
        print(f"  • MiniLM-L6-v2 (HuggingFace)")
        print(f"  • nomic-embed-text (via some configurations)")

    print(f"\nTo fix: Update config.yaml or .env to use original model:")
    print(f"  embeddings:")
    print(f"    provider: \"huggingface\"  # or \"openai\"")
    print(f"    model: \"<original-model-name>\"")
    print(f"\n# Then query with same model")
    print(f"python main.py query \"your question\"\n")

    print(f"\nOPTION C - START FRESH: Create new collection")
    print(f"{'─'*80}")
    print(f"# Build with new KB name")
    print(f"python main.py build --kb-name ragtest_kb_v2")
    print(f"\n# Query new collection")
    print(f"python main.py query \"your question\" --kb-name ragtest_kb_v2\n")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    diagnosis = diagnose_collection()
    show_fix_options(diagnosis)
