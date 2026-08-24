#!/usr/bin/env python
"""
Build knowledge base from PDF files in data/ folder.

Usage:
    python build_knowledge_base.py              # Build from data/ folder
    python build_knowledge_base.py --kb-name my_kb  # Custom KB name
"""

import sys
import argparse
from pathlib import Path

from src.knowledge_base import KnowledgeBaseBuilder


def main():
    """Build knowledge base from documents."""
    parser = argparse.ArgumentParser(
        description="Build RAG knowledge base from PDF files"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing PDF files (default: data)"
    )
    parser.add_argument(
        "--kb-name",
        default="ragtest_kb",
        help="Name of knowledge base (default: ragtest_kb)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in characters (default: 800)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Overlap between chunks (default: 150)"
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text:latest",
        help="Ollama embedding model (default: nomic-embed-text:latest)"
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to Chroma database (default: ./chroma_db)"
    )

    args = parser.parse_args()

    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        return 1

    # Count PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  Warning: No PDF files found in {data_dir}")
        return 1

    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  • {pdf.name} ({size_mb:.1f} MB)")

    print()

    try:
        # Create embeddings config
        embeddings_config = {
            "provider": "ollama",
            "model": args.model,
            "base_url": "http://localhost:11434"
        }

        # Build knowledge base
        builder = KnowledgeBaseBuilder(
            kb_name=args.kb_name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embeddings_config=embeddings_config,
            db_path=args.db_path
        )

        # Build from directory
        stats = builder.build_from_directory(str(data_dir))

        # Print statistics
        builder.print_stats()

        # Save metadata
        builder.save_metadata()

        print("✅ Knowledge base build completed successfully!\n")
        return 0

    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        print("\nAnd pull the model:")
        print(f"  ollama pull {args.model}")
        return 1

    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return 1

    except Exception as e:
        print(f"❌ Error building knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
