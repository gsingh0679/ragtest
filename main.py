#!/usr/bin/env python
"""
RAG Knowledge Base System - Main Entry Point

Usage:
    python main.py build --kb-name ragtest_kb --data-dir ./data
    python main.py query "Your question here" --kb-name ragtest_kb
"""

import argparse
import sys
from src.knowledge_base.builder import KnowledgeBaseBuilder


def build_kb(args):
    """Build knowledge base from documents."""
    print(f"\n🔧 Building Knowledge Base: {args.kb_name}\n")

    config = {
        "provider": "ollama",
        "model": args.model,
        "base_url": "http://localhost:11434"
    }

    try:
        builder = KnowledgeBaseBuilder(
            kb_name=args.kb_name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embeddings_config=config,
            db_path=args.db_path
        )

        stats = builder.build_from_directory(args.data_dir)
        builder.print_stats()
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


def query_kb(args):
    """Query the knowledge base (Phase 2 - not yet implemented)."""
    print("\n📚 Query Knowledge Base\n")
    print("⏳ Query Engine (Phase 2) - Coming soon!\n")
    return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build knowledge base from documents
  python main.py build --kb-name ragtest_kb --data-dir ./data

  # Build with custom settings
  python main.py build --chunk-size 1000 --overlap 200

  # Query knowledge base (Phase 2)
  python main.py query "What are tax deductions?"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge base from documents")
    build_parser.add_argument(
        "--kb-name",
        default="ragtest_kb",
        help="Name of the knowledge base (default: ragtest_kb)"
    )
    build_parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing documents (default: ./data)"
    )
    build_parser.add_argument(
        "--model",
        default="nomic-embed-text:latest",
        help="Ollama model for embeddings (default: nomic-embed-text:latest)"
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in characters (default: 800)"
    )
    build_parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Overlap between chunks in characters (default: 150)"
    )
    build_parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to Chroma database (default: ./chroma_db)"
    )
    build_parser.set_defaults(func=build_kb)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument(
        "query",
        help="Search query"
    )
    query_parser.add_argument(
        "--kb-name",
        default="ragtest_kb",
        help="Name of the knowledge base (default: ragtest_kb)"
    )
    query_parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to Chroma database (default: ./chroma_db)"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    query_parser.set_defaults(func=query_kb)

    args = parser.parse_args()

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
