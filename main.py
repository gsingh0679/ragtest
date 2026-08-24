#!/usr/bin/env python
"""
RAG Knowledge Base System - Main Entry Point

Usage:
    python main.py build --kb-name ragtest_kb --data-dir ./data
    python main.py query "Your question here" --kb-name ragtest_kb
"""

import argparse
import sys
import chromadb
from src.knowledge_base.builder import KnowledgeBaseBuilder
from src.retrieval import QueryEngine
from src.embeddings.factory import EmbeddingsFactory
from src.llm import OllamaClient
from src.config import get_config_loader


def build_kb(args):
    """Build knowledge base from documents."""
    try:
        # Load configuration
        config_loader = get_config_loader()
        kb_config = config_loader.get_kb_config()

        # Override with CLI arguments if provided
        kb_name = args.kb_name if args.kb_name != "ragtest_kb" else kb_config["name"]
        data_dir = args.data_dir if args.data_dir != "./data" else config_loader.get_data_config()["input_dir"]
        chunk_size = args.chunk_size if args.chunk_size != 800 else kb_config["chunk_size"]
        overlap = args.overlap if args.overlap != 150 else kb_config["overlap"]
        db_path = args.db_path if args.db_path != "./chroma_db" else kb_config["db_path"]

        print(f"\n🔧 Building Knowledge Base: {kb_name}")
        print(f"📂 Data directory: {data_dir}")
        print(f"📊 Chunk size: {chunk_size}, Overlap: {overlap}\n")

        builder = KnowledgeBaseBuilder(
            kb_name=kb_name,
            chunk_size=chunk_size,
            overlap=overlap,
            embeddings_config=config_loader.get_embeddings_config(),
            db_path=db_path
        )

        stats = builder.build_from_directory(data_dir)
        builder.print_stats()
        builder.save_metadata()

        print("✅ Knowledge base build completed successfully!\n")
        return 0

    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
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
    """Query the knowledge base."""
    print("\n📚 Query Knowledge Base\n")
    print(f"🔍 Query: {args.query}\n")

    try:
        # Load configuration
        config_loader = get_config_loader()
        kb_config = config_loader.get_kb_config()
        retrieval_config = config_loader.get_retrieval_config()
        llm_config = config_loader.get_llm_config()

        # Override with CLI arguments if provided
        kb_name = args.kb_name if args.kb_name != "ragtest_kb" else kb_config["name"]
        db_path = args.db_path if args.db_path != "./chroma_db" else kb_config["db_path"]
        top_k = args.top_k if args.top_k != 5 else retrieval_config["top_k"]
        min_score = args.min_score if args.min_score != 0.3 else retrieval_config["min_score"]
        use_llm = args.use_llm
        llm_model = args.llm_model if args.llm_model != "llama2" else llm_config["model"]
        temperature = args.temperature if args.temperature != 0.7 else llm_config["temperature"]

        # Initialize embeddings (same model used during KB building)
        embeddings = EmbeddingsFactory.create_from_config(
            config_loader.get_embeddings_config()
        )

        # Connect to Chroma collection
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=kb_name)

        # Initialize QueryEngine
        query_engine = QueryEngine(
            chroma_collection=collection,
            embeddings=embeddings,
            top_k=top_k,
            min_score=min_score
        )

        # Run query
        response = query_engine.query(args.query)

        # Display results
        query_engine.print_results(response)

        # Generate LLM answer if requested
        if use_llm:
            print("🤖 Generating answer with LLM...\n")
            try:
                llm = OllamaClient(
                    model=llm_model,
                    base_url=llm_config.get("base_url", "http://localhost:11434")
                )
                context = query_engine.get_context(args.query, top_k=top_k)
                answer = llm.generate_answer(args.query, context, temperature=temperature)
                print(f"Answer:\n{answer}\n")
            except ConnectionError as e:
                print(f"❌ LLM Error: {e}")
                print(f"Make sure Ollama is running: ollama serve\n")
                return 1
            except Exception as e:
                print(f"❌ Error generating answer: {e}\n")
                return 1

        return 0

    except Exception as e:
        print(f"❌ Error querying knowledge base: {e}")
        import traceback
        traceback.print_exc()
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
        "--embedding-model",
        default="nomic-embed-text:latest",
        help="Embedding model (default: nomic-embed-text:latest)"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    query_parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum similarity score (default: 0.3)"
    )
    query_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate answer using LLM"
    )
    query_parser.add_argument(
        "--llm-model",
        default="llama2",
        help="LLM model to use (default: llama2)"
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for answer generation (default: 0.7)"
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
