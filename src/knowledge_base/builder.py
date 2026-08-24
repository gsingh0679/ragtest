"""
Knowledge Base builder for RAG system.

Loads documents, chunks them, generates embeddings, and stores in vector database.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

import chromadb

from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.embeddings.base import EmbeddingsBase
from src.embeddings.factory import EmbeddingsFactory
from src.models import Document, Chunk
from src.config import get_config_loader
from src.chroma_utils import init_chroma_collection
from src.chunk_analyzer import ChunkAnalyzer
from src.metadata_extractor import MetadataExtractor


class KnowledgeBaseBuilder:
    """Build and manage a knowledge base from documents."""

    @classmethod
    def from_config(cls, config_path: str = "./config.yaml"):
        """
        Create KnowledgeBaseBuilder from config file.

        Args:
            config_path: Path to config.yaml

        Returns:
            Initialized KnowledgeBaseBuilder
        """
        config_loader = get_config_loader(config_path)
        kb_config = config_loader.get_kb_config()
        embeddings_config = config_loader.get_embeddings_config()

        return cls(
            kb_name=kb_config["name"],
            chunk_size=kb_config["chunk_size"],
            overlap=kb_config["overlap"],
            embeddings_config=embeddings_config,
            db_path=kb_config["db_path"],
            break_on_sentences=kb_config["break_on_sentences"]
        )

    def __init__(
        self,
        kb_name: str = "ragtest_kb",
        chunk_size: int = 800,
        overlap: int = 150,
        embeddings: Optional[EmbeddingsBase] = None,
        embeddings_config: Optional[Dict[str, Any]] = None,
        db_path: str = "./chroma_db",
        break_on_sentences: bool = True
    ):
        """
        Initialize Knowledge Base Builder.

        Args:
            kb_name: Name of the knowledge base
            chunk_size: Size of chunks in characters
            overlap: Overlap between chunks
            embeddings: Pre-initialized embeddings object
            embeddings_config: Config dict for embeddings (if embeddings not provided)
            db_path: Path to store Chroma database
            break_on_sentences: Break chunks at sentence boundaries

        Example:
            # Using pre-initialized embeddings
            emb = OllamaEmbeddings(model="nomic-embed-text:latest")
            kb = KnowledgeBaseBuilder(embeddings=emb)

            # Using config dict
            config = {"provider": "ollama", "model": "nomic-embed-text:latest"}
            kb = KnowledgeBaseBuilder(embeddings_config=config)
        """
        self.kb_name = kb_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.db_path = db_path
        self.break_on_sentences = break_on_sentences

        # Initialize components
        print("🔧 Initializing Knowledge Base Builder...")
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            break_on_sentences=break_on_sentences
        )

        # Initialize embeddings
        print("🔌 Initializing embeddings...")
        if embeddings is not None:
            self.embeddings = embeddings
            embedding_model = getattr(embeddings, "model", "custom")
        elif embeddings_config is not None:
            self.embeddings = EmbeddingsFactory.create_from_config(embeddings_config)
            embedding_model = embeddings_config.get("model", "unknown")
        else:
            # Default to Ollama
            self.embeddings = EmbeddingsFactory.create_ollama()
            embedding_model = "nomic-embed-text:latest"

        print("📚 Initializing Chroma database...")
        self._init_chroma(db_path)

        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "kb_name": kb_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": embedding_model,
            "total_documents": 0,
            "total_chunks": 0
        }

        print(f"✅ Knowledge Base Builder initialized: {kb_name}\n")

    def _init_chroma(self, db_path: str) -> None:
        """Initialize Chroma database."""
        os.makedirs(db_path, exist_ok=True)

        # Initialize collection with proper setup
        self.collection, self.client = init_chroma_collection(
            db_path=db_path,
            collection_name=self.kb_name,
            embeddings_model=self.embeddings
        )

    def build_from_directory(self, dir_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Build knowledge base from all documents in a directory.

        Args:
            dir_path: Directory containing documents
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with build statistics
        """
        print(f"📂 Building knowledge base from: {dir_path}\n")

        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")

        # Note: Embedding consistency is now handled by ChromaDB's embedding function
        # No manual verification needed

        stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents": []
        }

        # Load documents
        documents = self.loader.load_directory(str(dir_path))
        print(f"\n✓ Loaded {len(documents)} documents\n")

        # Process each document
        for doc in documents:
            doc_stats = self._add_document(doc, progress_callback)
            stats["documents_loaded"] += 1
            stats["chunks_created"] += doc_stats["chunks"]
            stats["embeddings_generated"] += doc_stats["embeddings"]
            stats["documents"].append({
                "name": doc.source,
                "chunks": doc_stats["chunks"],
                "size_bytes": doc.size_bytes
            })

        self.metadata["total_documents"] = stats["documents_loaded"]
        self.metadata["total_chunks"] = stats["chunks_created"]

        print(f"\n{'='*80}")
        print(f"✅ Knowledge Base Build Complete!")
        print(f"{'='*80}")
        print(f"Documents loaded: {stats['documents_loaded']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Embeddings generated: {stats['embeddings_generated']}")
        print()

        return stats

    def analyze_chunks(self) -> Dict[str, Any]:
        """
        Analyze chunks in the knowledge base for size mismatches.

        Returns:
            Analysis report with statistics
        """
        # Retrieve all chunks from collection
        all_items = self.collection.get()

        if not all_items or not all_items.get("ids"):
            print("No chunks in collection to analyze")
            return {}

        # Reconstruct chunk objects for analysis
        chunks = []
        for chunk_id, content, metadata in zip(
            all_items["ids"],
            all_items["documents"],
            all_items["metadatas"]
        ):
            chunk = Chunk(
                content=content,
                chunk_id=chunk_id,
                source_document=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0),
                token_count=metadata.get("token_count", 0),
            )
            chunks.append(chunk)

        # Run analysis
        ChunkAnalyzer.print_analysis(chunks, self.chunk_size)
        return ChunkAnalyzer.analyze_chunks(chunks, self.chunk_size)

    def _add_document(self, doc: Document, progress_callback=None) -> Dict[str, int]:
        """
        Add a document to the knowledge base.

        Args:
            doc: Document to add
            progress_callback: Optional progress callback

        Returns:
            Dictionary with chunk and embedding counts
        """
        print(f"📄 Processing: {doc.source}")

        chunks_list = []
        for chunk in self.chunker.chunk_stream(doc):
            chunks_list.append(chunk)

        if not chunks_list:
            print(f"  ⚠️  No chunks generated for {doc.source}\n")
            return {"chunks": 0, "embeddings": 0}

        print(f"  Chunks: {len(chunks_list)}")
        print(f"  Generating embeddings...")

        # Extract chunk contents for embedding
        chunk_texts = [chunk.content for chunk in chunks_list]

        # Prepare data for Chroma
        ids = [chunk.chunk_id for chunk in chunks_list]
        documents = [chunk.content for chunk in chunks_list]
        metadatas = []
        for chunk in chunks_list:
            # Extract metadata from chunk content
            extracted_meta = MetadataExtractor.extract_metadata(chunk.content)

            metadata = {
                "source": chunk.source_document,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                # Extracted metadata
                "sections": ",".join(extracted_meta["sections"]),
                "schedules": ",".join(extracted_meta["schedules"]),
                "forms": ",".join(extracted_meta["forms"]),
                "has_deduction": str(extracted_meta["has_deduction"]),
                "has_relief": str(extracted_meta["has_relief"]),
                "has_income": str(extracted_meta["has_income"]),
            }
            metadatas.append(metadata)

        # Add to Chroma collection
        # ChromaDB's embedding function will handle embedding generation
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"  ✓ Stored {len(ids)} documents with embeddings in Chroma")
        except Exception as e:
            raise RuntimeError(
                f"Failed to add documents to Chroma collection: {e}\n"
                f"Try: rm -rf {self.db_path} && python main.py build"
            )

        print(f"  ✓ Stored in Chroma database\n")

        return {"chunks": len(chunks_list), "embeddings": len(chunks_list)}

    def add_document(self, file_path: str) -> Dict[str, int]:
        """
        Add a single document to the knowledge base.

        Args:
            file_path: Path to document file

        Returns:
            Dictionary with chunk and embedding counts
        """
        doc = self.loader.load(file_path)
        return self._add_document(doc)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            **self.metadata,
            "collection_count": self.collection.count()
        }
        return stats

    def save_metadata(self, save_path: str = None) -> str:
        """
        Save knowledge base metadata to file.

        Args:
            save_path: Path to save metadata (default: chroma_db/metadata.json)

        Returns:
            Path where metadata was saved
        """
        if save_path is None:
            save_path = os.path.join(self.db_path, "metadata.json")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Metadata saved to: {save_path}")
        return save_path

    def print_stats(self) -> None:
        """Print knowledge base statistics."""
        stats = self.get_stats()

        print(f"\n{'='*80}")
        print("📊 Knowledge Base Statistics")
        print(f"{'='*80}")
        print(f"KB Name:           {stats['kb_name']}")
        print(f"Total Documents:   {stats['total_documents']}")
        print(f"Total Chunks:      {stats['total_chunks']}")
        print(f"Stored Chunks:     {stats['collection_count']}")
        print(f"Embedding Model:   {stats['embedding_model']}")
        print(f"Chunk Size:        {stats['chunk_size']} chars")
        print(f"Overlap:           {stats['overlap']} chars")
        print(f"Created At:        {stats['created_at']}")
        print(f"{'='*80}\n")
