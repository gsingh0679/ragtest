"""
Batch processor for handling large documents with minimal memory footprint.

This module provides utilities for processing large document collections
without accumulating all chunks in memory.
"""

from pathlib import Path
from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.models import Chunk
from typing import Callable, Iterator


class BatchProcessor:
    """Process documents in batches with configurable chunk handling."""

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 150,
        break_on_sentences: bool = True,
    ):
        """Initialize processor with chunking parameters."""
        self.chunker = TextChunker(chunk_size, overlap, break_on_sentences)
        self.loader = DocumentLoader()

    def process_file(
        self, file_path: str, on_chunk: Callable[[Chunk], None]
    ) -> int:
        """
        Process a single file and call on_chunk for each chunk.

        Args:
            file_path: Path to document
            on_chunk: Callback function for each chunk

        Returns:
            Number of chunks processed
        """
        doc = self.loader.load(file_path)
        chunk_count = 0

        for chunk in self.chunker.chunk_stream(doc):
            on_chunk(chunk)
            chunk_count += 1

        return chunk_count

    def process_directory(
        self, dir_path: str, on_chunk: Callable[[Chunk], None]
    ) -> int:
        """
        Process all documents in a directory.

        Args:
            dir_path: Path to directory
            on_chunk: Callback function for each chunk

        Returns:
            Total number of chunks processed
        """
        dir_path = Path(dir_path)
        total_chunks = 0

        for file_path in sorted(dir_path.rglob("*")):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.loader.SUPPORTED_FORMATS
            ):
                try:
                    count = self.process_file(str(file_path), on_chunk)
                    total_chunks += count
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return total_chunks

    def process_with_buffer(
        self,
        dir_path: str,
        on_batch: Callable[[list[Chunk]], None],
        batch_size: int = 100,
    ) -> int:
        """
        Process directory with buffered batches.

        Useful if you want to save chunks in batches (e.g., to database)
        instead of one at a time.

        Args:
            dir_path: Path to directory
            on_batch: Callback function for each batch of chunks
            batch_size: Number of chunks per batch

        Returns:
            Total number of chunks processed
        """
        batch_buffer = []
        total_chunks = 0

        def buffer_chunk(chunk: Chunk):
            nonlocal batch_buffer, total_chunks
            batch_buffer.append(chunk)
            total_chunks += 1

            if len(batch_buffer) >= batch_size:
                on_batch(batch_buffer)
                batch_buffer = []

        # Process all documents
        self.process_directory(dir_path, buffer_chunk)

        # Process remaining chunks
        if batch_buffer:
            on_batch(batch_buffer)

        return total_chunks

    def stream_chunks(self, dir_path: str) -> Iterator[Chunk]:
        """
        Stream all chunks from a directory.

        Useful as a generator for processing.

        Args:
            dir_path: Path to directory

        Yields:
            Chunk objects
        """
        dir_path = Path(dir_path)

        for file_path in sorted(dir_path.rglob("*")):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.loader.SUPPORTED_FORMATS
            ):
                try:
                    doc = self.loader.load(file_path)
                    yield from self.chunker.chunk_stream(doc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
