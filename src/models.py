"""Data models for the RAG system"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Chunk:
    """Represents a text chunk from a document"""
    content: str              # The actual text of the chunk
    chunk_id: str             # Unique ID (e.g., "document_0", "document_1")
    source_document: str      # Which document this came from (filename)
    chunk_index: int          # Position in sequence (0, 1, 2...)
    start_char: int           # Character position in original document
    end_char: int             # End character position
    token_count: int          # Approximate token count

    def preview(self, chars: int = 150) -> str:
        """Show first N characters for debugging"""
        if len(self.content) > chars:
            return self.content[:chars] + "..."
        return self.content

    def stats(self) -> dict:
        """Return chunk statistics"""
        return {
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "chunk_index": self.chunk_index,
            "content_length": len(self.content),
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class Document:
    """Represents a loaded document with metadata"""
    content: str          # The actual text extracted from file
    source: str           # Filename (e.g., "sample.pdf")
    file_path: Path       # Full file path
    file_type: str        # File format: 'pdf', 'txt', 'md'
    size_bytes: int       # Original file size in bytes
    loaded_at: datetime   # When the file was loaded

    def preview(self, chars: int = 200) -> str:
        """
        Show first N characters for debugging.

        Args:
            chars: Number of characters to show

        Returns:
            Preview string with ellipsis if text is longer
        """
        if len(self.content) > chars:
            return self.content[:chars] + "..."
        return self.content

    def stats(self) -> dict:
        """Return document statistics"""
        return {
            "source": self.source,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "content_length": len(self.content),
            "word_count": len(self.content.split()),
            "loaded_at": self.loaded_at.isoformat(),
        }
