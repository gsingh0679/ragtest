"""Text chunking and splitting logic"""

import re
from src.models import Document, Chunk


class TextChunker:
    """Split documents into overlapping chunks for RAG"""

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 150,
        break_on_sentences: bool = True,
    ):
        """
        Initialize chunker with parameters.

        Args:
            chunk_size: Target characters per chunk (soft limit, see break_on_sentences)
                When break_on_sentences=True: chunks may be smaller to respect sentence boundaries
                When break_on_sentences=False: chunks will be closer to this size
            overlap: Characters to overlap between chunks
            break_on_sentences: Quality vs Size trade-off
                True (default): Prioritize semantic coherence - break at sentence ends
                              (chunks ~70% of chunk_size due to early sentence breaks)
                False: Prioritize size consistency - break at hard char limit

        IMPORTANT: Actual chunk size depends on sentence lengths in your text.
        Use ChunkAnalyzer.print_analysis() after building to verify actual vs configured sizes.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.break_on_sentences = break_on_sentences

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count (rough approximation).

        Reasoning:
        - Average word is ~4-5 characters + space
        - Average token is ~4 characters
        - So tokens ≈ len(text) / 4

        Args:
            text: Text to estimate tokens for

        Returns:
            Approximate token count
        """
        return len(text) // 4

    def _break_at_sentence(self, text: str, max_pos: int) -> int:
        """
        Find best sentence boundary near max_pos.

        Reasoning:
        - Look for sentence endings (. ! ?) followed by space and capital letter
        - Search backwards from max_pos to find nearest sentence boundary
        - Fallback to max_pos if no sentence boundary found nearby

        Args:
            text: Full text to search
            max_pos: Maximum character position to consider

        Returns:
            Position to break at (preferably at sentence boundary)
        """
        if not self.break_on_sentences:
            return max_pos

        search_start = max(0, max_pos - 200)  # Look back up to 200 chars
        search_text = text[search_start:max_pos]

        sentence_pattern = r'[.!?]\s+'
        matches = list(re.finditer(sentence_pattern, search_text))

        if matches:
            last_match = matches[-1]
            end_pos = search_start + last_match.end()
            if end_pos > search_start:
                return end_pos

        return max_pos

    def chunk_stream(self, document: Document):
        """
        Split a document into overlapping chunks using a generator.

        Yields chunks one at a time instead of accumulating in memory.
        This allows processing large documents with constant memory usage.

        Args:
            document: Document object to chunk

        Yields:
            Chunk objects one at a time
        """
        text = document.content
        chunk_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, len(text))

            # If this is the last chunk, include all remaining text
            if end_pos == len(text):
                chunk_text = text[start_pos:end_pos]
            else:
                # Find sentence boundary near end_pos
                end_pos = self._break_at_sentence(text, end_pos)

                # Safety: if we couldn't find good boundary, use hard limit
                if end_pos <= start_pos:
                    end_pos = min(start_pos + self.chunk_size, len(text))

                chunk_text = text[start_pos:end_pos].strip()

            # Skip empty chunks
            if not chunk_text:
                start_pos += self.chunk_size
                continue

            # Create and yield chunk object
            chunk = Chunk(
                content=chunk_text,
                chunk_id=f"{document.source}_{chunk_index}",
                source_document=document.source,
                chunk_index=chunk_index,
                start_char=start_pos,
                end_char=end_pos,
                token_count=self._estimate_tokens(chunk_text),
            )

            yield chunk

            # If we've reached the end, break to avoid infinite loop with overlap
            if end_pos >= len(text):
                break

            # Move position for next iteration (sliding window with overlap)
            start_pos = end_pos - self.overlap
            chunk_index += 1

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into overlapping chunks.

        Returns a list for backward compatibility. Use chunk_stream()
        for memory-efficient processing of large documents.

        Args:
            document: Document object to chunk

        Returns:
            List of Chunk objects
        """
        return list(self.chunk_stream(document))

    def chunk_multiple_stream(self, documents: list[Document]):
        """
        Chunk multiple documents using a generator.

        Yields chunks from all documents one at a time.
        Memory usage stays constant regardless of total document size.

        Args:
            documents: List of Document objects

        Yields:
            Chunk objects from all documents
        """
        for doc in documents:
            yield from self.chunk_stream(doc)

    def chunk_multiple(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk multiple documents.

        Returns a list for backward compatibility. Use chunk_multiple_stream()
        for memory-efficient processing of multiple large documents.

        Args:
            documents: List of Document objects

        Returns:
            Combined list of chunks from all documents
        """
        return list(self.chunk_multiple_stream(documents))

    def stats(self, chunks: list[Chunk]) -> dict:
        """
        Calculate statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with stats
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "total_tokens": 0,
                "avg_chunk_size": 0,
                "avg_tokens": 0,
            }

        total_chars = sum(len(c.content) for c in chunks)
        total_tokens = sum(c.token_count for c in chunks)

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "avg_chunk_size": total_chars // len(chunks),
            "avg_tokens": total_tokens // len(chunks),
            "min_chunk_size": min(len(c.content) for c in chunks),
            "max_chunk_size": max(len(c.content) for c in chunks),
        }
