"""
Query engine for semantic search and retrieval from knowledge base.

Queries the existing Chroma collection built by KnowledgeBaseBuilder.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.embeddings.base import EmbeddingsBase


@dataclass
class RetrievalResult:
    """A single retrieved chunk with relevance score."""
    chunk_id: str
    content: str
    source: str
    chunk_index: int
    similarity_score: float
    start_char: int
    end_char: int
    token_count: int
    sections: str = ""  # Comma-separated section numbers
    schedules: str = ""  # Comma-separated schedule names
    forms: str = ""  # Comma-separated form numbers

    def preview(self, chars: int = 150) -> str:
        """Show first N characters for debugging."""
        if len(self.content) > chars:
            return self.content[:chars] + "..."
        return self.content


@dataclass
class QueryResponse:
    """Response from a query with retrieved results."""
    query: str
    results: List[RetrievalResult]
    query_embedding: Optional[List[float]] = None

    def __len__(self) -> int:
        return len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "result_count": len(self.results),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "source": r.source,
                    "chunk_index": r.chunk_index,
                    "similarity_score": r.similarity_score,
                    "start_char": r.start_char,
                    "end_char": r.end_char,
                    "token_count": r.token_count,
                }
                for r in self.results
            ]
        }


class QueryEngine:
    """
    Semantic search engine for querying a knowledge base.

    Works with an existing Chroma collection built by KnowledgeBaseBuilder.
    """

    def __init__(
        self,
        chroma_collection,
        embeddings: EmbeddingsBase,
        top_k: int = 5,
        min_score: float = 0.3
    ):
        """
        Initialize QueryEngine.

        Args:
            chroma_collection: Existing Chroma collection to query
            embeddings: Embeddings instance (same model used during KB building)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (0-1)
        """
        self.collection = chroma_collection
        self.embeddings = embeddings
        self.top_k = top_k
        self.min_score = min_score

    def query(self, query_text: str, top_k: Optional[int] = None, min_score: Optional[float] = None,
              filter_sections: Optional[list] = None) -> QueryResponse:
        """
        Query the knowledge base for relevant chunks.

        Args:
            query_text: User query string
            top_k: Override default top_k for this query
            min_score: Override default min_score for this query
            filter_sections: Optional list of sections to filter by (e.g., ['80C', '80D'])

        Returns:
            QueryResponse with retrieved chunks and scores
        """
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score

        # Embed the query
        query_embedding = self.embeddings.embed_text(query_text)

        # Search in Chroma (get more results to filter)
        search_k = int(top_k * 2) if filter_sections else top_k  # Get 2x results if filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_k
        )

        # Parse results
        retrieved_chunks = self._parse_results(results, min_score)

        # Apply section filtering if specified
        if filter_sections and retrieved_chunks:
            retrieved_chunks = self._filter_by_sections(retrieved_chunks, filter_sections)

        # Keep only top_k after filtering
        retrieved_chunks = retrieved_chunks[:top_k]

        return QueryResponse(
            query=query_text,
            results=retrieved_chunks,
            query_embedding=query_embedding
        )

    def _filter_by_sections(self, chunks: List[RetrievalResult], target_sections: list) -> List[RetrievalResult]:
        """
        Filter retrieval results by section numbers.

        Args:
            chunks: List of RetrievalResult objects
            target_sections: List of section numbers to keep (e.g., ['80C', '80D'])

        Returns:
            Filtered list of chunks (prioritizes exact matches)
        """
        target_sections_upper = [s.upper() for s in target_sections]

        exact_matches = []
        partial_matches = []

        for chunk in chunks:
            # Get sections from metadata
            chunk_sections = chunk.sections if hasattr(chunk, 'sections') else ""
            chunk_sections_list = [s.strip().upper() for s in chunk_sections.split(",") if s.strip()]

            # Check for exact matches
            if any(s in target_sections_upper for s in chunk_sections_list):
                exact_matches.append(chunk)
            # Check for partial matches (e.g., "80" matches "80C", "80D")
            elif any(
                s in target_sections_upper or
                any(target.startswith(s) for target in target_sections_upper if len(s) > 0)
                for s in chunk_sections_list
            ):
                partial_matches.append(chunk)

        # Return exact matches first, then partial matches
        return exact_matches + partial_matches

    def _parse_results(self, results: Dict[str, Any], min_score: float) -> List[RetrievalResult]:
        """
        Parse Chroma query results into RetrievalResult objects.

        Args:
            results: Raw Chroma query response
            min_score: Filter results by minimum similarity score

        Returns:
            List of RetrievalResult objects, filtered by min_score
        """
        retrieved = []

        if not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Convert distances to similarity scores (Chroma returns distances, not similarities)
        # For cosine distance: similarity = 1 - distance
        for chunk_id, content, distance, metadata in zip(ids, documents, distances, metadatas):
            similarity = 1 - distance

            if similarity < min_score:
                continue

            result = RetrievalResult(
                chunk_id=chunk_id,
                content=content,
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                similarity_score=similarity,
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0),
                token_count=metadata.get("token_count", 0),
                sections=metadata.get("sections", ""),
                schedules=metadata.get("schedules", ""),
                forms=metadata.get("forms", "")
            )
            retrieved.append(result)

        return retrieved

    def get_context(self, query_text: str, top_k: Optional[int] = None) -> str:
        """
        Get formatted context string for LLM.

        Args:
            query_text: User query
            top_k: Number of results

        Returns:
            Formatted context string with retrieved chunks
        """
        response = self.query(query_text, top_k=top_k)

        if not response.results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(response.results, 1):
            context_parts.append(
                f"[Source {i}: {result.source} (Relevance: {result.similarity_score:.2%})]\n{result.content}\n"
            )

        return "\n".join(context_parts)

    def print_results(self, response: QueryResponse, show_score: bool = True) -> None:
        """
        Pretty-print query results.

        Args:
            response: QueryResponse object
            show_score: Show similarity scores
        """
        print(f"\n📊 Retrieved {len(response.results)} results for: \"{response.query}\"\n")

        if not response.results:
            print("No relevant chunks found.\n")
            return

        for i, result in enumerate(response.results, 1):
            score_str = f" (Score: {result.similarity_score:.2%})" if show_score else ""
            print(f"[Result {i}] {result.source} - Chunk {result.chunk_index}{score_str}")
            print(f"  {result.preview(200)}")
            print()
