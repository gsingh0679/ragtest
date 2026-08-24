"""
Analyze chunking behavior and identify mismatches between config and actual chunks.
"""

from typing import Dict, Any, List
from src.models import Chunk


class ChunkAnalyzer:
    """Analyze chunk statistics and config mismatches."""

    @staticmethod
    def analyze_chunks(chunks: List[Chunk], config_chunk_size: int = 800) -> Dict[str, Any]:
        """
        Analyze chunks and compare against configured size.

        Args:
            chunks: List of Chunk objects
            config_chunk_size: Configured chunk size from config.yaml

        Returns:
            Analysis report with statistics and mismatch info
        """
        if not chunks:
            return {"error": "No chunks to analyze"}

        chunk_sizes = [len(c.content) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunks)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)

        # Calculate words (rough: split by spaces)
        avg_words = sum(len(c.content.split()) for c in chunks) / len(chunks)

        # Check for mismatch
        size_ratio = avg_size / config_chunk_size
        has_mismatch = size_ratio < 0.7  # Less than 70% of configured size

        return {
            "config_chunk_size": config_chunk_size,
            "avg_actual_size": round(avg_size, 1),
            "min_chunk_size": min_size,
            "max_chunk_size": max_size,
            "avg_words_per_chunk": round(avg_words, 1),
            "total_chunks": len(chunks),
            "size_ratio": round(size_ratio, 2),
            "has_mismatch": has_mismatch,
            "mismatch_severity": ChunkAnalyzer._get_severity(size_ratio),
        }

    @staticmethod
    def _get_severity(ratio: float) -> str:
        """Determine severity of size mismatch."""
        if ratio >= 0.9:
            return "NONE - chunks match config"
        elif ratio >= 0.7:
            return "MINOR - sentence breaking reduces size by 10-30%"
        elif ratio >= 0.5:
            return "MODERATE - sentence breaking reduces size by 30-50%"
        else:
            return "MAJOR - chunks significantly smaller than configured"

    @staticmethod
    def print_analysis(chunks: List[Chunk], config_chunk_size: int = 800) -> None:
        """Print formatted analysis report."""
        analysis = ChunkAnalyzer.analyze_chunks(chunks, config_chunk_size)

        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return

        print(f"\n{'='*80}")
        print("📊 Chunk Size Analysis")
        print(f"{'='*80}")
        print(f"Configured chunk size:     {analysis['config_chunk_size']} chars")
        print(f"Actual average size:       {analysis['avg_actual_size']} chars")
        print(f"Size ratio (actual/config):{analysis['size_ratio']}")
        print(f"\nChunk size range:")
        print(f"  Min: {analysis['min_chunk_size']} chars")
        print(f"  Max: {analysis['max_chunk_size']} chars")
        print(f"  Avg: {analysis['avg_actual_size']} chars")
        print(f"  Avg words per chunk: {analysis['avg_words_per_chunk']}")
        print(f"\nMismatch Status: {analysis['mismatch_severity']}")

        if analysis["has_mismatch"]:
            print(f"\n⚠️  RECOMMENDATION:")
            if analysis["size_ratio"] < 0.7:
                print(f"Your actual chunks are {int((1-analysis['size_ratio'])*100)}% smaller than configured.")
                print(f"\nChoose one:")
                print(f"  (A) Increase config chunk_size to {int(analysis['avg_actual_size'])} to match reality")
                print(f"  (B) Set break_on_sentences=False to enforce {analysis['config_chunk_size']} char limit")
                print(f"  (C) Accept current behavior (good for semantic coherence)")
            else:
                print(f"Minor size variation due to sentence boundary respecting.")
                print(f"This is usually acceptable for semantic search.")
        else:
            print(f"\n✅ Chunks match configuration well!")

        print(f"{'='*80}\n")

    @staticmethod
    def suggest_config(chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Suggest optimal config based on actual chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Suggested configuration
        """
        analysis = ChunkAnalyzer.analyze_chunks(chunks)

        if "error" in analysis:
            return {}

        avg_size = analysis["avg_actual_size"]
        avg_words = analysis["avg_words_per_chunk"]

        return {
            "suggested_chunk_size": int(avg_size),
            "suggested_overlap": int(avg_size * 0.2),  # 20% overlap
            "actual_avg_words": avg_words,
            "note": "Based on current document's actual chunk sizes"
        }
