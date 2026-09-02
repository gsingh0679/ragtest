#!/usr/bin/env python
"""
Comprehensive RAG Pipeline Test Suite
Tests embedding quality, retrieval quality, chunk size, and end-to-end queries

FIX: Manually embed queries before passing to ChromaDB to avoid embedding function issues
"""

import chromadb
from ollama import Client
import numpy as np
from typing import Dict, List, Tuple


class RAGPipelineTester:
    """Test suite for RAG pipeline"""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "ragtest_kb"):
        """Initialize tester with ChromaDB connection"""
        self.db_path = db_path
        self.collection_name = collection_name
        self.ollama_client = Client(host="http://localhost:11434")
        
        try:
            self.client_db = chromadb.PersistentClient(path=db_path)
            self.collection = self.client_db.get_collection(name=collection_name)
            print(f"✓ Connected to ChromaDB at {db_path}")
            print(f"✓ Loaded collection: {collection_name}")
            print(f"✓ Total documents in collection: {self.collection.count()}\n")
        except Exception as e:
            print(f"✗ Error connecting to ChromaDB: {e}")
            raise

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using Ollama nomic model"""
        response = self.ollama_client.embed(model="nomic-embed-text:latest", input=text)
        return np.array(response["embeddings"][0])

    def test_1_embedding_quality(self) -> Dict:
        """TEST 1: Check embedding model quality and discrimination"""
        print("=" * 80)
        print("TEST 1: EMBEDDING QUALITY CHECK")
        print("=" * 80)

        queries = {
            "80C": "Section 80C deduction limit for life insurance",
            "80D": "Section 80D deduction limit for medical insurance",
            "Tax Rate": "Tax rate slab for individuals earning 50 lakhs",
            "Schedule AA": "Schedule AA computation of total income",
            "Relief 89": "Relief u/s 89 income averaging for salaried"
        }

        # Generate embeddings
        embeddings = {}
        for label, query_text in queries.items():
            embeddings[label] = self.embed_text(query_text)
            print(f"✓ Generated embedding for {label}")

        print("\n" + "-" * 80)
        print("SIMILARITY COMPARISONS:")
        print("-" * 80)

        comparisons = [
            ("80C", "80D", "should be 0.80-0.90"),
            ("80C", "Tax Rate", "should be 0.50-0.65"),
            ("80C", "Schedule AA", "should be 0.55-0.70"),
            ("80D", "Relief 89", "should be 0.55-0.70"),
            ("Tax Rate", "Schedule AA", "should be 0.50-0.65"),
        ]

        results = {}
        for label1, label2, expected in comparisons:
            sim = self.cos_sim(embeddings[label1], embeddings[label2])
            results[f"{label1}_vs_{label2}"] = sim
            status = "✓" if (0.50 <= sim <= 0.95) else "⚠"
            print(f"{status} {label1:15} vs {label2:15} = {sim:.4f} ({expected})")

        print("\n" + "=" * 80)
        print("DIAGNOSIS:")
        print("=" * 80)

        if self.cos_sim(embeddings["80C"], embeddings["80D"]) > 0.93:
            print("✗ PROBLEM: 80C vs 80D similarity is too high (>0.93)")
            print("  → Embedding model can't distinguish between different tax sections")
            results["diagnosis"] = "POOR_DISCRIMINATION"
        elif self.cos_sim(embeddings["80C"], embeddings["80D"]) > 0.85:
            print("⚠ WARNING: 80C vs 80D similarity is borderline (0.85-0.93)")
            print("  → Model is generic; might retrieve noise")
            results["diagnosis"] = "BORDERLINE_DISCRIMINATION"
        else:
            print("✓ GOOD: 80C vs 80D have good separation")
            results["diagnosis"] = "GOOD_DISCRIMINATION"

        return results

    def test_2_retrieval_quality(self) -> Dict:
        """TEST 2: Check retrieval quality and relevance"""
        print("\n" + "=" * 80)
        print("TEST 2: RETRIEVAL QUALITY CHECK")
        print("=" * 80)

        test_queries = [
            {
                "query": "Section 80C deduction limit",
                "expected_keywords": ["80C", "deduction", "limit"],
                "unexpected_keywords": ["80D", "80E", "Schedule AA"],
                "description": "Deductions for life insurance and investments"
            },
            {
                "query": "Schedule AA computation of total income",
                "expected_keywords": ["Schedule AA", "computation", "income"],
                "unexpected_keywords": ["80C", "80D", "tax rate"],
                "description": "Applicable for certain individuals"
            },
            {
                "query": "ITR2 applicable person",
                "expected_keywords": ["ITR2", "applicable", "person"],
                "unexpected_keywords": ["ITR1", "ITR4", "ITR5"],
                "description": "Who should file ITR2"
            },
        ]

        overall_relevant = 0
        overall_total = 0
        detailed_results = []

        for test_idx, test in enumerate(test_queries, 1):
            query_text = test["query"]
            expected = test["expected_keywords"]
            unexpected = test["unexpected_keywords"]

            print(f"\n{'=' * 80}")
            print(f"Query {test_idx}: {query_text}")
            print(f"Description: {test['description']}")
            print(f"{'=' * 80}")

            try:
                # FIX: Manually embed the query first
                query_embedding = self.embed_text(query_text)

                # Query with pre-computed embedding (NOT query_texts)
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],  # ← USE THIS
                    n_results=5,
                    include=["documents", "distances", "metadatas"]
                )

                documents = results["documents"][0]
                distances = results["distances"][0]

                print(f"\nTop-5 Results:")
                print("-" * 80)

                for rank, (doc, dist) in enumerate(zip(documents, distances), 1):
                    similarity = 1 - dist

                    print(f"\n[Rank {rank}] Distance: {dist:.4f} | Similarity: {similarity:.4f}")
                    print(f"Content: {doc[:250]}...")

                    # Check relevance
                    has_expected = sum(1 for keyword in expected
                                      if keyword.lower() in doc.lower())
                    has_unexpected = sum(1 for keyword in unexpected
                                        if keyword.lower() in doc.lower())

                    if has_expected >= 2 and has_unexpected == 0:
                        status = "✓ RELEVANT"
                        relevance_score = 1.0
                        overall_relevant += 1
                    elif has_expected >= 2 and has_unexpected >= 1:
                        status = "⚠ PARTIALLY RELEVANT (has noise)"
                        relevance_score = 0.5
                        overall_relevant += 0.5
                    elif has_expected >= 1:
                        status = "⚠ MARGINALLY RELEVANT"
                        relevance_score = 0.25
                        overall_relevant += 0.25
                    else:
                        status = "✗ NOT RELEVANT"
                        relevance_score = 0.0

                    print(f"Status: {status}")
                    
                    detailed_results.append({
                        "query": query_text,
                        "rank": rank,
                        "similarity": similarity,
                        "relevance": relevance_score
                    })

                    overall_total += 1

            except Exception as e:
                print(f"✗ Error in query {test_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 80)
        print("OVERALL RESULTS:")
        print("=" * 80)
        
        relevance_pct = 100 * overall_relevant / overall_total if overall_total > 0 else 0
        print(f"Relevant chunks: {overall_relevant}/{overall_total} ({relevance_pct:.1f}%)")

        if overall_relevant / overall_total > 0.8:
            print("✓ Retrieval is working well")
            diagnosis = "GOOD_RETRIEVAL"
        elif overall_relevant / overall_total > 0.5:
            print("⚠ Retrieval needs tuning")
            print("  Recommendation: Check chunk size or improve embedding model")
            diagnosis = "FAIR_RETRIEVAL"
        else:
            print("✗ Retrieval is problematic")
            diagnosis = "POOR_RETRIEVAL"

        return {
            "relevance_percentage": relevance_pct,
            "diagnosis": diagnosis,
            "detailed_results": detailed_results
        }

    def test_3_chunk_size(self) -> Dict:
        """TEST 3: Analyze chunk size distribution"""
        print("\n" + "=" * 80)
        print("TEST 3: CHUNK SIZE ANALYSIS")
        print("=" * 80)

        # Get all documents from collection
        results = self.collection.get(include=["documents"])
        all_docs = results["documents"]

        # Analyze chunk sizes
        chunk_sizes = [len(doc.split()) for doc in all_docs]

        print(f"\nChunk Size Statistics:")
        print(f"  Total chunks: {len(all_docs)}")
        print(f"  Min size: {min(chunk_sizes)} words")
        print(f"  Max size: {max(chunk_sizes)} words")
        print(f"  Average size: {np.mean(chunk_sizes):.0f} words")
        print(f"  Median size: {np.median(chunk_sizes):.0f} words")
        print(f"  Std Dev: {np.std(chunk_sizes):.0f} words")

        # Distribution
        print(f"\nSize Distribution:")
        bins = [
            (0, 100, "<100"),
            (100, 200, "100-200"),
            (200, 300, "200-300"),
            (300, 400, "300-400"),
            (400, 600, "400-600"),
            (600, 800, "600-800"),
            (800, float('inf'), ">800"),
        ]

        for low, high, label in bins:
            count = sum(1 for s in chunk_sizes if low <= s < high)
            pct = 100 * count / len(chunk_sizes)
            if count > 0:
                print(f"  {label:12} words: {count:4} chunks ({pct:5.1f}%)")

        print("\n" + "=" * 80)
        print("DIAGNOSIS:")
        print("=" * 80)

        avg_size = np.mean(chunk_sizes)

        if avg_size > 600:
            print(f"⚠ WARNING: Average chunk size is {avg_size:.0f} words (potentially large)")
            print("  → May mix multiple concepts in single chunk")
            diagnosis = "TOO_LARGE"
        elif avg_size > 400:
            print(f"⚠ BORDERLINE: Average chunk size is {avg_size:.0f} words")
            print("  → Acceptable but consider reducing for better isolation")
            diagnosis = "BORDERLINE"
        elif avg_size < 50:
            print(f"⚠ WARNING: Average chunk size is {avg_size:.0f} words (too small)")
            print("  → May lose context and create fragmented chunks")
            diagnosis = "TOO_SMALL"
        else:
            print(f"✓ GOOD: Average chunk size is {avg_size:.0f} words (reasonable)")
            diagnosis = "OPTIMAL"

        return {
            "total_chunks": len(all_docs),
            "average_size": avg_size,
            "min_size": min(chunk_sizes),
            "max_size": max(chunk_sizes),
            "diagnosis": diagnosis
        }

    def test_4_actual_queries(self, queries: List[str] = None) -> Dict:
        """TEST 4: Test with actual queries"""
        print("\n" + "=" * 80)
        print("TEST 4: ACTUAL QUERY TESTS")
        print("=" * 80)

        if queries is None:
            queries = [
                "Section 80C deduction limit",
                "What is Schedule AA?",
                "Who should file ITR2?"
            ]

        results_summary = []

        for idx, query in enumerate(queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Query {idx}: {query}")
            print(f"{'=' * 80}")

            try:
                # FIX: Manually embed the query first
                query_embedding = self.embed_text(query)

                # Query with pre-computed embedding (NOT query_texts)
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],  # ← USE THIS
                    n_results=3,
                    include=["documents", "distances"]
                )

                documents = results["documents"][0]
                distances = results["distances"][0]

                print(f"\nTop-3 Results:")
                print("-" * 80)

                query_results = []
                for rank, (doc, dist) in enumerate(zip(documents, distances), 1):
                    similarity = 1 - dist
                    print(f"\n[Result {rank}] Similarity: {similarity:.4f}")
                    print(f"{doc}\n")
                    
                    query_results.append({
                        "rank": rank,
                        "similarity": similarity,
                        "document": doc[:100]
                    })

                results_summary.append({
                    "query": query,
                    "results": query_results,
                    "status": "✓ SUCCESS"
                })

            except Exception as e:
                print(f"✗ Error querying: {e}")
                import traceback
                traceback.print_exc()
                results_summary.append({
                    "query": query,
                    "error": str(e),
                    "status": "✗ FAILED"
                })

        print("\n" + "=" * 80)
        print("QUERY TEST SUMMARY:")
        print("=" * 80)
        
        for item in results_summary:
            print(f"\n{item['status']} - {item['query']}")
            if "error" in item:
                print(f"  Error: {item['error']}")
            else:
                for result in item["results"]:
                    print(f"  [{result['rank']}] Similarity: {result['similarity']:.4f}")

        return {"queries": results_summary}

    def run_all_tests(self, custom_queries: List[str] = None) -> Dict:
        """Run all tests and generate report"""
        print("\n" + "=" * 80)
        print("🧪 RAG PIPELINE COMPREHENSIVE TEST SUITE")
        print("=" * 80)

        all_results = {
            "test_1_embedding_quality": self.test_1_embedding_quality(),
            "test_2_retrieval_quality": self.test_2_retrieval_quality(),
            "test_3_chunk_size": self.test_3_chunk_size(),
            "test_4_actual_queries": self.test_4_actual_queries(custom_queries)
        }

        self.print_final_report(all_results)
        return all_results

    def print_final_report(self, results: Dict) -> None:
        """Print final test report with recommendations"""
        print("\n\n" + "=" * 80)
        print("📊 FINAL TEST REPORT")
        print("=" * 80)

        # Test 1 Summary
        print("\n[TEST 1] Embedding Quality:")
        diagnosis1 = results["test_1_embedding_quality"].get("diagnosis", "UNKNOWN")
        if diagnosis1 == "GOOD_DISCRIMINATION":
            print("  ✓ PASS - Model can distinguish concepts well")
        elif diagnosis1 == "BORDERLINE_DISCRIMINATION":
            print("  ⚠ WARNING - Model is generic, may retrieve noise")
        else:
            print("  ✗ FAIL - Model cannot distinguish concepts")

        # Test 2 Summary
        print("\n[TEST 2] Retrieval Quality:")
        relevance = results["test_2_retrieval_quality"].get("relevance_percentage", 0)
        diagnosis2 = results["test_2_retrieval_quality"].get("diagnosis", "UNKNOWN")
        print(f"  Relevance: {relevance:.1f}%")
        if diagnosis2 == "GOOD_RETRIEVAL":
            print("  ✓ PASS - Retrieval is working well")
        elif diagnosis2 == "FAIR_RETRIEVAL":
            print("  ⚠ WARNING - Retrieval needs tuning")
        else:
            print("  ✗ FAIL - Retrieval is problematic")

        # Test 3 Summary
        print("\n[TEST 3] Chunk Size:")
        chunk_stats = results["test_3_chunk_size"]
        diagnosis3 = chunk_stats.get("diagnosis", "UNKNOWN")
        print(f"  Average: {chunk_stats.get('average_size', 0):.0f} words")
        if diagnosis3 == "OPTIMAL":
            print("  ✓ PASS - Chunk size is reasonable")
        elif diagnosis3 in ["BORDERLINE", "TOO_LARGE", "TOO_SMALL"]:
            print(f"  ⚠ WARNING - {diagnosis3}: Consider adjustment")
        else:
            print("  ✗ FAIL - Chunk size needs review")

        # Test 4 Summary
        print("\n[TEST 4] Actual Queries:")
        queries_results = results["test_4_actual_queries"].get("queries", [])
        passed = sum(1 for q in queries_results if q.get("status") == "✓ SUCCESS")
        total = len(queries_results)
        print(f"  {passed}/{total} queries succeeded")
        if passed == total:
            print("  ✓ PASS - All queries executed successfully")
        else:
            print(f"  ⚠ WARNING - {total - passed} queries failed")

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)

        if diagnosis1 != "GOOD_DISCRIMINATION":
            print("1. Consider domain-specific embeddings for financial/tax documents")
            print("   - Or implement metadata filtering by section number")

        if diagnosis2 != "GOOD_RETRIEVAL":
            print("2. Implement re-ranking with LLM to filter noise after retrieval")
            print("   - Or reduce chunk size to isolate concepts better")

        if diagnosis3 != "OPTIMAL":
            print("3. Adjust chunk_size configuration to match actual behavior")
            print("   - Or review TextChunker logic (break_on_sentences setting)")

        print("\n" + "=" * 80)


def main():
    """Main entry point"""
    # Initialize tester
    tester = RAGPipelineTester(db_path="./chroma_db", collection_name="ragtest_kb")

    # Optional: Add your custom queries here
    custom_queries = [
        "Section 80C deduction limit",
        "What is Schedule AA?",
        "Tax rate for individuals",
        "ITR2 applicable person",
        "Relief u/s 89"
    ]

    # Run all tests
    results = tester.run_all_tests(custom_queries=custom_queries)


if __name__ == "__main__":
    main()
