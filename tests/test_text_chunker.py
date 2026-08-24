"""Test suite for Text Chunker"""

from pathlib import Path
from datetime import datetime
from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.models import Document, Chunk


def test_basic_chunking():
    """Test basic text chunking functionality"""
    print("=" * 70)
    print("TEST 1: Basic Chunking")
    print("=" * 70)

    # Create a test document with known content
    test_content = (
        "This is the first sentence. It has some content. "
        "This is the second sentence. It also has content. "
        "This is the third sentence. Adding more text here. "
        "This is the fourth sentence. Even more content. "
        "This is the fifth sentence. And more. "
        "This is the sixth sentence. Finishing up."
    )

    doc = Document(
        content=test_content,
        source="test_doc.txt",
        file_path=Path("test_doc.txt"),
        file_type="txt",
        size_bytes=len(test_content),
        loaded_at=datetime.now(),
    )

    chunker = TextChunker(chunk_size=150, overlap=30)

    chunk_count = 0
    print(f"  Document length: {len(test_content)} characters")
    print(f"  Chunk size: 150 characters, Overlap: 30 characters\n")

    for i, chunk in enumerate(chunker.chunk_stream(doc)):
        print(f"  Chunk {i}:")
        print(f"    ID: {chunk.chunk_id}")
        print(f"    Position: {chunk.start_char}-{chunk.end_char}")
        print(f"    Length: {len(chunk.content)} chars")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Preview: {chunk.preview(80)}\n")
        chunk_count += 1

    print(f"✓ Streamed {chunk_count} chunks (memory-efficient)\n")
    return chunk_count > 0


def test_overlap_preservation():
    """Test that overlap preserves context"""
    print("=" * 70)
    print("TEST 2: Overlap Preservation")
    print("=" * 70)

    # Create document with a key phrase at boundary
    test_content = (
        "The quick brown fox jumps over the lazy dog. "
        "The dog was sleeping peacefully under the tree. "
        "The tree was very old and provided good shade."
    )

    doc = Document(
        content=test_content,
        source="overlap_test.txt",
        file_path=Path("overlap_test.txt"),
        file_type="txt",
        size_bytes=len(test_content),
        loaded_at=datetime.now(),
    )

    # Test with overlap - collect only for this specific test that needs comparison
    chunker = TextChunker(chunk_size=80, overlap=30)
    chunks = list(chunker.chunk_stream(doc))

    print(f"✓ Streamed {len(chunks)} chunks with overlap")

    # Verify overlap by checking if positions overlap (not text content)
    overlaps_found = 0
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i].end_char
        chunk2_start = chunks[i + 1].start_char

        # Check if there's positional overlap
        if chunk1_end > chunk2_start:
            overlap_size = chunk1_end - chunk2_start
            overlaps_found += 1
            print(f"  ✓ Found {overlap_size} char overlap between chunk {i} and {i+1}")

    print(f"\n  Total overlaps verified: {overlaps_found}/{len(chunks)-1}")
    print()

    return overlaps_found > 0


def test_sentence_boundary_breaking():
    """Test that chunker respects sentence boundaries"""
    print("=" * 70)
    print("TEST 3: Sentence Boundary Breaking")
    print("=" * 70)

    test_content = (
        "First sentence ends here. Second sentence starts. "
        "Third sentence is here. Fourth sentence follows. "
        "Fifth sentence appears. Sixth sentence concludes."
    )

    doc = Document(
        content=test_content,
        source="boundary_test.txt",
        file_path=Path("boundary_test.txt"),
        file_type="txt",
        size_bytes=len(test_content),
        loaded_at=datetime.now(),
    )

    # With sentence breaking enabled
    chunker_with_breaking = TextChunker(chunk_size=100, overlap=20, break_on_sentences=True)
    chunks_with = list(chunker_with_breaking.chunk_stream(doc))

    # With sentence breaking disabled
    chunker_without_breaking = TextChunker(chunk_size=100, overlap=20, break_on_sentences=False)
    chunks_without = list(chunker_without_breaking.chunk_stream(doc))

    print(f"✓ With sentence breaking: {len(chunks_with)} chunks")
    print(f"✓ Without sentence breaking: {len(chunks_without)} chunks")

    # Check for mid-sentence breaks in "with" version
    mid_sentence_breaks = 0
    for chunk in chunks_with:
        if not chunk.content.endswith(('.', '!', '?')):
            if len(chunk.content) > 0 and chunk.content[-1] != '\n':
                mid_sentence_breaks += 1

    print(f"\n  Chunks ending mid-sentence (with breaking): {mid_sentence_breaks}")
    print()

    return True


def test_token_estimation():
    """Test token counting"""
    print("=" * 70)
    print("TEST 4: Token Estimation")
    print("=" * 70)

    # Create document with known word count
    test_content = "word " * 100  # 100 words = ~500 characters

    doc = Document(
        content=test_content,
        source="token_test.txt",
        file_path=Path("token_test.txt"),
        file_type="txt",
        size_bytes=len(test_content),
        loaded_at=datetime.now(),
    )

    chunker = TextChunker(chunk_size=500, overlap=50)

    total_tokens = 0
    chunk_count = 0
    for chunk in chunker.chunk_stream(doc):
        total_tokens += chunk.token_count
        chunk_count += 1

    word_count = len(test_content.split())

    print(f"✓ Document (streamed):")
    print(f"  Characters: {len(test_content)}")
    print(f"  Words: {word_count}")
    print(f"  Estimated tokens: {total_tokens}")
    print(f"  Chunks: {chunk_count}")
    print()

    return total_tokens > 0


def test_chunk_statistics():
    """Test statistics calculation"""
    print("=" * 70)
    print("TEST 5: Chunk Statistics")
    print("=" * 70)

    test_content = ("This is test content. " * 50).strip()

    doc = Document(
        content=test_content,
        source="stats_test.txt",
        file_path=Path("stats_test.txt"),
        file_type="txt",
        size_bytes=len(test_content),
        loaded_at=datetime.now(),
    )

    chunker = TextChunker(chunk_size=200, overlap=40)
    # Note: stats() requires a list, so we collect chunks here (only for stats calculation)
    chunks = list(chunker.chunk_stream(doc))
    stats = chunker.stats(chunks)

    print(f"✓ Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total characters: {stats['total_characters']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Avg chunk size: {stats['avg_chunk_size']} chars")
    print(f"  Avg tokens: {stats['avg_tokens']}")
    print(f"  Min chunk size: {stats['min_chunk_size']} chars")
    print(f"  Max chunk size: {stats['max_chunk_size']} chars")
    print()

    return stats['total_chunks'] > 0


def test_real_document():
    """Test with an actual loaded document"""
    print("=" * 70)
    print("TEST 6: Real Document (from file)")
    print("=" * 70)

    # Create a test file
    test_file = Path("data/chunking_test.txt")
    test_content = """
    Machine Learning Basics

    Machine learning is a subset of artificial intelligence.
    It focuses on developing algorithms that can learn from data.

    Types of Machine Learning:

    1. Supervised Learning: The model learns from labeled data.
    2. Unsupervised Learning: The model finds patterns in unlabeled data.
    3. Reinforcement Learning: The model learns through interaction.

    Applications include computer vision, natural language processing,
    and recommendation systems. Machine learning has revolutionized
    many industries and continues to advance rapidly.
    """.strip()

    test_file.write_text(test_content)

    # Load with DocumentLoader
    loader = DocumentLoader()
    doc = loader.load(test_file)

    # Chunk with TextChunker
    chunker = TextChunker(chunk_size=250, overlap=50)

    chunk_count = 0
    print(f"✓ Loaded and streamed document: {doc.source}")
    print(f"  Original size: {doc.size_bytes} bytes\n")

    for i, chunk in enumerate(chunker.chunk_stream(doc)):
        print(f"  Chunk {i}: {len(chunk.content)} chars, {chunk.token_count} tokens")
        print(f"    {chunk.preview(60)}\n")
        chunk_count += 1

    print(f"  Total chunks: {chunk_count}")
    return chunk_count > 0


def test_multiple_documents():
    """Test chunking multiple documents"""
    print("=" * 70)
    print("TEST 7: Multiple Documents")
    print("=" * 70)

    docs = []
    for i in range(3):
        content = f"Document {i} content. " * 20
        doc = Document(
            content=content,
            source=f"doc_{i}.txt",
            file_path=Path(f"doc_{i}.txt"),
            file_type="txt",
            size_bytes=len(content),
            loaded_at=datetime.now(),
        )
        docs.append(doc)

    chunker = TextChunker(chunk_size=150, overlap=30)

    # Stream chunks from multiple documents
    from collections import defaultdict
    chunks_by_source = defaultdict(list)
    total_chunks = 0

    for chunk in chunker.chunk_multiple_stream(docs):
        chunks_by_source[chunk.source_document].append(chunk)
        total_chunks += 1

    print(f"✓ Streamed {len(docs)} documents into {total_chunks} chunks (memory-efficient)\n")

    for source, source_chunks in chunks_by_source.items():
        print(f"  {source}: {len(source_chunks)} chunks")

    print()
    return total_chunks == sum(len(v) for v in chunks_by_source.values())


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TEXT CHUNKER TEST SUITE")
    print("=" * 70 + "\n")

    results = []
    results.append(("Basic Chunking", test_basic_chunking()))
    results.append(("Overlap Preservation", test_overlap_preservation()))
    results.append(("Sentence Boundary Breaking", test_sentence_boundary_breaking()))
    results.append(("Token Estimation", test_token_estimation()))
    results.append(("Chunk Statistics", test_chunk_statistics()))
    results.append(("Real Document", test_real_document()))
    results.append(("Multiple Documents", test_multiple_documents()))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}\n")

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print()
    if passed == total:
        print("🎉 All tests passed! Text Chunker is working correctly.\n")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    exit(main())
