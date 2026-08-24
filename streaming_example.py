"""
Example: Memory-efficient document chunking using generators

This demonstrates how to process large documents without loading all chunks
into memory at once. Chunks are yielded one at a time and can be processed
or saved as they're created.
"""

from pathlib import Path
from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker


def process_and_save_chunks(document_path: str):
    """
    Process a document and save chunks to a file without keeping all in memory.

    This approach:
    - Loads the document once
    - Streams chunks as they're created
    - Writes each chunk to disk immediately
    - Peak memory = 1 chunk + document content, not N chunks

    Args:
        document_path: Path to document to process
    """
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=800, overlap=150)

    # Load document (must load entire content for text splitting)
    doc = loader.load(document_path)
    print(f"Loaded {doc.source} ({doc.size_bytes} bytes)\n")

    output_file = Path("chunks_output.txt")
    chunk_count = 0
    total_chars = 0

    # Process chunks one at a time
    with open(output_file, "w") as f:
        for chunk in chunker.chunk_stream(doc):
            # Process each chunk immediately
            f.write(f"--- Chunk {chunk.chunk_index} ---\n")
            f.write(f"ID: {chunk.chunk_id}\n")
            f.write(f"Position: {chunk.start_char}-{chunk.end_char}\n")
            f.write(f"Tokens: {chunk.token_count}\n\n")
            f.write(chunk.content)
            f.write("\n\n")

            chunk_count += 1
            total_chars += len(chunk.content)

            # Print progress
            if chunk_count % 10 == 0:
                print(f"✓ Processed {chunk_count} chunks ({total_chars} chars)")

    print(f"\n✓ Saved {chunk_count} chunks to {output_file}")
    print(f"  Total characters: {total_chars}")


def process_large_directory(dir_path: str):
    """
    Process all documents in a directory with minimal memory usage.

    Each document is chunked and processed before moving to the next,
    so memory never accumulates across documents.

    Args:
        dir_path: Path to directory containing documents
    """
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=1000, overlap=200)

    print(f"Processing directory: {dir_path}\n")

    total_chunks = 0
    total_bytes = 0

    # Load documents one at a time
    for file_path in sorted(Path(dir_path).rglob("*")):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in {".txt", ".pdf", ".md", ".markdown"}
        ):
            try:
                doc = loader.load(file_path)
                total_bytes += doc.size_bytes

                # Stream chunks for this document
                doc_chunks = sum(1 for _ in chunker.chunk_stream(doc))
                total_chunks += doc_chunks

                print(f"✓ {file_path.name}: {doc_chunks} chunks")

            except Exception as e:
                print(f"✗ {file_path.name}: {e}")

    print(f"\n✓ Total: {total_chunks} chunks from {total_bytes} bytes")


def process_with_callback(document_path: str, callback):
    """
    Process chunks with a custom callback function.

    The callback is invoked for each chunk as it's created,
    allowing custom processing (save to DB, send to API, etc.)

    Args:
        document_path: Path to document
        callback: Function to call for each chunk
    """
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=800, overlap=150)

    doc = loader.load(document_path)
    print(f"Processing {doc.source}...\n")

    for chunk in chunker.chunk_stream(doc):
        # Apply custom processing
        callback(chunk)


def example_callback(chunk):
    """Example callback that could save to a database or API."""
    print(f"Chunk {chunk.chunk_index}: {len(chunk.content)} chars, {chunk.token_count} tokens")
    # In real usage, you might do:
    # - db.insert_chunk(chunk)
    # - api.post_chunk(chunk)
    # - cache.store_chunk(chunk)
    # etc.


if __name__ == "__main__":
    # Example 1: Save chunks to file (memory efficient)
    print("=" * 70)
    print("Example 1: Stream chunks to file")
    print("=" * 70 + "\n")

    # Create a sample document if it doesn't exist
    sample_file = Path("data/sample.txt")
    if sample_file.exists():
        process_and_save_chunks(sample_file)
    else:
        print(f"Sample file not found at {sample_file}")
        print("Creating a sample file...")
        sample_file.parent.mkdir(exist_ok=True)
        sample_file.write_text("This is sample text. " * 1000)
        process_and_save_chunks(sample_file)

    print("\n" + "=" * 70)
    print("Example 2: Process with callback")
    print("=" * 70 + "\n")

    if sample_file.exists():
        process_with_callback(sample_file, example_callback)

    print("\n" + "=" * 70)
    print("Memory usage tip:")
    print("=" * 70)
    print("""
When working with large documents:

✓ DO use chunk_stream() and chunk_multiple_stream()
  - Memory stays constant
  - Process/save chunks immediately

✗ DON'T use chunk() and chunk_multiple() for large files
  - All chunks are loaded into memory
  - Can cause OOM (out of memory) errors

Example:
    # GOOD - streaming
    for chunk in chunker.chunk_stream(doc):
        process(chunk)

    # BAD - loading all at once
    chunks = chunker.chunk(doc)  # allocates huge list!
    for chunk in chunks:
        process(chunk)
    """)
