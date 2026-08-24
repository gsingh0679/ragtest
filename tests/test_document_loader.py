"""Test suite for Document Loader"""

from src.document_loader import DocumentLoader
from pathlib import Path


def test_single_file():
    """Test loading a single file"""
    print("=" * 70)
    print("TEST 1: Load Single TXT File")
    print("=" * 70)

    loader = DocumentLoader()

    # Create a test TXT file
    test_file = Path("data/test.txt")
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text(
        "This is a test document.\n"
        "RAG is Retrieval-Augmented Generation.\n"
        "It combines retrieval with language generation."
    )

    try:
        doc = loader.load(test_file)
        print(f"✓ Successfully loaded: {doc.source}")
        print(f"  File type: {doc.file_type}")
        print(f"  File size: {doc.size_bytes} bytes")
        print(f"  Content length: {len(doc.content)} characters")
        print(f"  Preview:\n    {doc.preview(100)}\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


def test_batch_loading():
    """Test loading multiple files from directory"""
    print("=" * 70)
    print("TEST 2: Batch Load from Directory")
    print("=" * 70)

    loader = DocumentLoader()

    # Create test files
    Path("data/doc1.txt").write_text("First document about machine learning.")
    Path("data/doc2.txt").write_text(
        "Second document about natural language processing."
    )
    Path("data/doc3.md").write_text("# Markdown Document\n\nThis is markdown content.")

    try:
        docs = loader.load_directory("data/")
        print(f"✓ Successfully loaded {len(docs)} documents\n")
        for i, doc in enumerate(docs, 1):
            stats = doc.stats()
            print(f"  Document {i}: {doc.source}")
            print(f"    Type: {stats['file_type']}")
            print(f"    Size: {stats['size_bytes']} bytes")
            print(f"    Words: {stats['word_count']}")
        print()
        return len(docs) >= 3
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("=" * 70)
    print("TEST 3: Error Handling")
    print("=" * 70)

    loader = DocumentLoader()
    tests_passed = 0

    # Test 1: Missing file
    try:
        loader.load("data/nonexistent.txt")
        print("✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ Correctly caught missing file")
        tests_passed += 1

    # Test 2: Unsupported format (create file first to test format validation)
    unsupported_file = Path("data/test.unknown")
    unsupported_file.write_text("dummy content")
    try:
        loader.load(unsupported_file)
    except ValueError as e:
        print(f"✓ Correctly caught unsupported format")
        tests_passed += 1
    finally:
        unsupported_file.unlink()  # Clean up

    # Test 3: Not a directory
    try:
        loader.load_directory("data/test.txt")
    except NotADirectoryError as e:
        print(f"✓ Correctly caught non-directory path")
        tests_passed += 1

    print()
    return tests_passed == 3


def test_file_statistics():
    """Test document statistics"""
    print("=" * 70)
    print("TEST 4: Document Statistics")
    print("=" * 70)

    loader = DocumentLoader()

    # Create a test file
    content = "The quick brown fox jumps over the lazy dog. " * 5
    test_file = Path("data/stats_test.txt")
    test_file.write_text(content)

    try:
        doc = loader.load(test_file)
        stats = doc.stats()

        print(f"✓ Document statistics for: {stats['source']}")
        print(f"  File size: {stats['size_bytes']} bytes")
        print(f"  Content length: {stats['content_length']} characters")
        print(f"  Word count: {stats['word_count']} words")
        print(f"  Loaded at: {stats['loaded_at']}\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("DOCUMENT LOADER TEST SUITE")
    print("=" * 70 + "\n")

    results = []
    results.append(("Single File Loading", test_single_file()))
    results.append(("Batch Directory Loading", test_batch_loading()))
    results.append(("Error Handling", test_error_handling()))
    results.append(("File Statistics", test_file_statistics()))

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
        print("🎉 All tests passed! Document Loader is working correctly.\n")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    exit(main())
