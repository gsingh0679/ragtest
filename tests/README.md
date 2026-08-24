# Tests Directory

Comprehensive test suite for the RAG Document Chunking project.

## Running Tests

### From Project Root

```bash
# Run all tests
python run_tests.py

# Run quick tests (skip performance/memory intensive)
python run_tests.py quick

# Run specific test suite
python run_tests.py chunker      # Text chunker tests
python run_tests.py loader       # Document loader tests
python run_tests.py setup        # Setup verification
python run_tests.py memory       # Memory usage tests
python run_tests.py performance  # Performance benchmarks
```

### Using pytest

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_text_chunker.py -v

# Run specific test function
pytest tests/test_text_chunker.py::test_basic_chunking -v

# Run with markers
pytest tests/ -m "not slow" -v
```

### Direct Execution

```bash
# Run individual test file
python tests/test_setup.py
python tests/test_document_loader.py
python tests/test_text_chunker.py
python tests/test_memory_usage.py
python tests/test_performance.py
```

## Test Files

### test_setup.py
**Runtime:** <1 second  
**Purpose:** Verify Python environment and dependencies

Checks:
- Python version
- Required imports (Chroma, Ollama)
- System configuration

**Run:** `python run_tests.py setup`

### test_document_loader.py
**Runtime:** ~5 seconds  
**Purpose:** Verify document loading functionality

Tests:
- Single file loading (TXT)
- Batch directory loading
- Error handling
- File statistics

**Status:** ✅ 4/4 tests pass  
**Run:** `python run_tests.py loader`

### test_text_chunker.py
**Runtime:** ~10 seconds  
**Purpose:** Verify streaming chunk generation

Tests:
- Basic chunking (memory-efficient)
- Overlap preservation
- Sentence boundary breaking
- Token estimation
- Chunk statistics
- Real document processing
- Multiple documents

**Status:** ✅ 7/7 tests pass  
**Run:** `python run_tests.py chunker`

### test_memory_usage.py
**Runtime:** ~30 seconds  
**Purpose:** Verify memory efficiency improvements

Benchmarks:
- List approach (baseline)
- Streaming approach (optimized)
- Memory reduction verification
- Batch processor efficiency

**Results:** ✅ 98.4x less memory with streaming  
**Run:** `python run_tests.py memory`

### test_performance.py
**Runtime:** ~60 seconds  
**Purpose:** Performance benchmarking

Benchmarks:
1. Small document (100 KB)
2. Medium document (1 MB)
3. Large document (5 MB)
4. Batch processor (5 files)
5. Scalability analysis

**Results:** ✅ Streaming 1.4x-2.0x faster  
**Run:** `python run_tests.py performance`

## Test Organization

```
tests/
├── __init__.py           # Package marker
├── conftest.py           # Pytest configuration
├── README.md             # This file
├── test_setup.py         # Quick setup verification
├── test_document_loader.py
├── test_text_chunker.py
├── test_memory_usage.py
└── test_performance.py
```

## Configuration

### pytest.ini
Pytest configuration in project root:
```ini
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
```

### conftest.py
Handles import paths so tests can access `src/` module.

## Quick Test Commands

### Quick Verification (5 seconds)
```bash
python run_tests.py quick
```
Runs: setup, document_loader, text_chunker

### Full Test Suite (100+ seconds)
```bash
python run_tests.py
```
Runs: all tests including performance and memory

### Specific Component
```bash
python run_tests.py chunker    # Just chunker tests
python run_tests.py memory     # Just memory tests
python run_tests.py performance # Just performance
```

## Test Results Summary

| Test | Status | Runtime | Notes |
|------|--------|---------|-------|
| setup | ✅ Pass | <1s | Python environment |
| loader | ✅ Pass (4/4) | ~5s | Document loading |
| chunker | ✅ Pass (7/7) | ~10s | Streaming chunks |
| memory | ✅ Pass | ~30s | 98.4x improvement |
| performance | ✅ Pass | ~60s | 1.4-2.0x faster |

## Understanding Test Output

### Success Output
```
✅ PASS: test_name
```

### Failure Output
```
✗ FAIL: test_name
Error message here
```

### Performance Tests
```
100 KB document: 0.79 ms (233.0k chunks/sec)
Streaming is 2.0x FASTER ⚡
```

### Memory Tests
```
Streaming approach: 18.16 KB (PEAK)
Memory reduction: 98.4x less memory with streaming
```

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'src'`:
- Run tests from project root: `python run_tests.py`
- Or from root: `pytest tests/`
- Not from within tests folder

### Timeout Issues
If tests timeout:
- Skip performance tests: `python run_tests.py quick`
- Run individual tests: `python tests/test_setup.py`

### File Not Found Errors
- Ensure you're in project root
- Check data/ folder exists
- Run from: `/home/guru/projects/ragtest`

## Adding New Tests

1. Create `test_feature.py` in `tests/` folder
2. Import from `src.*` (relative imports work via conftest.py)
3. Use standard pytest naming: `def test_something():`
4. Run with: `pytest tests/test_feature.py -v`

Example:
```python
from src.text_chunker import TextChunker
from src.document_loader import DocumentLoader

def test_new_feature():
    loader = DocumentLoader()
    chunker = TextChunker()
    # Your test code here
    assert True
```

## CI/CD Integration

To integrate with CI/CD:

```bash
# Run quick tests (fast, minimal dependencies)
python run_tests.py quick

# Run all tests (comprehensive)
python run_tests.py

# Exit code 0 = all pass, 1 = any failure
```

## Performance Baseline

Expected performance on modern hardware:

| Test | Expected Time |
|------|---------------|
| quick | 15-20 seconds |
| all | 100-120 seconds |
| chunker | 10-15 seconds |
| memory | 30-40 seconds |
| performance | 60-80 seconds |

## Next Steps

1. Run `python run_tests.py quick` to verify setup
2. Review test results in output
3. For issues, see Troubleshooting section
4. Check individual test files for detailed behavior

---

**All tests can be run from project root.**  
**No need to navigate into tests/ folder.**
