# RAG System - Semantic Search & Knowledge Base

A production-ready Retrieval-Augmented Generation (RAG) system with memory-efficient document processing, semantic search, and optional LLM integration.

## Overview

**Complete RAG pipeline:**
1. 📄 **Document Loading** — Load PDF, TXT, Markdown files
2. ✂️ **Memory-Efficient Chunking** — Stream chunks (98.4x less memory)
3. 🔗 **Embedding Generation** — Ollama, HuggingFace, or OpenAI
4. 💾 **Knowledge Base** — Store chunks in Chroma vector database
5. 🔍 **Semantic Search** — Query and retrieve relevant chunks
6. 🤖 **Answer Generation** — Optional LLM-based responses (Ollama)

## Quick Start

### Setup (First Time Only)

```bash
# 1. Start Ollama in another terminal
ollama serve

# 2. Verify Ollama setup
python test_ollama.py

# 3. Build knowledge base from documents
python main.py build --kb-name ragtest_kb --data-dir ./data
```

### 3 Ways to Query

#### Option A: Web UI (Recommended for Interactive Use)

```bash
streamlit run assistant.py
# Opens browser at http://localhost:8501
# Features: Chat interface, real-time results, configurable settings
```

#### Option B: CLI (Quick One-Off Queries)

```bash
# Simple retrieval
python main.py query "What are the key features?"

# With LLM answer
python main.py query "What are the key features?" --use-llm

# Custom settings
python main.py query "What is RAG?" --top-k 10 --min-score 0.5 --use-llm
```

#### Option C: Python API (Programmatic Access)

```python
from src.knowledge_base.builder import KnowledgeBaseBuilder
from src.retrieval import QueryEngine
from src.embeddings.factory import EmbeddingsFactory
import chromadb

# Build knowledge base
builder = KnowledgeBaseBuilder(kb_name="ragtest_kb")
stats = builder.build_from_directory("./data")

# Query knowledge base
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="ragtest_kb")
embeddings = EmbeddingsFactory.create_ollama()
engine = QueryEngine(collection, embeddings, top_k=5)

response = engine.query("What is RAG?")
for result in response.results:
    print(f"Score: {result.similarity_score:.2%}")
    print(f"Source: {result.source}")
    print(result.preview(150))
```

## Key Features

### Document Processing
| Feature | Benefit |
|---------|---------|
| **Memory Efficient** | 98.4x less memory than list-based approaches |
| **Fast** | 1.4-2.0x faster chunking speed |
| **Scalable** | Handles documents up to available RAM |
| **Streaming** | Process chunks one-at-a-time, no accumulation |
| **Sentence-Aware** | Respects sentence boundaries for better chunks |

### Query & Retrieval
| Feature | Benefit |
|---------|---------|
| **Semantic Search** | Find relevant chunks using embeddings |
| **Multiple Providers** | Ollama, HuggingFace, OpenAI embeddings |
| **Similarity Scoring** | Ranked results with relevance scores |
| **Flexible Filtering** | Configurable top_k and min_score thresholds |
| **LLM Integration** | Optional answer generation with Ollama |
| **Context Formatting** | Ready-to-use context for LLM prompts |

## Performance Metrics

Benchmarked on real documents:

```
100 KB document:  0.79 ms  (233k chunks/sec) - 2.0x faster ⚡
1 MB document:    7.55 ms  (215k chunks/sec) - 1.4x faster ⚡
5 MB document:    40 ms    (205k chunks/sec) - Only option that works ✓

Memory (1 MB doc): 18 KB peak (vs 1.75 MB with list approach)
Scaling:          Linear with document size (10x size = 10.4x time)
```

## Project Structure

```
ragtest/
├── src/
│   ├── core/                  # Document loading & chunking
│   │   ├── document_loader.py    # Load PDF, TXT, Markdown
│   │   ├── text_chunker.py       # Streaming chunk generation
│   │   └── batch_processor.py    # Batch processing for directories
│   │
│   ├── embeddings/            # Embedding providers
│   │   ├── base.py               # Abstract base class
│   │   ├── factory.py            # Factory pattern for creation
│   │   └── implementations.py    # Ollama, HuggingFace, OpenAI
│   │
│   ├── knowledge_base/        # KB building
│   │   └── builder.py            # Build KB from documents
│   │
│   ├── retrieval/             # Query layer
│   │   ├── query_engine.py       # Semantic search & retrieval
│   │   └── __init__.py
│   │
│   ├── llm/                   # LLM integration
│   │   ├── ollama_client.py      # Ollama for answer generation
│   │   └── __init__.py
│   │
│   ├── models.py              # Data models (Document, Chunk)
│   └── __init__.py
│
├── tests/                     # Test suite
│   ├── test_text_chunker.py      # Chunker tests (7/7 pass)
│   ├── test_document_loader.py   # Loader tests (4/4 pass)
│   ├── test_memory_usage.py      # Memory verification
│   ├── test_performance.py       # Speed benchmarks
│   └── conftest.py
│
├── data/                      # Sample documents
├── chroma_db/                 # Vector database (created after build)
│
├── config.yaml                # Configuration file
├── main.py                    # CLI entry point
├── assistant.py               # Streamlit web UI (interactive chat)
├── rag_test_suite.py          # End-to-end RAG pipeline tests
├── test_ollama.py             # Ollama setup verification
│
├── streaming_example.py       # Usage examples
├── OPTIMIZATION_GUIDE.md      # Complete optimization guide
├── SETUP_GUIDE.md             # Setup and troubleshooting guide
└── README.md                  # This file
```

## Core APIs

### TextChunker - Single Document

```python
chunker = TextChunker(chunk_size=800, overlap=150, break_on_sentences=True)

# Stream chunks (memory-efficient)
for chunk in chunker.chunk_stream(doc):
    process(chunk)

# Multiple documents
for chunk in chunker.chunk_multiple_stream(docs):
    process(chunk)
```

### BatchProcessor - Directory Processing

```python
processor = BatchProcessor(chunk_size=800, overlap=150)

# Method 1: Stream all chunks
for chunk in processor.stream_chunks("data/documents/"):
    process(chunk)

# Method 2: Batch processing (recommended for DB)
def save_batch(chunks):
    db.insert_many(chunks)

processor.process_with_buffer("data/documents/", save_batch, batch_size=100)

# Method 3: Single file with callback
processor.process_file("file.pdf", on_chunk=lambda chunk: process(chunk))
```

### QueryEngine - Semantic Search

```python
from src.retrieval import QueryEngine
from src.embeddings.factory import EmbeddingsFactory
import chromadb

# Initialize embeddings and Chroma connection
embeddings = EmbeddingsFactory.create_ollama(model="nomic-embed-text:latest")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="ragtest_kb")

# Create query engine
engine = QueryEngine(
    chroma_collection=collection,
    embeddings=embeddings,
    top_k=5,
    min_score=0.3
)

# Query and get results
response = engine.query("What is RAG?")

# Results include: chunk_id, content, source, similarity_score, etc.
for result in response.results:
    print(f"[{result.source}] Score: {result.similarity_score:.2%}")
    print(result.content)

# Get formatted context for LLM
context = engine.get_context("What is RAG?", top_k=5)

# Pretty print results
engine.print_results(response, show_score=True)
```

### OllamaClient - LLM Answer Generation

```python
from src.llm import OllamaClient

llm = OllamaClient(model="llama2", base_url="http://localhost:11434")

# Generate answer from context
answer = llm.generate_answer(
    question="What is RAG?",
    context=retrieved_context,
    temperature=0.7
)

print(answer)
```

## CLI Usage

### Build Command

```bash
# Basic build
python main.py build --kb-name ragtest_kb --data-dir ./data

# With custom settings
python main.py build \
  --kb-name my_kb \
  --data-dir ./documents \
  --chunk-size 1000 \
  --overlap 200 \
  --model nomic-embed-text:latest \
  --db-path ./vector_db
```

**Options:**
- `--kb-name` — Knowledge base name (default: ragtest_kb)
- `--data-dir` — Directory with documents (default: ./data)
- `--model` — Embedding model (default: nomic-embed-text:latest)
- `--chunk-size` — Chunk size in characters (default: 800)
- `--overlap` — Overlap between chunks (default: 150)
- `--db-path` — Chroma database path (default: ./chroma_db)

### Query Command

```bash
# Basic query (retrieval only)
python main.py query "What is RAG?"

# Customize retrieval
python main.py query "What is RAG?" --top-k 10 --min-score 0.5

# With LLM answer generation
python main.py query "What is RAG?" --use-llm

# Full example
python main.py query "What is RAG?" \
  --kb-name ragtest_kb \
  --embedding-model nomic-embed-text:latest \
  --top-k 5 \
  --min-score 0.3 \
  --use-llm \
  --llm-model llama2 \
  --temperature 0.7
```

**Options:**
- `query` — Your search query (required)
- `--kb-name` — Knowledge base name (default: ragtest_kb)
- `--embedding-model` — Model for query embedding (default: nomic-embed-text:latest)
- `--top-k` — Number of results (default: 5)
- `--min-score` — Minimum similarity threshold (default: 0.3)
- `--use-llm` — Generate LLM answer
- `--llm-model` — Model for LLM (default: llama2)
- `--temperature` — LLM temperature (default: 0.7)

## Configuration

### Chunk Size

```python
# Smaller chunks (more API calls, faster processing)
chunker = TextChunker(chunk_size=400, overlap=50)

# Larger chunks (fewer API calls, more context)
chunker = TextChunker(chunk_size=2000, overlap=300)

# Balanced (default, recommended)
chunker = TextChunker(chunk_size=800, overlap=150)
```

### Sentence Breaking

```python
# Respect sentence boundaries (recommended for quality)
chunker = TextChunker(break_on_sentences=True)

# Cut anywhere (faster, less clean)
chunker = TextChunker(break_on_sentences=False)
```

## Common Workflows

### Workflow 1: Build KB and Query

```bash
# Step 1: Build KB
python main.py build --kb-name my_kb --data-dir ./docs

# Step 2: Query (retrieval only)
python main.py query "What is the main topic?"

# Step 3: Query with LLM answer
python main.py query "What is the main topic?" --use-llm
```

### Workflow 2: Python API - End-to-End

```python
from src.knowledge_base.builder import KnowledgeBaseBuilder
from src.retrieval import QueryEngine
from src.embeddings.factory import EmbeddingsFactory
from src.llm import OllamaClient
import chromadb

# 1. Build KB
builder = KnowledgeBaseBuilder(kb_name="my_kb")
builder.build_from_directory("./documents")

# 2. Connect and query
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="my_kb")
embeddings = EmbeddingsFactory.create_ollama()
engine = QueryEngine(collection, embeddings, top_k=5)

# 3. Semantic search
response = engine.query("Your question here")
print(f"Found {len(response.results)} results")

# 4. Optional: Generate answer with LLM
if response.results:
    llm = OllamaClient(model="llama2")
    context = engine.get_context("Your question here")
    answer = llm.generate_answer("Your question here", context)
    print(f"Answer: {answer}")
```

### Workflow 3: Custom Chunking Pattern

```python
from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker

# Load and chunk for processing
loader = DocumentLoader()
chunker = TextChunker(chunk_size=800, overlap=150)

doc = loader.load("document.pdf")
for chunk in chunker.chunk_stream(doc):
    # Process each chunk (embed, save, etc)
    embedding = embeddings.embed_text(chunk.content)
    save_to_db(chunk, embedding)
```

### Workflow 4: Retrieve and Rerank

```python
# Get more results, then rerank locally
response = engine.query("Question", top_k=20)

# Filter by score
high_relevance = [r for r in response.results if r.similarity_score > 0.7]

# Use top results for LLM
context = "\n".join([r.content for r in high_relevance[:5]])
```

## Supported Formats

- **PDF** (.pdf) - Extracts text from all pages
- **Plain Text** (.txt) - UTF-8 and Latin-1 encoding
- **Markdown** (.md, .markdown) - Preserves formatting

## Memory Characteristics

### Peak Memory = Document Size + 18 KB

```
100 KB doc:  100 KB + 18 KB = 118 KB
1 MB doc:    1 MB + 18 KB = 1.018 MB
5 GB doc:    5 GB + 18 KB = 5 GB
```

### Maximum Document Size

Limited only by available RAM:
- 2 GB system: Process up to 2 GB documents
- 8 GB system: Process up to 8 GB documents
- 16 GB system: Process up to 16 GB documents

## Setup & Prerequisites

### Python Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `chromadb` — Vector database
- `pypdf` — PDF text extraction
- `sentence-transformers` — For HuggingFace embeddings
- `requests` — HTTP client for Ollama

### Embedding Providers Setup

#### Ollama (Recommended for local development)

1. Install Ollama: https://ollama.ai
2. Run Ollama server:
   ```bash
   ollama serve
   ```
3. Pull embedding model:
   ```bash
   ollama pull nomic-embed-text:latest
   ```
4. (Optional) Pull LLM model for answer generation:
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```

#### HuggingFace

Models download automatically on first use. Set environment variable if needed:
```bash
export HF_HOME=/path/to/cache
```

#### OpenAI

Set API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

## Testing & Verification

### Run Tests

All tests are organized in the `tests/` folder. Run from project root:

```bash
# Run all tests
python run_tests.py

# Run quick tests (skip performance/memory intensive)
python run_tests.py quick

# Run specific test suite
python run_tests.py chunker      # Text chunker (7/7 pass)
python run_tests.py loader       # Document loader (4/4 pass)
python run_tests.py setup        # Setup verification
python run_tests.py memory       # Memory usage (98.4x improvement)
python run_tests.py performance  # Performance (1.4x-2.0x faster)
```

### Test Results

```
✅ test_setup.py           - Python environment & dependencies
✅ test_document_loader.py - Document loading (4/4 tests pass)
✅ test_text_chunker.py    - Streaming chunks (7/7 tests pass)
✅ test_memory_usage.py    - 98.4x memory reduction verified
✅ test_performance.py     - 1.4x-2.0x speedup verified
```

See `tests/README.md` for detailed testing documentation.

### Example Usage

```bash
# Run streaming examples
python streaming_example.py
```

## Design Decisions

### Why Streaming?

1. **Memory**: No accumulation of chunks - constant 18 KB overhead
2. **Speed**: Better cache utilization, less GC pressure
3. **Scale**: Handles any document size up to available RAM
4. **Simplicity**: Generator pattern is Pythonic and clean

### Why Sentence Breaking?

1. **Quality**: Chunks end at sentence boundaries, not mid-sentence
2. **Context**: Avoids splitting related ideas
3. **Readability**: Results are more human-friendly
4. **Semantic**: Better for semantic search and embeddings

### Why Overlap?

1. **Context**: Previous chunk's end appears in next chunk's start
2. **Retrieval**: Important info at boundaries isn't lost
3. **Semantic Continuity**: Embeddings have better similarity
4. **Configurable**: Adjust overlap to your needs (default 150 chars)

## Troubleshooting

### Memory issues?
- Ensure you're using `chunk_stream()`, not `chunk()`
- Check available RAM is >= document size
- Use `test_memory_usage.py` to verify

### Slow processing?
- Verify you're using streaming pattern
- Check `test_performance.py` for baseline
- Consider batch processing for directories

### Need to compare chunks?
```python
# Only collect when necessary
chunks = list(chunker.chunk_stream(doc))
# Now you can compare chunks[i] with chunks[i+1]
```

### Processing large directory?
```python
# Use batch processor - handles thousands of files efficiently
processor.process_with_buffer("data/", save_batch, batch_size=100)
```

## Architecture

### Full RAG Pipeline

```
BUILD PHASE:
  Raw Documents
        ↓
  DocumentLoader (PDF, TXT, MD)
        ↓
  Document Objects
        ↓
  TextChunker (streaming, with overlap)
        ↓
  Chunk Objects
        ↓
  EmbeddingsFactory (Ollama/HuggingFace/OpenAI)
        ↓
  Vector Embeddings
        ↓
  KnowledgeBaseBuilder
        ↓
  Chroma Vector Database


QUERY PHASE:
  User Query
        ↓
  EmbeddingsFactory (same model as build)
        ↓
  Query Embedding
        ↓
  QueryEngine (semantic search)
        ↓
  Chroma Collection
        ↓
  Retrieved Chunks (ranked by score)
        ↓
  [Optional] OllamaClient (LLM answer generation)
        ↓
  Final Response
```

### Layered Design

```
Application Layer
       ↓
Retrieval Layer (QueryEngine, OllamaClient)
       ↓
Knowledge Base Layer (Builder, Embeddings)
       ↓
Core Layer (Chunking, Loading)
       ↓
Vector Storage (Chroma)
```

## Migration Guide

If upgrading from list-based chunking:

1. Replace `chunker.chunk(doc)` with `chunker.chunk_stream(doc)`
2. Change tests to iterate over stream instead of checking list length
3. Run `test_memory_usage.py` to verify 98.4x improvement
4. Run `test_performance.py` to verify 1.4-2.0x speedup

## Performance Summary

| Metric | Value |
|--------|-------|
| Throughput | 195-233k chunks/sec |
| Time to first chunk | 37-47 microseconds |
| Memory reduction | 98.4x less |
| Speed improvement | 1.4-2.0x faster |
| Max document size | Up to available RAM |
| Scaling | Perfect linear |

## FAQ

### Document Processing

**Q: Can I handle 1 GB documents?**
A: Yes, if your system has 1 GB+ RAM available. Memory usage will be ~1 GB + 18 KB.

**Q: Is streaming slower?**
A: No, streaming is 1.4-2.0x faster due to less memory pressure.

**Q: What's the chunk_size vs overlap trade-off?**
A: Larger chunks = fewer API calls but less granular. Overlap prevents losing context at boundaries.

**Q: Do I need to collect chunks into a list?**
A: Only if you must compare or analyze multiple chunks. Otherwise, stream directly.

**Q: What about PDFs with images?**
A: Currently extracts text only. Images are not processed.

### Query & Retrieval

**Q: What embedding models are supported?**
A: Ollama (local), HuggingFace (local/cached), OpenAI (API). Configure in config.yaml or via factory.

**Q: How does similarity scoring work?**
A: Uses cosine distance: `similarity = 1 - distance`. Scores range 0-1 (1 = perfect match).

**Q: What's min_score and how do I tune it?**
A: Minimum relevance threshold. Start at 0.3, increase if getting irrelevant results.

**Q: Can I use different embedding models for build vs query?**
A: Yes, but not recommended. The embedding space must be consistent.

**Q: How do I integrate with my LLM API?**
A: OllamaClient is built-in. For OpenAI/other APIs, extend the LLM interface.

**Q: Can I get chunks without LLM?**
A: Yes! QueryEngine returns `QueryResponse` with full chunk details regardless of LLM.

**Q: How do I speed up queries?**
A: Reduce `top_k`, use higher `min_score`, or pre-filter by source.

**Q: What if my Ollama model is on a different machine?**
A: Set `base_url` in config to the remote address: `http://192.168.1.100:11434`

## Interface Comparison

| Feature | Web UI | CLI | Python API |
|---------|--------|-----|-----------|
| **Interface** | Streamlit browser | Terminal | Code |
| **Learning Curve** | Easiest | Medium | Hardest |
| **Real-time Chat** | ✅ Yes | ❌ No | ✅ Yes |
| **Configuration** | GUI Sidebar | CLI flags | Programmatic |
| **Best For** | Exploration, demos | Automation, scripts | Integration, custom logic |
| **Launch** | `streamlit run assistant.py` | `python main.py` | `import src.*` |

## Documentation

For detailed information:
- **SETUP_GUIDE.md** — Installation, model setup, troubleshooting
- **OPTIMIZATION_GUIDE.md** — Best practices, configuration tuning, performance
- **Code examples** — `streaming_example.py`, `rag_test_suite.py`

## Requirements

### Core Requirements

- Python 3.8+
- `chromadb` — Vector database for embeddings storage
- `pypdf` — PDF text extraction
- `pyyaml` — Configuration file parsing

### Embedding Providers (choose at least one)

- **Ollama** (Recommended)
  - Install from https://ollama.ai
  - No additional Python packages needed
  - Runs locally, fully private

- **HuggingFace**
  - `sentence-transformers` — For embedding models
  - Models auto-download on first use
  
- **OpenAI**
  - `openai` package (if using OpenAI embeddings)
  - Requires API key

### Optional Dependencies

- `requests` — For Ollama HTTP client (auto-installed with chromadb)
- `transformers` — For HuggingFace embeddings

### Install All

```bash
pip install chromadb pypdf pyyaml sentence-transformers requests
```

## License

MIT License

## Getting Started

### 1. Initial Setup (One Time)

```bash
# Install Python dependencies
pip install chromadb pypdf pyyaml sentence-transformers requests streamlit

# Start Ollama in another terminal
ollama serve

# Verify setup
python test_ollama.py
```

### 2. Build Knowledge Base

```bash
# Place your documents in ./data (PDF, TXT, or Markdown)
python main.py build --kb-name ragtest_kb --data-dir ./data
```

### 3. Query Your Knowledge Base

**Choose your interface:**

```bash
# Option A: Interactive Web UI (Recommended)
streamlit run assistant.py

# Option B: CLI (One-off queries)
python main.py query "Your question" --use-llm

# Option C: Python Code (Custom logic)
# See "Python API" section above
```

### Learn More

- **Setup Help** → See `SETUP_GUIDE.md` for installation and troubleshooting
- **Performance Tips** → Read `OPTIMIZATION_GUIDE.md` for best practices
- **Test Your Setup** → Run `test_ollama.py` to verify Ollama is ready
- **See Examples** → Check `streaming_example.py` and `rag_test_suite.py`

### Production Deployment

For production use:
1. Configure `config.yaml` for your environment
2. Use appropriate embedding model (Ollama, HuggingFace, or OpenAI)
3. Set up persistent Chroma database
4. Monitor performance with built-in analytics
5. Consider deploying Web UI with Streamlit Cloud or Docker
