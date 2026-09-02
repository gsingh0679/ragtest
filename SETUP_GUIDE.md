# 🚀 RAG Assistant Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: At least 4 GB (8 GB+ recommended)
- **Disk Space**: 10-20 GB (for Ollama models)
- **Ollama**: Latest version from https://ollama.ai

### Install Python Dependencies

```bash
# Install required packages
pip install chromadb pypdf pyyaml sentence-transformers requests streamlit
```

---

## Quick Start

### 1. Install Ollama and Start It

```bash
# In a terminal, start Ollama
ollama serve
```

### 2. Install Required Models

The assistant needs:
- **Embedding Model** (for search) - Already pulled: `nomic-embed-text:latest`
- **LLM Model** (for answer generation) - You need to pull one

#### Pull an LLM Model (choose one):

**Option A: Mistral (Recommended - Fast & Good Quality)**
```bash
ollama pull mistral
```

**Option B: Llama 2 (Classic)**
```bash
ollama pull llama2
```

**Option C: Neural Chat (Lightweight)**
```bash
ollama pull neural-chat
```

**Option D: Orca Mini (Ultra-Lightweight)**
```bash
ollama pull orca-mini
```

### 3. Verify Setup (Optional but Recommended)

```bash
# Run automatic verification script
python test_ollama.py
```

This will check:
- ✅ Ollama is running
- ✅ Models are available
- ✅ Embedding model works
- ✅ LLM model works

### 4. Check Available Models

```bash
# List all pulled models
ollama list
```

Expected output:
```
NAME                    ID              SIZE    MODIFIED
nomic-embed-text:latest abc123...       274 MB  2 minutes ago
mistral:latest          def456...       4.1 GB  Just now
```

### 5. Add Your Documents

```bash
# Place your documents in the data directory
# Supported formats: PDF, TXT, Markdown
cp your_documents.pdf ./data/
```

### 6. Build Your Knowledge Base

```bash
# From project root
python main.py build --kb-name ragtest_kb --data-dir ./data
```

### 7. Start the Assistant

```bash
streamlit run assistant.py
```

Opens at: http://localhost:8501

---

## Configuration (Optional)

### Customize config.yaml

The `config.yaml` file contains settings for:
- Knowledge base name and chunk sizes
- Embedding model selection
- Retrieval settings (top_k, min_score)
- LLM settings (model, temperature)

Example:
```yaml
knowledge_base:
  name: ragtest_kb
  chunk_size: 800
  overlap: 150
  db_path: ./chroma_db

embeddings:
  provider: ollama
  model: nomic-embed-text:latest

retrieval:
  top_k: 5
  min_score: 0.3

llm:
  model: mistral
  temperature: 0.7
```

See `config.example.yaml` for all options.

---

## Troubleshooting

### ❌ "404 Client Error: Not Found for url: http://localhost:11434/api/generate"

**Cause**: LLM model not available

**Fix**:
```bash
# Check what you have
ollama list

# Pull a model if missing
ollama pull mistral

# Then select it in the sidebar
```

### ❌ "Could not connect to Ollama"

**Cause**: Ollama not running

**Fix**:
```bash
# In another terminal
ollama serve
```

### ⚠️ "No models found in Ollama"

**Cause**: No models pulled yet

**Fix**:
```bash
ollama pull mistral
ollama pull nomic-embed-text:latest
```

### 🐢 Generation is very slow

**Cause**: Running a large model (7B+ parameters)

**Solution**:
```bash
# Use a smaller model
ollama pull orca-mini  # Only ~3.3 GB
```

### ❌ Knowledge Base build fails

**Common causes and fixes**:

**Out of Memory**:
- Use smaller chunk_size in config.yaml
- Reduce number of documents
- Check available RAM: `free -h`

**Document parsing errors**:
- Ensure PDFs are not corrupted
- Check file permissions: `ls -l ./data/`
- Try converting PDF to text first

**Embedding dimension mismatch**:
- Ensure embedding model is consistent
- Rebuild KB from scratch: `rm -rf ./chroma_db`

### 🔍 No results when querying

**Causes**:
1. min_score threshold too high (try 0.2-0.3)
2. Knowledge base is empty (rebuild with `python main.py build`)
3. Question doesn't match document content (rephrase query)

**Debug**:
```bash
# Analyze KB chunks
python main.py analyze

# Adjust settings in sidebar and retry
```

---

## Model Recommendations

| Use Case | Model | Size | Speed | Quality |
|----------|-------|------|-------|---------|
| **Fast Testing** | orca-mini | 1.4 GB | ⚡⚡⚡ | ⭐⭐ |
| **Balanced** | mistral | 4.1 GB | ⚡⚡ | ⭐⭐⭐ |
| **Best Quality** | llama2 | 3.8 GB | ⚡ | ⭐⭐⭐⭐ |
| **Lightweight** | neural-chat | 4.1 GB | ⚡⚡ | ⭐⭐⭐ |

---

## Full Example Setup

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Setup models (while Ollama is running)
ollama pull mistral
ollama pull nomic-embed-text:latest

# Terminal 3: Build knowledge base
python main.py build --kb-name ragtest_kb --data-dir ./data

# Terminal 4: Start assistant
streamlit run assistant.py

# Open browser: http://localhost:8501
```

---

## Using the Assistant

1. **Configure** (left sidebar):
   - Knowledge base settings
   - Retrieval settings (top-k, min score)
   - LLM settings (model, temperature)

2. **Chat** (main area):
   - Type your question at the bottom
   - Get semantic search results + LLM answers
   - View retrieved chunks on the right

3. **Inspect Results** (right sidebar):
   - Expand chunks to see full content
   - Check similarity scores (% = relevance)
   - Verify sources

---

## Tips & Tricks

### Improving Answer Quality

1. **Increase top-k** in sidebar (get more context)
2. **Lower min similarity score** to include more results
3. **Lower temperature** for focused answers (0.3-0.5)

### Faster Generation

1. **Use smaller model** (orca-mini, neural-chat)
2. **Reduce top-k** to use less context
3. **Lower temperature** (less sampling = faster)

### Better Retrieval

1. Ask **specific questions** (not generic)
2. Use **document terms** from your KB
3. Adjust **min similarity score** up if too many irrelevant results

---

## Command Reference

```bash
# Check Ollama status
curl http://localhost:11434/api/tags | jq

# List models
ollama list

# Pull a model
ollama pull mistral

# Run a model directly
ollama run mistral "What is AI?"

# Remove a model
ollama rm mistral

# Rebuild knowledge base
python main.py build --kb-name ragtest_kb --data-dir ./data

# Start assistant
streamlit run assistant.py

# Run tests
python run_tests.py
```

---

## Next Steps

✅ Pull an LLM model  
✅ Build knowledge base  
✅ Start assistant  
✅ Try asking questions  
✅ Fine-tune settings in sidebar  

Enjoy! 🚀
