import sys
print(f"Python: {sys.version}")

# Test Chroma
import chromadb
print("✓ Chroma DB imported")

# Test Ollama
import ollama
print("✓ Ollama imported")

# Test connection to Ollama
try:
    response = ollama.generate(model="neural-chat", prompt="What is 2+2?", stream=False)
    print("✓ Ollama is running and responding")
    print(f"  Response: {response['response'][:100]}")
except Exception as e:
    print(f"✗ Ollama error: {e}")
    print("  Make sure Ollama is running: run 'ollama serve' in another terminal")

