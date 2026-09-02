#!/usr/bin/env python
"""
Quick test script to verify Ollama setup and model availability
"""

import requests
import json


def test_ollama_connection():
    """Test connection to Ollama"""
    print("🔍 Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Connected to Ollama\n")
        return True
    except requests.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("   Run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_available_models():
    """Get and display available models"""
    print("📋 Available Models:")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        if not data.get("models"):
            print("   ❌ No models found")
            return False

        for model in data.get("models", []):
            name = model["name"]
            size_gb = model.get("size", 0) / (1024**3)
            print(f"   • {name} ({size_gb:.2f} GB)")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_embedding_model():
    """Test embedding model"""
    print("\n🧠 Testing Embedding Model (nomic-embed-text)...")
    try:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text:latest",
                "input": "test text"
            },
            timeout=30
        )
        response.raise_for_status()
        print("✅ Embedding model works\n")
        return True
    except requests.HTTPError as e:
        print(f"❌ Embedding model error: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_llm_model(model_name: str = "neural-chat:latest"):
    """Test LLM model"""
    print(f"🤖 Testing LLM Model ({model_name})...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Say 'Hello' in one word.",
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        print(f"✅ LLM works! Response: '{answer[:50]}...'\n")
        return True
    except requests.HTTPError as e:
        print(f"❌ LLM model error: {e.response.status_code} {e.response.reason}")
        print(f"   URL: {e.response.url}")
        if e.response.status_code == 404:
            print("   ⚠️  Model not found - try: ollama pull neural-chat")
        return False
    except requests.Timeout:
        print(f"❌ LLM timeout - model might be loading or too slow")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("🔧 Ollama Setup Verification")
    print("=" * 60 + "\n")

    # Test connection
    if not test_ollama_connection():
        return

    # List models
    test_available_models()

    # Test embedding
    test_embedding_model()

    # Test LLM
    test_llm_model("neural-chat:latest")

    print("=" * 60)
    print("Summary:")
    print("  ✅ If all tests pass, your setup is ready!")
    print("  ❌ If any test fails, see messages above for fixes")
    print("=" * 60)


if __name__ == "__main__":
    main()
