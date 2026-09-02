"""
Streamlit web UI for RAG Assistant
Interactive chat interface for querying the knowledge base
"""

import streamlit as st
import chromadb
from pathlib import Path
from typing import Optional

from src.retrieval import QueryEngine
from src.embeddings.factory import EmbeddingsFactory
from src.llm import OllamaClient
from src.config import get_config_loader


def load_config():
    """Load configuration"""
    config_loader = get_config_loader()
    return config_loader


def initialize_session():
    """Initialize Streamlit session state"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.query_engine = None
        st.session_state.llm_client = None
        st.session_state.config_loader = None
        st.session_state.messages = []


@st.cache_resource
def load_knowledge_base(kb_name: str, db_path: str, embedding_model: str):
    """Load and cache knowledge base and embeddings"""
    try:
        # Initialize embeddings
        embeddings = EmbeddingsFactory.create_ollama(model=embedding_model)

        # Connect to Chroma
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=kb_name)

        return collection, embeddings
    except Exception as e:
        st.error(f"❌ Failed to load knowledge base: {e}")
        return None, None


def initialize_llm(llm_model: str, base_url: str = "http://localhost:11434"):
    """Initialize LLM client (no caching - need fresh instance each time)"""
    try:
        return OllamaClient(model=llm_model, base_url=base_url)
    except ValueError as e:
        st.error(f"❌ {e}")
        return None
    except ConnectionError as e:
        st.error(f"❌ {e}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {e}")
        return None


def get_available_models(base_url: str = "http://localhost:11434"):
    """Get available models from Ollama (returns full names like 'neural-chat:latest')"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]  # Keep full name with :latest
        return sorted(models)
    except Exception:
        return ["llama2:latest", "mistral:latest", "neural-chat:latest", "orca-mini:latest"]  # Fallback options


def format_retrieval_results(response):
    """Format retrieval results for display"""
    results_html = ""
    for i, result in enumerate(response.results, 1):
        score_color = "🟢" if result.similarity_score > 0.7 else "🟡" if result.similarity_score > 0.5 else "🔴"
        results_html += f"""
        <div style="border-left: 4px solid #4CAF50; padding: 12px; margin: 8px 0; background-color: #f9f9f9; border-radius: 4px;">
            <strong>{score_color} Result {i}</strong> | Score: {result.similarity_score:.2%} | Source: {result.source}
            <p style="margin: 8px 0; color: #333;">{result.preview(300)}</p>
        </div>
        """
    return results_html


def main():
    # Page config
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.title("🤖 RAG Knowledge Base Assistant")
    st.markdown("*Chat with your documents using semantic search and AI*")

    # Initialize session
    initialize_session()

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check Ollama connectivity
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                st.success(f"✅ Ollama connected ({len(models)} models)")
            else:
                st.warning("⚠️ Ollama not responding properly")
        except requests.ConnectionError:
            st.error("❌ Cannot connect to Ollama at http://localhost:11434")
            st.info("Run in another terminal: `ollama serve`")
        except Exception as e:
            st.warning(f"⚠️ Error checking Ollama: {e}")

        # Knowledge Base Settings
        st.subheader("Knowledge Base")
        config_loader = load_config()
        kb_config = config_loader.get_kb_config()
        retrieval_config = config_loader.get_retrieval_config()
        llm_config = config_loader.get_llm_config()

        kb_name = st.text_input(
            "KB Name",
            value=kb_config["name"],
            help="Name of the knowledge base collection"
        )

        db_path = st.text_input(
            "Database Path",
            value=kb_config["db_path"],
            help="Path to Chroma database"
        )

        # Retrieval Settings
        st.subheader("Retrieval Settings")
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=retrieval_config["top_k"],
            help="Number of chunks to retrieve"
        )

        min_score = st.slider(
            "Min Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=retrieval_config["min_score"],
            step=0.05,
            help="Minimum relevance threshold"
        )

        # Embedding Settings
        st.subheader("Embeddings")
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["nomic-embed-text:latest", "mistral-embed:latest", "all-minilm:latest"],
            index=0,
            help="Model used to embed documents and queries"
        )

        # LLM Settings
        st.subheader("Answer Generation")
        use_llm = st.checkbox(
            "🤖 Enable LLM Answers",
            value=True,
            help="Generate natural language answers using LLM"
        )

        # Define variables outside conditional so they're always available
        llm_base_url = st.text_input(
            "LLM Base URL",
            value=llm_config.get("base_url", "http://localhost:11434"),
            help="Ollama server address"
        )

        available_models = get_available_models(llm_base_url)

        if not available_models:
            st.warning("⚠️ No LLM models found in Ollama. Pull one with: `ollama pull mistral`")
            llm_model = "mistral:latest"
        else:
            # Find preferred model index
            default_idx = 0
            for i, model in enumerate(available_models):
                if "neural-chat" in model:
                    default_idx = i
                    break
                elif "mistral" in model:
                    default_idx = i
                elif "llama2" in model:
                    default_idx = i

            llm_model = st.selectbox(
                "LLM Model",
                options=available_models,
                index=default_idx,
                help="Model used to generate answers (includes version tag)"
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=llm_config["temperature"],
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )

        # Status
        st.subheader("Status")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload KB", key="reload_kb"):
                st.cache_resource.clear()
                st.success("✅ Cache cleared!")
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.markdown("""
        ### 💡 Tips
        - Ask specific questions for better results
        - Adjust similarity score to filter irrelevant results
        - Check retrieval scores to gauge relevance
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat")

        # Load knowledge base
        collection, embeddings = load_knowledge_base(kb_name, db_path, embedding_model)

        if collection is None or embeddings is None:
            st.error("⚠️ Failed to load knowledge base. Check configuration and try reloading.")
            return

        # Initialize query engine
        query_engine = QueryEngine(
            chroma_collection=collection,
            embeddings=embeddings,
            top_k=top_k,
            min_score=min_score
        )

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["content"])

        # Query input
        query = st.chat_input("Ask a question about your knowledge base...")

        if query:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user", avatar="👤"):
                st.write(query)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Retrieving relevant chunks..."):
                    # Query knowledge base
                    response = query_engine.query(query)

                    if not response.results:
                        message = "❌ No relevant chunks found. Try a different query."
                        st.write(message)
                        st.session_state.messages.append({"role": "assistant", "content": message})
                    else:
                        # Generate LLM answer if enabled
                        if use_llm:
                            try:
                                with st.spinner("✍️ Generating answer..."):
                                    st.write(f"*Using model: {llm_model}*")
                                    llm = initialize_llm(llm_model, llm_base_url)
                                    if llm:
                                        context = query_engine.get_context(query, top_k=top_k)
                                        answer = llm.generate_answer(
                                            query,
                                            context,
                                            temperature=temperature
                                        )
                                        st.write(answer)
                                        st.session_state.messages.append({"role": "assistant", "content": answer})
                                    else:
                                        # Fallback: Show just the chunks
                                        st.warning("⚠️ LLM unavailable. Showing retrieved chunks instead.")
                                        chunks_text = "\n\n".join([r.preview(200) for r in response.results])
                                        st.write(chunks_text)
                                        st.session_state.messages.append({"role": "assistant", "content": chunks_text})
                            except Exception as e:
                                import traceback
                                st.error(f"❌ Error generating answer: {e}")
                                with st.expander("📋 Debug Info"):
                                    st.write(f"**Model**: {llm_model}")
                                    st.write(f"**Base URL**: {llm_base_url}")
                                    st.write(f"**Error Trace**:")
                                    st.code(traceback.format_exc())
                        else:
                            # Just show chunks
                            chunks_text = "\n\n".join([r.preview(300) for r in response.results])
                            st.write(chunks_text)
                            st.session_state.messages.append({"role": "assistant", "content": chunks_text})

    with col2:
        st.subheader("📊 Retrieved Chunks")

        # Display retrieval results in sidebar
        if query and collection and embeddings:
            response = query_engine.query(query)

            if response.results:
                st.metric("Total Results", len(response.results))
                st.divider()

                for i, result in enumerate(response.results, 1):
                    score_pct = result.similarity_score * 100
                    score_color = "🟢" if result.similarity_score > 0.7 else "🟡" if result.similarity_score > 0.5 else "🔴"

                    with st.expander(f"{score_color} Chunk {i} ({score_pct:.0f}%) - {result.source}", expanded=(i==1)):
                        st.metric("Similarity Score", f"{score_pct:.1f}%")
                        st.metric("Source", result.source)
                        st.metric("Chunk Index", result.chunk_index)
                        st.text_area(
                            "Content",
                            value=result.content,
                            height=200,
                            disabled=True,
                            key=f"chunk_{i}"
                        )


if __name__ == "__main__":
    main()
