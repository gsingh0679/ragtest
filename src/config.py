"""
Configuration loader for RAG system.

Reads from config.yaml and .env file with precedence:
1. Environment variables (.env)
2. config.yaml
3. Hardcoded defaults
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load configuration from multiple sources."""

    def __init__(self, config_path: str = "./config.yaml", env_path: str = "./.env"):
        """
        Initialize config loader.

        Args:
            config_path: Path to config.yaml
            env_path: Path to .env file
        """
        # Load .env file first
        load_dotenv(env_path)

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration with env var precedence."""
        embeddings_config = self.config.get("embeddings", {})

        # Override with environment variables if present (check runtime vars first, then defaults)
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("DEFAULT_OLLAMA_BASE_URL")
        if base_url:
            embeddings_config["base_url"] = base_url
        elif "base_url" not in embeddings_config:
            embeddings_config["base_url"] = "http://localhost:11434"

        model = os.getenv("OLLAMA_EMBEDDING_MODEL") or os.getenv("DEFAULT_OLLAMA_EMBEDDING_MODEL")
        if model:
            embeddings_config["model"] = model
        elif "model" not in embeddings_config:
            embeddings_config["model"] = "nomic-embed-text:latest"

        return embeddings_config

    def get_kb_config(self) -> Dict[str, Any]:
        """Get knowledge base configuration with env var precedence."""
        kb_config = self.config.get("knowledge_base", {})
        return {
            "name": os.getenv("KB_NAME", os.getenv("DEFAULT_KB_NAME", kb_config.get("name", "ragtest_kb"))),
            "db_path": os.getenv("KB_DB_PATH", os.getenv("DEFAULT_KB_DB_PATH", kb_config.get("db_path", "./chroma_db"))),
            "chunk_size": int(os.getenv("KB_CHUNK_SIZE", os.getenv("DEFAULT_KB_CHUNK_SIZE", kb_config.get("chunk_size", 800)))),
            "overlap": int(os.getenv("KB_OVERLAP", os.getenv("DEFAULT_KB_OVERLAP", kb_config.get("overlap", 150)))),
            "break_on_sentences": os.getenv("KB_BREAK_ON_SENTENCES", os.getenv("DEFAULT_KB_BREAK_ON_SENTENCES", kb_config.get("break_on_sentences", True))) in (True, "true", "True"),
        }

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration with env var precedence."""
        retrieval_config = self.config.get("retrieval", {})
        return {
            "top_k": int(os.getenv("RETRIEVAL_TOP_K", os.getenv("DEFAULT_RETRIEVAL_TOP_K", retrieval_config.get("top_k", 5)))),
            "min_score": float(os.getenv("RETRIEVAL_MIN_SCORE", os.getenv("DEFAULT_RETRIEVAL_MIN_SCORE", retrieval_config.get("min_score", 0.3)))),
            "include_metadata": os.getenv("RETRIEVAL_INCLUDE_METADATA", os.getenv("DEFAULT_RETRIEVAL_INCLUDE_METADATA", retrieval_config.get("include_metadata", True))) in (True, "true", "True"),
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration with env var precedence."""
        data_config = self.config.get("data", {})

        # Parse supported formats from env var (comma-separated)
        supported_formats_env = os.getenv("DATA_SUPPORTED_FORMATS") or os.getenv("DEFAULT_DATA_SUPPORTED_FORMATS")
        if supported_formats_env:
            supported_formats = [fmt.strip() for fmt in supported_formats_env.split(",")]
        else:
            supported_formats = data_config.get("supported_formats", [".pdf", ".txt", ".md"])

        return {
            "input_dir": os.getenv("DATA_INPUT_DIR", os.getenv("DEFAULT_DATA_INPUT_DIR", data_config.get("input_dir", "./data"))),
            "supported_formats": supported_formats,
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with env var precedence."""
        llm_config = self.config.get("llm", {})
        return {
            "model": os.getenv("OLLAMA_LLM_MODEL", os.getenv("DEFAULT_OLLAMA_LLM_MODEL", "llama2")),
            "base_url": os.getenv("OLLAMA_BASE_URL", os.getenv("DEFAULT_OLLAMA_BASE_URL", "http://localhost:11434")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", os.getenv("DEFAULT_LLM_TEMPERATURE", llm_config.get("temperature", 0.7)))),
            "top_k": int(os.getenv("LLM_TOP_K", os.getenv("DEFAULT_LLM_TOP_K", llm_config.get("top_k", 40)))),
            "top_p": float(os.getenv("LLM_TOP_P", os.getenv("DEFAULT_LLM_TOP_P", llm_config.get("top_p", 0.9)))),
        }

    def get_app_config(self) -> Dict[str, Any]:
        """Get application-level configuration."""
        return {
            "debug": os.getenv("DEBUG", "true").lower() in ("true", "1", "yes"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "streamlit_port": int(os.getenv("STREAMLIT_PORT", 8501)),
        }

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return {
            "embeddings": self.get_embeddings_config(),
            "knowledge_base": self.get_kb_config(),
            "retrieval": self.get_retrieval_config(),
            "data": self.get_data_config(),
            "llm": self.get_llm_config(),
            "app": self.get_app_config(),
        }


# Global config loader instance
_config_loader = None


def get_config_loader(config_path: str = "./config.yaml") -> ConfigLoader:
    """Get or create global config loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
