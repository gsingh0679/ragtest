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

        # Override with environment variables if present
        if os.getenv("OLLAMA_BASE_URL"):
            embeddings_config["base_url"] = os.getenv("OLLAMA_BASE_URL")

        if os.getenv("OLLAMA_EMBEDDING_MODEL"):
            embeddings_config["model"] = os.getenv("OLLAMA_EMBEDDING_MODEL")

        return embeddings_config

    def get_kb_config(self) -> Dict[str, Any]:
        """Get knowledge base configuration."""
        kb_config = self.config.get("knowledge_base", {})
        return {
            "name": kb_config.get("name", "ragtest_kb"),
            "db_path": kb_config.get("db_path", "./chroma_db"),
            "chunk_size": kb_config.get("chunk_size", 800),
            "overlap": kb_config.get("overlap", 150),
            "break_on_sentences": kb_config.get("break_on_sentences", True),
        }

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        retrieval_config = self.config.get("retrieval", {})
        return {
            "top_k": retrieval_config.get("top_k", 5),
            "min_score": retrieval_config.get("min_score", 0.3),
            "include_metadata": retrieval_config.get("include_metadata", True),
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        data_config = self.config.get("data", {})
        return {
            "input_dir": data_config.get("input_dir", "./data"),
            "supported_formats": data_config.get("supported_formats", [".pdf", ".txt", ".md"]),
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            "model": os.getenv("OLLAMA_LLM_MODEL", "llama2"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature": self.config.get("llm", {}).get("temperature", 0.7),
        }

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return {
            "embeddings": self.get_embeddings_config(),
            "knowledge_base": self.get_kb_config(),
            "retrieval": self.get_retrieval_config(),
            "data": self.get_data_config(),
            "llm": self.get_llm_config(),
        }


# Global config loader instance
_config_loader = None


def get_config_loader(config_path: str = "./config.yaml") -> ConfigLoader:
    """Get or create global config loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
