"""
Factory for creating embedding models from configuration.

Supports loading different embedding implementations based on config.
"""

from typing import Dict, Any, Optional
import yaml
import json

from src.embeddings.base import EmbeddingsBase
from src.embeddings.implementations import OllamaEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings
from src.config import get_config_loader


class EmbeddingsFactory:
    """Factory for creating embedding models."""

    _IMPLEMENTATIONS = {
        "ollama": OllamaEmbeddings,
        "huggingface": HuggingFaceEmbeddings,
        "openai": OpenAIEmbeddings,
    }

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> EmbeddingsBase:
        """
        Create embedding model from configuration dictionary.

        Args:
            config: Configuration dictionary with 'provider' and 'model' keys

        Returns:
            Initialized embeddings model

        Example:
            config = {
                "provider": "ollama",
                "model": "nomic-embed-text:latest",
                "base_url": "http://localhost:11434"
            }
            embeddings = EmbeddingsFactory.create_from_config(config)
        """
        provider = config.get("provider", "ollama").lower()

        if provider not in cls._IMPLEMENTATIONS:
            raise ValueError(
                f"Unknown embedding provider: {provider}\n"
                f"Supported: {list(cls._IMPLEMENTATIONS.keys())}"
            )

        implementation = cls._IMPLEMENTATIONS[provider]

        # Extract model-specific parameters
        params = {k: v for k, v in config.items() if k not in ["provider", "type"]}

        try:
            return implementation(**params)
        except Exception as e:
            raise RuntimeError(f"Failed to create {provider} embeddings: {e}")

    @classmethod
    def create_from_file(cls, config_file: str) -> EmbeddingsBase:
        """
        Create embedding model from configuration file.

        Args:
            config_file: Path to YAML or JSON config file

        Returns:
            Initialized embeddings model

        Example config file (config.yaml):
            embeddings:
              provider: ollama
              model: nomic-embed-text:latest
              base_url: http://localhost:11434
        """
        # Load configuration file
        if config_file.endswith(".yaml") or config_file.endswith(".yml"):
            with open(config_file, "r") as f:
                full_config = yaml.safe_load(f)
                config = full_config.get("embeddings", {})
        elif config_file.endswith(".json"):
            with open(config_file, "r") as f:
                full_config = json.load(f)
                config = full_config.get("embeddings", {})
        else:
            raise ValueError("Config file must be YAML or JSON")

        if not config:
            raise ValueError(f"No 'embeddings' section found in {config_file}")

        return cls.create_from_config(config)

    @classmethod
    def create_ollama(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> OllamaEmbeddings:
        """
        Create Ollama embeddings with config defaults.

        Args:
            model: Model name (uses config default if not provided)
            base_url: Ollama server URL (uses config default if not provided)
        """
        if model is None or base_url is None:
            config_loader = get_config_loader()
            embeddings_config = config_loader.get_embeddings_config()
            model = model or embeddings_config.get("model", "nomic-embed-text:latest")
            base_url = base_url or embeddings_config.get("base_url", "http://localhost:11434")

        return OllamaEmbeddings(model=model, base_url=base_url)

    @classmethod
    def create_huggingface(cls, model: Optional[str] = None) -> HuggingFaceEmbeddings:
        """
        Create HuggingFace embeddings with config defaults.

        Args:
            model: Model name (uses config default if not provided)
        """
        if model is None:
            import os
            model = os.getenv("HUGGINGFACE_MODEL") or os.getenv("DEFAULT_HUGGINGFACE_MODEL", "all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model=model)

    @classmethod
    def create_openai(
        cls,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> OpenAIEmbeddings:
        """
        Create OpenAI embeddings with config defaults.

        Args:
            model: Model name (uses config default if not provided)
            api_key: OpenAI API key (uses env var if not provided)
        """
        if model is None:
            import os
            model = os.getenv("OPENAI_MODEL") or os.getenv("DEFAULT_OPENAI_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=api_key)

    @classmethod
    def list_providers(cls) -> list:
        """List available embedding providers."""
        return list(cls._IMPLEMENTATIONS.keys())
