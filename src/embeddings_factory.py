"""
Factory for creating embedding models from configuration.

Supports loading different embedding implementations based on config.
"""

from typing import Dict, Any, Optional
import yaml
import json

from src.embeddings_base import EmbeddingsBase
from src.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings


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
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434"
    ) -> OllamaEmbeddings:
        """Create Ollama embeddings quickly."""
        return OllamaEmbeddings(model=model, base_url=base_url)

    @classmethod
    def create_huggingface(cls, model: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
        """Create HuggingFace embeddings quickly."""
        return HuggingFaceEmbeddings(model=model)

    @classmethod
    def create_openai(
        cls,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ) -> OpenAIEmbeddings:
        """Create OpenAI embeddings quickly."""
        return OpenAIEmbeddings(model=model, api_key=api_key)

    @classmethod
    def list_providers(cls) -> list:
        """List available embedding providers."""
        return list(cls._IMPLEMENTATIONS.keys())
