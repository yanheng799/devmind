"""Configuration settings for DEVMIND."""

import os
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Loads from environment variables with fallback to .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DEVMIND_",
        case_sensitive=False,
        extra="ignore",
    )

    # Application paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "data")
    cache_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "data" / "cache")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "data" / "logs")

    # LLM settings
    llm_provider: str = Field(default="deepseek", description="LLM provider: deepseek, openai, etc.")
    llm_api_key: str = Field(default="", description="LLM API key")
    llm_api_base: str = Field(default="https://api.deepseek.com/v1", description="LLM API base URL")
    llm_model: str = Field(default="deepseek-chat", description="LLM model name")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    llm_max_tokens: int = Field(default=4096, ge=1, description="LLM max tokens")
    llm_timeout: int = Field(default=120, ge=1, description="LLM request timeout in seconds")

    # Embedding settings
    embedding_model: str = Field(default="text2vec-base-chinese", description="Embedding model name")
    embedding_device: str = Field(default="cpu", description="Embedding device: cpu or cuda")
    embedding_batch_size: int = Field(default=32, ge=1, description="Embedding batch size")
    embedding_dim: int = Field(default=768, ge=1, description="Embedding dimension")

    # Milvus settings
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, ge=1, le=65535, description="Milvus port")
    milvus_collection_name: str = Field(default="devmind_events", description="Milvus collection name")
    milvus_index_type: str = Field(default="IVF_FLAT", description="Milvus index type")
    milvus_metric_type: str = Field(default="COSINE", description="Milvus metric type")

    # Database settings
    db_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "data" / "devmind.db")
    db_pool_size: int = Field(default=5, ge=1, description="Database connection pool size")

    # News collector settings
    news_sources_config: str = Field(
        default="config/sources.yaml",
        description="Path to news sources config file",
    )
    news_fetch_interval: int = Field(default=300, ge=10, description="News fetch interval in seconds")
    news_max_retries: int = Field(default=3, ge=0, description="Max retries for news fetching")
    news_retry_delay: int = Field(default=5, ge=1, description="Retry delay in seconds")

    # Market data settings
    market_data_source: str = Field(default="akshare", description="Market data source")
    market_fetch_interval: int = Field(default=60, ge=10, description="Market data fetch interval in seconds")

    # Agent settings
    agent_max_iterations: int = Field(default=10, ge=1, description="Max agent iterations")
    agent_timeout: int = Field(default=180, ge=10, description="Agent timeout in seconds")
    agent_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for predictions",
    )
    agent_rag_top_k: int = Field(default=5, ge=1, description="Top K for RAG retrieval")

    # Prediction settings
    prediction_default_horizon: str = Field(default="short", description="Default prediction horizon")
    prediction_min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to save prediction",
    )

    # Logging settings
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    log_format: str = Field(default="json", description="Log format: json or text")
    log_rotation: str = Field(default="10 MB", description="Log rotation size")
    log_retention: int = Field(default=30, ge=1, description="Log retention days")

    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")

    @field_validator("data_dir", "cache_dir", "logs_dir")
    @classmethod
    def create_dirs(cls, value: Path) -> Path:
        """Create directories if they don't exist."""
        value.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("db_path")
    @classmethod
    def create_db_dir(cls, value: Path) -> Path:
        """Create parent directory for database if it doesn't exist."""
        value.parent.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("llm_api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """Validate API key is set."""
        if not value:
            raise ValueError(
                "DEVMIND_LLM_API_KEY must be set. "
                "Please set it in .env file or environment variable."
            )
        return value

    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from environment with validation."""
        return cls()

    def get_llm_config(self) -> dict[str, Any]:
        """Get LLM configuration dict."""
        return {
            "api_key": self.llm_api_key,
            "base_url": self.llm_api_base,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "timeout": self.llm_timeout,
        }

    def get_milvus_config(self) -> dict[str, Any]:
        """Get Milvus configuration dict."""
        return {
            "host": self.milvus_host,
            "port": self.milvus_port,
            "collection_name": self.milvus_collection_name,
        }

    def get_embedding_config(self) -> dict[str, Any]:
        """Get embedding configuration dict."""
        return {
            "model_name": self.embedding_model,
            "device": self.embedding_device,
            "batch_size": self.embedding_batch_size,
            "dim": self.embedding_dim,
        }


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance.

    Creates instance on first call and reuses it.
    """
    global _settings
    if _settings is None:
        _settings = Settings.load_from_env()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment.

    Useful for testing or when environment changes.
    """
    global _settings
    _settings = Settings.load_from_env()
    return _settings
