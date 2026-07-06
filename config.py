"""
config.py — Centralized application configuration using pydantic-settings.
All environment variables are loaded from .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    local_model_path: str = Field(default="", env="LOCAL_MODEL_PATH")

    # Embeddings
    embedding_provider: str = Field(default="sentence_transformers", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")

    # Vector Store
    vector_store_path: str = Field(default="vector_store/", env="VECTOR_STORE_PATH")
    faiss_index_name: str = Field(default="main_index", env="FAISS_INDEX_NAME")

    # Backend
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    debug: bool = Field(default=True, env="DEBUG")

    # Database
    database_url: str = Field(default="", env="DATABASE_URL")

    # Cache
    cache_dir: str = Field(default=".cache", env="CACHE_DIR")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="logs/", env="LOG_DIR")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
