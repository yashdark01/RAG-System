"""
Configuration management for the RAG system.
Uses pydantic-settings for environment variable loading and validation.
"""

import os
from pathlib import Path
from typing import Optional, Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    
    # Application
    APP_NAME: str = "RAG System"
    APP_VERSION: str = "1.0.0"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = Field(default=False)
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: list = Field(default=["*"])
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = Field(default=["*"])
    CORS_HEADERS: list = Field(default=["*"])
    
    # Database
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./data/rag.db")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Vector Store
    VECTOR_STORE_PATH: str = "./data/vector_store"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # LLM Configuration
    LLM_PROVIDER: Literal["gemini", "google", "groq", "openai", "anthropic", "ollama"] = "gemini"
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    
    # RAG Settings
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    RERANKING_ENABLED: bool = True
    
    # Agent Settings
    MAX_AGENT_ITERATIONS: int = 3
    AGENT_TIMEOUT_SECONDS: int = 300
    
    # Cache
    CACHE_TTL_SECONDS: int = 3600
    CACHE_ENABLED: bool = True
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = "logs/app.log"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Path("./data")
    UPLOAD_DIR: Path = Path("./data/uploads")
    MODELS_DIR: Path = Path("./models")
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        """Parse debug flag flexibly."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "debug", "dev", "development")
        return bool(v)

    @field_validator("GOOGLE_API_KEY", mode="before")
    @classmethod
    def fallback_google_key(cls, v, info):
        """Fallback to GEMINI_API_KEY if GOOGLE_API_KEY is not set."""
        if not v and hasattr(info, "data"):
            return info.data.get("GEMINI_API_KEY")
        return v
    
    @field_validator("DATABASE_URL", mode='before')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if v and not v.startswith(("postgresql://", "postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL or SQLite connection string")
        return v
    
    @field_validator("CORS_ORIGINS", mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string to list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    def model_post_init(self, __context):
        """Create necessary directories and harmonize API keys after initialization."""
        if not self.GEMINI_API_KEY and self.GOOGLE_API_KEY:
            self.GEMINI_API_KEY = self.GOOGLE_API_KEY
        elif not self.GOOGLE_API_KEY and self.GEMINI_API_KEY:
            self.GOOGLE_API_KEY = self.GEMINI_API_KEY

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_PATH, exist_ok=True)
        os.makedirs("logs", exist_ok=True)


settings = Settings()