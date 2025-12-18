import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Settings:
    """Configuration values loaded from environment variables."""

    app_name: str = os.getenv("APP_NAME", "fraud-detector")
    model_path: Path = Path(os.getenv("MODEL_PATH", "training/models/model.joblib"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # auto: try Redis then fallback to memory; memory: use in-memory only; redis: require Redis
    cache_backend: str = os.getenv("CACHE_BACKEND", "auto").lower()
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
