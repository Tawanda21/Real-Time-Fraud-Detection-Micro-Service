import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, predict, openapi_overrides
from app.cache import InMemoryCache, RedisCache
from app.config import get_settings
from app.predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = settings.cache_backend
    if backend == "memory":
        cache = InMemoryCache(ttl_seconds=settings.cache_ttl_seconds)
    else:
        cache = RedisCache.from_url(settings.redis_url, ttl_seconds=settings.cache_ttl_seconds)
    app.state.predictor = Predictor(settings.model_path, cache=cache)  # type: ignore[attr-defined]

    yield

    # Teardown if necessary


logging.basicConfig(level=get_settings().log_level)
app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(openapi_overrides.router)
