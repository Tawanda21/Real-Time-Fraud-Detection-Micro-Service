import logging
import time
from threading import Lock
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class CacheBackend(Protocol):
    """Minimal cache protocol used by the predictor."""

    def get(self, key: str) -> Optional[Any]:
        ...

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ...


class InMemoryCache:
    """Thread-safe in-memory cache used as a fallback when Redis is unavailable."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            if key not in self._store:
                return None
            expires_at, value = self._store[key]
            if expires_at < now:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds or self._ttl_seconds
        expires_at = time.time() + ttl
        with self._lock:
            self._store[key] = (expires_at, value)


class RedisCache:
    """Redis-backed cache. Falls back to in-memory if Redis is not installed or reachable."""

    def __init__(self, client: Any, ttl_seconds: int = 300) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int = 300) -> CacheBackend:
        try:
            import redis
        except ImportError:
            logger.warning("redis package not installed; using in-memory cache")
            return InMemoryCache(ttl_seconds=ttl_seconds)

        client = redis.Redis.from_url(url)
        try:
            client.ping()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable (%s); using in-memory cache", exc)
            return InMemoryCache(ttl_seconds=ttl_seconds)

        logger.info("Redis cache enabled (%s)", url)
        return cls(client=client, ttl_seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        return raw if raw is None else raw

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds or self._ttl_seconds
        self._client.setex(key, ttl, value)


class RateLimiter:
    """Simple per-key fixed-window rate limiter using Redis when available.

    If a Redis client is available via RedisCache, it uses `INCR` with expiry for
    atomic counters. Otherwise falls back to a thread-safe in-memory counter.
    """

    def __init__(self, limit_per_window: int, window_seconds: int = 60, backend: CacheBackend | None = None) -> None:
        self.limit = limit_per_window
        self.window_seconds = window_seconds
        self.backend = backend
        self._mem_counts: dict[str, tuple[int, float]] = {}
        self._lock = Lock()

    def _redis_incr(self, client: Any, key: str, ttl: int) -> int:
        pipe = client.pipeline(True)
        pipe.incr(key, 1)
        pipe.expire(key, ttl)
        count, _ = pipe.execute()
        return int(count)

    def allow(self, key: str) -> bool:
        """Return True if the request for `key` is within the rate limit."""

        # Redis path
        if isinstance(self.backend, RedisCache):
            try:
                count = self._redis_incr(self.backend._client, f"rl:{key}", self.window_seconds)
                return count <= self.limit
            except Exception as exc:  # noqa: BLE001
                logger.debug("RateLimiter Redis path failed: %s; falling back to memory", exc)

        # In-memory path
        now = time.time()
        window_start = now - (now % self.window_seconds)
        with self._lock:
            current = self._mem_counts.get(key)
            if current is None or current[1] != window_start:
                self._mem_counts[key] = (1, window_start)
                return True
            count = current[0] + 1
            self._mem_counts[key] = (count, window_start)
            return count <= self.limit
