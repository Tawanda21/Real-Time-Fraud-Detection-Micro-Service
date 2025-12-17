import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Optional

from app.cache import CacheBackend, InMemoryCache
from training.features import vectorize_features

try:
    import joblib
except ImportError as exc:  # noqa: W0706, BLE001
    raise ImportError("joblib is required for model serialization/deserialization") from exc

logger = logging.getLogger(__name__)


class Predictor:
    """Wrapper around the trained model with optional caching."""

    def __init__(
        self,
        model_path: Path | str,
        cache: Optional[CacheBackend] = None,
        threshold: float = 0.5,
        feature_order_path: Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.cache = cache or InMemoryCache()
        self.threshold = threshold
        self._lock = Lock()
        self._model: Any | None = None
        self._feature_order: list[str] | None = None
        # Attempt to infer feature order file alongside the model if not provided
        self._feature_order_path = feature_order_path or self.model_path.parent / "feature_order.txt"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Model file not found at {self.model_path}")
                logger.info("Loading model from %s", self.model_path)
                self._model = joblib.load(self.model_path)
                # Load feature order if available
                try:
                    if self._feature_order_path.exists():
                        self._feature_order = [
                            line.strip() for line in self._feature_order_path.read_text().splitlines() if line.strip()
                        ]
                        logger.info("Loaded feature order with %d features", len(self._feature_order))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to load feature order: %s", exc)
        return self._model

    @property
    def model_loaded(self) -> bool:
        """Return True when the model has been loaded into memory."""

        return self._model is not None

    def _cache_key(self, vector: list[float]) -> str:
        return json.dumps({"v": vector})

    def predict(self, features: Mapping[str, float]) -> dict[str, Any]:
        model = self._load_model()
        vector = vectorize_features(features, self._feature_order)
        cache_key = self._cache_key(vector)

        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            try:
                if isinstance(cached, bytes):
                    cached = cached.decode("utf-8")
                cached_value = json.loads(cached) if isinstance(cached, str) else cached
                if isinstance(cached_value, dict):
                    cached_value.setdefault("cached", True)
                return cached_value
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to decode cached value: %s", exc)

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba([vector])[0][1])
        else:
            prediction = model.predict([vector])[0]
            proba = float(prediction)

        prediction = int(proba >= self.threshold)
        response = {"prediction": prediction, "score": proba, "cached": False}

        if self.cache:
            try:
                self.cache.set(cache_key, json.dumps(response))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to write to cache: %s", exc)

        return response
