# Project Overview

This document explains what each part of the codebase does, how the pieces fit together, and why specific design decisions were made. It complements the top-level README with deeper, file-by-file context.

## High-Level Architecture

- Inference service built with FastAPI that loads a serialized ML model (joblib), exposes `/predict` and `/health`, and optionally uses Redis for caching and rate-limiting support.
- Training pipeline that can learn a model from the Kaggle `creditcard.csv` dataset (or a synthetic fallback) and exports both a `model.joblib` and a `feature_order.txt` to keep feature ordering consistent between training and serving.
- Docker-based dev and prod-like runs with an optional Redis sidecar.

Data flow:
1. Client sends POST `/predict` with a JSON `{ features: { ... } }` map.
2. The API validates input and passes it to `Predictor`.
3. `Predictor` aligns features to the training order, computes a cache key, optionally serves from cache, otherwise calls the model and returns `{ prediction, score }`.
4. Response is optionally cached.

---

## Application (FastAPI)

- [app/main.py](app/main.py)
  - Creates the FastAPI app and wires lifespan startup to initialize shared resources.
  - At startup, resolves settings via `get_settings()`, configures cache backend:
    - `CACHE_BACKEND=memory` → in-memory cache only
    - otherwise it tries Redis via `RedisCache.from_url(REDIS_URL)` with graceful fallback
  - Stores a single `Predictor` instance on `app.state.predictor` for request handlers to use.
  - Registers routers from [app/api/](app/api).

- [app/api/health.py](app/api/health.py)
  - GET `/health` endpoint.
  - Returns `{ status: "ok", model_loaded: bool }`. `model_loaded` is `True` once the model has been loaded in memory by `Predictor`.

- [app/api/predict.py](app/api/predict.py)
  - POST `/predict` endpoint.
  - Accepts a `PredictionRequest` (features dict and optional `correlation_id`).
  - If the model has a known feature order, it checks for missing feature keys and warns or errors accordingly; then calls `Predictor.predict()` and returns a `PredictionResponse`.

- [app/api/openapi_overrides.py](app/api/openapi_overrides.py)
  - Adds a simple root route `/` with a friendly message, excluded from the OpenAPI schema.

- [app/schemas.py](app/schemas.py)
  - Pydantic models for request/response validation:
    - `PredictionRequest` (features map; validator ensures it is not empty)
    - `PredictionResponse` (prediction, score, optional model_version, cached flag)
    - `HealthResponse` (status and model readiness flag)

- [app/predictor.py](app/predictor.py)
  - The serving wrapper around the trained model.
  - Lazily loads the model from `model_path` and attempts to read `feature_order.txt` next to it.
  - `predict(features)`:
    - Calls `training.features.vectorize_features` to align inputs to the exact training order.
    - Builds a cache key from the numeric vector.
    - Returns cached response if present; otherwise runs model:
      - If estimator has `predict_proba`, uses class-1 probability; else uses `predict()` and casts.
    - Applies threshold (default 0.5) to produce a binary prediction.
    - Writes response to cache and returns `{ prediction, score, cached }`.
  - Exposes `model_loaded` and `feature_order` properties.

- [app/cache.py](app/cache.py)
  - `CacheBackend` protocol: minimal `get`/`set` contract used by the predictor.
  - `InMemoryCache`: thread-safe TTL cache for local/dev or fallback mode.
  - `RedisCache`: uses `redis-py` if available; pings Redis on startup; falls back to in-memory if unreachable.
  - `RateLimiter`: optional, simple fixed-window limiter that uses Redis atomics if available, otherwise an in-memory counter.

- [app/config.py](app/config.py)
  - Centralized settings loaded from environment variables with defaults:
    - `APP_NAME`, `MODEL_PATH`, `REDIS_URL`
    - `CACHE_BACKEND` (auto | memory | redis)
    - `CACHE_TTL_SECONDS`, `RATE_LIMIT_PER_MINUTE`, `LOG_LEVEL`
  - `get_settings()` caches the `Settings` object for fast reuse.

---

## Training Pipeline

- [training/train_model.py](training/train_model.py)
  - CLI to train a model and export artifacts.
  - If `training/data/creditcard.csv` exists (or `--csv` is passed), loads the Kaggle dataset. Otherwise, synthesizes a small dummy dataset so the pipeline is runnable without data.
  - Uses scikit-learn `LogisticRegression` as the default model.
  - Saves outputs to [training/models/](training/models):
    - `model.joblib` (the trained estimator)
    - `feature_order.txt` (line-separated feature names in the exact order used to train) so the API can reproduce the same feature vector layout during inference.

- [training/features.py](training/features.py)
  - `vectorize_features(mapping, order)` turns a dict of features into a stable numeric list, using either the given order or an alphabetical fallback.
  - `ensure_feature_order(order, observed)` ensures we always get a concrete order (provided or derived).

- [training/requirements.txt](training/requirements.txt)
  - Dependencies required only for training (e.g., scikit-learn, numpy, pandas, kaggle).

- [training/data/README.md](training/data/README.md)
  - Instructions for obtaining `creditcard.csv` from Kaggle.

- [training/data/download_creditcard.py](training/data/download_creditcard.py)
  - Small helper using the Kaggle API to download `creditcard.csv` directly into `training/data/`.

- [training/notebooks/eda_creditcard.ipynb](training/notebooks/eda_creditcard.ipynb)
  - Lightweight exploratory notebook to preview the dataset (loaded locally). Not used in production.

- [training/models/](training/models)
  - Output directory for serialized models and accompanying metadata.

### Why feature_order.txt?
Serving must construct the exact same feature vector as training, in the same order. To avoid subtle bugs from dict ordering, the training pipeline writes `feature_order.txt`. At runtime, `Predictor` reads it and strictly orders input features to match training, minimizing train-serve skew.

---

## Tests and Utilities

- [tests/unit/test_features.py](tests/unit/test_features.py)
  - Exercises the feature vectorization and ordering helpers to ensure deterministic behavior.

- [tests/integration/test_api_smoke.py](tests/integration/test_api_smoke.py)
  - Basic API smoke tests using FastAPI’s `TestClient`.

- [tests/load/locustfile.py](tests/load/locustfile.py)
  - Minimal Locust load test hitting `/predict` to sanity check throughput and tail latencies.

- [tests/smoke_test.py](tests/smoke_test.py)
  - Simple script using `requests` to probe `/health` and `/predict` of a running instance.

- [utils/logging.py](utils/logging.py)
  - Helper to set sane logging defaults (format and level), if needed by scripts.

- [utils/sample_payload.py](utils/sample_payload.py)
  - Reads `training/models/feature_order.txt` and generates a schema-correct `/predict` payload (all features present), optionally writing it to a file.

---

## Infrastructure & Packaging

- [Dockerfile](Dockerfile)
  - Python 3.10-slim base, installs API requirements, copies `app/`, the full `training/` package (for `features.py` and models), and `utils/`.
  - Default command runs Uvicorn on port 8000.

- [.dockerignore](.dockerignore)
  - Excludes `.venv/`, git metadata, OS junk, notebooks, and local data from the build context.

- [infra/docker-compose.yml](infra/docker-compose.yml)
  - `app` (prod-like) and `app-dev` (dev with `--reload`) services.
  - `redis` service for caching and rate-limit support.
  - Sets `PYTHONPATH=/app` for imports inside the container.
  - In `app-dev`, mounts `../app`, `../training`, and `../utils` for live reload.

- [.env](.env)
  - Central place for defaults like `MODEL_PATH`, `CACHE_BACKEND`, `REDIS_URL`, etc. You can override these per environment or via compose.

- [requirements.txt](requirements.txt)
  - Runtime dependencies for the API and shared tooling (FastAPI, Uvicorn, Redis client, sklearn, etc.).

---

## Operational Behavior

- Caching strategy:
  - When Redis is configured and reachable, the API uses Redis for response caching (keyed by the numeric feature vector) and for the `RateLimiter` if you choose to wire it in.
  - If Redis isn’t available—and `CACHE_BACKEND=auto`—the app falls back to a thread-safe in-memory cache with TTL.
  - Set `CACHE_BACKEND=memory` to avoid even attempting a Redis connection (useful for local/offline).

- Model loading:
  - `Predictor` loads the model lazily upon first request (or earlier if you call properties that force load). This minimizes startup time and memory if the service runs but isn’t used.

- Validation & safety:
  - Input features are validated to be non-empty; the predict endpoint can enforce presence of all trained features (and warn about extras).
  - Errors during prediction are logged and returned as a `500` with a clear message.

---

## Extending the Service

- Swap the model:
  - Replace `training/train_model.py` with your algorithm (e.g., XGBoost, LightGBM). Continue to output `model.joblib` and `feature_order.txt`.
  - Ensure your runtime has the correct inference dependencies and that `predictor.py` knows how to call your estimator (`predict_proba` vs `predict`).

- Add features or preprocessing:
  - Extend [training/features.py](training/features.py) and ensure the training script writes the new order.
  - At inference, the API will read `feature_order.txt` and require (or impute) the same fields.

- Rate limiting in the API:
  - Use `RateLimiter` from [app/cache.py](app/cache.py). For example, enforce a per-IP or per-key limit at the start of `/predict` using the Redis path if available.

- Monitoring & metrics:
  - Add Prometheus metrics or request logging middleware. The `utils/logging.py` file provides a simple baseline.

---

## Common Pitfalls & Troubleshooting

- Redis warning at startup: Set `CACHE_BACKEND=memory` to suppress Redis attempts, or run the `redis` service via compose.
- Feature mismatch errors: Ensure requests include all features listed in `training/models/feature_order.txt`. Use `utils/sample_payload.py` to generate a correct skeleton.
- Import errors in Docker dev: The compose file sets `PYTHONPATH=/app` and mounts `../training`. If you change structure, keep PYTHONPATH and mounts aligned.
- Model not found: Train via `python -m training.train_model` so `training/models/model.joblib` exists before starting the API image.

---

## Why These Choices?

- FastAPI: async, type-friendly, excellent OpenAPI and validation.
- Redis (optional): simple, reliable distributed cache; in-memory fallback keeps local/dev friction low.
- `feature_order.txt`: explicit, debuggable contract between training and serving removing ambiguity around feature ordering.
- Single `Predictor` instance: reduces load time and memory churn; model usually fits in memory with negligible load overhead once cached.
- Compose-based dev: one command to bring up the whole stack; code mounts + `--reload` optimize iteration speed.

---

## Quick References

- Start dev stack (reload) + Redis:
  - From `infra/`: `docker compose up -d --build app-dev redis`
- Start prod-like + Redis:
  - From `infra/`: `docker compose up -d --build app redis`
- Local run (no Docker):
  - `python -m uvicorn app.main:app --reload`
- Train model:
  - `python -m training.train_model` (or pass `--csv path/to/creditcard.csv`)
- Generate valid payload:
  - `python utils/sample_payload.py --out .tmp/payload.json`
