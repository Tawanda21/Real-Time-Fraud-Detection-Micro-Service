import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_endpoint_schema():
    payload = {"features": {"amount": 100.0, "distance": 10.0, "night": 0}}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        body = resp.json()
        assert {"prediction", "score", "cached"}.issubset(body.keys())
