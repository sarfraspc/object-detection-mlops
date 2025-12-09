from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"