import sys
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


SAMPLE_PAYLOAD = {
    "age": 30,
    "gender": "Male",
    "subscription_type": "Basic",
    "watch_hours": 1.0,
    "last_login_days": 2,
    "region": "Europe",
    "device": "Laptop",
    "monthly_fee": 13.99,
    "payment_method": "Credit Card",
    "number_of_profiles": 2,
    "avg_watch_time_per_day": 1.4,
    "favorite_genre": "Action",
}


# fixture for creating a mock inference function instead of the actual predict function
@pytest.fixture(scope="module")
def client():
    # slot a fake module into sys.modules
    mock_inference = MagicMock()
    mock_inference.predict.return_value = "Not likely to churn"
    sys.modules["src.service.inference"] = mock_inference

    # remove cached imports so the fixture controls what main.py sees
    for mod in ("src.app.main", "src.app"):
        sys.modules.pop(mod, None)

    # import the application
    from src.app.main import app
    with TestClient(app) as c:
        yield c

    sys.modules.pop("src.service.inference", None)
    sys.modules.pop("src.app.main", None)


def test_root_returns_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_prediction_key(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert "prediction" in response.json()


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_calls_inference(client):
    from src.service.inference import predict as mock_predict
    client.post("/predict", json=SAMPLE_PAYLOAD)
    mock_predict.assert_called()
