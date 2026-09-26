"""API tests for the current flash-flood ML service contract."""
from pathlib import Path
import sys

import joblib
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ARTIFACTS, app, model_state, _risk_level_from_prob


VALID_RECORD = {
    "grid_id": "G001",
    "elevation_m": 1420.0,
    "slope_deg": 38.5,
    "aspect_deg": 225.0,
    "plan_curvature": -0.002,
    "profile_curvature": -0.003,
    "twi": 8.5,
    "dist_to_stream_m": 320.0,
    "land_cover_code": 10,
    "soil_type_enc": 1,
    "rainfall_1h_mm": 18.0,
    "rainfall_3h_mm": 42.0,
    "rainfall_6h_mm": 58.0,
    "rainfall_12h_mm": 71.0,
    "rainfall_24h_mm": 85.0,
    "rainfall_3d_accum_mm": 210.0,
    "rainfall_7d_accum_mm": 420.0,
    "rainfall_intensity_mm_h": 12.5,
    "soil_moisture_pct": 72.0,
    "historical_landslide_density": 2,
}

EXPECTED_FEATURES = {
    "elevation_m", "slope_deg", "aspect_deg", "plan_curvature", "profile_curvature",
    "twi", "dist_to_stream_m", "land_cover_code", "soil_type_enc", "rainfall_1h_mm",
    "rainfall_3h_mm", "rainfall_6h_mm", "rainfall_12h_mm", "rainfall_24h_mm",
    "rainfall_3d_accum_mm", "rainfall_7d_accum_mm", "rainfall_intensity_mm_h",
    "soil_moisture_pct", "historical_landslide_density", "rain_moisture_index",
    "slope_wetness_index", "rainfall_runoff_proxy",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_risk_level_boundaries():
    threshold = 0.46
    assert _risk_level_from_prob(0.2799, threshold) == "Low"
    assert _risk_level_from_prob(0.28, threshold) == "Moderate"
    assert _risk_level_from_prob(threshold, threshold) == "High"
    assert _risk_level_from_prob(0.75, threshold) == "Critical"
    assert _risk_level_from_prob(0.7501, threshold) == "Critical"


def test_predict_accepts_full_record_and_returns_22_features(client):
    response = client.post("/predict", json={"records": [VALID_RECORD]})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction["grid_id"] == "G001"
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["risk_level"] in {"Low", "Moderate", "High", "Critical"}
    assert set(prediction["features_used"]) == EXPECTED_FEATURES


def test_engineered_features_are_calculated_and_can_be_overridden(client):
    calculated_response = client.post("/predict", json={"records": [VALID_RECORD]})
    assert calculated_response.status_code == 200
    calculated = calculated_response.json()["predictions"][0]["features_used"]
    assert calculated["rain_moisture_index"] == pytest.approx(12.5 * 72.0)
    assert calculated["slope_wetness_index"] == pytest.approx(38.5 * 72.0)
    assert calculated["rainfall_runoff_proxy"] == pytest.approx(85.0 * 8.5)

    explicit_record = {
        **VALID_RECORD,
        "rain_moisture_index": 123.0,
        "slope_wetness_index": 456.0,
        "rainfall_runoff_proxy": 789.0,
    }
    explicit_response = client.post("/predict", json={"records": [explicit_record]})
    assert explicit_response.status_code == 200
    explicit = explicit_response.json()["predictions"][0]["features_used"]
    assert explicit["rain_moisture_index"] == 123.0
    assert explicit["slope_wetness_index"] == 456.0
    assert explicit["rainfall_runoff_proxy"] == 789.0


def test_health_reports_loaded_model_and_artifact_threshold(client):
    health = client.get("/health")

    assert health.status_code == 200
    body = health.json()
    assert body["loaded"] is True
    assert model_state["loaded"] is True
    assert body["features_count"] == 22
    assert body["threshold"] == pytest.approx(float(joblib.load(ARTIFACTS["threshold"])))


def test_predict_batches_multiple_records(client):
    records = [
        {**VALID_RECORD, "grid_id": f"G{i:03d}", "slope_deg": 20.0 + i * 3}
        for i in range(1, 4)
    ]
    response = client.post("/predict", json={"records": records})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == len(records)
    assert [prediction["grid_id"] for prediction in predictions] == [record["grid_id"] for record in records]
    assert all(set(prediction["features_used"]) == EXPECTED_FEATURES for prediction in predictions)
