"""
ml-service/main.py
FastAPI microservice wrapping the XGBoost landslide risk model.

Startup: loads all 4 model artifacts from ../models/ once.
Endpoints:
  POST /predict   — batch or single prediction
  GET  /health    — model load status + metadata
  GET  /explain/{grid_id}  — SHAP top-5 feature contributions (stretch goal)
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
MODELS_DIR = HERE.parent / "models"

ARTIFACTS = {
    "model":      MODELS_DIR / "xgboost_landslide_model_v2.json",
    "features":   MODELS_DIR / "feature_list_v2.pkl",
    "encoders":   MODELS_DIR / "label_encoders_v2.pkl",
    "thresholds": MODELS_DIR / "risk_thresholds_v2.pkl",
    "metadata":   MODELS_DIR / "model_metadata_v2.json",
}

LOG_PATH = HERE / "predictions.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ml-service")

pred_logger = logging.getLogger("predictions")
pred_logger.setLevel(logging.INFO)
pred_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
pred_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
pred_logger.addHandler(pred_handler)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
model_state: dict[str, Any] = {}


def load_artifacts() -> None:
    """Load all model artifacts at startup. Fails loudly if anything is missing."""
    missing = [name for name, path in ARTIFACTS.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Missing model artifacts: {missing}. "
            f"Expected in: {MODELS_DIR}. Cannot start."
        )

    logger.info("Loading model artifacts from %s", MODELS_DIR)
    t0 = time.time()

    booster = xgb.Booster()
    booster.load_model(str(ARTIFACTS["model"]))

    feature_list: list[str] = joblib.load(ARTIFACTS["features"])
    label_encoders: dict = joblib.load(ARTIFACTS["encoders"])
    thresholds: dict = joblib.load(ARTIFACTS["thresholds"])
    metadata: dict = json.loads(ARTIFACTS["metadata"].read_text(encoding="utf-8"))

    # Validate feature list matches metadata
    if feature_list != metadata["features"]:
        raise RuntimeError(
            "feature_list_v2.pkl does not match model_metadata_v2.json features. "
            "Artifacts may be mismatched."
        )

    model_state.update({
        "booster": booster,
        "feature_list": feature_list,
        "label_encoders": label_encoders,
        "thresholds": thresholds,
        "metadata": metadata,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "load_time_s": round(time.time() - t0, 3),
        "loaded": True,
    })

    logger.info(
        "Model loaded in %.3fs | features=%d | model=%s",
        model_state["load_time_s"],
        len(feature_list),
        metadata.get("model", "unknown"),
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    logger.info("ml-service shutting down")


app = FastAPI(
    title="DHARA AI Landslide Risk ML Service",
    description="XGBoost inference service for landslide risk scoring (PS 26001)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Land cover and soil type mappings (from label_encoders_v2.pkl alphabetical order)
# These are the ONLY valid string values accepted by the model.
# ---------------------------------------------------------------------------
LAND_COVER_CLASSES = [
    "Agriculture / Jhum (Slash & Burn)",
    "Barren / Degraded Slope",
    "Built-up Urban Slope",
    "Dense Forest",
    "Open Shrubland",
    "Tea Plantation / Terrace",
]

SOIL_TYPE_CLASSES = [
    "Disang Shale & Soft Clay",
    "Granite Wash Residual Loam",
    "Granitic Gneiss Collapsible",
    "High-Grade Gneissic Debris",
    "Loose Residual Hill Clay",
    "Phyllite & Quartzite Slates",
    "Quartzitic Phyllites & Red Clay",
    "Red Lateritic Sandy Loam",
    "Sandstone Overlain Karst & Silt",
    "Sandstone-Shale Intercalation",
    "Siltstone & Friable Shale",
    "Splintery Disang Shale",
    "Weathered Basalt & Sandstone",
    "Weathered Gneiss & Mica Schist",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FeatureInput(BaseModel):
    grid_id: str | None = None

    # Terrain (static)
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    plan_curvature: float
    profile_curvature: float
    twi: float
    dist_to_road_m: float
    dist_to_stream_m: float

    # Categorical (human-readable strings — encoded server-side)
    land_cover_type: str
    soil_type: str

    # Dynamic / sensor
    rainfall_24h_mm: float
    rainfall_3d_accum_mm: float
    rainfall_7d_accum_mm: float
    rainfall_intensity_mm_h: float
    soil_moisture_pct: float

    # Derived
    historical_landslide_density: float

    @model_validator(mode="after")
    def validate_categorical(self):
        if self.land_cover_type not in LAND_COVER_CLASSES:
            raise ValueError(
                f"land_cover_type '{self.land_cover_type}' is not a valid class. "
                f"Valid: {LAND_COVER_CLASSES}"
            )
        if self.soil_type not in SOIL_TYPE_CLASSES:
            raise ValueError(
                f"soil_type '{self.soil_type}' is not a valid class. "
                f"Valid: {SOIL_TYPE_CLASSES}"
            )
        return self


class PredictRequest(BaseModel):
    records: list[FeatureInput]


class PredictionResult(BaseModel):
    grid_id: str | None
    probability: float
    risk_level: str
    model_version: str
    features_used: dict


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model_version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------
def _risk_level_from_prob(prob: float, thresholds: dict) -> str:
    low_edge = thresholds["low_edge"]
    high_edge = thresholds["high_edge"]
    critical_edge = thresholds["critical_edge"]

    if prob >= critical_edge:
        return "Critical"
    elif prob >= high_edge:
        return "High"
    elif prob >= low_edge:
        return "Medium"
    else:
        return "Low"


def _encode_and_predict(records: list[FeatureInput]) -> list[PredictionResult]:
    state = model_state
    booster: xgb.Booster = state["booster"]
    feature_list: list[str] = state["feature_list"]
    label_encoders: dict = state["label_encoders"]
    thresholds: dict = state["thresholds"]
    metadata: dict = state["metadata"]

    model_version = f"{metadata.get('model', 'XGBClassifier')}_v2"

    rows = []
    for rec in records:
        # Encode categorical features using LabelEncoders
        land_cover_code = int(
            label_encoders["land_cover_type"].transform([rec.land_cover_type])[0]
        )
        soil_type_enc = int(
            label_encoders["soil_type"].transform([rec.soil_type])[0]
        )

        feature_map = {
            "elevation_m":                  rec.elevation_m,
            "slope_deg":                    rec.slope_deg,
            "aspect_deg":                   rec.aspect_deg,
            "plan_curvature":               rec.plan_curvature,
            "profile_curvature":            rec.profile_curvature,
            "twi":                          rec.twi,
            "dist_to_road_m":               rec.dist_to_road_m,
            "dist_to_stream_m":             rec.dist_to_stream_m,
            "land_cover_code":              land_cover_code,
            "rainfall_24h_mm":              rec.rainfall_24h_mm,
            "rainfall_3d_accum_mm":         rec.rainfall_3d_accum_mm,
            "rainfall_7d_accum_mm":         rec.rainfall_7d_accum_mm,
            "rainfall_intensity_mm_h":      rec.rainfall_intensity_mm_h,
            "soil_moisture_pct":            rec.soil_moisture_pct,
            "historical_landslide_density": rec.historical_landslide_density,
            "soil_type_enc":                soil_type_enc,
        }

        # Assemble in exact feature order from feature_list_v2.pkl
        vector = [feature_map[f] for f in feature_list]
        rows.append(vector)

    dmat = xgb.DMatrix(np.array(rows, dtype=np.float32), feature_names=feature_list)
    probs_2d = booster.predict(dmat)  # shape: (n, 2) for binary, or (n,) prob of class 1

    # Handle both binary output shapes
    if probs_2d.ndim == 2:
        probs = probs_2d[:, 1]
    else:
        probs = probs_2d

    results = []
    for i, (rec, prob) in enumerate(zip(records, probs)):
        prob_f = float(prob)
        risk_level = _risk_level_from_prob(prob_f, thresholds)
        results.append(PredictionResult(
            grid_id=rec.grid_id,
            probability=round(prob_f, 6),
            risk_level=risk_level,
            model_version=model_version,
            features_used={
                "land_cover_type": rec.land_cover_type,
                "soil_type": rec.soil_type,
                "elevation_m": rec.elevation_m,
                "slope_deg": rec.slope_deg,
                "rainfall_24h_mm": rec.rainfall_24h_mm,
            },
        ))

        pred_logger.info(
            json.dumps({
                "grid_id": rec.grid_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "probability": prob_f,
                "risk_level": risk_level,
                "model_version": model_version,
            })
        )

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if not model_state.get("loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    meta = model_state["metadata"]
    return {
        "model_loaded": True,
        "loaded_at": model_state["loaded_at"],
        "load_time_s": model_state["load_time_s"],
        "model": meta.get("model"),
        "model_version": "v2",
        "features": len(model_state["feature_list"]),
        "xgboost_version": meta.get("library_versions", {}).get("xgboost"),
        "metrics": {
            "roc_auc_final_test": meta["metrics"]["final_test_at_0.5"]["roc_auc"],
            "accuracy_final_test": meta["metrics"]["final_test_at_0.5"]["accuracy"],
            "recall_final_test":   meta["metrics"]["final_test_at_0.5"]["landslide_recall"],
        },
        "thresholds": model_state["thresholds"],
        "caveats": meta.get("caveats", []),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model_state.get("loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.records:
        raise HTTPException(status_code=422, detail="No records provided")

    try:
        predictions = _encode_and_predict(request.records)
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        predictions=predictions,
        model_version="v2",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/explain/{grid_id}")
def explain(grid_id: str):
    """
    Returns SHAP-based top-5 feature contributions for a grid cell's last prediction.
    NOTE: This is a stretch-goal endpoint. Requires shap to be installed.
    """
    return {
        "note": "SHAP explanation endpoint — provide a feature vector via POST /predict first",
        "grid_id": grid_id,
        "status": "not_yet_implemented",
    }
