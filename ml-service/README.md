# ml-service — Flash Flood ML Inference Service

FastAPI microservice wrapping the trained 22-feature XGBoost flash-flood model (SIH 26192).

## Run the Application on Windows

Start each service in a separate PowerShell terminal. Run commands from the repository root unless a step says otherwise.

### 1. Install dependencies

```powershell
npm install
python -m venv ml-service/venv
.\ml-service\venv\Scripts\Activate.ps1
python -m pip install -r ml-service/requirements.txt
```

### 2. Start the ML service

In PowerShell terminal 1, from the repository root:

```powershell
.\ml-service\venv\Scripts\Activate.ps1
Set-Location ml-service
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Wait for the startup message confirming the model loaded with 22 features. Keep this terminal running.

### 3. Start the backend

In PowerShell terminal 2, from the repository root:

```powershell
npm run dev --workspace=backend
```

The backend calls the ML service as its primary predictor. If it is unavailable, risk scoring falls back to local XGBoost tree inference.

### 4. Start the frontend

In PowerShell terminal 3, from the repository root:

```powershell
npm run dev --workspace=frontend
```

Open the Vite URL printed in terminal 3, usually `http://localhost:5173`.

### 5. Check service health

In another PowerShell terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:3001/api/admin/health
Invoke-RestMethod http://localhost:3001/api/risk/scores
```

The ML health response should report `loaded: true` and `features_count: 22`. Admin health reports the predictor as `primary` or `fallback active`.

## Run ML Tests

From the repository root:

```powershell
.\ml-service\venv\Scripts\python.exe -m pytest ml-service/tests/test_predict.py -v
```

The service expects these artifacts in the repository's `models/` directory:
- `xgboost_flash_flood_model.json`
- `feature_list_flash_flood.pkl`
- `label_encoders_flash_flood.pkl`
- `flash_flood_threshold.pkl`
- `flash_flood_model_metadata.json`

**Startup will fail loudly if any artifact is missing.**

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model status, version, and performance metrics |
| POST | `/predict` | Batch or single grid-cell risk prediction |
| GET | `/explain/{grid_id}` | SHAP feature contributions (stretch goal) |

## POST /predict — Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "grid_id": "G001",
      "elevation_m": 1420,
      "slope_deg": 38.5,
      "aspect_deg": 225,
      "plan_curvature": -0.002,
      "profile_curvature": -0.003,
      "twi": 8.5,
      "dist_to_stream_m": 320,
      "land_cover_code": 10,
      "soil_type_enc": 1,
      "rainfall_1h_mm": 18,
      "rainfall_3h_mm": 42,
      "rainfall_6h_mm": 58,
      "rainfall_12h_mm": 71,
      "rainfall_24h_mm": 85,
      "rainfall_3d_accum_mm": 210,
      "rainfall_7d_accum_mm": 420,
      "rainfall_intensity_mm_h": 12.5,
      "soil_moisture_pct": 72,
      "historical_landslide_density": 2
    }]
  }'
```

## Notes

- `land_cover_code` and `soil_type_enc` are numeric encoded model inputs.
- The three engineered features are calculated by the service when omitted.
- Predictions are logged to `ml-service/predictions.log`.
- Risk thresholds: Low < 0.28 ≤ Moderate < configured threshold ≤ High < 0.75 ≤ Critical.
- These are prototype heuristics, **not** validated disaster-response standards.

## Model Caveats

- Dataset is a 2024 prototype; validation is internal to it only
- Reported metrics (ROC-AUC ~0.99) are NOT evidence of real-world performance
- `historical_landslide_density` temporal provenance requires confirmation
- Risk-bucket thresholds must be re-validated against confirmed real incidents
