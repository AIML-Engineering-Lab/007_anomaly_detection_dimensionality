"""FastAPI serving endpoint for Isolation Forest anomaly detection."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Anomaly Detection API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "iforest_jet.pkl"
_model = None


class PredictionInput(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    anomaly_label: float
    model: str = "IsolationForest"


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "healthy", "model": "IsolationForest"}


@app.get("/info")
def info_endpoint():
    return {"project": "007_anomaly_detection_dimensionality", "task": "anomaly_detection"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        model = get_model()
        df = pd.DataFrame([input_data.features])
        pred = model.predict(df)[0]
        return PredictionResponse(anomaly_label=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
