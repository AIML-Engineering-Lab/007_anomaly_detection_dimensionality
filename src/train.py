"""Train Isolation Forest anomaly detectors for both datasets."""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    "jet": {
        "csv": "jet_engine_sensors.csv",
        "model": "iforest_jet.pkl",
        "drop_cols": ["label"],
        "contamination": 0.05,
    },
    "silicon": {
        "csv": "silicon_thermal_hotspots.csv",
        "model": "iforest_silicon.pkl",
        "drop_cols": ["label"],
        "contamination": 0.05,
    },
}


def train(name: str, cfg: dict):
    df = pd.read_csv(DATA_DIR / cfg["csv"])
    X = df.drop(columns=cfg["drop_cols"], errors="ignore")
    X = X.select_dtypes(include="number")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            contamination=cfg["contamination"],
            random_state=42,
            n_estimators=200,
        )),
    ])
    pipe.fit(X)

    model_path = MODEL_DIR / cfg["model"]
    joblib.dump(pipe, model_path)

    preds = pipe.predict(X)
    n_anomalies = (preds == -1).sum()
    print(f"{name:>8s} | Anomalies={n_anomalies}/{len(X)} ({n_anomalies/len(X):.1%}) | {model_path.name}")
    return pipe


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        train(name, cfg)
