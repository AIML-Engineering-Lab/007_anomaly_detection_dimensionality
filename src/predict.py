"""Inference for Isolation Forest anomaly detection."""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    if model_path is None:
        model_path = str(MODEL_DIR / "iforest_jet.pkl")
    pipe = joblib.load(model_path)
    X = data.select_dtypes(include="number")
    return pipe.predict(X).tolist()


if __name__ == "__main__":
    from train import DATASETS

    for name, cfg in DATASETS.items():
        df = pd.read_csv(ROOT / "data" / cfg["csv"])
        features = df.drop(columns=cfg["drop_cols"], errors="ignore").head(5)
        preds = predict(features, str(MODEL_DIR / cfg["model"]))
        print(f"{name}: {preds}  (-1=anomaly, 1=normal)")
