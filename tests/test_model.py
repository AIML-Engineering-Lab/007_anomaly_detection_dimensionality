"""Tests for Isolation Forest anomaly detection models."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_jet_model_exists():
    assert (ROOT / "models" / "iforest_jet.pkl").exists()


def test_silicon_model_exists():
    assert (ROOT / "models" / "iforest_silicon.pkl").exists()


def test_jet_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "jet_engine_sensors.csv")
    features = df.drop(columns=["label"], errors="ignore").head(5)
    preds = predict(features, str(ROOT / "models" / "iforest_jet.pkl"))
    assert len(preds) == 5
    assert all(p in (-1, 1) for p in preds)


def test_silicon_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "silicon_thermal_hotspots.csv")
    features = df.drop(columns=["label"], errors="ignore").head(5)
    preds = predict(features, str(ROOT / "models" / "iforest_silicon.pkl"))
    assert len(preds) == 5
    assert all(p in (-1, 1) for p in preds)


if __name__ == "__main__":
    test_jet_model_exists()
    test_silicon_model_exists()
    test_jet_prediction()
    test_silicon_prediction()
    print("All 4 tests passed.")
