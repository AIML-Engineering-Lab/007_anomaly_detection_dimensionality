# Anomaly Detection & Dimensionality Reduction: t-SNE, UMAP, Isolation Forest

---

## Overview

This project demonstrates **Isolation Forest** anomaly detection with **t-SNE**, **UMAP**, and **PCA** dimensionality reduction — applied to two real-world fault detection problems where anomalies are rare and unlabeled.

| Concept | Description |
|---|---|
| **t-SNE** | Non-linear dimensionality reduction that preserves local neighborhood structure |
| **Perplexity** | t-SNE hyperparameter: controls the balance between local and global structure |
| **UMAP** | Faster alternative to t-SNE that also preserves global topology |
| **Isolation Forest** | Anomaly detection by isolating outliers through random partitioning |
| **Contamination** | Isolation Forest parameter: expected fraction of anomalies in the data |
| **Anomaly Score** | Higher score = more isolated = more anomalous |

---

## Datasets

### Dataset A: Jet Engine Sensor Anomaly Detection
Detects anomalous sensor readings from jet engine telemetry.

| Feature | Description |
|---|---|
| `turbine_temp_c` | Turbine temperature (°C) |
| `oil_pressure_psi` | Oil pressure (PSI) |
| `vibration_hz` | Dominant vibration frequency (Hz) |
| `fuel_flow_rate` | Fuel flow rate |
| `exhaust_gas_temp_c` | Exhaust gas temperature (°C) |
| `fan_speed_rpm` | Fan speed (RPM) |
| `compressor_ratio` | Compressor pressure ratio |
| `n1_speed_pct` | N1 speed percentage |
| **`label`** | **0 = Normal, 1 = Anomaly** |

- **Rows:** 4,650 | **Anomaly rate:** 3.2% | **Anomaly types:** Turbine overheat, Oil pressure loss, Fan blade imbalance

### Dataset B: Silicon Thermal Hotspot Detection
Identifies thermal anomalies in semiconductor chips.

| Feature | Description |
|---|---|
| `core_temp_c` | Core temperature (°C) |
| `power_density_mw_mm2` | Power density (mW/mm²) |
| `thermal_resistance` | Thermal resistance |
| `leakage_current_ua` | Leakage current (μA) |
| `clock_freq_ghz` | Clock frequency (GHz) |
| `vdd_mv` | Supply voltage (mV) |
| `ir_drop_mv` | IR drop (mV) |
| `thermal_gradient` | Thermal gradient |
| **`label`** | **0 = Normal, 1 = Anomaly** |

- **Rows:** 9,300 | **Anomaly rate:** 3.2% | **Anomaly types:** Thermal runaway, IR drop hotspot, Leakage spike

---

## Repository Structure

```
007_anomaly_detection_dimensionality/
├── assets/
│   ├── proj1_jet_eda.png                          # Jet: feature distributions + correlations
│   ├── proj1_jet_pca_explained_variance.png       # Jet: PCA cumulative explained variance
│   ├── proj1_jet_tsne_vs_pca.png                  # Jet: t-SNE vs PCA comparison
│   ├── proj1_jet_umap_vs_tsne.png                 # Jet: UMAP vs t-SNE comparison
│   ├── proj1_jet_tsne_perplexity.png              # Jet: t-SNE perplexity sweep (5→50)
│   ├── proj1_jet_dim_reduction_comparison.png     # Jet: PCA/t-SNE/UMAP side-by-side
│   ├── proj1_jet_isolation_forest.png             # Jet: Isolation Forest anomaly detection
│   ├── proj1_jet_anomaly_scores.png               # Jet: anomaly score distribution
│   ├── proj1_jet_3d_anomaly.png                   # Jet: 3D anomaly visualization
│   ├── proj1_jet_model_heatmap.png                # Jet: model performance heatmap
│   ├── proj1_jet_flowchart.png                    # Jet: AI-generated pipeline flowchart
│   ├── proj2_silicon_eda.png                      # Silicon: feature distributions + correlations
│   ├── proj2_silicon_dim_reduction.png            # Silicon: t-SNE/PCA projections
│   ├── proj2_silicon_hotspot.png                  # Silicon: thermal hotspot visualization
│   ├── proj2_silicon_anomaly_scores.png           # Silicon: anomaly score distribution
│   ├── proj2_silicon_3d_thermal.png               # Silicon: 3D thermal visualization
│   └── proj2_silicon_flowchart.png                # Silicon: AI-generated pipeline flowchart
├── data/
│   ├── jet_engine_sensors.csv                     # 4,650 jet engine records (8 features + label)
│   └── silicon_thermal_hotspots.csv               # 9,300 silicon records (8 features + label)
├── deploy/
│   ├── Dockerfile                                 # Container image for FastAPI server
│   └── docker-compose.yml                         # Single-command deployment
├── docs/
│   ├── Anomaly_Detection_Dimensionality_Report.html  # Full report with embedded visualizations
│   └── Anomaly_Detection_Dimensionality_Report.pdf   # Print-ready A4 report
├── models/
│   ├── iforest_jet.pkl                            # Trained IsolationForest pipeline (jet engine)
│   └── iforest_silicon.pkl                        # Trained IsolationForest pipeline (silicon)
├── notebooks/
│   ├── 01_anomaly_jet_engine.ipynb                # Jet EDA, t-SNE/UMAP, Isolation Forest
│   └── 02_anomaly_silicon_thermal.ipynb           # Silicon EDA, anomaly detection, hotspot analysis
├── src/
│   ├── train.py                                   # Train both IsolationForest models (DATASETS dict)
│   ├── predict.py                                 # Inference: load model, predict anomalies
│   ├── api.py                                     # FastAPI endpoint (/health, /info, /predict)
│   └── data_generator.py                          # Generate synthetic datasets
├── tests/
│   └── test_model.py                              # 4 tests: existence + prediction per model
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core runtime |
| scikit-learn | ≥ 1.3 | IsolationForest, PCA, StandardScaler, Pipeline |
| umap-learn | ≥ 0.5 | UMAP dimensionality reduction |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| matplotlib / seaborn | ≥ 3.7 / 0.12 | Visualizations |
| FastAPI | latest | REST API serving |
| joblib | built-in | Model serialization |

---

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/007_anomaly_detection_dimensionality.git
cd 007_anomaly_detection_dimensionality
pip install -r requirements.txt

# Train both models
python src/train.py

# Run inference
python src/predict.py

# Run tests
python tests/test_model.py

# Explore notebooks
jupyter notebook notebooks/
```
