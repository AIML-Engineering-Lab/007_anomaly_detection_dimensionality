# Post 007 — Anomaly Detection & Dimensionality Reduction: t-SNE, UMAP, Isolation Forest

**AI Engineering Lab Series** | Era 1: Classic Machine Learning

## Overview

This project demonstrates t-SNE, UMAP, and Isolation Forest for anomaly detection and advanced dimensionality reduction — applied to two real-world fault detection problems where anomalies are rare and unlabeled.

| Concept | Description |
|---|---|
| **t-SNE** | Non-linear dimensionality reduction that preserves local neighborhood structure |
| **Perplexity** | t-SNE hyperparameter: controls the balance between local and global structure |
| **UMAP** | Faster alternative to t-SNE that also preserves global topology |
| **Isolation Forest** | Anomaly detection by isolating outliers through random partitioning |
| **Contamination** | Isolation Forest parameter: expected fraction of anomalies in the data |
| **Anomaly Score** | Higher score = more isolated = more anomalous |

## Datasets

### Dataset A: Jet Engine Sensor Anomaly Detection
- **Rows:** 4,650 | **Anomaly rate:** 3.2% | **Features:** 8 sensor readings
- **Anomaly types:** Turbine overheat, Oil pressure loss, Fan blade imbalance

### Dataset B: Silicon Thermal Hotspot Detection (Post-Silicon Validation)
- **Rows:** 9,300 | **Anomaly rate:** 3.2% | **Features:** 8 thermal/electrical metrics
- **Anomaly types:** Thermal runaway, IR drop hotspot, Leakage spike

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/007_anomaly_detection_dimensionality.git
cd 007_anomaly_detection_dimensionality
pip install -r requirements.txt
python src/data_generator.py
jupyter notebook notebooks/
```

*Part of the [AI Engineering Lab](https://github.com/AIML-Engineering-Lab) series.*
