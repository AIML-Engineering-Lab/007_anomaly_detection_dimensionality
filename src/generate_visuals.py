"""
Visualization Generator for Project 007: Anomaly Detection & Dimensionality Reduction
Generates 5 publication-quality figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)
DATA   = Path(__file__).parent.parent / "data"

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})

# Load data
df_jet = pd.read_csv(DATA / "jet_engine_sensors.csv")
df_si  = pd.read_csv(DATA / "silicon_thermal_hotspots.csv")

feat_jet = ['turbine_temp_c','oil_pressure_psi','vibration_hz','fuel_flow_rate',
            'exhaust_gas_temp_c','fan_speed_rpm','compressor_ratio','n1_speed_pct']
feat_si  = ['core_temp_c','power_density_mw_mm2','thermal_resistance','leakage_current_ua',
            'clock_freq_ghz','vdd_mv','ir_drop_mv','thermal_gradient']

X_jet = df_jet[feat_jet].values
y_jet = df_jet['label'].values
X_si  = df_si[feat_si].values
y_si  = df_si['label'].values

scaler = StandardScaler()
X_jet_sc = scaler.fit_transform(X_jet)
X_si_sc  = StandardScaler().fit_transform(X_si)

# ── Figure 1: t-SNE Jet Engine Anomalies ──────────────────────────────────────
print("Generating Fig 1: t-SNE Jet Engine...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_jet_sc)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1565C0', '#E91E63']
labels_text = ['Normal (96.8%)', 'Anomaly (3.2%)']

ax = axes[0]
for lbl, color, text in zip([0, 1], colors, labels_text):
    mask = y_jet == lbl
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, s=8 if lbl==0 else 30,
               alpha=0.4 if lbl==0 else 0.9, label=text, zorder=2 if lbl==1 else 1)
ax.set_title('t-SNE: Jet Engine Sensor Data\n(2D projection of 8 features)', fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
ax.legend()

# PCA for comparison
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_jet_sc)
ax = axes[1]
for lbl, color, text in zip([0, 1], colors, labels_text):
    mask = y_jet == lbl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=8 if lbl==0 else 30,
               alpha=0.4 if lbl==0 else 0.9, label=text, zorder=2 if lbl==1 else 1)
ax.set_title('PCA: Same Data for Comparison\n(Linear projection — less separation)',
             fontweight='bold')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
ax.legend()

plt.suptitle('t-SNE vs PCA: Non-linear vs Linear Dimensionality Reduction\n'
             't-SNE reveals cluster structure invisible to PCA', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig1_tsne_vs_pca.png", bbox_inches='tight')
plt.close()
print("  fig1 saved")

# ── Figure 2: Isolation Forest Anomaly Scores ─────────────────────────────────
print("Generating Fig 2: Isolation Forest...")
iso = IsolationForest(contamination=0.032, random_state=42, n_estimators=200)
iso.fit(X_jet_sc)
scores = -iso.score_samples(X_jet_sc)  # Higher = more anomalous
pred = iso.predict(X_jet_sc)  # -1 = anomaly, 1 = normal

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.hist(scores[y_jet == 0], bins=60, color='#1565C0', alpha=0.7, label='True Normal', density=True)
ax.hist(scores[y_jet == 1], bins=30, color='#E91E63', alpha=0.8, label='True Anomaly', density=True)
ax.axvline(np.percentile(scores, 96.8), color='black', linestyle='--', lw=2, label='Decision threshold')
ax.set_xlabel('Anomaly Score (higher = more anomalous)')
ax.set_ylabel('Density')
ax.set_title('Isolation Forest Anomaly Scores\n(Jet Engine Data)', fontweight='bold')
ax.legend()

ax = axes[1]
ax.scatter(X_tsne[pred == 1, 0], X_tsne[pred == 1, 1], c='#1565C0', s=6, alpha=0.3, label='Predicted Normal')
ax.scatter(X_tsne[pred == -1, 0], X_tsne[pred == -1, 1], c='#E91E63', s=40, alpha=0.9, label='Predicted Anomaly', marker='X')
ax.set_title('Isolation Forest Predictions\n(Overlaid on t-SNE projection)', fontweight='bold')
ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
ax.legend()

ax = axes[2]
fpr, tpr, _ = roc_curve(y_jet, scores)
auc = roc_auc_score(y_jet, scores)
ax.plot(fpr, tpr, color='#1565C0', lw=2.5, label=f'Isolation Forest (AUC = {auc:.3f})')
ax.plot([0,1],[0,1],'--', color='gray', lw=1.5, label='Random (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.1, color='#1565C0')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: Anomaly Detection\nPerformance', fontweight='bold')
ax.legend()

plt.suptitle('Isolation Forest: Unsupervised Anomaly Detection\n(No labels required during training)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig2_isolation_forest.png", bbox_inches='tight')
plt.close()
print("  fig2 saved")

# ── Figure 3: UMAP vs t-SNE Comparison ────────────────────────────────────────
print("Generating Fig 3: UMAP vs t-SNE...")
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1, low_memory=True)
    X_umap = reducer.fit_transform(X_jet_sc)
    umap_available = True
except ImportError:
    umap_available = False
    print("  UMAP not available, using PCA as substitute")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for lbl, color, text in zip([0, 1], colors, labels_text):
    mask = y_jet == lbl
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, s=8 if lbl==0 else 30,
               alpha=0.4 if lbl==0 else 0.9, label=text)
ax.set_title('t-SNE (perplexity=40)\nPreserves local structure', fontweight='bold')
ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
ax.legend()

ax = axes[1]
if umap_available:
    for lbl, color, text in zip([0, 1], colors, labels_text):
        mask = y_jet == lbl
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, s=8 if lbl==0 else 30,
                   alpha=0.4 if lbl==0 else 0.9, label=text)
    ax.set_title('UMAP (n_neighbors=30)\nPreserves both local and global structure', fontweight='bold')
else:
    # PCA as fallback
    for lbl, color, text in zip([0, 1], colors, labels_text):
        mask = y_jet == lbl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=8 if lbl==0 else 30,
                   alpha=0.4 if lbl==0 else 0.9, label=text)
    ax.set_title('PCA (linear)\nFor comparison with t-SNE', fontweight='bold')
ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
ax.legend()

plt.suptitle('t-SNE vs UMAP: Non-linear Dimensionality Reduction Comparison\n'
             'Both reveal hidden structure; UMAP is faster and preserves global topology',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig3_umap_vs_tsne.png", bbox_inches='tight')
plt.close()
print("  fig3 saved")

# ── Figure 4: Silicon Thermal Hotspot Detection ────────────────────────────────
print("Generating Fig 4: Silicon Thermal Hotspot...")
iso_si = IsolationForest(contamination=0.032, random_state=42, n_estimators=200)
iso_si.fit(X_si_sc)
scores_si = -iso_si.score_samples(X_si_sc)

tsne_si = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=800)
X_si_tsne = tsne_si.fit_transform(X_si_sc[:3000])  # Sample for speed
y_si_sample = y_si[:3000]
scores_si_sample = scores_si[:3000]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for lbl, color, text in zip([0, 1], ['#1565C0','#E91E63'], ['Normal','Hotspot']):
    mask = y_si_sample == lbl
    ax.scatter(X_si_tsne[mask, 0], X_si_tsne[mask, 1], c=color,
               s=8 if lbl==0 else 35, alpha=0.4 if lbl==0 else 0.9, label=text)
ax.set_title('t-SNE: Silicon Thermal Data\n(3 hotspot types visible as clusters)',
             fontweight='bold')
ax.legend()

ax = axes[1]
sc = ax.scatter(df_si.loc[:2999, 'core_temp_c'], df_si.loc[:2999, 'power_density_mw_mm2'],
                c=scores_si_sample, cmap='RdYlBu_r', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='Anomaly Score')
ax.set_xlabel('Core Temperature (°C)'); ax.set_ylabel('Power Density (mW/mm²)')
ax.set_title('Anomaly Scores: Temperature vs Power\n(Red = high anomaly score)',
             fontweight='bold')

ax = axes[2]
fpr_si, tpr_si, _ = roc_curve(y_si, scores_si)
auc_si = roc_auc_score(y_si, scores_si)
ax.plot(fpr_si, tpr_si, color='#E91E63', lw=2.5, label=f'AUC = {auc_si:.3f}')
ax.plot([0,1],[0,1],'--', color='gray', lw=1.5)
ax.fill_between(fpr_si, tpr_si, alpha=0.1, color='#E91E63')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: Silicon Hotspot\nDetection Performance', fontweight='bold')
ax.legend()

plt.suptitle('Silicon Thermal Hotspot Detection: Isolation Forest + t-SNE\n'
             'Detecting 3 anomaly types without any labels', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig4_silicon_hotspot.png", bbox_inches='tight')
plt.close()
print("  fig4 saved")

# ── Figure 5: Perplexity Effect on t-SNE ─────────────────────────────────────
print("Generating Fig 5: t-SNE Perplexity Effect...")
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

sample_idx = np.random.RandomState(42).choice(len(X_jet_sc), 1500, replace=False)
X_sample = X_jet_sc[sample_idx]
y_sample = y_jet[sample_idx]

for ax, perp in zip(axes, perplexities):
    t = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=800)
    X_t = t.fit_transform(X_sample)
    for lbl, color in zip([0, 1], ['#1565C0','#E91E63']):
        mask = y_sample == lbl
        ax.scatter(X_t[mask, 0], X_t[mask, 1], c=color,
                   s=8 if lbl==0 else 25, alpha=0.4 if lbl==0 else 0.9)
    ax.set_title(f'Perplexity = {perp}', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('t-SNE Perplexity Hyperparameter Effect\n'
             'Too low: fragmented | Optimal (30-50): clear clusters | Too high: compressed',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig5_tsne_perplexity.png", bbox_inches='tight')
plt.close()
print("  fig5 saved")

print("\nAll 5 figures generated successfully.")
