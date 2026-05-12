"""
01_pca.py
=========
Análisis de Componentes Principales (PCA) sobre la matriz estandarizada.
Genera:
  - outputs/figures/01a_scree_plot.png
  - outputs/figures/01b_biplot_pc1_pc2.png
  - outputs/figures/01c_varianza_acumulada.png
  - outputs/tables/01_pca_loadings.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, FIG_DIR, TABLE_DIR, FEATURE_COLS,
                    PCA_VARIANCE, MPL_STYLE, CLUSTER_COLORS, DATA_URLS)

plt.rcParams.update(MPL_STYLE)


# ── 1. Cargar datos ─────────────────────────────────────────────────────────────
def cargar_datos():
    local = DATA_DIR / "matriz_escalada.csv"
    if local.exists():
        print(f"Cargando desde local: {local}")
        return pd.read_csv(local)
    print(f"Cargando desde GitHub: {DATA_URLS['escalada']}")
    return pd.read_csv(DATA_URLS["escalada"])


df_scaled = cargar_datos()
df_master = pd.read_csv(DATA_DIR / "matriz_maestra.csv") if (DATA_DIR / "matriz_maestra.csv").exists() \
            else pd.read_csv(DATA_URLS["maestra"])

X = df_scaled[FEATURE_COLS].values
unis = df_scaled["universidad"].values
es_publico = df_master["es_publico"].values
print(f"Matriz cargada: {X.shape[0]} universidades × {X.shape[1]} features")


# ── 2. Ajustar PCA ──────────────────────────────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X)

var_ratio     = pca_full.explained_variance_ratio_
var_cumul     = np.cumsum(var_ratio)
n_comp_90     = int(np.argmax(var_cumul >= PCA_VARIANCE) + 1)

print(f"\nVarianza explicada por componente:")
for i, v in enumerate(var_ratio[:8]):
    print(f"  PC{i+1}: {v*100:.1f}%  (acum: {var_cumul[i]*100:.1f}%)")
print(f"\nComponentes para retener {PCA_VARIANCE*100:.0f}% varianza: {n_comp_90}")

pca = PCA(n_components=n_comp_90, random_state=42)
X_pca = pca.fit_transform(X)

np.save(DATA_DIR / "X_pca.npy", X_pca)
print(f"X_pca guardado en data/X_pca.npy  ({X_pca.shape})")


# ── 3. Scree plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
n_show = min(12, len(var_ratio))
bars = ax.bar(
    range(1, n_show + 1), var_ratio[:n_show] * 100,
    color="#AFA9EC", edgecolor="#534AB7", linewidth=0.5, label="Individual variance"
)
ax2 = ax.twinx()
ax2.plot(
    range(1, n_show + 1), var_cumul[:n_show] * 100,
    color="#D85A30", marker="o", markersize=5, linewidth=1.5, label="Cumulative variance"
)
ax2.axhline(90, color="#D85A30", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.axvline(n_comp_90, color="#888780", linestyle=":", linewidth=0.8)
ax2.text(n_comp_90 + 0.1, 91, f"PC{n_comp_90} = 90%", fontsize=9, color="#D85A30")
ax.set_xlabel("Principal component")
ax.set_ylabel("Individual variance (%)")
ax2.set_ylabel("Cumulative variance (%)")
ax2.set_ylim(0, 105)
ax.set_xticks(range(1, n_show + 1))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
ax.set_title("Scree plot — Explained variance by principal component", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "01a_scree_plot.png", bbox_inches="tight")
plt.close()
print("Guardado: 01a_scree_plot.png")


# ── 4. Varianza acumulada ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(var_cumul) + 1), var_cumul * 100,
        color="#7F77DD", marker="o", markersize=4, linewidth=1.5)
ax.axhline(90, color="#D85A30", linestyle="--", linewidth=0.8)
ax.axvline(n_comp_90, color="#D85A30", linestyle="--", linewidth=0.8)
ax.fill_between(range(1, n_comp_90 + 1), var_cumul[:n_comp_90] * 100,
                alpha=0.15, color="#7F77DD")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative variance (%)")
ax.set_title("Cumulative variance explained by PCA", fontsize=12)
ax.set_ylim(0, 105)
ax.set_xticks(range(1, len(var_ratio) + 1))
fig.tight_layout()
fig.savefig(FIG_DIR / "01c_varianza_acumulada.png", bbox_inches="tight")
plt.close()
print("Guardado: 01c_varianza_acumulada.png")


# ── 5. Biplot PC1 vs PC2 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))

colors = ["#7F77DD" if p == 1 else "#D85A30" for p in es_publico]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=60,
                     alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)

# Etiquetas selectivas (IES con valores extremos en PC1 o PC2)
pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
threshold = 1.8
for i, nombre in enumerate(unis):
    if abs(pc1[i]) > threshold or abs(pc2[i]) > threshold:
        short = nombre[:28] + "…" if len(nombre) > 28 else nombre
        ax.annotate(short, (pc1[i], pc2[i]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

# Vectores de loadings (flechas de variables)
loadings = pca.components_.T
scale = 3.5
for j, feat in enumerate(FEATURE_COLS):
    lx, ly = loadings[j, 0] * scale, loadings[j, 1] * scale
    if np.sqrt(lx**2 + ly**2) > 0.8:
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#1D9E75",
                                   lw=1.2, alpha=0.7))
        ax.text(lx * 1.08, ly * 1.08, feat.replace("pct_", "").replace("_", " "),
                fontsize=7, color="#085041", alpha=0.85)

ax.axhline(0, color="#B4B2A9", linewidth=0.5)
ax.axvline(0, color="#B4B2A9", linewidth=0.5)
ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("PCA Biplot — Peruvian universities (PC1 vs PC2)", fontsize=13)

leg = [mpatches.Patch(color="#7F77DD", label="Public"),
       mpatches.Patch(color="#D85A30", label="Private")]
ax.legend(handles=leg, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "01b_biplot_pc1_pc2.png", bbox_inches="tight")
plt.close()
print("Guardado: 01b_biplot_pc1_pc2.png")


# ── 6. Tabla de loadings ────────────────────────────────────────────────────────
loadings_df = pd.DataFrame(
    pca.components_.T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
).round(3)
loadings_df["comunalidad"] = (pca.components_.T ** 2).sum(axis=1).round(3)
loadings_df.index.name = "feature"
loadings_df.to_csv(TABLE_DIR / "01_pca_loadings.csv")
print("Guardado: 01_pca_loadings.csv")

print(f"\n{'='*50}")
print(f"PCA completado. Componentes retenidos: {n_comp_90}")
print(f"Varianza explicada: {var_cumul[n_comp_90-1]*100:.1f}%")
print(f"Archivos en outputs/figures/ y outputs/tables/")
