"""
03_clustering.py
================
Aplica cuatro algoritmos de clustering y valida su consistencia.
  1. K-Means++
  2. Clustering jerárquico (Ward)
  3. DBSCAN
  4. Gaussian Mixture Model (GMM)

Genera:
  - outputs/figures/03a_dendrograma.png
  - outputs/figures/03b_scatter_kmeans.png
  - outputs/figures/03c_scatter_todos.png
  - outputs/tables/03_asignacion_clusters.csv
  - outputs/tables/03_validacion_cruzada.csv
  - data/labels_final.npy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, FIG_DIR, TABLE_DIR, RANDOM_STATE,
                    MPL_STYLE, CLUSTER_COLORS, DATA_URLS, K_FINAL)

plt.rcParams.update(MPL_STYLE)


# ── 1. Cargar datos ─────────────────────────────────────────────────────────────
X_pca  = np.load(DATA_DIR / "X_pca.npy")
k_arr  = np.load(DATA_DIR / "k_optimo.npy")
K      = int(K_FINAL) if K_FINAL else int(k_arr[0])

local_m = DATA_DIR / "matriz_maestra.csv"
df_master = pd.read_csv(local_m) if local_m.exists() else pd.read_csv(DATA_URLS["maestra"])
unis      = df_master["universidad"].values
es_pub    = df_master["es_publico"].values

print(f"Datos cargados: {X_pca.shape[0]} IES  |  k = {K}")


# ── 2. K-Means++ ────────────────────────────────────────────────────────────────
km = KMeans(n_clusters=K, init="k-means++", n_init=50,
            max_iter=500, random_state=RANDOM_STATE)
labels_km = km.fit_predict(X_pca)
print(f"\nK-Means:  distribución por clúster → {dict(zip(*np.unique(labels_km, return_counts=True)))}")


# ── 3. Ward jerárquico + dendrograma ───────────────────────────────────────────
Z = linkage(X_pca, method="ward")
labels_ward = fcluster(Z, t=K, criterion="maxclust") - 1

fig, ax = plt.subplots(figsize=(14, 6))
short_names = [n[:22] + "…" if len(n) > 22 else n for n in unis]
dend = dendrogram(
    Z, labels=short_names, ax=ax,
    leaf_font_size=6.5, color_threshold=Z[-K+1, 2],
    above_threshold_color="#B4B2A9"
)
ax.set_title(f"Dendrograma Ward — Clustering jerárquico (k={K})", fontsize=12)
ax.set_xlabel("Universidad")
ax.set_ylabel("Distancia euclidiana (espacio PCA)")
ax.axhline(Z[-K+1, 2], color="#D85A30", linestyle="--", linewidth=1, alpha=0.7)
ax.text(2, Z[-K+1, 2] * 1.02, f"corte k={K}", color="#D85A30", fontsize=9)
plt.xticks(rotation=90)
fig.tight_layout()
fig.savefig(FIG_DIR / "03a_dendrograma.png", bbox_inches="tight", dpi=180)
plt.close()
print("Guardado: 03a_dendrograma.png")
print(f"Ward:     distribución → {dict(zip(*np.unique(labels_ward, return_counts=True)))}")


# ── 4. DBSCAN ───────────────────────────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_pca)
distances, _ = nbrs.kneighbors(X_pca)
eps_auto = np.percentile(np.sort(distances[:, -1]), 90)

db = DBSCAN(eps=eps_auto, min_samples=3)
labels_db = db.fit_predict(X_pca)
n_noise = (labels_db == -1).sum()
n_clus  = len(set(labels_db)) - (1 if -1 in labels_db else 0)
print(f"DBSCAN:   eps={eps_auto:.3f}  →  {n_clus} clústeres + {n_noise} outliers (ruido)")


# ── 5. GMM ──────────────────────────────────────────────────────────────────────
gmm = GaussianMixture(n_components=K, covariance_type="full",
                      n_init=20, random_state=RANDOM_STATE)
gmm.fit(X_pca)
labels_gmm  = gmm.predict(X_pca)
probs_gmm   = gmm.predict_proba(X_pca)
bic_gmm     = gmm.bic(X_pca)
print(f"GMM:      BIC={bic_gmm:.1f}  →  distribución {dict(zip(*np.unique(labels_gmm, return_counts=True)))}")


# ── 6. Validación cruzada entre algoritmos ─────────────────────────────────────
pares = [
    ("K-Means", "Ward",    labels_km,   labels_ward),
    ("K-Means", "GMM",     labels_km,   labels_gmm),
    ("Ward",    "GMM",     labels_ward, labels_gmm),
]
print("\nValidación cruzada (ARI y NMI):")
valid_rows = []
for n1, n2, l1, l2 in pares:
    ari = adjusted_rand_score(l1, l2)
    nmi = normalized_mutual_info_score(l1, l2)
    print(f"  {n1:8s} vs {n2:8s}:  ARI={ari:.3f}  NMI={nmi:.3f}"
          + ("  ✓ robusto" if ari > 0.70 else "  ⚠ revisar"))
    valid_rows.append({"par": f"{n1} vs {n2}", "ARI": round(ari, 3), "NMI": round(nmi, 3)})

pd.DataFrame(valid_rows).to_csv(TABLE_DIR / "03_validacion_cruzada.csv", index=False)
print("Guardado: 03_validacion_cruzada.csv")


# ── 7. Figura comparativa 4 algoritmos ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
algo_results = [
    ("K-Means++", labels_km),
    ("Ward (jerárquico)", labels_ward),
    ("DBSCAN", labels_db),
    ("GMM", labels_gmm),
]

for ax, (title, labs) in zip(axes.flat, algo_results):
    unique = sorted(set(labs))
    cmap   = {l: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
              for i, l in enumerate(u for u in unique if u != -1)}
    cmap[-1] = "#B4B2A9"
    c_list = [cmap[l] for l in labs]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=c_list,
               s=55, alpha=0.82, edgecolors="white", linewidths=0.4)
    handles = [mpatches.Patch(color=cmap[l],
               label=("Outlier" if l == -1 else f"Clúster {l+1}")) for l in unique]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)

fig.suptitle("Comparación de algoritmos de clustering — Espacio PCA (PC1 vs PC2)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "03c_scatter_todos.png", bbox_inches="tight")
plt.close()
print("Guardado: 03c_scatter_todos.png")


# ── 8. Scatter K-Means detallado (con nombres) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
for cl in range(K):
    mask = labels_km == cl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[cl], s=70, label=f"Clúster {cl+1}",
               alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)

threshold_label = 1.5
for i, nombre in enumerate(unis):
    if abs(X_pca[i, 0]) > threshold_label or abs(X_pca[i, 1]) > threshold_label:
        short = nombre[:26] + "…" if len(nombre) > 26 else nombre
        ax.annotate(short, (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

ax.axhline(0, color="#B4B2A9", linewidth=0.5)
ax.axvline(0, color="#B4B2A9", linewidth=0.5)
ax.set_xlabel("PC1", fontsize=11)
ax.set_ylabel("PC2", fontsize=11)
ax.set_title(f"K-Means (k={K}) en espacio PCA — Universidades peruanas 2025", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "03b_scatter_kmeans.png", bbox_inches="tight")
plt.close()
print("Guardado: 03b_scatter_kmeans.png")


# ── 9. Guardar asignaciones ─────────────────────────────────────────────────────
df_asig = pd.DataFrame({
    "universidad":  unis,
    "es_publico":   es_pub,
    "cluster_kmeans": labels_km + 1,
    "cluster_ward":   labels_ward + 1,
    "cluster_gmm":    labels_gmm + 1,
    "cluster_dbscan": labels_db,
    "prob_cluster1_gmm": probs_gmm[:, 0].round(3),
    "prob_cluster2_gmm": probs_gmm[:, 1].round(3) if K > 1 else 0,
})
df_asig.to_csv(TABLE_DIR / "03_asignacion_clusters.csv", index=False)
np.save(DATA_DIR / "labels_final.npy", labels_km)
print("Guardado: 03_asignacion_clusters.csv  |  data/labels_final.npy")

print(f"\n{'='*50}")
print(f"Clustering completado. k={K} clústeres.")
print(f"Usar K-Means como etiqueta principal (más estable y reproducible).")
