"""
02_k_optimo.py
==============
Determina el número óptimo de clústeres con tres criterios:
  1. Elbow / Within-SS
  2. Silhouette score
  3. Gap statistic (bootstrap n=100)

Genera:
  - outputs/figures/02a_elbow.png
  - outputs/figures/02b_silhouette.png
  - outputs/figures/02c_gap_statistic.png
  - outputs/tables/02_k_scores.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIG_DIR, TABLE_DIR, K_RANGE, RANDOM_STATE, MPL_STYLE

plt.rcParams.update(MPL_STYLE)


# ── 1. Cargar X_pca ─────────────────────────────────────────────────────────────
X_pca = np.load(DATA_DIR / "X_pca.npy")
print(f"X_pca cargado: {X_pca.shape}")


# ── 2. Elbow (inercia) ──────────────────────────────────────────────────────────
print("\nCalculando inercia (elbow)...")
inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    print(f"  k={k}: inercia={km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_RANGE), inertias, marker="o", color="#7F77DD",
        linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Inercia (Within-cluster SS)")
ax.set_title("Elbow method — Selección de k", fontsize=12)
ax.set_xticks(list(K_RANGE))

# Marcar el codo automáticamente (diferencia de segunda derivada)
diffs2 = np.diff(np.diff(inertias))
elbow_k = list(K_RANGE)[np.argmax(diffs2) + 2]
ax.axvline(elbow_k, color="#D85A30", linestyle="--", linewidth=1, alpha=0.7)
ax.text(elbow_k + 0.1, max(inertias) * 0.95, f"k={elbow_k}", color="#D85A30", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "02a_elbow.png", bbox_inches="tight")
plt.close()
print(f"Guardado: 02a_elbow.png  (codo automático detectado en k={elbow_k})")


# ── 3. Silhouette score global ──────────────────────────────────────────────────
print("\nCalculando silhouette scores...")
sil_scores = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    print(f"  k={k}: silhouette={score:.4f}")

best_k_sil = list(K_RANGE)[np.argmax(sil_scores)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_RANGE), sil_scores, marker="s", color="#1D9E75",
        linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
ax.axvline(best_k_sil, color="#D85A30", linestyle="--", linewidth=1, alpha=0.7)
ax.text(best_k_sil + 0.1, min(sil_scores) + 0.005,
        f"k={best_k_sil} (max)", color="#D85A30", fontsize=10)
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("Silhouette score — Cohesión y separación", fontsize=12)
ax.set_xticks(list(K_RANGE))
fig.tight_layout()
fig.savefig(FIG_DIR / "02b_silhouette.png", bbox_inches="tight")
plt.close()
print(f"Guardado: 02b_silhouette.png  (mejor k={best_k_sil})")


# ── 4. Silhouette diagram para k óptimo ────────────────────────────────────────
k_plot = best_k_sil
km = KMeans(n_clusters=k_plot, random_state=RANDOM_STATE, n_init=30)
labels_k = km.fit_predict(X_pca)
sil_vals  = silhouette_samples(X_pca, labels_k)

fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
colors_sil = plt.cm.tab10(np.linspace(0, 1, k_plot))
for i in range(k_plot):
    ith = np.sort(sil_vals[labels_k == i])
    size = len(ith)
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith,
                     facecolor=colors_sil[i], edgecolor=colors_sil[i], alpha=0.8)
    ax.text(-0.03, y_lower + size / 2, f"C{i+1}", fontsize=8)
    y_lower = y_upper + 5

avg = sil_vals.mean()
ax.axvline(avg, color="#D85A30", linestyle="--", linewidth=1)
ax.text(avg + 0.01, y_lower - 10, f"media={avg:.3f}", color="#D85A30", fontsize=9)
ax.set_xlabel("Silhouette coefficient")
ax.set_title(f"Diagrama silhouette — k={k_plot}", fontsize=12)
ax.set_yticks([])
fig.tight_layout()
fig.savefig(FIG_DIR / "02b2_silhouette_diagram.png", bbox_inches="tight")
plt.close()


# ── 5. Gap Statistic ────────────────────────────────────────────────────────────
print("\nCalculando Gap statistic (n_bootstrap=100)...")

def gap_statistic(X, k_range, n_bootstrap=100, random_state=42):
    rng = np.random.default_rng(random_state)
    gaps, sk = [], []
    X_min, X_max = X.min(axis=0), X.max(axis=0)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        km.fit(X)
        Wk = km.inertia_

        # Bootstrap sobre distribución uniforme
        Wk_boots = []
        for _ in range(n_bootstrap):
            X_rand = rng.uniform(X_min, X_max, size=X.shape)
            km_b = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km_b.fit(X_rand)
            Wk_boots.append(np.log(km_b.inertia_))
        l_bar = np.mean(Wk_boots)
        sdk   = np.std(Wk_boots) * np.sqrt(1 + 1 / n_bootstrap)
        gaps.append(l_bar - np.log(Wk))
        sk.append(sdk)
        print(f"  k={k}: gap={gaps[-1]:.4f}  sk={sdk:.4f}")

    # Criterio de Tibshirani: menor k tal que gap(k) >= gap(k+1) - s(k+1)
    best_k_gap = list(k_range)[0]
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sk[i + 1]:
            best_k_gap = list(k_range)[i]
            break
    return gaps, sk, best_k_gap

gaps, sks, best_k_gap = gap_statistic(X_pca, K_RANGE)

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(list(K_RANGE), gaps, yerr=sks, marker="D", color="#BA7517",
            linewidth=2, markersize=6, capsize=4, elinewidth=1,
            markerfacecolor="white", markeredgewidth=2)
ax.axvline(best_k_gap, color="#D85A30", linestyle="--", linewidth=1, alpha=0.7)
ax.text(best_k_gap + 0.1, min(gaps) + 0.01, f"k={best_k_gap}", color="#D85A30", fontsize=10)
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Gap statistic")
ax.set_title("Gap statistic — Criterio de Tibshirani et al. (2001)", fontsize=12)
ax.set_xticks(list(K_RANGE))
fig.tight_layout()
fig.savefig(FIG_DIR / "02c_gap_statistic.png", bbox_inches="tight")
plt.close()
print(f"Guardado: 02c_gap_statistic.png  (mejor k={best_k_gap})")


# ── 6. Tabla resumen de k scores ────────────────────────────────────────────────
df_scores = pd.DataFrame({
    "k":          list(K_RANGE),
    "inertia":    [round(v, 2) for v in inertias],
    "silhouette": [round(v, 4) for v in sil_scores],
    "gap":        [round(v, 4) for v in gaps],
    "gap_sk":     [round(v, 4) for v in sks],
})
df_scores.to_csv(TABLE_DIR / "02_k_scores.csv", index=False)
print("Guardado: 02_k_scores.csv")


# ── 7. Resumen ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Resultados de los tres criterios:")
print(f"  Elbow (codo 2ª derivada) : k = {elbow_k}")
print(f"  Silhouette (máximo)      : k = {best_k_sil}")
print(f"  Gap statistic (Tibshirani): k = {best_k_gap}")
votes = [elbow_k, best_k_sil, best_k_gap]
from collections import Counter
k_consenso = Counter(votes).most_common(1)[0][0]
print(f"\n  >> Recomendación: k = {k_consenso} <<")
print(f"     (editar K_FINAL en config.py si decides otro valor)")
print(f"{'='*50}")

# Guardar k recomendado para uso en siguientes scripts
np.save(DATA_DIR / "k_optimo.npy", np.array([k_consenso]))
