"""
07_disciplinar.py
=================
Option A — Disciplinary profile: for each HEI computes the percentage
of enrolled students in each of the 10 SUNEDU knowledge areas, then
clusters HEIs by their academic specialisation profile.

Generates:
  - outputs/figures/07a_radar_disciplinar.png
  - outputs/figures/07b_scatter_disciplinar.png
  - outputs/figures/07c_heatmap_areas.png
  - outputs/figures/07d_barras_especializacion.png
  - outputs/tables/07_perfil_disciplinar.csv
  - outputs/tables/07_clusters_disciplinar.csv
  - data/labels_disciplinar.npy
"""

import csv as _csv, sys, zipfile, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIG_DIR, TABLE_DIR, MPL_STYLE, CLUSTER_COLORS, DATA_URLS, RANDOM_STATE

plt.rcParams.update(MPL_STYLE)


def open_csv_smart(path, encoding="latin1", delimiter="|"):
    """Opens a CSV that might be a plain file, a ZIP, or a nested ZIP.
    Also searches /tmp/data/ as fallback location."""
    name = Path(path).name
    candidates = [path,
                  Path(__file__).parent.parent / "data" / name,
                  Path("/tmp/data") / name]
    real_path = next((str(c) for c in candidates if Path(str(c)).exists()), None)
    if real_path is None:
        raise FileNotFoundError(
            f"\n  Cannot find '{name}'.\n"
            f"  Copy it to the project data/ folder and re-run."
        )
    with open(real_path, "rb") as fh:
        sig = fh.read(4)
    if sig[:2] == b"PK":
        print(f"  ZIP detected — extracting {name}...")
        with zipfile.ZipFile(real_path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            zip_names = [n for n in zf.namelist() if n.endswith(".zip")]
            if csv_names:
                raw = zf.read(csv_names[0])
            elif zip_names:
                inner = io.BytesIO(zf.read(zip_names[0]))
                with zipfile.ZipFile(inner) as zf2:
                    csv2 = [n for n in zf2.namelist() if n.endswith(".csv")]
                    raw  = zf2.read(csv2[0])
            else:
                raise ValueError(f"No CSV found inside ZIP {real_path}")
        return _csv.DictReader(
            io.TextIOWrapper(io.BytesIO(raw), encoding=encoding), delimiter=delimiter)
    else:
        return _csv.DictReader(open(real_path, encoding=encoding), delimiter=delimiter)

# ── Short English labels for knowledge areas ────────────────────────────────────
AREA_LABELS = {
    "Ciencias Administrativas y Derecho":           "Business & Law",
    "Ingeniería, Industria y Construcción":          "Engineering",
    "Ciencias Sociales, Periodismo e Información":  "Social Sciences",
    "Salud y bienestar":                            "Health",
    "Tecnología de la Información y la Comunicación":"ICT",
    "Educación":                                    "Education",
    "Ciencias Naturales, Matemáticas y Estadística":"Natural Sciences",
    "Agricultura, Silvicultura, Pesca y Veterinaria":"Agriculture",
    "Arte y Humanidades":                           "Arts & Humanities",
    "Servicios":                                    "Services",
}
AREAS_ES = list(AREA_LABELS.keys())
AREAS_EN = list(AREA_LABELS.values())

DISC_COLORS = ["#7F77DD","#1D9E75","#D85A30","#BA7517","#378ADD"]


# ── 1. Compute % enrollment per area per HEI ────────────────────────────────────
print("Loading pre-computed disciplinary profiles...")
# Reads the small pre-computed file (127 rows) instead of the full 643MB matriculado
precomp_path = DATA_DIR / "precomputed_areas.csv"
if not precomp_path.exists():
    raise FileNotFoundError(
        "Missing: data/precomputed_areas.csv\n"
        "Copy this file (provided separately) into the data/ folder."
    )
df_disc = pd.read_csv(precomp_path)
# Rename columns to match AREAS_EN labels
col_map = {
    "Business_Law": "Business & Law",
    "Engineering": "Engineering",
    "Social_Sciences": "Social Sciences",
    "Health": "Health",
    "ICT": "ICT",
    "Education": "Education",
    "Natural_Sciences": "Natural Sciences",
    "Agriculture": "Agriculture",
    "Arts_Humanities": "Arts & Humanities",
    "Services": "Services",
}
df_disc = df_disc.rename(columns=col_map)
df_disc = df_disc.rename(columns={"total_matriculados": "total_mat"})
print(f"  HEIs with disciplinary data: {len(df_disc)}")


# ── 2. Merge with existing cluster assignments ──────────────────────────────────
local_m = DATA_DIR / "matriz_maestra.csv"
df_master = pd.read_csv(local_m) if local_m.exists() else pd.read_csv(DATA_URLS["maestra"])
labels_main = np.load(DATA_DIR / "labels_final.npy")
df_master["cluster_main"] = labels_main + 1

df_disc = df_disc.merge(
    df_master[["universidad","cluster_main","es_publico","es_licenciada"]],
    on="universidad", how="inner"
)
print(f"  HEIs after merge with master matrix: {len(df_disc)}")

df_disc.to_csv(TABLE_DIR / "07_perfil_disciplinar.csv", index=False)
print("Saved: 07_perfil_disciplinar.csv")

# Feature matrix for new clustering
X_disc = df_disc[AREAS_EN].values
scaler = StandardScaler()
X_s    = scaler.fit_transform(X_disc)


# ── 3. Optimal k for disciplinary clustering ────────────────────────────────────
print("\nFinding optimal k for disciplinary clustering...")
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, init="k-means++", n_init=30, random_state=RANDOM_STATE)
    lbl = km.fit_predict(X_s)
    sil_scores.append(silhouette_score(X_s, lbl))
    print(f"  k={k}: silhouette={sil_scores[-1]:.4f}")

best_k = list(range(2, 8))[np.argmax(sil_scores)]
print(f"  Best k = {best_k}")


# ── 4. Fit final disciplinary clustering ────────────────────────────────────────
km_disc = KMeans(n_clusters=best_k, init="k-means++",
                 n_init=50, random_state=RANDOM_STATE)
df_disc["cluster_disc"] = km_disc.fit_predict(X_s) + 1

np.save(DATA_DIR / "labels_disciplinar.npy", km_disc.labels_)
df_disc.to_csv(TABLE_DIR / "07_clusters_disciplinar.csv", index=False)
print(f"\nDisciplinary clusters distribution:")
for cl in range(1, best_k + 1):
    sub = df_disc[df_disc["cluster_disc"] == cl]
    pub = sub["es_publico"].sum()
    print(f"  Cluster {cl}: {len(sub)} HEIs  ({pub} pub / {len(sub)-pub} priv)")

# Auto-label clusters by dominant area
disc_labels = {}
for cl in range(1, best_k + 1):
    sub = df_disc[df_disc["cluster_disc"] == cl]
    dom_area = sub[AREAS_EN].mean().idxmax()
    disc_labels[cl] = dom_area
    print(f"    C{cl} dominant area: {dom_area}")


# ── 5. Figure A: radar charts per disciplinary cluster ─────────────────────────
def radar_chart(ax, values, labels_r, color, title):
    N = len(labels_r)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    max_v = max(values) if max(values) > 0 else 1
    norm  = [v / max_v for v in values] + [values[0] / max_v]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=7.5)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.plot(angles, norm, color=color, linewidth=2)
    ax.fill(angles, norm, color=color, alpha=0.22)
    ax.set_title(title, size=9, pad=14, fontweight="500")

cols_r = 3 if best_k > 3 else best_k
rows_r = (best_k + cols_r - 1) // cols_r
fig, axes = plt.subplots(rows_r, cols_r,
                         figsize=(cols_r * 4.5, rows_r * 4.5),
                         subplot_kw=dict(polar=True))
axes_list = list(np.array(axes).flatten()) if best_k > 1 else [axes]

for cl in range(1, best_k + 1):
    sub    = df_disc[df_disc["cluster_disc"] == cl]
    vals   = sub[AREAS_EN].mean().values.tolist()
    n_ies  = len(sub)
    label  = disc_labels[cl]
    color  = DISC_COLORS[cl - 1]
    radar_chart(axes_list[cl - 1], vals, AREAS_EN, color,
                f"Cluster {cl}: {label}\n(n={n_ies})")

for ax in axes_list[best_k:]:
    ax.set_visible(False)

fig.suptitle("Disciplinary profile clusters — % enrollment by knowledge area (normalized)",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "07a_radar_disciplinar.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 07a_radar_disciplinar.png")


# ── 6. Figure B: PCA scatter — disciplinary clusters ───────────────────────────
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca2.fit_transform(X_s)

fig, ax = plt.subplots(figsize=(9, 7))
for cl in range(1, best_k + 1):
    mask = df_disc["cluster_disc"] == cl
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=DISC_COLORS[cl - 1], s=65, alpha=0.82,
               edgecolors="white", linewidths=0.4,
               label=f"C{cl}: {disc_labels[cl]}")

for i, uni in enumerate(df_disc["universidad"]):
    if abs(X_2d[i, 0]) > 1.8 or abs(X_2d[i, 1]) > 1.8:
        short = uni[:24] + "…" if len(uni) > 24 else uni
        ax.annotate(short, (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=6.5, xytext=(4, 4),
                    textcoords="offset points", alpha=0.85)

ax.axhline(0, color="#B4B2A9", linewidth=0.5)
ax.axvline(0, color="#B4B2A9", linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("HEI disciplinary clusters — PCA space", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "07b_scatter_disciplinar.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 07b_scatter_disciplinar.png")


# ── 7. Figure C: heatmap areas × disciplinary cluster ──────────────────────────
heat = df_disc.groupby("cluster_disc")[AREAS_EN].mean().round(1)
heat.index = [f"C{i}: {disc_labels[i][:20]}" for i in heat.index]

fig, ax = plt.subplots(figsize=(12, max(3, best_k * 1.2) + 1))
sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.4, linecolor="#E8E6DF", ax=ax,
            cbar_kws={"shrink": 0.5, "label": "% enrollment"},
            annot_kws={"size": 9})
ax.set_title("Mean % enrollment by knowledge area per disciplinary cluster",
             fontsize=11, pad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "07c_heatmap_areas.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 07c_heatmap_areas.png")


# ── 8. Figure D: top areas per cluster (horizontal bars) ───────────────────────
fig, axes = plt.subplots(1, best_k, figsize=(best_k * 4, 5), sharey=True)
if best_k == 1:
    axes = [axes]

for cl, ax in zip(range(1, best_k + 1), axes):
    sub   = df_disc[df_disc["cluster_disc"] == cl]
    means = sub[AREAS_EN].mean().sort_values(ascending=True)
    colors_bar = [DISC_COLORS[cl - 1] if v == means.max()
                  else "#D3D1C7" for v in means.values]
    ax.barh(means.index, means.values,
            color=colors_bar, edgecolor="white", linewidth=0.4)
    ax.set_title(f"C{cl}: {disc_labels[cl][:22]}\n(n={len(sub)})", fontsize=9)
    ax.set_xlabel("% enrollment")
    for i, v in enumerate(means.values):
        if v > 0.5:
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7.5)

fig.suptitle("Dominant knowledge areas by disciplinary cluster", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "07d_barras_especializacion.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 07d_barras_especializacion.png")


# ── 9. Cross-tab: main cluster vs disciplinary cluster ─────────────────────────
cross = pd.crosstab(df_disc["cluster_main"], df_disc["cluster_disc"],
                    margins=True)
cross.index   = [f"Main C{i}" if i != "All" else "Total" for i in cross.index]
cross.columns = [f"Disc C{i}" if i != "All" else "Total" for i in cross.columns]
print("\nCross-tabulation: Main cluster vs Disciplinary cluster")
print(cross.to_string())

print(f"\n{'='*55}")
print("Disciplinary analysis complete.")
print(f"Best k = {best_k} disciplinary clusters found.")
print(f"{'='*55}")
