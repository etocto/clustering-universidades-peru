"""
04_perfilado.py
===============
Perfilado e interpretación de los clústeres resultantes.
  - Radar charts por clúster
  - Test Kruskal-Wallis por variable
  - Test post-hoc de Dunn (corrección Bonferroni)
  - Tabla descriptiva media ± SD por clúster

Genera:
  - outputs/figures/04a_radar_clusters.png
  - outputs/figures/04b_heatmap_features.png
  - outputs/figures/04c_boxplots_clave.png
  - outputs/tables/04_descriptiva_por_cluster.csv
  - outputs/tables/04_kruskal_wallis.csv
  - outputs/tables/04_dunn_posthoc.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import stats
from itertools import combinations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, FIG_DIR, TABLE_DIR, FEATURE_COLS,
                    MPL_STYLE, CLUSTER_COLORS, DATA_URLS)

plt.rcParams.update(MPL_STYLE)


# ── 1. Cargar datos ─────────────────────────────────────────────────────────────
local_m = DATA_DIR / "matriz_maestra.csv"
df = pd.read_csv(local_m) if local_m.exists() else pd.read_csv(DATA_URLS["maestra"])
labels = np.load(DATA_DIR / "labels_final.npy")
K = len(np.unique(labels))

df["cluster"] = labels + 1
df["cluster_label"] = df["cluster"].map(
    lambda c: {1: "Investigación\nconsolidada",
               2: "Docencia\nestabilizada",
               3: "Masificadas\nflexibilizadas",
               4: "Regionales en\ndesarrollo",
               5: f"Clúster 5"}.get(c, f"Clúster {c}")
)

print(f"Datos cargados: {len(df)} IES, {K} clústeres")
for cl in range(1, K + 1):
    sub = df[df["cluster"] == cl]
    pub = sub["es_publico"].sum()
    print(f"  Clúster {cl} ({sub['cluster_label'].iloc[0].replace(chr(10),' ')}): "
          f"{len(sub)} IES  ({pub} pub / {len(sub)-pub} priv)")


# ── 2. Features clave para figuras (subconjunto informativo) ───────────────────
FEATS_RADAR = [
    "pct_doctorado", "pct_renacyt_doc", "pct_exclusiva",
    "pct_contratado", "nota_prom_egr", "puntaje_medio",
    "pct_prod_rec", "pct_discap", "ratio_mat_doc",
]

FEATS_CLAVE = [
    "pct_doctorado", "pct_renacyt_doc", "pct_exclusiva",
    "pct_contratado", "puntaje_medio", "nota_prom_egr",
    "pct_prod_rec", "ratio_mat_doc", "n_departamentos",
]


# ── 3. Tabla descriptiva ────────────────────────────────────────────────────────
rows = []
for cl in range(1, K + 1):
    sub = df[df["cluster"] == cl][FEATURE_COLS]
    row = {"cluster": cl, "n": len(df[df["cluster"] == cl])}
    for col in FEATURE_COLS:
        row[f"{col}_mean"] = round(sub[col].mean(), 3)
        row[f"{col}_sd"]   = round(sub[col].std(), 3)
    rows.append(row)

df_desc = pd.DataFrame(rows)
df_desc.to_csv(TABLE_DIR / "04_descriptiva_por_cluster.csv", index=False)
print("Guardado: 04_descriptiva_por_cluster.csv")


# ── 4. Kruskal-Wallis ──────────────────────────────────────────────────────────
print("\nKruskal-Wallis:")
kw_rows = []
for col in FEATURE_COLS:
    groups = [df.loc[df["cluster"] == cl, col].dropna().values for cl in range(1, K + 1)]
    stat, p = stats.kruskal(*groups)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    kw_rows.append({"feature": col, "H": round(stat, 3), "p_value": round(p, 6), "sig": sig})

df_kw = pd.DataFrame(kw_rows).sort_values("p_value")
df_kw.to_csv(TABLE_DIR / "04_kruskal_wallis.csv", index=False)
sig_feats = df_kw[df_kw["p_value"] < 0.05]["feature"].tolist()
print(f"  Variables significativas (p<0.05): {len(sig_feats)}/{len(FEATURE_COLS)}")
print(df_kw.head(10).to_string(index=False))


# ── 5. Test post-hoc de Dunn ────────────────────────────────────────────────────
def dunn_bonferroni(data, labels_col, value_col):
    groups  = data[labels_col].unique()
    pares   = list(combinations(sorted(groups), 2))
    n_total = len(data)
    n_tests = len(pares)
    rows    = []
    all_vals = data[value_col].rank().values
    n_i_dict = data.groupby(labels_col).size()
    for g1, g2 in pares:
        r1 = data.loc[data[labels_col] == g1, value_col].rank(method="average")
        r2 = data.loc[data[labels_col] == g2, value_col].rank(method="average")
        n1, n2 = len(r1), len(r2)
        u, p = stats.mannwhitneyu(
            data.loc[data[labels_col] == g1, value_col].values,
            data.loc[data[labels_col] == g2, value_col].values,
            alternative="two-sided"
        )
        p_adj = min(p * n_tests, 1.0)
        rows.append({"group1": g1, "group2": g2,
                     "U": round(u, 1), "p_raw": round(p, 6),
                     "p_bonferroni": round(p_adj, 6),
                     "sig": "***" if p_adj < 0.001 else "**" if p_adj < 0.01
                            else "*" if p_adj < 0.05 else "n.s."})
    return pd.DataFrame(rows)

dunn_rows = []
for col in sig_feats[:10]:
    df_d = dunn_bonferroni(df, "cluster", col)
    df_d.insert(0, "feature", col)
    dunn_rows.append(df_d)

if dunn_rows:
    pd.concat(dunn_rows).to_csv(TABLE_DIR / "04_dunn_posthoc.csv", index=False)
    print("Guardado: 04_dunn_posthoc.csv")


# ── 6. Radar charts ─────────────────────────────────────────────────────────────
def radar_chart(ax, values, labels_radar, color, title, max_vals, min_vals):
    N = len(labels_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    norm = [(v - mn) / (mx - mn + 1e-9)
            for v, mx, mn in zip(values, max_vals, min_vals)]
    norm += norm[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    short = [l.replace("pct_","").replace("_"," ") for l in labels_radar]
    ax.set_xticklabels(short, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    ax.plot(angles, norm, color=color, linewidth=2)
    ax.fill(angles, norm, color=color, alpha=0.25)
    ax.set_title(title, size=10, pad=15, fontweight="500")

max_vals = [df[f].max() for f in FEATS_RADAR]
min_vals = [df[f].min() for f in FEATS_RADAR]

cols_fig = 2 if K <= 4 else 3
rows_fig = (K + cols_fig - 1) // cols_fig
fig, axes = plt.subplots(rows_fig, cols_fig,
                         figsize=(cols_fig * 5, rows_fig * 4.5),
                         subplot_kw=dict(polar=True))
axes_flat = list(axes.flat) if K > 1 else [axes]

for cl in range(1, K + 1):
    ax = axes_flat[cl - 1]
    vals   = df[df["cluster"] == cl][FEATS_RADAR].mean().values
    label  = df[df["cluster"] == cl]["cluster_label"].iloc[0].replace("\n", " ")
    n_ies  = (df["cluster"] == cl).sum()
    radar_chart(ax, vals, FEATS_RADAR, CLUSTER_COLORS[cl - 1],
                f"Clúster {cl} — {label}\n(n={n_ies})", max_vals, min_vals)

# Ocultar ejes sobrantes
for ax in list(axes_flat)[K:]:
    ax.set_visible(False)

fig.suptitle("Perfil de clústeres — Radar charts (valores normalizados 0–1)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "04a_radar_clusters.png", bbox_inches="tight")
plt.close()
print("Guardado: 04a_radar_clusters.png")


# ── 7. Heatmap de features por clúster ─────────────────────────────────────────
heat_data = df.groupby("cluster")[FEATS_CLAVE].mean()
heat_data_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)

fig, ax = plt.subplots(figsize=(12, K * 1.2 + 2))
sns.heatmap(
    heat_data_norm,
    annot=heat_data.round(2),
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.4,
    linecolor="#E8E6DF",
    ax=ax,
    cbar_kws={"shrink": 0.6, "label": "Valor normalizado (0–1)"},
    annot_kws={"size": 9}
)
ax.set_yticklabels([f"Clúster {i}" for i in range(1, K + 1)], rotation=0, fontsize=10)
ax.set_xticklabels([f.replace("pct_","").replace("_"," ") for f in FEATS_CLAVE],
                   rotation=35, ha="right", fontsize=9)
ax.set_title("Heatmap — Media por clúster (valores originales anotados)", fontsize=12, pad=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "04b_heatmap_features.png", bbox_inches="tight")
plt.close()
print("Guardado: 04b_heatmap_features.png")


# ── 8. Boxplots de 6 variables clave ───────────────────────────────────────────
BOXPLOT_FEATS = ["pct_doctorado","pct_renacyt_doc","pct_contratado",
                 "puntaje_medio","nota_prom_egr","pct_prod_rec"]
BOXPLOT_LABELS = ["% doctorado","% RENACYT","% contratado",
                  "Puntaje RENACYT","Nota prom. egresado","% prod. reciente"]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, feat, label in zip(axes.flat, BOXPLOT_FEATS, BOXPLOT_LABELS):
    data_cl = [df.loc[df["cluster"] == cl, feat].dropna().values
               for cl in range(1, K + 1)]
    bp = ax.boxplot(data_cl, patch_artist=True, notch=False,
                    medianprops=dict(color="#2C2C2A", linewidth=2))
    for patch, color in zip(bp["boxes"], CLUSTER_COLORS[:K]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"C{i}" for i in range(1, K + 1)])
    ax.set_title(label, fontsize=10)

fig.suptitle("Distribución de variables clave por clúster", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "04c_boxplots_clave.png", bbox_inches="tight")
plt.close()
print("Guardado: 04c_boxplots_clave.png")

print(f"\n{'='*50}")
print("Perfilado completado. Revisa outputs/figures/ y outputs/tables/")
print("Edita 'cluster_label' en este script según los clústeres que encuentres.")
