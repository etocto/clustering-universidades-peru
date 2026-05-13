"""
06_regional.py
==============
Option A — Geographic analysis: adds department and macro-region
to the existing cluster assignments and shows how the 4 clusters
distribute across Peru's 25 departments and 5 macro-regions.

Generates:
  - outputs/figures/06a_clusters_por_departamento.png
  - outputs/figures/06b_clusters_por_macroregion.png
  - outputs/figures/06c_mapa_burbujas_peru.png
  - outputs/figures/06d_heatmap_region_cluster.png
  - outputs/tables/06_ies_con_region.csv
  - outputs/tables/06_perfil_macroregion.csv
"""

import csv as _csv, sys, zipfile, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIG_DIR, TABLE_DIR, MPL_STYLE, CLUSTER_COLORS, DATA_URLS, \
                   CLUSTER_NAMES  # ← nombres centralizados desde config.py

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


# ── 1. Macro-region mapping ───────────────────────────────────────────────────
MACROREGION = {
    "Lima":         "Lima Metropolitana",
    "Callao":       "Lima Metropolitana",
    "Arequipa":     "Sur",
    "Moquegua":     "Sur",
    "Tacna":        "Sur",
    "Puno":         "Sur",
    "Cusco":        "Sur",
    "Apurímac":     "Sur",
    "Ayacucho":     "Sur",
    "Ica":          "Centro",
    "Junín":        "Centro",
    "Huancavelica": "Centro",
    "Pasco":        "Centro",
    "Áncash":       "Norte",
    "La Libertad":  "Norte",
    "Lambayeque":   "Norte",
    "Piura":        "Norte",
    "Tumbes":       "Norte",
    "Cajamarca":    "Norte",
    "Amazonas":     "Oriente",
    "San Martín":   "Oriente",
    "Loreto":       "Oriente",
    "Ucayali":      "Oriente",
    "Madre de Dios":"Oriente",
    "Huánuco":      "Oriente",
}

NATURAL_REGION = {
    "Lima": "Costa", "Callao": "Costa", "Arequipa": "Costa",
    "Moquegua": "Costa", "Tacna": "Costa", "Ica": "Costa",
    "La Libertad": "Costa", "Lambayeque": "Costa", "Piura": "Costa",
    "Tumbes": "Costa",
    "Puno": "Sierra", "Cusco": "Sierra", "Apurímac": "Sierra",
    "Ayacucho": "Sierra", "Junín": "Sierra", "Huancavelica": "Sierra",
    "Pasco": "Sierra", "Áncash": "Sierra", "Cajamarca": "Sierra",
    "Huánuco": "Sierra",
    "Amazonas": "Selva", "San Martín": "Selva", "Loreto": "Selva",
    "Ucayali": "Selva", "Madre de Dios": "Selva",
}

# Approximate centroids for bubble map (longitude, latitude)
DEPT_COORDS = {
    "Lima": (-76.9, -12.0), "Callao": (-77.1, -12.1),
    "Arequipa": (-71.5, -16.4), "Cusco": (-71.9, -13.5),
    "Puno": (-70.0, -15.8), "Tacna": (-70.2, -18.0),
    "Moquegua": (-70.9, -17.2), "Ica": (-75.7, -14.0),
    "Ayacucho": (-74.2, -13.2), "Apurímac": (-73.1, -14.0),
    "Junín": (-75.2, -11.2), "Huancavelica": (-74.8, -12.8),
    "Pasco": (-75.7, -10.7), "Áncash": (-77.8, -9.5),
    "La Libertad": (-78.5, -8.1), "Lambayeque": (-79.9, -6.8),
    "Piura": (-80.6, -5.2), "Tumbes": (-80.4, -3.6),
    "Cajamarca": (-78.5, -7.2), "Amazonas": (-77.9, -5.5),
    "San Martín": (-76.4, -6.5), "Loreto": (-74.6, -4.9),
    "Ucayali": (-74.9, -8.4), "Madre de Dios": (-70.8, -11.8),
    "Huánuco": (-76.2, -9.9),
}


# ── 2. Get dominant department per HEI from matriculado ──────────────────────
print("Loading pre-computed department data...")
precomp_path = DATA_DIR / "precomputed_departamentos.csv"
if not precomp_path.exists():
    raise FileNotFoundError(
        "Missing: data/precomputed_departamentos.csv\n"
        "Copy this file (provided separately) into the data/ folder."
    )
df_precomp = pd.read_csv(precomp_path)
ies_main_dept = dict(zip(df_precomp["universidad"], df_precomp["departamento_principal"]))
print(f"  {len(ies_main_dept)} HEIs with department data")


# ── 3. Merge with cluster assignments ─────────────────────────────────────────
local_m = DATA_DIR / "matriz_maestra.csv"
df = pd.read_csv(local_m) if local_m.exists() else pd.read_csv(DATA_URLS["maestra"])
labels = np.load(DATA_DIR / "labels_final.npy")
df["cluster"] = labels + 1

# Usa CLUSTER_NAMES importado desde config.py (fuente única de verdad)
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

# Add geographic variables
df["departamento"]   = df["universidad"].map(ies_main_dept)
df["macroregion"]    = df["departamento"].map(MACROREGION)
df["region_natural"] = df["departamento"].map(NATURAL_REGION)

# Fill missing with fuzzy approach
missing = df["departamento"].isna().sum()
print(f"  HEIs without department match: {missing}/99")
df["departamento"]   = df["departamento"].fillna("Unknown")
df["macroregion"]    = df["macroregion"].fillna("Unknown")
df["region_natural"] = df["region_natural"].fillna("Unknown")

df.to_csv(TABLE_DIR / "06_ies_con_region.csv", index=False)
print("Saved: 06_ies_con_region.csv")

print("\nCluster distribution by macro-region:")
pivot = pd.crosstab(df["macroregion"], df["cluster_name"])
print(pivot.to_string())


# ── 4. Figure A: stacked bar — clusters per department ────────────────────────
dept_order = (df[df["departamento"] != "Unknown"]
              .groupby("departamento")["universidad"].count()
              .sort_values(ascending=True).index.tolist())

fig, ax = plt.subplots(figsize=(10, max(6, len(dept_order)*0.45)))
bottom = np.zeros(len(dept_order))

for cl in range(1, 5):
    counts = [len(df[(df["departamento"]==d) & (df["cluster"]==cl)]) for d in dept_order]
    bars = ax.barh(dept_order, counts, left=bottom,
                   color=CLUSTER_COLORS[cl-1], alpha=0.85,
                   label=f"C{cl}: {CLUSTER_NAMES[cl]}")
    bottom += np.array(counts)

ax.set_xlabel("Number of HEIs")
ax.set_title("Distribution of clusters across Peru's departments", fontsize=12)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "06a_clusters_por_departamento.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 06a_clusters_por_departamento.png")


# ── 5. Figure B: grouped bar — clusters per macro-region ─────────────────────
macro_order = ["Lima Metropolitana", "Norte", "Centro", "Sur", "Oriente"]
macro_order = [m for m in macro_order if m in df["macroregion"].values]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(macro_order))
width = 0.2

for i, cl in enumerate(range(1, 5)):
    counts = [len(df[(df["macroregion"]==m) & (df["cluster"]==cl)]) for m in macro_order]
    ax.bar(x + i*width, counts, width, label=f"C{cl}: {CLUSTER_NAMES[cl]}",
           color=CLUSTER_COLORS[cl-1], alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(macro_order, fontsize=10)
ax.set_ylabel("Number of HEIs")
ax.set_title("Cluster distribution by macro-region", fontsize=12)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "06b_clusters_por_macroregion.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 06b_clusters_por_macroregion.png")


# ── 6. Figure C: bubble map of Peru ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 11))

ax.set_xlim(-82, -68)
ax.set_ylim(-19, -0.5)
ax.set_facecolor("#EAF3DE")
ax.set_aspect("equal")

for dept, (lon, lat) in DEPT_COORDS.items():
    sub = df[df["departamento"] == dept]
    if len(sub) == 0:
        ax.scatter(lon, lat, s=40, color="#D3D1C7", zorder=2, alpha=0.5)
        ax.annotate(dept, (lon, lat), fontsize=5.5, ha="center",
                    va="bottom", xytext=(0, 5), textcoords="offset points",
                    color="#888780")
        continue
    dom_cl = sub["cluster"].value_counts().idxmax()
    n_ies  = len(sub)
    color  = CLUSTER_COLORS[dom_cl - 1]
    ax.scatter(lon, lat, s=n_ies*60 + 80, color=color,
               alpha=0.75, zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(f"{dept}\n(n={n_ies})", (lon, lat), fontsize=6,
                ha="center", va="center", color="white", fontweight="500", zorder=4)

handles = [mpatches.Patch(color=CLUSTER_COLORS[i], alpha=0.85,
           label=f"C{i+1}: {CLUSTER_NAMES[i+1]}") for i in range(4)]
ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic distribution of HEI clusters — Peru 2025\n"
             "(bubble size = number of HEIs; color = dominant cluster)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "06c_mapa_burbujas_peru.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 06c_mapa_burbujas_peru.png")


# ── 7. Figure D: heatmap region × cluster ────────────────────────────────────
import seaborn as sns

heat = pd.crosstab(df["macroregion"], df["cluster"],
                   values=df["universidad"], aggfunc="count").fillna(0)
heat.columns = [f"C{c}: {CLUSTER_NAMES[c][:18]}" for c in heat.columns]

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.4, linecolor="#E8E6DF", ax=ax,
            cbar_kws={"shrink": 0.6, "label": "Nº HEIs"},
            annot_kws={"size": 10})
ax.set_title("HEI count by macro-region and cluster", fontsize=12, pad=10)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "06d_heatmap_region_cluster.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: 06d_heatmap_region_cluster.png")


# ── 8. Profile table by macro-region ─────────────────────────────────────────
FEAT_COLS = ["pct_doctorado", "pct_renacyt_doc", "pct_contratado",
             "puntaje_medio", "nota_prom_egr", "pct_prod_rec", "ratio_mat_doc"]

prof = df.groupby("macroregion")[FEAT_COLS].mean().round(2)
prof["n_ies"]   = df.groupby("macroregion")["universidad"].count()
prof["pct_pub"] = df.groupby("macroregion")["es_publico"].mean().round(2) * 100
prof.to_csv(TABLE_DIR / "06_perfil_macroregion.csv")
print("Saved: 06_perfil_macroregion.csv")

print(f"\n{'='*55}")
print("Regional analysis complete.")
print("Key finding: check how clusters concentrate in Lima vs. regions.")
print(f"{'='*55}")
