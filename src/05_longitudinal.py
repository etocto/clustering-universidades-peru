"""
05_longitudinal.py
==================
Análisis longitudinal: replica el clustering para los tres periodos
disponibles de docentes (2024-II, 2025-I, 2025-II) y traza las
transiciones institucionales entre periodos.

Genera:
  - outputs/figures/05a_sankey_transiciones.png
  - outputs/figures/05b_evolucion_clusters.png
  - outputs/figures/05c_casos_transicion.png
  - outputs/tables/05_transiciones.csv
  - outputs/tables/05_ies_cambiaron_cluster.csv

NOTA: Para este script necesitas los CSV originales de docente en data/:
  data/docente_2024_II.csv
  data/docente_2025_I.csv
  data/docente_2025_II.csv
Si no los tienes localmente, el script usa la matriz_maestra.csv como base
y solo analiza el periodo 2025-I (ya procesado).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, FIG_DIR, TABLE_DIR, FEATURE_COLS,
                    RANDOM_STATE, MPL_STYLE, CLUSTER_COLORS, DATA_URLS)

plt.rcParams.update(MPL_STYLE)


# ── 1. Cargar k óptimo y datos base ────────────────────────────────────────────
K       = int(np.load(DATA_DIR / "k_optimo.npy")[0])
labels_base = np.load(DATA_DIR / "labels_final.npy")

local_m = DATA_DIR / "matriz_maestra.csv"
df_base = pd.read_csv(local_m) if local_m.exists() else pd.read_csv(DATA_URLS["maestra"])
df_base["cluster_2025I"] = labels_base + 1
unis_base = set(df_base["universidad"].values)

print(f"Periodo base (2025-I): {len(df_base)} IES, k={K}")


# ── 2. Función para agregar features de docentes ────────────────────────────────
DOC_FEATURES = [
    "pct_doctorado", "pct_maestria", "pct_renacyt_doc",
    "pct_exclusiva", "pct_tc", "pct_contratado", "pct_ordinario",
    "edad_media_doc", "pct_fem_doc",
]

def agregar_docente(path):
    import csv
    from collections import defaultdict
    unis = defaultdict(lambda: dict(
        total=0, exclusiva=0, tc=0,
        doctorado=0, maestria=0, renacyt=0,
        ordinario=0, contratado=0,
        edad_sum=0, edad_n=0, masc=0, fem=0
    ))
    with open(path, encoding="latin1") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            u = row["ENTIDAD"].strip()
            d = unis[u]
            d["total"] += 1
            if row.get("REGIMEN_DEDICACION","") == "Dedicación Exclusiva": d["exclusiva"] += 1
            if row.get("REGIMEN_DEDICACION","") == "Tiempo Completo":      d["tc"] += 1
            if row.get("NIVEL_ACADEMICO","") == "Doctorado":               d["doctorado"] += 1
            if row.get("NIVEL_ACADEMICO","") in ("Maestro","Maestría"):    d["maestria"] += 1
            if row.get("NIVEL_INVESTIGADOR",""):                           d["renacyt"] += 1
            if "Ordinario" in row.get("CATEGORIA_DOCENTE",""):             d["ordinario"] += 1
            if "Contratado" in row.get("CATEGORIA_DOCENTE",""):            d["contratado"] += 1
            if row.get("SEXO","") == "Femenino":  d["fem"] += 1
            try:
                d["edad_sum"] += int(row.get("EDAD",""))
                d["edad_n"]   += 1
            except: pass

    rows = []
    for u, d in unis.items():
        t = d["total"]
        rows.append({
            "universidad":       u,
            "doc_total":         t,
            "pct_doctorado":     round(d["doctorado"]/t*100,2) if t else 0,
            "pct_maestria":      round(d["maestria"]/t*100,2)  if t else 0,
            "pct_renacyt_doc":   round(d["renacyt"]/t*100,2)   if t else 0,
            "pct_exclusiva":     round(d["exclusiva"]/t*100,2) if t else 0,
            "pct_tc":            round(d["tc"]/t*100,2)        if t else 0,
            "pct_contratado":    round(d["contratado"]/t*100,2)if t else 0,
            "pct_ordinario":     round(d["ordinario"]/t*100,2) if t else 0,
            "edad_media_doc":    round(d["edad_sum"]/d["edad_n"],2) if d["edad_n"] else 0,
            "pct_fem_doc":       round(d["fem"]/t*100,2) if t else 0,
        })
    return pd.DataFrame(rows)


# ── 3. Procesar cada periodo disponible ────────────────────────────────────────
PERIODS = {
    "2024-II": DATA_DIR / "docente_2024_II.csv",
    "2025-I":  DATA_DIR / "docente_2025_I.csv",
    "2025-II": DATA_DIR / "docente_2025_II.csv",
}

period_dfs   = {}
period_labels = {}

for period, path in PERIODS.items():
    if not path.exists():
        print(f"  {period}: archivo no encontrado en data/ — omitido")
        continue
    print(f"  Procesando {period}...")
    df_p = agregar_docente(path)

    X_p = df_p[DOC_FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_ps   = scaler.fit_transform(X_p)

    km_p = KMeans(n_clusters=K, init="k-means++", n_init=50, random_state=RANDOM_STATE)
    df_p[f"cluster_{period.replace('-','_')}"] = km_p.fit_predict(X_ps) + 1

    period_dfs[period]   = df_p
    period_labels[period] = df_p[["universidad", f"cluster_{period.replace('-','_')}"]].copy()
    n = len(df_p)
    col_name = f"cluster_{period.replace('-','_')}"
    dist = dict(zip(*np.unique(df_p[col_name], return_counts=True)))
    print(f"    {n} IES  |  distribución: {dist}")

if len(period_labels) < 2:
    print("\nMenos de 2 periodos disponibles — análisis longitudinal requiere los CSV originales.")
    print("Descarga docente_2024_II.csv, docente_2025_I.csv, docente_2025_II.csv en data/")
    exit(0)


# ── 4. Matriz de transiciones entre periodos ───────────────────────────────────
sorted_periods = [p for p in ["2024-II","2025-I","2025-II"] if p in period_labels]
transition_tables = {}

for i in range(len(sorted_periods) - 1):
    p1, p2 = sorted_periods[i], sorted_periods[i+1]
    c1 = f"cluster_{p1.replace('-','_')}"
    c2 = f"cluster_{p2.replace('-','_')}"
    merged = period_labels[p1].merge(period_labels[p2], on="universidad", how="inner")
    trans  = pd.crosstab(merged[c1], merged[c2])
    transition_tables[f"{p1}→{p2}"] = trans
    n_cambio = (merged[c1] != merged[c2]).sum()
    print(f"\nTransición {p1}→{p2}: {n_cambio}/{len(merged)} IES cambiaron de clúster")
    print(trans.to_string())


# ── 5. Figura: evolución de tamaño de clústeres ────────────────────────────────
fig, axes = plt.subplots(1, len(sorted_periods), figsize=(4*len(sorted_periods), 5),
                         sharey=True)
for ax, period in zip(axes if len(sorted_periods) > 1 else [axes], sorted_periods):
    c_col  = f"cluster_{period.replace('-','_')}"
    counts = period_labels[period][c_col].value_counts().sort_index()
    bars   = ax.bar(counts.index, counts.values,
                    color=[CLUSTER_COLORS[i-1] for i in counts.index],
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Period {period}", fontsize=11)
    ax.set_xlabel("Cluster")
    if ax == axes[0] if len(sorted_periods) > 1 else axes:
        ax.set_ylabel("No. of HEIs")
    ax.set_xticks(range(1, K+1))
    ax.set_xticklabels([f"C{i}" for i in range(1, K+1)])

fig.suptitle("Cluster size evolution by academic period", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "05b_evolucion_clusters.png", bbox_inches="tight")
plt.close()
print("Guardado: 05b_evolucion_clusters.png")


# ── 6. IES que cambiaron de clúster ────────────────────────────────────────────
if len(sorted_periods) >= 2:
    p1, p2 = sorted_periods[0], sorted_periods[-1]
    c1 = f"cluster_{p1.replace('-','_')}"
    c2 = f"cluster_{p2.replace('-','_')}"
    merged_all = period_labels[p1].merge(period_labels[p2], on="universidad", how="inner")
    cambios    = merged_all[merged_all[c1] != merged_all[c2]].copy()
    cambios    = cambios.rename(columns={c1: f"cluster_{p1}", c2: f"cluster_{p2}"})
    cambios["transicion"] = cambios.apply(
        lambda r: f"C{r[f'cluster_{p1}']} → C{r[f'cluster_{p2}']}", axis=1
    )
    cambios.to_csv(TABLE_DIR / "05_ies_cambiaron_cluster.csv", index=False)
    print(f"\nIES que cambiaron de clúster ({p1}→{p2}): {len(cambios)}")
    print(cambios[["universidad","transicion"]].to_string(index=False))
    print("Guardado: 05_ies_cambiaron_cluster.csv")


# ── 7. Guardar tabla de transiciones ───────────────────────────────────────────
trans_rows = []
for key, trans in transition_tables.items():
    for from_cl in trans.index:
        for to_cl in trans.columns:
            trans_rows.append({"periodos": key, "de": from_cl,
                               "hacia": to_cl, "n": trans.loc[from_cl, to_cl]})
pd.DataFrame(trans_rows).to_csv(TABLE_DIR / "05_transiciones.csv", index=False)
print("Guardado: 05_transiciones.csv")


# ── 8. Sankey manual (matplotlib) ──────────────────────────────────────────────
if len(sorted_periods) >= 2:
    p1, p2 = sorted_periods[0], sorted_periods[1]
    c1 = f"cluster_{p1.replace('-','_')}"
    c2 = f"cluster_{p2.replace('-','_')}"
    merged = period_labels[p1].merge(period_labels[p2], on="universidad", how="inner")
    trans  = pd.crosstab(merged[c1], merged[c2])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(f"Flow of HEIs between clusters: {p1} → {p2}", fontsize=12)

    left_tot  = {cl: merged[c1].tolist().count(cl) for cl in range(1, K+1)}
    right_tot = {cl: merged[c2].tolist().count(cl) for cl in range(1, K+1)}

    def bar_positions(totals, x, width=0.6):
        positions = {}
        y = 1.0
        for cl in sorted(totals):
            h = totals[cl] / len(merged) * 8
            positions[cl] = (y, y + h)
            rect = plt.Rectangle((x - width/2, y), width, h,
                                  facecolor=CLUSTER_COLORS[cl-1], alpha=0.8,
                                  edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x, y + h/2, f"C{cl}\n({totals[cl]})",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="500")
            y += h + 0.2
        return positions

    pos_left  = bar_positions(left_tot,  x=2)
    pos_right = bar_positions(right_tot, x=8)

    from matplotlib.patches import FancyArrowPatch
    y_left  = {cl: pos_left[cl][0]  for cl in pos_left}
    y_right = {cl: pos_right[cl][0] for cl in pos_right}

    for from_cl in range(1, K+1):
        for to_cl in range(1, K+1):
            if from_cl not in trans.index or to_cl not in trans.columns: continue
            n = trans.loc[from_cl, to_cl]
            if n == 0: continue
            h_norm = n / len(merged) * 8 * 0.8
            alpha  = 0.15 + 0.5 * (n / len(merged))
            color  = CLUSTER_COLORS[from_cl - 1]
            y0 = y_left[from_cl]
            y1 = y_right[to_cl]
            xs = [2.3, 5, 5, 7.7]
            ys = [y0 + 0.05, y0 + 0.05, y1 + 0.05, y1 + 0.05]
            ax.fill_betweenx(
                np.linspace(y0, y0 + h_norm, 50),
                np.interp(np.linspace(y0, y0 + h_norm, 50),
                          [y0, y0 + h_norm], [2.3, 2.3]),
                np.interp(np.linspace(y1, y1 + h_norm, 50),
                          [y1, y1 + h_norm], [7.7, 7.7]),
                alpha=alpha, color=color
            )
            y_left[from_cl]  += h_norm + 0.02
            y_right[to_cl]   += h_norm + 0.02

    ax.text(2, 0.3, p1, ha="center", fontsize=10, fontweight="500")
    ax.text(8, 0.3, p2, ha="center", fontsize=10, fontweight="500")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05a_sankey_transiciones.png", bbox_inches="tight")
    plt.close()
    print("Guardado: 05a_sankey_transiciones.png")

print(f"\n{'='*50}")
print("Análisis longitudinal completado.")
print(f"Revisa 05_ies_cambiaron_cluster.csv para identificar casos de estudio.")
