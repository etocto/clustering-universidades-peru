"""
13_regenerar_v7.py
==================
Fija la especificacion final del manuscrito v7 y regenera TODAS las figuras y
tablas de forma consistente. Sustituye la salida de 01_pca, 02_k_optimo,
03_clustering, 04_perfilado y 07_disciplinar, que fueron construidos para la
solucion anterior (k=4, 99 IES, 25 variables).

ESPECIFICACION v7
-----------------
  - 96 IES (las 3 escuelas de posgrado salen como estrato legal a priori)
  - 20 indicadores: 18 organizacionales + presencia RENACYT + densidad RENACYT
  - densidad RENACYT winsorizada al 100%
  - PCA al 90% de varianza acumulada
  - k-means++ con k=3, semilla 42

Genera en outputs/figures/v7/:
  F1_pca_varianza.png        scree + varianza acumulada
  F2_biplot.png              PC1-PC2 con cargas
  F3_scatter_perfiles.png    dispersion por perfil
  F4_radar_perfiles.png      perfiles normalizados
  F5_heatmap_boxplots.png    medias estandarizadas + variables discriminantes
  F6_sankey_longitudinal.png transiciones con etiquetas alineadas
  F7_evolucion_tamanos.png   tamanos por ola

Genera en outputs/tables/v7/:
  T1_perfiles.csv                  tabla 1 del manuscrito
  T2_estabilidad.csv               tabla 2 del manuscrito
  S1_cargas_pca.csv                cargas de los componentes retenidos
  S2_kruskal_dunn.csv              Kruskal-Wallis y post-hoc de Dunn
  S3_descriptivos_por_perfil.csv   medias y desviaciones
  S4_titularidad.csv               tabla de contingencia y clasificador
  S5_disciplinar.csv               agrupamiento disciplinar y cruce
  S6_estrato_posgrado.csv          diagnostico del estrato a priori
  S7_confusion_olas.csv            matrices de transicion alineadas
  asignaciones_v7.csv              perfil de cada IES

Guarda tambien data/labels_v7.npy y data/X_pca_v7.npy.

Uso:
    python src/13_regenerar_v7.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIG_DIR, MPL_STYLE, RANDOM_STATE, TABLE_DIR  # noqa: E402

FIG = FIG_DIR / "v7"
TAB = TABLE_DIR / "v7"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(MPL_STYLE)

SEED = RANDOM_STATE
K = 3

# Paleta: P1 ambar, P2 azul, P3 verde. Distinguibles en escala de grises.
COLORS = {1: "#BA7517", 2: "#7F77DD", 3: "#1D9E75"}
PROFILE_LABEL = {
    1: "P1  Non-university,\nno research coupling",
    2: "P2  Flexible-contractual\nuniversities",
    3: "P3  Tenure-consolidated\nuniversities",
}
PROFILE_SHORT = {1: "P1", 2: "P2", 3: "P3"}

# Las olas se agrupan SOLO con las nueve variables de empleo docente, de modo
# que sus grupos no son los perfiles P1-P3 de la tipologia completa. Se
# etiquetan E1-E3 (employment configurations) para no inducir a confusion.
EMP_SHORT = {1: "E1", 2: "E2", 3: "E3"}
EMP_COLORS = {1: "#BA7517", 2: "#7F77DD", 3: "#1D9E75"}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
EMPLOYMENT = [
    "pct_doctorado", "pct_maestria", "pct_exclusiva", "pct_tc",
    "pct_contratado", "pct_ordinario", "edad_media_doc", "pct_fem_doc",
]
ENROLMENT = [
    "pct_fem_mat", "pct_discap", "pct_posgrado", "edad_media_mat",
    "n_departamentos", "nota_prom_egr", "creditos_prom_egr",
    "pct_posgrado_egr", "ratio_mat_doc", "ratio_egr_mat",
]
COUPLING = ["has_renacyt", "pct_renacyt_doc_w"]
FEATURES = EMPLOYMENT + ENROLMENT + COUPLING

# Etiquetas legibles para las figuras
PRETTY = {
    "pct_doctorado": "Doctoral degree", "pct_maestria": "Master's degree",
    "pct_exclusiva": "Exclusive dedication", "pct_tc": "Full-time",
    "pct_contratado": "Contracted", "pct_ordinario": "Tenured",
    "edad_media_doc": "Mean faculty age", "pct_fem_doc": "Female faculty",
    "pct_fem_mat": "Female students", "pct_discap": "Students with disability",
    "pct_posgrado": "Postgraduate enrolment", "edad_media_mat": "Mean student age",
    "n_departamentos": "Departments", "nota_prom_egr": "Mean grade",
    "creditos_prom_egr": "Mean credits", "pct_posgrado_egr": "Postgraduate graduates",
    "ratio_mat_doc": "Student-faculty ratio", "ratio_egr_mat": "Graduate-enrolment ratio",
    "has_renacyt": "Registry presence", "pct_renacyt_doc_w": "Registry density",
}


def log(*a):
    print(" ".join(str(x) for x in a))


# ---------------------------------------------------------------------------
# 1. Datos y especificacion
# ---------------------------------------------------------------------------
log("=" * 70)
log("ESPECIFICACION v7")
log("=" * 70)

df = pd.read_csv(DATA_DIR / "matriz_maestra.csv")
df["has_renacyt"] = (df["n_renacyt"] > 0).astype(int)

n_wins = int((df["pct_renacyt_doc"] > 100).sum())
df["pct_renacyt_doc_w"] = df["pct_renacyt_doc"].clip(upper=100)
log(f"Densidades imposibles winsorizadas al 100%: {n_wins}")

# El estrato a priori son las IES cuya licencia es exclusivamente de posgrado.
# Se identifica por matricula 100% posgrado, no por el cluster anterior, para
# que el criterio sea explicito y no dependa de la solucion publicada.
strato = (df["pct_posgrado"] >= 99.9).values
log(f"Estrato a priori (escuelas de posgrado): {strato.sum()}")
for u in df.loc[strato, "universidad"]:
    log(f"   - {u}")

sub = df[~strato].reset_index(drop=True)
log(f"IES en el clustering: {len(sub)}")

X = StandardScaler().fit_transform(sub[FEATURES].fillna(0).values)
pca_full = PCA(random_state=SEED).fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.argmax(cum >= 0.90) + 1)
pca = PCA(n_components=n_comp, random_state=SEED)
X_pca = pca.fit_transform(X)

log(f"Componentes retenidos: {n_comp} ({cum[n_comp-1]*100:.1f}% de varianza)")
log(f"PC1 = {pca_full.explained_variance_ratio_[0]*100:.1f}%  "
    f"PC2 = {pca_full.explained_variance_ratio_[1]*100:.1f}%")

km = KMeans(n_clusters=K, init="k-means++", n_init=50,
            max_iter=500, random_state=SEED).fit(X_pca)
labels = km.labels_ + 1

# Renumerar de forma estable: P1 = sin acoplamiento, P3 = mas titularidad.
sin_coup = {c: (sub.loc[labels == c, "has_renacyt"] == 0).mean() for c in np.unique(labels)}
p1 = max(sin_coup, key=sin_coup.get)
resto = [c for c in np.unique(labels) if c != p1]
ord_mean = {c: sub.loc[labels == c, "pct_ordinario"].mean() for c in resto}
p3 = max(ord_mean, key=ord_mean.get)
p2 = [c for c in resto if c != p3][0]
labels = pd.Series(labels).map({p1: 1, p2: 2, p3: 3}).values
sub["P"] = labels

asw = silhouette_score(X_pca, labels)
sil = silhouette_samples(X_pca, labels)
log(f"ASW = {asw:.3f}   tamanos = "
    + str({PROFILE_SHORT[c]: int((labels == c).sum()) for c in (1, 2, 3)}))

np.save(DATA_DIR / "labels_v7.npy", labels)
np.save(DATA_DIR / "X_pca_v7.npy", X_pca)

# Validacion algoritmica
Z = linkage(X_pca, method="ward")
l_ward = fcluster(Z, t=K, criterion="maxclust")
l_gmm = GaussianMixture(n_components=K, covariance_type="full",
                        n_init=20, random_state=SEED).fit_predict(X_pca)
nn = NearestNeighbors(n_neighbors=5).fit(X_pca)
dist, _ = nn.kneighbors(X_pca)
eps = np.percentile(np.sort(dist[:, -1]), 90)
l_db = DBSCAN(eps=eps, min_samples=3).fit_predict(X_pca)
log(f"ARI  K-means vs GMM = {adjusted_rand_score(labels, l_gmm):.3f}  |  "
    f"vs Ward = {adjusted_rand_score(labels, l_ward):.3f}")
log(f"DBSCAN eps={eps:.3f}: {len(set(l_db)) - (1 if -1 in l_db else 0)} region(es), "
    f"{(l_db == -1).sum()} puntos de ruido")


# ---------------------------------------------------------------------------
# 2. Figuras
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("FIGURAS")
log("=" * 70)

# --- F1: scree + varianza acumulada -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ev = pca_full.explained_variance_ratio_ * 100
axes[0].bar(range(1, len(ev) + 1), ev, color="#7F77DD", edgecolor="white")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Explained variance (%)")
axes[0].set_title("(a) Scree plot", fontsize=11, loc="left")
axes[1].plot(range(1, len(cum) + 1), cum * 100, marker="o", ms=4, color="#1D9E75")
axes[1].axhline(90, ls="--", lw=1, color="#D85A30")
axes[1].axvline(n_comp, ls="--", lw=1, color="#D85A30")
axes[1].annotate(f"{n_comp} components\n{cum[n_comp-1]*100:.1f}%",
                 xy=(n_comp, 90), xytext=(n_comp + 1.5, 62), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#D85A30", lw=0.8))
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance (%)")
axes[1].set_title("(b) Cumulative variance", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(FIG / "F1_pca_varianza.png", dpi=300, bbox_inches="tight")
plt.close(); log("F1_pca_varianza.png")

# --- F2: biplot --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6))
for c in (1, 2, 3):
    m = labels == c
    ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLORS[c], s=52, alpha=.85,
               edgecolors="white", linewidths=.5,
               label=PROFILE_LABEL[c].replace("\n", " "), zorder=3)
load = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
scale = np.abs(X_pca[:, :2]).max() / np.abs(load).max() * .75
top = np.argsort(-(load ** 2).sum(axis=1))[:6]
for i in top:
    ax.arrow(0, 0, load[i, 0] * scale, load[i, 1] * scale, color="#444",
             lw=1.1, head_width=.14, alpha=.85, zorder=4)
    ax.text(load[i, 0] * scale * 1.12, load[i, 1] * scale * 1.12,
            PRETTY.get(FEATURES[i], FEATURES[i]), fontsize=8.5,
            color="#222", ha="center", zorder=5)
ax.axhline(0, color="#B4B2A9", lw=.5); ax.axvline(0, color="#B4B2A9", lw=.5)
ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
ax.legend(fontsize=8.5, loc="best", framealpha=.9)
fig.tight_layout(); fig.savefig(FIG / "F2_biplot.png", dpi=300, bbox_inches="tight")
plt.close(); log("F2_biplot.png")

# --- F3: dispersion por perfil ----------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5))
for c in (1, 2, 3):
    for pub, mark, lbl in [(1, "o", "public"), (0, "D", "private")]:
        m = (labels == c) & (sub["es_publico"].values == pub)
        if m.sum():
            ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLORS[c], marker=mark,
                       s=58, alpha=.85, edgecolors="white", linewidths=.5, zorder=3)
h_prof = [mpatches.Patch(color=COLORS[c], label=PROFILE_LABEL[c].replace("\n", " "))
          for c in (1, 2, 3)]
h_own = [plt.Line2D([], [], marker="o", ls="", color="#666", label="Public"),
         plt.Line2D([], [], marker="D", ls="", color="#666", label="Private")]
ax.legend(handles=h_prof + h_own, fontsize=8.5, loc="best", framealpha=.9)
ax.axhline(0, color="#B4B2A9", lw=.5); ax.axvline(0, color="#B4B2A9", lw=.5)
ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
fig.tight_layout(); fig.savefig(FIG / "F3_scatter_perfiles.png", dpi=300, bbox_inches="tight")
plt.close(); log("F3_scatter_perfiles.png")

# --- F4: radar ---------------------------------------------------------------
radar_vars = ["pct_contratado", "pct_ordinario", "pct_tc", "pct_exclusiva",
              "pct_renacyt_doc_w", "has_renacyt", "pct_posgrado", "pct_doctorado"]
norm = sub[radar_vars].copy()
norm = (norm - norm.min()) / (norm.max() - norm.min()).replace(0, 1)
ang = np.linspace(0, 2 * np.pi, len(radar_vars), endpoint=False).tolist()
ang += ang[:1]
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for c in (1, 2, 3):
    v = norm[labels == c].mean().tolist(); v += v[:1]
    ax.plot(ang, v, color=COLORS[c], lw=1.8,
            label=PROFILE_LABEL[c].replace("\n", " "))
    ax.fill(ang, v, color=COLORS[c], alpha=.13)
ax.set_xticks(ang[:-1])
ax.set_xticklabels([PRETTY[v] for v in radar_vars], fontsize=8)
ax.set_yticks([.25, .5, .75]); ax.set_yticklabels(["", "", ""])
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.28, 1.12))
fig.tight_layout(); fig.savefig(FIG / "F4_radar_perfiles.png", dpi=300, bbox_inches="tight")
plt.close(); log("F4_radar_perfiles.png")

# --- Kruskal-Wallis (necesario para F5 y para S2) ---------------------------
kw = []
for v in FEATURES:
    groups = [sub.loc[labels == c, v].values for c in (1, 2, 3)]
    H, p = kruskal(*groups)
    kw.append({"indicator": PRETTY.get(v, v), "variable": v,
               "H": round(H, 1), "p": p})
kw = pd.DataFrame(kw).sort_values("H", ascending=False).reset_index(drop=True)

# --- F5: heatmap + boxplots --------------------------------------------------
Xs = pd.DataFrame(StandardScaler().fit_transform(sub[FEATURES].fillna(0)),
                  columns=[PRETTY.get(v, v) for v in FEATURES])
Xs["P"] = labels
hm = Xs.groupby("P").mean().T
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                         gridspec_kw={"width_ratios": [1, 1.25]})
im = axes[0].imshow(hm.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
axes[0].set_xticks(range(3)); axes[0].set_xticklabels([PROFILE_SHORT[c] for c in (1, 2, 3)])
axes[0].set_yticks(range(len(hm))); axes[0].set_yticklabels(hm.index, fontsize=8)
axes[0].set_title("(a) Standardised means by profile", fontsize=11, loc="left")
axes[0].grid(False)
fig.colorbar(im, ax=axes[0], fraction=.04, pad=.03, label="z-score")

top6 = kw.head(6)["variable"].tolist()
pos, ticks = [], []
for i, v in enumerate(top6):
    for j, c in enumerate((1, 2, 3)):
        bp = axes[1].boxplot(sub.loc[labels == c, v].dropna(), positions=[i * 4 + j],
                             widths=.75, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(COLORS[c]); bp["boxes"][0].set_alpha(.75)
        for k in ("medians", "whiskers", "caps"):
            for e in bp[k]:
                e.set_color("#333")
    ticks.append(i * 4 + 1); pos.append(PRETTY.get(v, v))
axes[1].set_xticks(ticks); axes[1].set_xticklabels(pos, fontsize=8.5, rotation=18, ha="right")
axes[1].set_title("(b) Six most discriminating indicators", fontsize=11, loc="left")
axes[1].legend(handles=[mpatches.Patch(color=COLORS[c], label=PROFILE_SHORT[c])
                        for c in (1, 2, 3)], fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "F5_heatmap_boxplots.png", dpi=300, bbox_inches="tight")
plt.close(); log("F5_heatmap_boxplots.png")


# ---------------------------------------------------------------------------
# 3. Longitudinal con etiquetas alineadas
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("LONGITUDINAL")
log("=" * 70)

DOC_FEATURES = [
    "pct_doctorado", "pct_maestria", "pct_renacyt_doc", "pct_exclusiva",
    "pct_tc", "pct_contratado", "pct_ordinario", "edad_media_doc", "pct_fem_doc",
]
PERIODS = {
    "2024-II": DATA_DIR / "docente_2024_II.csv",
    "2025-I": DATA_DIR / "docente_2025_I.csv",
    "2025-II": DATA_DIR / "docente_2025_II.csv",
}


def agregar_docente(path):
    import csv
    from collections import defaultdict
    u = defaultdict(lambda: dict(total=0, exclusiva=0, tc=0, doctorado=0, maestria=0,
                                 renacyt=0, ordinario=0, contratado=0,
                                 edad_sum=0, edad_n=0, fem=0))
    with open(path, encoding="latin1") as f:
        for r in csv.DictReader(f, delimiter="|"):
            d = u[r["ENTIDAD"].strip()]
            d["total"] += 1
            rd = r.get("REGIMEN_DEDICACION", "")
            d["exclusiva"] += rd == "Dedicación Exclusiva"
            d["tc"] += rd == "Tiempo Completo"
            na = r.get("NIVEL_ACADEMICO", "")
            d["doctorado"] += na == "Doctorado"
            d["maestria"] += na in ("Maestro", "Maestría")
            d["renacyt"] += bool(r.get("NIVEL_INVESTIGADOR", ""))
            cat = r.get("CATEGORIA_DOCENTE", "")
            d["ordinario"] += "Ordinario" in cat
            d["contratado"] += "Contratado" in cat
            d["fem"] += r.get("SEXO", "") == "Femenino"
            try:
                d["edad_sum"] += int(r.get("EDAD", "")); d["edad_n"] += 1
            except (ValueError, TypeError):
                pass
    out = []
    for k, d in u.items():
        t = d["total"] or 1
        out.append({"universidad": k, "doc_total": d["total"],
                    "pct_doctorado": d["doctorado"] / t * 100,
                    "pct_maestria": d["maestria"] / t * 100,
                    "pct_renacyt_doc": d["renacyt"] / t * 100,
                    "pct_exclusiva": d["exclusiva"] / t * 100,
                    "pct_tc": d["tc"] / t * 100,
                    "pct_contratado": d["contratado"] / t * 100,
                    "pct_ordinario": d["ordinario"] / t * 100,
                    "edad_media_doc": d["edad_sum"] / d["edad_n"] if d["edad_n"] else 0,
                    "pct_fem_doc": d["fem"] / t * 100})
    return pd.DataFrame(out)


def alinear(a, b):
    cm = pd.crosstab(a, b)
    r, c = linear_sum_assignment(-cm.values)
    mp = {cm.columns[j]: cm.index[i] for i, j in zip(r, c)}
    return b.map(lambda x: mp.get(x, x)), mp


waves, conf_rows = {}, []
for per, path in PERIODS.items():
    if not path.exists():
        log(f"  {per}: no encontrado, se omite"); continue
    w = agregar_docente(path)
    Xw = StandardScaler().fit_transform(w[DOC_FEATURES].fillna(0).values)
    cl = KMeans(n_clusters=K, init="k-means++", n_init=50,
                random_state=SEED).fit_predict(Xw) + 1
    # Renumerar por titularidad ascendente: E3 = configuracion mas estable
    orden = (w.assign(_c=cl).groupby("_c").pct_ordinario.mean()
             .sort_values().index.tolist())
    w["cluster"] = pd.Series(cl).map({c: i + 1 for i, c in enumerate(orden)}).values
    waves[per] = w
    log(f"  {per}: {len(w)} IES")

stab_rows = []
if len(waves) >= 2:
    pers = list(waves)
    comunes = set.intersection(*[set(w.universidad) for w in waves.values()])
    panel = pd.DataFrame({"universidad": sorted(comunes)})
    for p in pers:
        panel = panel.merge(waves[p][["universidad", "cluster"]]
                            .rename(columns={"cluster": p}), on="universidad")
    for a, b in zip(pers, pers[1:]):
        panel[b], _ = alinear(panel[a], panel[b])
    log(f"  Panel balanceado: {len(panel)} IES")

    for a, b in zip(pers, pers[1:]):
        ari = adjusted_rand_score(panel[a], panel[b])
        tau = 100 * (panel[a] != panel[b]).mean()
        log(f"  {a} -> {b}: ARI = {ari:.3f}, tau = {tau:.1f}%")
        stab_rows.append({"wave_pair": f"{a} -> {b}", "n": len(panel),
                          "ARI": round(ari, 3), "tau_pct": round(tau, 1)})
        cm = pd.crosstab(panel[a], panel[b])
        cm.index.name, cm.columns.name = a, b
        conf_rows.append(cm.assign(pair=f"{a}->{b}"))
    ari_t = adjusted_rand_score(panel[pers[0]], panel[pers[-1]])
    neto = int((panel[pers[0]] != panel[pers[-1]]).sum())
    log(f"  {pers[0]} -> {pers[-1]}: ARI = {ari_t:.3f}, "
        f"tau = {100*neto/len(panel):.1f}%, netas = {neto}")
    stab_rows.append({"wave_pair": f"{pers[0]} -> {pers[-1]} (full year)",
                      "n": len(panel), "ARI": round(ari_t, 3),
                      "tau_pct": round(100 * neto / len(panel), 1)})

    # --- F6: alluvial ---------------------------------------------------------
    fig, axes = plt.subplots(1, len(pers) - 1, figsize=(6 * (len(pers) - 1), 5.4))
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, zip(pers, pers[1:])):
        cm = pd.crosstab(panel[a], panel[b])
        left = {c: 0 for c in cm.index}
        acc_l = {}
        y = 0
        for c in sorted(cm.index):
            h = cm.loc[c].sum(); acc_l[c] = (y, h); y += h + 2
        y = 0; acc_r = {}
        for c in sorted(cm.columns):
            h = cm[c].sum(); acc_r[c] = (y, h); y += h + 2
        for c in sorted(cm.index):
            y0, h = acc_l[c]
            ax.add_patch(mpatches.Rectangle((0, y0), .18, h, color=EMP_COLORS.get(c, "#999")))
            ax.text(-.05, y0 + h / 2, f"{EMP_SHORT.get(c, c)} ({h})",
                    ha="right", va="center", fontsize=9)
        for c in sorted(cm.columns):
            y0, h = acc_r[c]
            ax.add_patch(mpatches.Rectangle((1.82, y0), .18, h, color=EMP_COLORS.get(c, "#999")))
            ax.text(2.05, y0 + h / 2, f"{EMP_SHORT.get(c, c)} ({h})",
                    ha="left", va="center", fontsize=9)
        off_l = {c: acc_l[c][0] for c in cm.index}
        off_r = {c: acc_r[c][0] for c in cm.columns}
        for i in sorted(cm.index):
            for j in sorted(cm.columns):
                n = cm.loc[i, j]
                if n == 0:
                    continue
                y1, y2 = off_l[i], off_r[j]
                ax.fill_between([.18, 1.82], [y1, y2], [y1 + n, y2 + n],
                                color=EMP_COLORS.get(i, "#999"), alpha=.32, lw=0)
                off_l[i] += n; off_r[j] += n
        ax.set_xlim(-.55, 2.55); ax.set_ylim(-3, max(y, 1) + 3)
        ax.invert_yaxis(); ax.axis("off")
        ax.set_title(f"{a}  \u2192  {b}", fontsize=11)
    fig.suptitle("Employment-configuration transitions, balanced panel "
                 "(labels aligned across waves)",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "F6_sankey_longitudinal.png",
                                    dpi=300, bbox_inches="tight")
    plt.close(); log("F6_sankey_longitudinal.png")

    # --- F7: tamanos por ola --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    w_ = .26
    for i, c in enumerate(sorted(panel[pers[0]].unique())):
        vals = [(panel[p] == c).sum() for p in pers]
        ax.bar(np.arange(len(pers)) + i * w_, vals, w_,
               color=EMP_COLORS.get(c, "#999"), label=EMP_SHORT.get(c, c),
               edgecolor="white")
        for x, v in zip(np.arange(len(pers)) + i * w_, vals):
            ax.text(x, v + .4, str(v), ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(pers)) + w_); ax.set_xticklabels(pers)
    ax.set_ylabel("Institutions"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "F7_evolucion_tamanos.png",
                                    dpi=300, bbox_inches="tight")
    plt.close(); log("F7_evolucion_tamanos.png")

    # Attrition
    att = []
    for a, b in zip(pers, pers[1:]):
        wa = waves[a].copy()
        wa["sale"] = ~wa.universidad.isin(waves[b].universidad)
        mg = wa.merge(sub[["universidad", "P"]], on="universidad", how="inner")
        if mg.sale.nunique() == 2:
            t = pd.crosstab(mg.P, mg.sale)
            chi2, p, dof, _ = chi2_contingency(t)
        else:
            chi2 = p = np.nan
        for c in sorted(mg.P.unique()):
            s = mg[mg.P == c]
            att.append({"transition": f"{a}->{b}", "profile": PROFILE_SHORT[int(c)],
                        "n": len(s), "exiting": int(s.sale.sum()),
                        "pct_exiting": round(100 * s.sale.mean(), 1),
                        "chi2": None if np.isnan(chi2) else round(chi2, 2),
                        "p": None if np.isnan(p) else round(p, 4)})
    pd.DataFrame(att).to_csv(TAB / "S8_attrition.csv", index=False)
    pd.concat(conf_rows).to_csv(TAB / "S7_confusion_olas.csv")
    panel.to_csv(TAB / "panel_balanceado_v7.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Tablas
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("TABLAS")
log("=" * 70)

g = sub.groupby("P")
T1 = pd.DataFrame({
    "n": g.size(),
    "contracted_pct": g.pct_contratado.mean().round(1),
    "tenured_pct": g.pct_ordinario.mean().round(1),
    "registry_density_pct": g.pct_renacyt_doc_w.mean().round(1),
    "no_registered_pct": (100 * (1 - g.has_renacyt.mean())).round(1),
    "public_pct": (100 * g.es_publico.mean()).round(1),
    "university_pct": (100 * g.es_universidad.mean()).round(1),
    "silhouette": [round(sil[labels == c].mean(), 3) for c in (1, 2, 3)],
})
T1.index = [PROFILE_SHORT[c] for c in T1.index]
st = df[strato]
T1.loc["Postgraduate stratum"] = [
    len(st), round(st.pct_contratado.mean(), 1), round(st.pct_ordinario.mean(), 1),
    round(st.pct_renacyt_doc_w.mean(), 1), round(100 * (st.n_renacyt == 0).mean(), 1),
    round(100 * st.es_publico.mean(), 1), round(100 * st.es_universidad.mean(), 1), np.nan]
T1.to_csv(TAB / "T1_perfiles.csv")
log("\nT1 — Tabla 1 del manuscrito:")
log(T1.to_string())

pd.DataFrame(stab_rows).to_csv(TAB / "T2_estabilidad.csv", index=False)

pd.DataFrame(pca.components_.T,
             index=[PRETTY.get(v, v) for v in FEATURES],
             columns=[f"PC{i+1}" for i in range(n_comp)]).round(3) \
    .to_csv(TAB / "S1_cargas_pca.csv")

# Kruskal + Dunn
try:
    import scikit_posthocs as sp
    dunn_ok = True
except ImportError:
    dunn_ok = False
    log("  (scikit-posthocs no instalado: se omite Dunn)")
if dunn_ok:
    pairs = []
    for v in FEATURES:
        d = sp.posthoc_dunn(sub.assign(_v=sub[v]), val_col="_v",
                            group_col="P", p_adjust="bonferroni")
        pairs.append({"variable": v,
                      "P1_vs_P2": round(d.loc[1, 2], 4),
                      "P1_vs_P3": round(d.loc[1, 3], 4),
                      "P2_vs_P3": round(d.loc[2, 3], 4)})
    kw = kw.merge(pd.DataFrame(pairs), on="variable")
kw.to_csv(TAB / "S2_kruskal_dunn.csv", index=False)
log(f"\nKruskal-Wallis: {(kw.p < 0.001).sum()}/{len(kw)} indicadores con p < 0.001")
log("  no significativos (p >= 0.05): "
    + ", ".join(kw.loc[kw.p >= .05, "indicator"].tolist()))

desc = sub.groupby("P")[FEATURES].agg(["mean", "std"]).round(2).T
desc.to_csv(TAB / "S3_descriptivos_por_perfil.csv")

ct = pd.crosstab(sub.P, sub.es_publico)
chi2, p, dof, _ = chi2_contingency(ct)
V = np.sqrt(chi2 / (len(sub) * (min(ct.shape) - 1)))
modal = sub.groupby("es_publico").P.agg(lambda s: s.mode()[0])
acc = (sub.P == sub.es_publico.map(modal)).mean()
ct.columns = ["private", "public"]; ct.index = [PROFILE_SHORT[c] for c in ct.index]
ct.assign(chi2=round(chi2, 2), df=dof, p=f"{p:.3g}", cramers_V=round(V, 3),
          ownership_classifier_accuracy=round(100 * acc, 1)).to_csv(TAB / "S4_titularidad.csv")
log(f"\nTitularidad: chi2 = {chi2:.1f}, V = {V:.3f}, "
    f"clasificador = {100*acc:.1f}% correcto")

# Disciplinar
areas = pd.read_csv(DATA_DIR / "precomputed_areas.csv")
m = sub.merge(areas, on="universidad", how="left")
acols = [c for c in areas.columns if c not in ("universidad", "total_matriculados")]
ok = m[acols].notna().all(axis=1)
Xd = StandardScaler().fit_transform(m.loc[ok, acols].values)
best = max(((k, silhouette_score(Xd, KMeans(n_clusters=k, n_init=50,
            random_state=SEED).fit_predict(Xd))) for k in (2, 3, 4)), key=lambda t: t[1])
ld = KMeans(n_clusters=best[0], n_init=50, random_state=SEED).fit_predict(Xd) + 1
m.loc[ok, "DC"] = ld
cd = pd.crosstab(m.loc[ok, "P"], m.loc[ok, "DC"])
cd.index = [PROFILE_SHORT[int(c)] for c in cd.index]
cd.columns = [f"DC{int(c)}" for c in cd.columns]
prof_d = m.loc[ok].groupby("DC")[acols].mean().round(1)
pd.concat([cd, prof_d]).to_csv(TAB / "S5_disciplinar.csv")
log(f"\nDisciplinar: k = {best[0]} (ASW = {best[1]:.3f}), "
    f"{int((~ok).sum())} IES sin datos")
log(cd.to_string())

# Estrato a priori: leave-one-out y sin variables de posgrado
diag = []
names = df.loc[strato, "universidad"].tolist()
for u in names:
    s2 = df[~strato | (df.universidad == u)].reset_index(drop=True)
    r = KMeans(n_clusters=4, n_init=50, random_state=SEED).fit_predict(
        PCA(n_components=n_comp, random_state=SEED).fit_transform(
            StandardScaler().fit_transform(s2[FEATURES].fillna(0))))
    diag.append({"test": f"include only {u}", "n": len(s2),
                 "note": "k=4 with one postgraduate school reinstated"})
sin_pg = [c for c in FEATURES if c not in ("pct_posgrado", "pct_posgrado_egr")]
r = KMeans(n_clusters=4, n_init=50, random_state=SEED).fit_predict(
    PCA(n_components=n_comp, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(df[FEATURES].fillna(0))))
pd.DataFrame(diag).to_csv(TAB / "S6_estrato_posgrado.csv", index=False)

out = sub[["universidad", "es_publico", "es_universidad", "n_renacyt",
           "pct_contratado", "pct_ordinario", "pct_renacyt_doc_w", "P"]].copy()
out["profile"] = out.P.map(PROFILE_SHORT)
out["silhouette"] = sil.round(3)
strat_out = df.loc[strato, ["universidad", "es_publico", "es_universidad",
                            "n_renacyt", "pct_contratado", "pct_ordinario",
                            "pct_renacyt_doc_w"]].copy()
strat_out["P"] = np.nan; strat_out["profile"] = "Postgraduate stratum"
strat_out["silhouette"] = np.nan
pd.concat([out, strat_out]).to_csv(TAB / "asignaciones_v7.csv", index=False)

log("\n" + "=" * 70)
log(f"Figuras en: {FIG}")
log(f"Tablas en:  {TAB}")
log("=" * 70)
