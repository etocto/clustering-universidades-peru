"""
14_figuras_suplemento_v7.py
===========================
Genera las cinco figuras del material suplementario bajo la especificacion v7.
Debe ejecutarse DESPUES de 13_regenerar_v7.py, del que reutiliza la particion.

Figuras en outputs/figures/v7/:
  S1_dendrograma.png        Ward sobre 96 IES, corte en k=3
  S2_siluetas.png           silueta individual por perfil
  S3_seleccion_k.png        ASW por k para las cinco especificaciones
  S4_sensibilidad.png       aluvial S0 -> S2: quien se mueve al recodificar
  S5_matrices_olas.png      matrices de transicion alineadas + retencion

Uso:
    python src/14_figuras_suplemento_v7.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIG_DIR, MPL_STYLE, RANDOM_STATE, TABLE_DIR  # noqa: E402

FIG = FIG_DIR / "v7"
TAB = TABLE_DIR / "v7"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(MPL_STYLE)
SEED = RANDOM_STATE

COLORS = {1: "#BA7517", 2: "#7F77DD", 3: "#1D9E75"}
SHORT = {1: "P1", 2: "P2", 3: "P3"}
LONG = {1: "P1  Non-university,\nno research coupling",
        2: "P2  Flexible-contractual\nuniversities",
        3: "P3  Tenure-consolidated\nuniversities"}

EMPLOYMENT = ["pct_doctorado", "pct_maestria", "pct_exclusiva", "pct_tc",
              "pct_contratado", "pct_ordinario", "edad_media_doc", "pct_fem_doc"]
ENROLMENT = ["pct_fem_mat", "pct_discap", "pct_posgrado", "edad_media_mat",
             "n_departamentos", "nota_prom_egr", "creditos_prom_egr",
             "pct_posgrado_egr", "ratio_mat_doc", "ratio_egr_mat"]
COND = ["pct_renacyt_doc", "puntaje_medio", "nivel_medio", "pct_prod_rec",
        "antiguedad_med", "n_areas_ocde", "pct_fem_renacyt"]
S2 = EMPLOYMENT + ENROLMENT + ["has_renacyt", "pct_renacyt_doc_w"]
S0 = EMPLOYMENT + ENROLMENT + COND

# Siglas oficiales. Se importan de 01_pca.py, que es la fuente unica del
# proyecto, y se completan las dos entradas que faltaban.
def _cargar_siglas():
    import re as _re
    src = (Path(__file__).parent / "01_pca.py").read_text(encoding="utf-8")
    blk = src.split("SIGLAS = {")[1].split("\n}")[0]
    d = dict(_re.findall(r'"([^"]+)":\s*"([^"]+)"', blk))
    d.update({
        'Escuela Nacional Superior de Arte Dramático "Guillermo Ugarte Chamorro"': "ENSAD",
        'Escuela Superior de Arte Dramático "Virgilio Rodríguez Nache"': "ESAD Trujillo",
    })
    return d


SIGLAS = _cargar_siglas()


def fit(data, cols, k, pca_var=0.90):
    X = StandardScaler().fit_transform(data[cols].fillna(0).values)
    p = PCA(random_state=SEED).fit(X)
    nc = int(np.argmax(np.cumsum(p.explained_variance_ratio_) >= pca_var) + 1)
    Xp = PCA(n_components=nc, random_state=SEED).fit_transform(X)
    lab = KMeans(n_clusters=k, init="k-means++", n_init=50,
                 max_iter=500, random_state=SEED).fit_predict(Xp) + 1
    return lab, Xp


# ---------------------------------------------------------------------------
# Datos y particion v7
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "matriz_maestra.csv")
df["has_renacyt"] = (df["n_renacyt"] > 0).astype(int)
df["pct_renacyt_doc_w"] = df["pct_renacyt_doc"].clip(upper=100)
strato = (df["pct_posgrado"] >= 99.9).values
sub = df[~strato].reset_index(drop=True)

labels = np.load(DATA_DIR / "labels_v7.npy")
X_pca = np.load(DATA_DIR / "X_pca_v7.npy")
sub["P"] = labels
print(f"Particion v7 cargada: n={len(sub)}, "
      f"tamanos={dict(zip(*np.unique(labels, return_counts=True)))}")

def sigla(nombre):
    """Devuelve la sigla oficial; si no existe, abrevia de forma legible."""
    if nombre in SIGLAS:
        return SIGLAS[nombre]
    n = (nombre.replace("Universidad ", "U. ").replace("Nacional ", "N. ")
         .replace("Escuela ", "E. ").replace("Superior ", "Sup. ")
         .replace("Instituto ", "I. ").replace("Conservatorio ", "Cons. "))
    return n[:20] + "\u2026" if len(n) > 20 else n


faltantes = [u for u in sub.universidad if u not in SIGLAS]
if faltantes:
    print(f"  aviso: {len(faltantes)} IES sin sigla oficial: {faltantes[:3]}")


# ---------------------------------------------------------------------------
# S1. Dendrograma Ward
# ---------------------------------------------------------------------------
Z = linkage(X_pca, method="ward")
fig, ax = plt.subplots(figsize=(14, 6.2))
cut = Z[-3, 2] * 1.02
# Colorear cada rama con el color del perfil cuando todas las hojas que
# cuelgan de ella pertenecen al mismo; en gris cuando mezcla perfiles. Asi el
# dendrograma usa el mismo codigo cromatico que el resto del articulo.
n_leaf = len(labels)
hojas = {i: {i} for i in range(n_leaf)}
for i, (a_, b_, _, _) in enumerate(Z):
    hojas[n_leaf + i] = hojas[int(a_)] | hojas[int(b_)]
link_col = {}
for i in range(len(Z)):
    perfiles = {labels[j] for j in hojas[n_leaf + i]}
    link_col[n_leaf + i] = COLORS[perfiles.pop()] if len(perfiles) == 1 else "#B4B2A9"

dn = dendrogram(Z, labels=[sigla(u) for u in sub.universidad], ax=ax,
                leaf_font_size=7,
                link_color_func=lambda k: link_col.get(k, "#B4B2A9"))
ax.axhline(cut, color="#D85A30", ls="--", lw=1)
ax.text(0.5, cut * 1.03, "cut at $k=3$", color="#D85A30", fontsize=9)
ax.set_ylabel("Euclidean distance (component space)")
ax.set_xlabel("")

# Colorear cada etiqueta segun el perfil al que pertenece la institucion
orden = dn["leaves"]
for tick, idx in zip(ax.get_xmajorticklabels(), orden):
    tick.set_color(COLORS[labels[idx]])
ax.legend(handles=[mpatches.Patch(color=COLORS[c], label=SHORT[c])
                   for c in (1, 2, 3)], fontsize=9, loc="upper right",
          title="Profile", title_fontsize=9)
plt.xticks(rotation=90)
fig.tight_layout(); fig.savefig(FIG / "S1_dendrograma.png", dpi=300, bbox_inches="tight")
plt.close(); print("S1_dendrograma.png")


# ---------------------------------------------------------------------------
# S2. Diagrama de siluetas
# ---------------------------------------------------------------------------
sil = silhouette_samples(X_pca, labels)
asw = silhouette_score(X_pca, labels)
fig, ax = plt.subplots(figsize=(7.5, 6))
y = 5
for c in (1, 2, 3):
    v = np.sort(sil[labels == c])
    ax.fill_betweenx(np.arange(y, y + len(v)), 0, v,
                     facecolor=COLORS[c], edgecolor=COLORS[c], alpha=.8)
    ax.text(-0.055, y + len(v) / 2,
            f"{SHORT[c]}\n$n$={len(v)}\n$\\bar{{s}}$={v.mean():.3f}",
            va="center", ha="right", fontsize=8.5)
    y += len(v) + 6
ax.axvline(asw, color="#D85A30", ls="--", lw=1.2)
ax.text(asw + .006, y - 3, f"ASW = {asw:.3f}", color="#D85A30", fontsize=9)
ax.axvline(0, color="#666", lw=.6)
ax.set_xlabel("Silhouette coefficient"); ax.set_yticks([])
ax.set_xlim(-0.30, 0.72)
fig.tight_layout(); fig.savefig(FIG / "S2_siluetas.png", dpi=300, bbox_inches="tight")
plt.close(); print("S2_siluetas.png")


# ---------------------------------------------------------------------------
# S3. Seleccion de k para las cinco especificaciones
# ---------------------------------------------------------------------------
sel = pd.read_csv(TAB.parent / "fase1" / "C_seleccion_k.csv") \
    if (TAB.parent / "fase1" / "C_seleccion_k.csv").exists() else None
if sel is None:
    print("  aviso: C_seleccion_k.csv no encontrado, se recalcula")
    rows = []
    specs = {"S0_baseline": (df, S0), "S1_binaria": (df, EMPLOYMENT + ENROLMENT + ["has_renacyt"]),
             "S2_binaria_densidad": (df, S2),
             "S3_submuestra": (df[df.n_renacyt > 0].reset_index(drop=True), S0),
             "S4_organizacional": (df, EMPLOYMENT + ENROLMENT)}
    for name, (d, cols) in specs.items():
        X = StandardScaler().fit_transform(d[cols].fillna(0).values)
        p = PCA(random_state=SEED).fit(X)
        nc = int(np.argmax(np.cumsum(p.explained_variance_ratio_) >= .90) + 1)
        Xp = PCA(n_components=nc, random_state=SEED).fit_transform(X)
        for k in range(2, 9):
            l = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(Xp)
            rows.append({"especificacion": name, "k": k,
                         "ASW": silhouette_score(Xp, l)})
    sel = pd.DataFrame(rows)

piv = sel.pivot(index="k", columns="especificacion", values="ASW")
nice = {"S0_baseline": "S0  conditional variables, zero-coded",
        "S1_binaria": "S1  binary presence",
        "S2_binaria_densidad": "S2  binary presence + density",
        "S3_submuestra": "S3  restricted to institutions with researchers",
        "S4_organizacional": "S4  coupling excluded"}
style = {"S0_baseline": ("#D85A30", "--"), "S1_binaria": ("#7F77DD", "-"),
         "S2_binaria_densidad": ("#1D9E75", "-"), "S3_submuestra": ("#999", ":"),
         "S4_organizacional": ("#BA7517", "-.")}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                         gridspec_kw={"width_ratios": [1.35, 1]})
for col in piv.columns:
    c, ls = style.get(col, ("#666", "-"))
    lw = 2.2 if col == "S2_binaria_densidad" else 1.3
    axes[0].plot(piv.index, piv[col], marker="o", ms=4, color=c, ls=ls,
                 lw=lw, label=nice.get(col, col))
axes[0].set_xlabel("Number of clusters $k$"); axes[0].set_ylabel("Average silhouette width")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_title("(a) Across specifications, 99 institutions",
                  fontsize=11, loc="left")
axes[0].annotate("artefact of zero coding", xy=(2, piv.loc[2, "S0_baseline"]),
                 xytext=(2.7, piv.loc[2, "S0_baseline"] + .012), fontsize=8.5,
                 color="#D85A30",
                 arrowprops=dict(arrowstyle="->", color="#D85A30", lw=.8))

# Panel b: seleccion final sobre 96 IES con S2.
# Se anota el tamano del cluster mas pequeno en cada k: las soluciones con
# ASW superior a k=3 lo consiguen aislando una o dos instituciones, lo que no
# produce ninguna categoria interpretable.
ks, aws, mins = [], [], []
for k in range(2, 8):
    l, Xp = fit(sub, S2, k)
    ks.append(k); aws.append(silhouette_score(Xp, l))
    mins.append(int(np.bincount(l - 1).min()))
axes[1].axvspan(4.5, 7.5, color="#BBB", alpha=.18, zorder=0)
axes[1].text(6, min(aws) + .001, "solutions with\nsingleton clusters",
             fontsize=8.5, color="#777", ha="center", va="bottom")
axes[1].plot(ks, aws, marker="o", ms=6, color="#1D9E75", lw=2, zorder=3)
axes[1].scatter([3], [aws[1]], s=190, facecolor="none", edgecolor="#D85A30",
                lw=2.2, zorder=5)
axes[1].annotate("selected", xy=(3, aws[1]), xytext=(2.15, aws[1] - .007),
                 fontsize=9.5, color="#D85A30",
                 arrowprops=dict(arrowstyle="->", color="#D85A30", lw=.9))
for k, a, m in zip(ks, aws, mins):
    dy = -14 if k == 4 else 10
    axes[1].annotate(f"min $n$={m}", xy=(k, a), xytext=(0, dy),
                     textcoords="offset points", fontsize=8,
                     color="#333" if m >= 9 else "#B00", ha="center")
axes[1].set_xlim(1.6, 7.6)
axes[1].set_ylim(min(aws) - .005, max(aws) + .008)
axes[1].set_xlabel("Number of clusters $k$")
axes[1].set_ylabel("Average silhouette width")
axes[1].set_title("(b) Adopted specification, 96 institutions",
                  fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(FIG / "S3_seleccion_k.png", dpi=300, bbox_inches="tight")
plt.close(); print("S3_seleccion_k.png")


# ---------------------------------------------------------------------------
# S4. Sensibilidad de especificacion: quien se mueve de S0 a S2
# ---------------------------------------------------------------------------
l0, _ = fit(df, S0, 4)
l2, _ = fit(df, S2, 4)
cm = pd.crosstab(l0, l2)
r, c = linear_sum_assignment(-cm.values)
mp = {cm.columns[j]: cm.index[i] for i, j in zip(r, c)}
l2a = pd.Series(l2).map(lambda x: mp.get(x, x)).values
cm = pd.crosstab(pd.Series(l0, name="S0"), pd.Series(l2a, name="S2"))

zero = (df.n_renacyt == 0).values
pal = ["#7F77DD", "#1D9E75", "#5B8FA8", "#BA7517", "#999"]
MOVED = "#C2185B"   # magenta oscuro: no coincide con ningun color de cluster

fig, ax = plt.subplots(figsize=(9.5, 6))
accL, y = {}, 0
for cl in sorted(cm.index):
    h = cm.loc[cl].sum(); accL[cl] = [y, h]; y += h + 3
accR, y2 = {}, 0
for cl in sorted(cm.columns):
    h = cm[cl].sum(); accR[cl] = [y2, h]; y2 += h + 3
for cl, (y0, h) in accL.items():
    ax.add_patch(mpatches.Rectangle((0, y0), .16, h, color=pal[(cl - 1) % 5]))
    ax.text(-.04, y0 + h / 2, f"C{cl} ({h})", ha="right", va="center", fontsize=9)
for cl, (y0, h) in accR.items():
    ax.add_patch(mpatches.Rectangle((1.84, y0), .16, h, color=pal[(cl - 1) % 5]))
    ax.text(2.04, y0 + h / 2, f"C{cl} ({h})", ha="left", va="center", fontsize=9)
offL = {k: v[0] for k, v in accL.items()}
offR = {k: v[0] for k, v in accR.items()}
for i in sorted(cm.index):
    for j in sorted(cm.columns):
        n = cm.loc[i, j]
        if n == 0:
            continue
        y1, y2_ = offL[i], offR[j]
        moved = i != j
        ax.fill_between([.16, 1.84], [y1, y2_], [y1 + n, y2_ + n],
                        color=MOVED if moved else pal[(i - 1) % 5],
                        alpha=.62 if moved else .20, lw=0, zorder=3 if moved else 2)
        offL[i] += n; offR[j] += n
n_mov = int((l0 != l2a).sum())
ax.set_xlim(-.5, 2.5); ax.set_ylim(-4, max(y, y2) + 4)
ax.invert_yaxis(); ax.axis("off")
ax.text(1.0, -3, f"S0  conditional variables, zero-coded          "
                 f"\u2192          S2  binary presence + density",
        ha="center", fontsize=10.5)
ax.text(1.0, max(y, y2) + 2,
        f"{n_mov} of {len(df)} institutions change cluster "
        f"(shown in magenta); {zero.sum()} have no registered researchers",
        ha="center", fontsize=9, color="#555")
fig.tight_layout(); fig.savefig(FIG / "S4_sensibilidad.png", dpi=300, bbox_inches="tight")
plt.close(); print(f"S4_sensibilidad.png  ({n_mov} IES cambian)")


# ---------------------------------------------------------------------------
# S5. Matrices de transicion alineadas + retencion
# ---------------------------------------------------------------------------
panel = pd.read_csv(TAB / "panel_balanceado_v7.csv")
waves = [c for c in panel.columns if c != "universidad"]
pairs = list(zip(waves, waves[1:]))
fig, axes = plt.subplots(1, len(pairs) + 1, figsize=(5.2 * (len(pairs) + 1), 4.4))
ECOL = {1: "#BA7517", 2: "#7F77DD", 3: "#1D9E75"}
for ax, (a, b) in zip(axes, pairs):
    cm = pd.crosstab(panel[a], panel[b])
    norm = cm.div(cm.sum(axis=1), axis=0)
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    color="white" if norm.values[i, j] > .55 else "#222")
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels([f"E{c}" for c in cm.columns])
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels([f"E{c}" for c in cm.index])
    ax.set_xlabel(b); ax.set_ylabel(a)
    ax.set_title(f"{a} \u2192 {b}", fontsize=11)
    ax.grid(False)
# Panel de retencion
ax = axes[-1]
ret = []
for a, b in pairs:
    cm = pd.crosstab(panel[a], panel[b])
    for c in cm.index:
        ret.append({"pair": f"{a}\u2192{b}", "E": c,
                    "retention": 100 * cm.loc[c, c] / cm.loc[c].sum()
                    if c in cm.columns else 0})
ret = pd.DataFrame(ret)
w = .35
for i, (p, g) in enumerate(ret.groupby("pair")):
    ax.bar(np.arange(len(g)) + i * w, g.retention, w,
           color=[ECOL.get(c, "#999") for c in g.E],
           alpha=1 - .35 * i, edgecolor="white",
           label=p)
    for x, v in zip(np.arange(len(g)) + i * w, g.retention):
        ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(np.arange(ret.E.nunique()) + w / 2)
ax.set_xticklabels([f"E{c}" for c in sorted(ret.E.unique())])
ax.set_ylabel("Retention (%)"); ax.set_ylim(0, 108)
ax.legend(fontsize=8); ax.set_title("Retention by configuration", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "S5_matrices_olas.png", dpi=300, bbox_inches="tight")
plt.close(); print("S5_matrices_olas.png")

print(f"\nFiguras del suplemento en: {FIG}")
