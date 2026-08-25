"""
10_fase1_sensibilidad.py
========================
FASE 1 — Análisis que deciden qué puede afirmar el paper.

Responde a las observaciones del rechazo en Studies in Higher Education:
  R2.5a-c : la dimensión de investigación se codifica como 0 para las IES sin
            RENACYT, lo que puede fabricar el cluster C4.
  R2.6    : k=4 se justifica por utilidad de política, pero k=2/k=3 son mejores
            estadisticamente y C3 tiene solo 3 instituciones.
  R2.7    : C3 puede ser un conjunto de outliers agrupados por exclusion.
  R2.9    : la caida 113 -> 99 -> 75 puede explicar los cambios longitudinales.
  R1.5    : regimen de empleo (atributo organizacional) y RENACYT (acoplamiento
            sistemico) ocupan niveles conceptuales distintos.

Bloques:
  A. Especificaciones alternativas de la dimension de investigacion
  B. Diagnostico de C3 (leave-one-out y sin variables de posgrado)
  C. Re-seleccion de k sobre cada especificacion
  D. Attrition entre olas y panel balanceado

Salidas en outputs/tables/fase1/:
  A_especificaciones.csv        resumen por especificacion
  A_crosswalk_<spec>.csv        tabla cruzada baseline x especificacion
  A_asignaciones.csv            etiqueta de cada IES bajo cada especificacion
  B_c3_leave_one_out.csv        diagnostico de C3
  C_seleccion_k.csv             elbow / ASW / gap por especificacion
  D_attrition.csv               composicion de las IES que salen de cada ola
  D_panel_balanceado.csv        transiciones sobre el panel balanceado
  fase1_resumen.txt             log completo

Uso:
    python src/10_fase1_sensibilidad.py

Requiere data/matriz_maestra.csv, data/matriz_escalada.csv y, para el bloque D,
data/docente_2024_II.csv, docente_2025_I.csv, docente_2025_II.csv.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, RANDOM_STATE, TABLE_DIR  # noqa: E402

OUT = TABLE_DIR / "fase1"
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "fase1_resumen.txt", "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")


def rule(title):
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


# ---------------------------------------------------------------------------
# Definicion de bloques de variables
# ---------------------------------------------------------------------------
# Variables condicionales: solo estan definidas si la IES tiene >=1 investigador
# RENACYT. En la matriz actual se les asigna 0, que es exactamente la decision
# que el Revisor 2 cuestiona.
RESEARCH_COND = [
    "pct_renacyt_doc",
    "puntaje_medio",
    "nivel_medio",
    "pct_prod_rec",
    "antiguedad_med",
    "n_areas_ocde",
    "pct_fem_renacyt",
]

EMPLOYMENT = [
    "pct_doctorado", "pct_maestria", "pct_exclusiva", "pct_tc",
    "pct_contratado", "pct_ordinario", "edad_media_doc", "pct_fem_doc",
]

ENROLMENT = [
    "pct_fem_mat", "pct_discap", "pct_posgrado", "edad_media_mat",
    "n_departamentos", "nota_prom_egr", "creditos_prom_egr",
    "pct_posgrado_egr", "ratio_mat_doc", "ratio_egr_mat",
]

ORGANIZATIONAL = EMPLOYMENT + ENROLMENT           # 18 variables
BASELINE_COLS = ORGANIZATIONAL + RESEARCH_COND    # 25 variables


# ---------------------------------------------------------------------------
# Pipeline reutilizable: estandarizar -> PCA(90%) -> Ward -> K-means
# ---------------------------------------------------------------------------
def run_pipeline(df, cols, k=4, pca_var=0.90, seed=RANDOM_STATE, init="kmeans++"):
    """Devuelve etiquetas (1..k), matriz PCA, ASW y metadatos.

    init='kmeans++' reproduce exactamente 03_clustering.py del repositorio.
    init='ward'     aplica la inicializacion jerarquica que describe el
                    manuscrito (Punj & Stewart 1983). Ambas se reportan para
                    documentar la discrepancia entre codigo y redaccion.
    """
    X = StandardScaler().fit_transform(df[cols].fillna(0).values)

    pca_full = PCA(random_state=seed).fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum >= pca_var) + 1)
    X_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    if init == "ward":
        Z = linkage(X_pca, method="ward")
        ward = fcluster(Z, t=k, criterion="maxclust") - 1
        centroids = np.vstack([X_pca[ward == c].mean(axis=0) for c in range(k)])
        km = KMeans(n_clusters=k, init=centroids, n_init=1,
                    max_iter=500, random_state=seed)
    else:
        km = KMeans(n_clusters=k, init="k-means++", n_init=50,
                    max_iter=500, random_state=seed)

    labels = km.fit_predict(X_pca)
    asw = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else np.nan
    return {
        "labels": labels + 1,
        "X_pca": X_pca,
        "n_comp": n_comp,
        "cum_var": float(cum[n_comp - 1]),
        "asw": float(asw),
        "n_vars": len(cols),
    }


def crosswalk(a, b, name_a="baseline", name_b="spec"):
    return pd.crosstab(pd.Series(a, name=name_a), pd.Series(b, name=name_b))


def concentration(labels, mask):
    """Proporcion del subconjunto `mask` que cae en su cluster modal.
    1.0 = el subconjunto sigue siendo un cluster intacto."""
    sub = labels[mask]
    if len(sub) == 0:
        return np.nan, None
    vals, counts = np.unique(sub, return_counts=True)
    modal = vals[np.argmax(counts)]
    return counts.max() / len(sub), int(modal)


def purity(labels, mask, modal):
    """Proporcion del cluster modal que pertenece al subconjunto.
    1.0 = el cluster no contiene nada mas que el subconjunto."""
    if modal is None:
        return np.nan
    in_cluster = labels == modal
    return mask[in_cluster].sum() / in_cluster.sum()


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
master = pd.read_csv(DATA_DIR / "matriz_maestra.csv")
scaled = pd.read_csv(DATA_DIR / "matriz_escalada.csv")

# La matriz escalada ya viene en z-scores; usamos la maestra como fuente para
# poder construir variables nuevas (binaria de RENACYT) sobre la escala original.
df = master.copy()
df["has_renacyt"] = (df["n_renacyt"] > 0).astype(int)

zero_mask = (df["n_renacyt"] == 0).values
say(f"IES totales: {len(df)}")
say(f"IES con n_RENACYT = 0: {zero_mask.sum()}")

baseline = run_pipeline(df, BASELINE_COLS, k=4)

# Renumerar las etiquetas a la nomenclatura del paper:
#   C1 = flexible/contratado (el mas grande), C2 = consolidado/ordinario,
#   C3 = posgrado especializado, C4 = sin RENACYT.
tmp = baseline["labels"]
c4_raw = concentration(tmp, zero_mask)[1]
sizes_raw = {c: int((tmp == c).sum()) for c in np.unique(tmp)}
c3_raw = min((c for c in sizes_raw if c != c4_raw), key=lambda c: sizes_raw[c])
resto = [c for c in sizes_raw if c not in (c3_raw, c4_raw)]
means_ord = {c: df.loc[tmp == c, "pct_ordinario"].mean() for c in resto}
c2_raw = max(means_ord, key=means_ord.get)
c1_raw = [c for c in resto if c != c2_raw][0]
remap = {c1_raw: 1, c2_raw: 2, c3_raw: 3, c4_raw: 4}
df["cl_baseline"] = pd.Series(tmp).map(remap).values
baseline["labels"] = df["cl_baseline"].values

rule("REPLICACION DE LA SOLUCION PUBLICADA")
say(f"Componentes retenidos: {baseline['n_comp']}  "
    f"(varianza acumulada {baseline['cum_var']*100:.2f}%)")
say(f"ASW k=4: {baseline['asw']:.3f}")
say("Tamanos de cluster (nomenclatura del paper): "
    + str({f"C{c}": int((df.cl_baseline == c).sum()) for c in range(1, 5)}))

pub = DATA_DIR / "labels_final.npy"
if pub.exists():
    lp = np.load(pub)
    say(f"ARI contra labels_final.npy del repositorio: "
        f"{adjusted_rand_score(lp, df['cl_baseline']):.3f}")

ward = run_pipeline(df, BASELINE_COLS, k=4, init="ward")
say(f"\nCHEQUEO DE REPRODUCIBILIDAD — inicializacion Ward (la que describe Methods):")
say(f"  tamanos = {dict(zip(*np.unique(ward['labels'], return_counts=True)))}, "
    f"ASW = {ward['asw']:.3f}")
say(f"  ARI Ward-init vs k-means++ = "
    f"{adjusted_rand_score(ward['labels'], df['cl_baseline']):.3f}")
say("  El codigo publicado usa k-means++ directo; el manuscrito describe")
say("  inicializacion jerarquica Ward. Hay que alinear texto y codigo.")

conc_base = concentration(df["cl_baseline"].values, zero_mask)[0]
say(f"\nIES sin RENACYT: concentracion en C4 = {conc_base:.2f}, "
    f"pureza de C4 = {purity(df['cl_baseline'].values, zero_mask, 4):.2f}")

c3_base = 3
pg_mask = (df["cl_baseline"] == 3).values


# ===========================================================================
# BLOQUE A — Especificaciones alternativas de la dimension de investigacion
# ===========================================================================
rule("BLOQUE A — SENSIBILIDAD DE LA DIMENSION DE INVESTIGACION (R2.5a-c)")

specs = {
    "S0_baseline": dict(
        cols=BASELINE_COLS, subset=None,
        desc="25 indicadores; 7 variables de investigacion en 0 para IES sin RENACYT",
    ),
    "S1_binaria": dict(
        cols=ORGANIZATIONAL + ["has_renacyt"], subset=None,
        desc="18 organizacionales + indicador binario de presencia RENACYT",
    ),
    "S2_binaria_densidad": dict(
        cols=ORGANIZATIONAL + ["has_renacyt", "pct_renacyt_doc"], subset=None,
        desc="18 organizacionales + binaria + densidad relativa de investigadores",
    ),
    "S3_submuestra": dict(
        cols=BASELINE_COLS, subset="con_renacyt",
        desc="25 indicadores, solo IES con al menos un investigador RENACYT",
    ),
    "S4_organizacional": dict(
        cols=ORGANIZATIONAL, subset=None,
        desc="solo atributos organizacionales; RENACYT como validacion externa",
    ),
}

rows = []
assign = df[["universidad", "es_publico", "n_renacyt", "cl_baseline"]].copy()

for name, cfg in specs.items():
    if cfg["subset"] == "con_renacyt":
        sub = df[df["n_renacyt"] > 0].reset_index(drop=True)
        sub_mask = np.zeros(len(sub), dtype=bool)   # no hay IES con cero aqui
    else:
        sub = df
        sub_mask = zero_mask

    res = run_pipeline(sub, cfg["cols"], k=4)
    lab = res["labels"]

    # ARI contra baseline sobre las IES comunes
    if cfg["subset"] == "con_renacyt":
        common = df["n_renacyt"] > 0
        ari = adjusted_rand_score(df.loc[common, "cl_baseline"].values, lab)
    else:
        ari = adjusted_rand_score(df["cl_baseline"].values, lab)

    if sub_mask.sum() > 0:
        conc, modal = concentration(lab, sub_mask)
        pur = purity(lab, sub_mask, modal)
    else:
        conc, modal, pur = np.nan, None, np.nan

    sizes = dict(zip(*np.unique(lab, return_counts=True)))

    say(f"\n[{name}] {cfg['desc']}")
    say(f"  n = {len(sub)}  |  variables = {res['n_vars']}  |  "
        f"componentes = {res['n_comp']} ({res['cum_var']*100:.1f}%)")
    say(f"  ASW = {res['asw']:.3f}  |  ARI vs baseline = {ari:.3f}")
    say(f"  tamanos = {sizes}")
    if sub_mask.sum() > 0:
        say(f"  IES sin RENACYT: concentracion = {conc:.2f} en C{modal}, pureza = {pur:.2f}")

    rows.append({
        "especificacion": name,
        "descripcion": cfg["desc"],
        "n_IES": len(sub),
        "n_variables": res["n_vars"],
        "n_componentes": res["n_comp"],
        "varianza_%": round(res["cum_var"] * 100, 2),
        "ASW_k4": round(res["asw"], 3),
        "ARI_vs_baseline": round(ari, 3),
        "tamanos": str(sizes),
        "conc_sin_renacyt": None if np.isnan(conc) else round(conc, 3),
        "pureza_cluster_sin_renacyt": None if np.isnan(pur) else round(pur, 3),
    })

    if cfg["subset"] is None:
        assign[name] = lab
        cw = crosswalk(df["cl_baseline"].values, lab, "baseline", name)
        cw.to_csv(OUT / f"A_crosswalk_{name}.csv")

pd.DataFrame(rows).to_csv(OUT / "A_especificaciones.csv", index=False)
assign.to_csv(OUT / "A_asignaciones.csv", index=False)
say(f"\nGuardado: A_especificaciones.csv, A_asignaciones.csv, A_crosswalk_*.csv")

# Validacion externa para S4: si la dimension de investigacion NO entra al
# clustering, ¿los perfiles organizacionales siguen separandose por RENACYT?
rule("BLOQUE A bis — RENACYT COMO VALIDACION EXTERNA (nivel analitico, R1.5)")
s4 = assign["S4_organizacional"]
ext = df.groupby(s4).agg(
    n=("universidad", "size"),
    media_pct_renacyt=("pct_renacyt_doc", "mean"),
    media_n_renacyt=("n_renacyt", "mean"),
    pct_sin_renacyt=("has_renacyt", lambda s: 100 * (1 - s.mean())),
    pct_publico=("es_publico", lambda s: 100 * s.mean()),
).round(2)
say(ext.to_string())
ext.to_csv(OUT / "A_validacion_externa_S4.csv")

from scipy.stats import kruskal  # noqa: E402

# (a) presencia/ausencia de RENACYT entre perfiles organizacionales
tab_bin = pd.crosstab(s4, df["has_renacyt"])
chi2_b, p_b, dof_b, _ = chi2_contingency(tab_bin)
say(f"\nPresencia de RENACYT x perfil organizacional: chi2 = {chi2_b:.2f}, "
    f"gl = {dof_b}, p = {p_b:.4g}")

# (b) intensidad relativa, solo entre las IES que si tienen investigadores
con = df["has_renacyt"] == 1
groups = [df.loc[con & (s4 == c), "pct_renacyt_doc"].values
          for c in sorted(s4.unique())
          if (con & (s4 == c)).sum() > 0]
H, p = kruskal(*groups)
say(f"Densidad RENACYT entre perfiles (solo IES con investigadores, n={con.sum()}): "
    f"H = {H:.2f}, p = {p:.4g}")
say("Si ambos p < 0.05, el acoplamiento con RENACYT se predice desde la")
say("configuracion organizacional sin haberlo usado para construirla: el eje")
say("es externo y no circular, que es lo que pide R1.5.")

# (c) ¿C4 es un tipo emergente o una categoria legal redescubierta?
rule("BLOQUE A ter — ¿ES C4 UN TIPO EMPIRICO O UNA CATEGORIA LEGAL? (R2.1)")
if "es_universidad" in df.columns:
    tab_legal = pd.crosstab(df["cl_baseline"], df["es_universidad"],
                            rownames=["cluster"], colnames=["es_universidad"])
    say(tab_legal.to_string())
    z = df[zero_mask]
    say(f"\nDe las {len(z)} IES sin RENACYT, {int((z.es_universidad == 0).sum())} "
        f"no son universidades (escuelas de arte, conservatorios, escuelas de posgrado).")
    no_uni = (df["es_universidad"] == 0)
    say(f"De las {int(no_uni.sum())} IES que no son universidades, "
        f"{int((no_uni & zero_mask).sum())} no tienen RENACYT.")
    inter = (no_uni & zero_mask).sum()
    jac = inter / (no_uni | zero_mask).sum()
    say(f"Solapamiento (Jaccard) entre 'no universidad' y 'sin RENACYT': {jac:.2f}")
    say("Un solapamiento alto implica que C4 no es un tipo emergente de los datos")
    say("sino la categoria legal que la Ley 30220 incorporo bajo el mismo regimen;")
    say("decirlo asi es mas honesto y refuerza el argumento del paper.")
    tab_legal.to_csv(OUT / "A_categoria_legal.csv")

# (d) Asociacion empleo <-> investigacion sin pasar por el clustering.
# Es la prueba limpia de la tesis central del paper: si el eje laboral y el eje
# de acoplamiento estan asociados, debe verse a nivel de variables, no solo
# porque ambos entraron al mismo espacio de distancias.
rule("BLOQUE A quater — ASOCIACION EMPLEO x INVESTIGACION SIN CLUSTERING")
from scipy.stats import spearmanr  # noqa: E402

emp_vars = ["pct_ordinario", "pct_contratado", "pct_exclusiva", "pct_tc"]
res_vars = ["pct_renacyt_doc", "n_renacyt", "n_areas_ocde", "puntaje_medio"]
bi_rows = []
for muestra, sel in [("todas (n=99)", df.index),
                     ("solo con RENACYT (n=%d)" % con.sum(), df.index[con])]:
    say(f"\nSpearman rho — {muestra}")
    for ev in emp_vars:
        line = []
        for rv in res_vars:
            r, pv = spearmanr(df.loc[sel, ev], df.loc[sel, rv])
            star = "*" if pv < 0.05 else " "
            line.append(f"{rv}={r:+.2f}{star}")
            bi_rows.append({"muestra": muestra, "empleo": ev, "investigacion": rv,
                            "rho": round(r, 3), "p": round(pv, 4)})
        say(f"  {ev:16s} " + "  ".join(line))
say("\n* p < 0.05. Si la asociacion se sostiene en la submuestra con RENACYT,")
say("el eje laboral y el acoplamiento sistemico son empiricamente distintos y")
say("estan asociados; si solo aparece en la muestra completa, la asociacion la")
say("producen los ceros estructurales.")
pd.DataFrame(bi_rows).to_csv(OUT / "A_bivariado_empleo_investigacion.csv", index=False)


# ===========================================================================
# BLOQUE B — Diagnostico de C3 (R2.6 y R2.7)
# ===========================================================================
rule("BLOQUE B — ¿ES C3 UN TIPO O TRES OUTLIERS? (R2.7)")

c3_ies = df.loc[df["cl_baseline"] == c3_base, "universidad"].tolist()
say("Instituciones en C3: " + "; ".join(c3_ies))

sil_vals = silhouette_samples(baseline["X_pca"], baseline["labels"])
say("\nSilueta individual de las IES de C3:")
for u, s in zip(c3_ies, sil_vals[df["cl_baseline"] == c3_base]):
    say(f"  {s:+.3f}  {u}")

b_rows = []

# B1. Leave-one-out: quitar cada IES de C3 por separado
for u in c3_ies:
    sub = df[df["universidad"] != u].reset_index(drop=True)
    res = run_pipeline(sub, BASELINE_COLS, k=4)
    lab = res["labels"]
    resto = sub["universidad"].isin([x for x in c3_ies if x != u]).values
    conc, modal = concentration(lab, resto)
    tam_modal = int((lab == modal).sum()) if modal else np.nan
    ari = adjusted_rand_score(
        df.loc[df["universidad"] != u, "cl_baseline"].values, lab)
    say(f"\n[LOO] sin {u}")
    say(f"  ARI vs baseline (n={len(sub)}) = {ari:.3f}  |  ASW = {res['asw']:.3f}")
    say(f"  las otras 2 IES de C3 quedan juntas: {conc == 1.0} "
        f"(cluster C{modal}, tamano {tam_modal})")
    b_rows.append({
        "prueba": f"leave_one_out::{u}", "n": len(sub),
        "ARI_vs_baseline": round(ari, 3), "ASW": round(res["asw"], 3),
        "restantes_C3_juntas": bool(conc == 1.0),
        "tamano_cluster_modal": tam_modal,
    })

# B2. Sin las variables de posgrado
cols_sin_pg = [c for c in BASELINE_COLS if c not in ("pct_posgrado", "pct_posgrado_egr")]
res = run_pipeline(df, cols_sin_pg, k=4)
lab = res["labels"]
conc, modal = concentration(lab, pg_mask)
ari = adjusted_rand_score(df["cl_baseline"].values, lab)
say(f"\n[sin pct_posgrado y pct_posgrado_egr] variables = {len(cols_sin_pg)}")
say(f"  ARI vs baseline = {ari:.3f}  |  ASW = {res['asw']:.3f}")
say(f"  tamanos = {dict(zip(*np.unique(lab, return_counts=True)))}")
say(f"  las 3 IES de posgrado siguen juntas: {conc == 1.0} "
    f"(concentracion {conc:.2f}, cluster C{modal}, tamano {(lab==modal).sum()})")
b_rows.append({
    "prueba": "sin_variables_posgrado", "n": len(df),
    "ARI_vs_baseline": round(ari, 3), "ASW": round(res["asw"], 3),
    "restantes_C3_juntas": bool(conc == 1.0),
    "tamano_cluster_modal": int((lab == modal).sum()),
})

# B3. Tratar las escuelas de posgrado como estrato legal a priori y clusterizar
# el resto con k=3 (la alternativa de diseno que elimina la tension de R2.6)
sub = df[~pg_mask].reset_index(drop=True)
res3 = run_pipeline(sub, BASELINE_COLS, k=3)
ari3 = adjusted_rand_score(df.loc[~pg_mask, "cl_baseline"].values, res3["labels"])
say(f"\n[estrato a priori] 3 escuelas de posgrado fuera del clustering; k=3 sobre n={len(sub)}")
say(f"  ARI vs baseline = {ari3:.3f}  |  ASW = {res3['asw']:.3f}")
say(f"  tamanos = {dict(zip(*np.unique(res3['labels'], return_counts=True)))}")
b_rows.append({
    "prueba": "estrato_a_priori_k3", "n": len(sub),
    "ARI_vs_baseline": round(ari3, 3), "ASW": round(res3["asw"], 3),
    "restantes_C3_juntas": None, "tamano_cluster_modal": None,
})

pd.DataFrame(b_rows).to_csv(OUT / "B_c3_leave_one_out.csv", index=False)
say("\nGuardado: B_c3_leave_one_out.csv")


# ===========================================================================
# BLOQUE C — Re-seleccion de k sobre cada especificacion (R2.6)
# ===========================================================================
rule("BLOQUE C — SELECCION DE k POR ESPECIFICACION (R2.6)")

def gap_statistic(X, k, n_ref=50, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    def wk(data, kk):
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(data)
        return np.log(km.inertia_) if km.inertia_ > 0 else 0.0
    obs = wk(X, k)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    refs = [wk(rng.uniform(mins, maxs, size=X.shape), k) for _ in range(n_ref)]
    return float(np.mean(refs) - obs)

c_rows = []
for name, cfg in specs.items():
    sub = df[df["n_renacyt"] > 0].reset_index(drop=True) if cfg["subset"] else df
    X = StandardScaler().fit_transform(sub[cfg["cols"]].fillna(0).values)
    p = PCA(random_state=RANDOM_STATE).fit(X)
    n_comp = int(np.argmax(np.cumsum(p.explained_variance_ratio_) >= 0.90) + 1)
    Xp = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(X)

    say(f"\n[{name}]")
    inertias = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(Xp)
        asw = silhouette_score(Xp, km.labels_)
        gap = gap_statistic(Xp, k)
        inertias[k] = km.inertia_
        say(f"  k={k}: ASW={asw:.3f}  gap={gap:.3f}  inercia={km.inertia_:.1f}")
        c_rows.append({"especificacion": name, "k": k,
                       "ASW": round(asw, 3), "gap": round(gap, 3),
                       "inercia": round(km.inertia_, 1)})

pd.DataFrame(c_rows).to_csv(OUT / "C_seleccion_k.csv", index=False)
say("\nGuardado: C_seleccion_k.csv")


# ===========================================================================
# BLOQUE D — Attrition entre olas y panel balanceado (R2.9)
# ===========================================================================
rule("BLOQUE D — ATTRITION ENTRE OLAS Y PANEL BALANCEADO (R2.9)")

DOC_FEATURES = [
    "pct_doctorado", "pct_maestria", "pct_renacyt_doc", "pct_exclusiva",
    "pct_tc", "pct_contratado", "pct_ordinario", "edad_media_doc", "pct_fem_doc",
]


def agregar_docente(path):
    unis = defaultdict(lambda: dict(total=0, exclusiva=0, tc=0, doctorado=0,
                                    maestria=0, renacyt=0, ordinario=0,
                                    contratado=0, edad_sum=0, edad_n=0, fem=0))
    with open(path, encoding="latin1") as f:
        for row in csv.DictReader(f, delimiter="|"):
            u = row["ENTIDAD"].strip()
            d = unis[u]
            d["total"] += 1
            rd = row.get("REGIMEN_DEDICACION", "")
            if rd == "Dedicación Exclusiva":
                d["exclusiva"] += 1
            if rd == "Tiempo Completo":
                d["tc"] += 1
            na = row.get("NIVEL_ACADEMICO", "")
            if na == "Doctorado":
                d["doctorado"] += 1
            if na in ("Maestro", "Maestría"):
                d["maestria"] += 1
            if row.get("NIVEL_INVESTIGADOR", ""):
                d["renacyt"] += 1
            cat = row.get("CATEGORIA_DOCENTE", "")
            if "Ordinario" in cat:
                d["ordinario"] += 1
            if "Contratado" in cat:
                d["contratado"] += 1
            if row.get("SEXO", "") == "Femenino":
                d["fem"] += 1
            try:
                d["edad_sum"] += int(row.get("EDAD", ""))
                d["edad_n"] += 1
            except (ValueError, TypeError):
                pass
    out = []
    for u, d in unis.items():
        t = d["total"] or 1
        out.append({
            "universidad": u, "doc_total": d["total"],
            "pct_doctorado": d["doctorado"] / t * 100,
            "pct_maestria": d["maestria"] / t * 100,
            "pct_renacyt_doc": d["renacyt"] / t * 100,
            "pct_exclusiva": d["exclusiva"] / t * 100,
            "pct_tc": d["tc"] / t * 100,
            "pct_contratado": d["contratado"] / t * 100,
            "pct_ordinario": d["ordinario"] / t * 100,
            "edad_media_doc": d["edad_sum"] / d["edad_n"] if d["edad_n"] else 0,
            "pct_fem_doc": d["fem"] / t * 100,
        })
    return pd.DataFrame(out)


PERIODS = {
    "2024-II": DATA_DIR / "docente_2024_II.csv",
    "2025-I": DATA_DIR / "docente_2025_I.csv",
    "2025-II": DATA_DIR / "docente_2025_II.csv",
}

waves = {}
for per, path in PERIODS.items():
    if not path.exists():
        say(f"  {per}: archivo no encontrado, se omite")
        continue
    w = agregar_docente(path)
    X = StandardScaler().fit_transform(w[DOC_FEATURES].fillna(0).values)
    km = KMeans(n_clusters=4, init="k-means++", n_init=50,
                random_state=RANDOM_STATE).fit(X)
    w["cluster"] = km.labels_ + 1
    waves[per] = w
    say(f"  {per}: {len(w)} IES  |  tamanos "
        f"{dict(zip(*np.unique(w.cluster, return_counts=True)))}")

if len(waves) >= 2:
    # D1. ¿Quien sale y desde que perfil?
    say("\n--- Composicion de las IES que salen de cada ola ---")
    d_rows = []
    pares = [("2024-II", "2025-I"), ("2025-I", "2025-II")]
    for a, b in pares:
        if a not in waves or b not in waves:
            continue
        wa, wb = waves[a], waves[b]
        salen = set(wa.universidad) - set(wb.universidad)
        entran = set(wb.universidad) - set(wa.universidad)
        comunes = set(wa.universidad) & set(wb.universidad)
        say(f"\n{a} -> {b}: {len(wa)} -> {len(wb)}  "
            f"| salen {len(salen)}, entran {len(entran)}, comunes {len(comunes)}")

        wa2 = wa.copy()
        wa2["sale"] = wa2.universidad.isin(salen)
        tab = pd.crosstab(wa2.cluster, wa2.sale)
        say("Perfil en la ola de origen de las IES que salen:")
        say(tab.to_string())
        if tab.shape[1] == 2 and tab.values.min() >= 0 and tab.values.sum() > 0:
            try:
                chi2, p, dof, exp = chi2_contingency(tab)
                aleatorio = p >= 0.05
                say(f"chi2 = {chi2:.2f}, gl = {dof}, p = {p:.4f}  -> "
                    + ("la salida NO se concentra en un perfil (aleatoria)"
                       if aleatorio else
                       "la salida SE CONCENTRA en perfiles especificos"))
            except ValueError:
                chi2, p = np.nan, np.nan
        else:
            chi2, p = np.nan, np.nan

        # Cruce con la tipologia principal (solo IES del universo de 99)
        merge = wa2.merge(df[["universidad", "cl_baseline", "es_publico"]],
                          on="universidad", how="left")
        con_tipo = merge[merge.cl_baseline.notna()]
        if len(con_tipo):
            say("\nCruce con la tipologia publicada (C1-C4) de las IES del universo de 99:")
            tab_t = pd.crosstab(con_tipo.cl_baseline, con_tipo.sale)
            say(tab_t.to_string())
            if tab_t.shape[1] == 2:
                c2t, pt, doft, _ = chi2_contingency(tab_t)
                say(f"chi2 = {c2t:.2f}, gl = {doft}, p = {pt:.4f} -> "
                    + ("la perdida NO se concentra en ningun perfil publicado"
                       if pt >= 0.05 else
                       "la perdida SE CONCENTRA en perfiles publicados"))

        for cl in sorted(wa2.cluster.unique()):
            sub = wa2[wa2.cluster == cl]
            d_rows.append({
                "transicion": f"{a}->{b}", "cluster_origen": int(cl),
                "n_origen": len(sub), "n_salen": int(sub.sale.sum()),
                "pct_salen": round(100 * sub.sale.mean(), 1),
                "chi2": None if np.isnan(chi2) else round(chi2, 2),
                "p": None if np.isnan(p) else round(p, 4),
            })

    pd.DataFrame(d_rows).to_csv(OUT / "D_attrition.csv", index=False)

    # D2. Panel balanceado
    say("\n--- Panel balanceado (IES presentes en las tres olas) ---")
    comunes = set.intersection(*[set(w.universidad) for w in waves.values()])
    say(f"IES en las tres olas: {len(comunes)}")
    panel = pd.DataFrame({"universidad": sorted(comunes)})
    for per, w in waves.items():
        panel = panel.merge(
            w[["universidad", "cluster"]].rename(columns={"cluster": per}),
            on="universidad", how="left")

    # Alinear etiquetas entre olas con el algoritmo hungaro
    pers = list(waves.keys())
    for a, b in zip(pers, pers[1:]):
        cm = pd.crosstab(panel[a], panel[b]).values
        r, c = linear_sum_assignment(-cm)
        mapping = {int(cb) + 1: int(ra) + 1 for ra, cb in zip(r, c)}
        panel[b] = panel[b].map(lambda x: mapping.get(int(x), int(x)))

    for a, b in zip(pers, pers[1:]):
        ari = adjusted_rand_score(panel[a], panel[b])
        tau = 100 * (panel[a] != panel[b]).mean()
        say(f"{a} -> {b}  (panel balanceado, n={len(panel)}): "
            f"ARI = {ari:.3f}, tau = {tau:.1f}%")

    if len(pers) >= 3:
        ari_total = adjusted_rand_score(panel[pers[0]], panel[pers[-1]])
        tau_total = 100 * (panel[pers[0]] != panel[pers[-1]]).mean()
        neto = int((panel[pers[0]] != panel[pers[-1]]).sum())
        say(f"{pers[0]} -> {pers[-1]} (ano completo): ARI = {ari_total:.3f}, "
            f"tau = {tau_total:.1f}%, reclasificadas netas = {neto}")

    panel.to_csv(OUT / "D_panel_balanceado.csv", index=False)
    say("\nGuardado: D_attrition.csv, D_panel_balanceado.csv")

    say("\nNOTA DE REPRODUCIBILIDAD: el script 05_longitudinal.py del repositorio")
    say("construye las olas con 9 variables docentes y sin PCA, no con las 25")
    say("variables y PCA como afirma el manuscrito. Hay que corregir la redaccion")
    say("de Methods o rehacer la replicacion longitudinal con el pipeline completo.")

rule("FIN")
LOG.close()
print(f"\nResultados en: {OUT}")
