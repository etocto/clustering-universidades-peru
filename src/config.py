"""
config.py
=========
Parámetros globales del proyecto. Edita GITHUB_USER antes de ejecutar.
"""

from pathlib import Path

# ── GitHub ────────────────────────────────────────────────────────────────────
# Reemplaza con tu usuario y nombre de repositorio en GitHub
GITHUB_USER   = "etocto"
GITHUB_REPO   = "clustering-universidades-peru"
GITHUB_BRANCH = "main"

GITHUB_RAW = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data"
)

DATA_URLS = {
    "maestra":  f"{GITHUB_RAW}/matriz_maestra.csv",
    "escalada": f"{GITHUB_RAW}/matriz_escalada.csv",
}

# ── Rutas locales ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
FIG_DIR    = ROOT / "outputs" / "figures"
TABLE_DIR  = ROOT / "outputs" / "tables"

for d in [DATA_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Parámetros de análisis ────────────────────────────────────────────────────
RANDOM_STATE = 42
K_RANGE      = range(2, 11)       # rango de k a evaluar
K_FINAL      = 4                  # se asigna en 02_k_optimo.py; sobreescribir aquí si ya lo sabes
PCA_VARIANCE = 0.90               # varianza acumulada a retener en PCA
FUZZY_THRESHOLD = 80              # umbral mínimo de similitud para el join

# Features para clustering (en el mismo orden que matriz_escalada.csv)
FEATURE_COLS = [
    # Docente
    "pct_doctorado", "pct_maestria", "pct_renacyt_doc",
    "pct_exclusiva", "pct_tc", "pct_contratado", "pct_ordinario",
    "edad_media_doc", "pct_fem_doc",
    # Matriculado
    "pct_fem_mat", "pct_discap", "pct_posgrado",
    "edad_media_mat", "n_departamentos",
    # Egresado
    "nota_prom_egr", "creditos_prom_egr", "pct_posgrado_egr",
    # CONCYTEC
    "puntaje_medio", "nivel_medio", "pct_prod_rec",
    "antiguedad_med", "n_areas_ocde", "pct_fem_renacyt",
    # Ratios derivados
    "ratio_mat_doc", "ratio_egr_mat",
]

# Etiquetas de grupo para figuras
FEATURE_GROUPS = {
    "Docente": [
        "pct_doctorado", "pct_maestria", "pct_renacyt_doc",
        "pct_exclusiva", "pct_tc", "pct_contratado", "pct_ordinario",
        "edad_media_doc", "pct_fem_doc",
    ],
    "Matriculado": [
        "pct_fem_mat", "pct_discap", "pct_posgrado",
        "edad_media_mat", "n_departamentos",
    ],
    "Egresado": [
        "nota_prom_egr", "creditos_prom_egr", "pct_posgrado_egr",
    ],
    "CONCYTEC": [
        "puntaje_medio", "nivel_medio", "pct_prod_rec",
        "antiguedad_med", "n_areas_ocde", "pct_fem_renacyt",
    ],
    "Ratios": [
        "ratio_mat_doc", "ratio_egr_mat",
    ],
}

# ── Nombres de clústeres (fuente única de verdad) ─────────────────────────────
# Español — usado en tablas CSV y figuras en español
CLUSTER_NAMES = {
    1: "Universidades Estabilizadas",
    2: "Escuelas de Posgrado",
    3: "Universidades Masivas con Docencia Flexible",
    4: "Instituciones Regionales en Desarrollo",
}

# Inglés — usado en figuras y papers en inglés
CLUSTER_NAMES_EN = {
    1: "Stabilized Universities",
    2: "Postgraduate Schools",
    3: "Mass Universities\nFlexible Teaching",
    4: "Regional Developing\nInstitutions",
}

# Colores por clúster (paleta daltónica)
CLUSTER_COLORS = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517", "#378ADD"]

# Estilo global para matplotlib
MPL_STYLE = {
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
}
