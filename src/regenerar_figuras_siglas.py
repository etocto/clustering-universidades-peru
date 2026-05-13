"""
regenerar_figuras_siglas.py
============================
Regenera los 3 gráficos usando las siglas oficiales de cada universidad:
  - outputs/figures/03a_dendrograma.png
  - outputs/figures/03b_scatter_kmeans.png
  - outputs/figures/07b_scatter_disciplinar.png

Uso:
    python src/regenerar_figuras_siglas.py

Requisitos previos (deben existir en data/):
    - X_pca.npy
    - k_optimo.npy
    - labels_final.npy
    - labels_disciplinar.npy
    - matriz_maestra.csv
    - matriculado_2025_I.csv  (o .zip)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FIG_DIR, TABLE_DIR, RANDOM_STATE, MPL_STYLE, \
                   CLUSTER_COLORS, DATA_URLS, K_FINAL

plt.rcParams.update(MPL_STYLE)

# ── Diccionario de siglas  ───────────────────────────────────────────────────
SIGLAS = {
    "Universidad Nacional Agraria La Molina":                              "UNALM",
    "Universidad Nacional Agraria de la Selva":                            "UNAS",
    "Universidad Marcelino Champagnat":                                    "UMCH",
    "Universidad Autónoma del Perú S.A.C.":                                "Autónoma",
    "Universidad para el Desarrollo Andino":                               "UDEA",
    "Universidad Nacional de Tumbes":                                      "UNTUMBES",
    "Universidad de Piura":                                                "UDEP",
    "Universidad Tecnológica del Perú S.A.C.":                            "UTP",
    "Universidad de Huánuco":                                              "UDH",
    "Universidad Andina del Cusco":                                        "UAC",
    "Universidad Nacional de Cajamarca":                                   "UNC",
    "Universidad Privada de Huancayo Franklin Roosevelt S.A.C.":          "U. Roosevelt",
    "Universidad del Pacífico":                                            "UP",
    "Universidad ESAN":                                                    "ESAN",
    "Universidad Nacional Intercultural de Quillabamba":                   "UNIQ",
    "Universidad Católica de Trujillo Benedicto XVI":                      "UCT",
    "Universidad Nacional Autónoma de Tayacaja Daniel Hernández Morillo":  "UNAT",
    "Universidad Nacional de Juliaca":                                     "UNAJ",
    "Universidad San Pedro":                                               "USP",
    "Universidad Nacional de Frontera":                                    "UNF",
    "Universidad Nacional José María Arguedas":                            "UNAJMA",
    "Universidad José Carlos Mariátegui":                                  "UJCM",
    "Universidad César Vallejo S.A.C.":                                    "UCV",
    "Universidad Nacional Micaela Bastidas de Apurímac":                   "UNAMBA",
    "Universidad Católica Santo Toribio de Mogrovejo":                     "USAT",
    "Universidad Privada de Pucallpa S.A.C.":                             "UPP",
    "Universidad Nacional del Altiplano":                                  "UNA Puno",
    "Universidad Católica de Santa María":                                 "UCSM",
    "Universidad Inca Garcilaso de la Vega Asociación Civil":              "UIGV",
    "Universidad Señor de Sipán S.A.C.":                                  "USS",
    "Universidad Nacional Toribio Rodríguez de Mendoza de Amazonas":       "UNTRM",
    "Universidad Antonio Ruiz de Montoya":                                 "UARM",
    "Universidad Católica San Pablo":                                      "UCSP",
    "Universidad Nacional Federico Villarreal":                            "UNFV",
    "Universidad Nacional Amazónica de Madre de Dios":                     "UNAMAD",
    "Universidad Privada Antenor Orrego":                                  "UPAO",
    "Universidad Católica Los Ángeles de Chimbote":                        "ULADECH",
    "Universidad Femenina del Sagrado Corazón":                            "UNIFE",
    "Universidad de Lima":                                                 "ULIMA",
    "Escuela de Postgrado Gerens S.A.":                                    "GERENS",
    "Universidad Nacional Hermilio Valdizán de Huánuco":                   "UNHEVAL",
    "Universidad Continental S.A.C.":                                     "Continental",
    "Pontificia Universidad Católica del Perú":                            "PUCP",
    "Universidad Privada del Norte S.A.C.":                               "UPN",
    "Universidad Científica del Sur S.A.C.":                              "UCSUR",
    "Universidad Católica Sedes Sapientiae":                               "UCSS",
    "Universidad Nacional del Callao":                                     "UNAC",
    "Universidad Ricardo Palma":                                           "URP",
    "Universidad Privada San Juan Bautista S.A.C.":                       "UPSJB",
    "Universidad Nacional de Cañete":                                      "UNDC",
    "Universidad Nacional Tecnológica de Lima Sur":                        "UNTELS",
    "Universidad Nacional Santiago Antúnez de Mayolo":                     "UNASAM",
    "Universidad de Ingeniería y Tecnología":                              "UTEC",
    "Universidad Peruana de Ciencias Aplicadas S.A.C.":                   "UPC",
    "Universidad Privada Norbert Wiener S.A.":                            "U. Wiener",
    "Facultad de Teología Pontificia y Civil de Lima":                     "FTPCL",
    "Universidad Nacional de Moquegua":                                    "UNAM",
    "Universidad Nacional Daniel Alcides Carrión":                         "UNDAC",
    "Universidad Nacional de San Agustín de Arequipa":                     "UNSA",
    "Universidad Nacional de Jaén":                                        "UNJ",
    "Universidad Peruana Los Andes":                                       "UPLA",
    "Universidad Le Cordon Bleu S.A.C.":                                  "ULCB",
    "Asociación Civil Universidad de Ciencias y Humanidades":              "UCH",
    "Universidad Privada de Tacna":                                        "UPT",
    "Universidad de San Martín de Porres":                                 "USMP",
    "Universidad Nacional de Piura":                                       "UNP",
    "Universidad Nacional de San Martín":                                  "UNSM",
    "Universidad Nacional Autónoma de Chota":                              "UNACH",
    "Universidad Nacional José Faustino Sánchez Carrión":                  "UNJFSC",
    "Universidad Nacional Intercultural de la Selva Central Juan Santos Atahualpa": "UNISCJSA",
    "Universidad Nacional Autónoma de Huanta":                             "UNAH",
    "Universidad María Auxiliadora S.A.C.":                               "UMA",
    "Universidad Peruana Unión":                                           "UPeU",
    "Universidad Nacional de Huancavelica":                                "UNH",
    "Universidad Andina Néstor Cáceres Velásquez":                         "UANCV",
    "Universidad Nacional Pedro Ruiz Gallo":                               "UNPRG",
    "Universidad Alas Peruanas S.A.":                                     "UAP",
    "Universidad Peruana Cayetano Heredia":                                "UPCH",
    "Universidad Jaime Bausate y Meza":                                    "UJBM",
    "Universidad Nacional de San Cristóbal de Huamanga":                   "UNSCH",
    "Universidad Nacional Mayor de San Marcos":                            "UNMSM",
    "Universidad Nacional de la Amazonía Peruana":                         "UNAP",
    "Universidad Científica del Perú":                                     "UCP",
    "Universidad de Ciencias y Artes de América Latina S.A.C.":           "UCAL",
    "Universidad Nacional del Centro del Perú":                            "UNCP",
    "Escuela de Posgrado Newman S.A.C.":                                  "Newman",
    "Universidad San Ignacio de Loyola S.R.L.":                           "USIL",
    'Escuela Nacional Superior de Arte Dramático "Guillermo Ugarte Chamorro"': "ENSAD",
    "Escuela Nacional Superior de Folklore José María Arguedas":           "ENSF",
    "Escuela Superior de Guerra Naval":                                    "ESGUERN",
    "Conservatorio Regional de Música Luis Duncker Lavalle":               "Duncker Lavalle",
    "Instituto Superior de Música Público Daniel Alomía Robles de Huánuco": "UNDAR",
    "Escuela Superior de Formación Artística Sérvulo Gutiérrez Alarcón de Ica": "ESFA Ica",
    "Escuela Superior de Música Pública José María Valle Riestra de Piura": "ESMU Piura",
    "Escuela Nacional Superior de Ballet":                                 "ENSB",
    'Escuela Superior de Arte Dramático "Virgilio Rodríguez Nache"':       "ESAD Trujillo",
    "Escuela Superior de Formación Artística Pública Mario Urteaga Alvarado de Cajamarca": "ESFAP Cajamarca",
    "Escuela Superior de Arte Pública Ignacio Merino de Piura":            "ESFAP Piura",
    "Instituto Superior de Música Público Leandro Alviña Miranda del Cusco": "IM Cusco",
}

def get_sigla(nombre):
    """Retorna la sigla de una universidad dado su nombre completo."""
    return SIGLAS.get(nombre, nombre[:12] + "…")


# ── 1. Cargar datos base ─────────────────────────────────────────────────────
print("Cargando datos...")
X_pca     = np.load(DATA_DIR / "X_pca.npy")
k_arr     = np.load(DATA_DIR / "k_optimo.npy")
K         = int(K_FINAL) if K_FINAL else int(k_arr[0])
labels_km = np.load(DATA_DIR / "labels_final.npy")

local_m   = DATA_DIR / "matriz_maestra.csv"
df_master = pd.read_csv(local_m) if local_m.exists() else pd.read_csv(DATA_URLS["maestra"])
unis      = df_master["universidad"].values
siglas    = [get_sigla(u) for u in unis]

print(f"  IES: {len(unis)}  |  k = {K}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURA 03a — DENDROGRAMA WARD
# ════════════════════════════════════════════════════════════════════════════════
print("\nGenerando 03a_dendrograma.png ...")

Z = linkage(X_pca, method="ward")
cut_height = Z[-K + 1, 2]

fig, ax = plt.subplots(figsize=(22, 7))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
for sp in ax.spines.values():
    sp.set_linewidth(0.6)
    sp.set_color("#CCCCCC")
ax.tick_params(width=0.6, color="#CCCCCC")
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.7)
ax.set_axisbelow(True)

dendrogram(
    Z,
    labels=siglas,
    ax=ax,
    leaf_font_size=7,
    color_threshold=cut_height,
    above_threshold_color="#B4B2A9",
    leaf_rotation=90,
)

ax.axhline(cut_height, color="#D85A30", linestyle="--",
           linewidth=1.2, alpha=0.85, zorder=5)
ax.text(4, cut_height * 1.03, f"cut k={K}",
        color="#D85A30", fontsize=9, va="bottom")
ax.set_title(f"Ward dendrogram — Hierarchical clustering (k={K})",
             fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Universidad", fontsize=10)
ax.set_ylabel("Distancia euclidiana (espacio PCA)", fontsize=10)

plt.tight_layout()
fig.savefig(FIG_DIR / "03a_dendrograma.png", bbox_inches="tight", dpi=180, facecolor="white")
plt.close()
print("  ✓ Guardado: outputs/figures/03a_dendrograma.png")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURA 03b — SCATTER K-MEANS
# ════════════════════════════════════════════════════════════════════════════════
print("\nGenerando 03b_scatter_kmeans.png ...")

fig, ax = plt.subplots(figsize=(11, 9))
for cl in range(K):
    mask = labels_km == cl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[cl], s=70, label=f"Cluster {cl+1}",
               alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)

threshold_label = 1.5
for i, nombre in enumerate(unis):
    if abs(X_pca[i, 0]) > threshold_label or abs(X_pca[i, 1]) > threshold_label:
        ax.annotate(
            get_sigla(nombre),
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=7, alpha=0.85,
            xytext=(4, 4), textcoords="offset points",
        )

ax.axhline(0, color="#B4B2A9", linewidth=0.5)
ax.axvline(0, color="#B4B2A9", linewidth=0.5)
ax.set_xlabel("PC1", fontsize=11)
ax.set_ylabel("PC2", fontsize=11)
ax.set_title(f"K-Means (k={K}) in PCA space — Peruvian universities 2025", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "03b_scatter_kmeans.png", bbox_inches="tight", dpi=150)
plt.close()
print("  ✓ Guardado: outputs/figures/03b_scatter_kmeans.png")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURA 07b — SCATTER DISCIPLINAR
# ════════════════════════════════════════════════════════════════════════════════
print("\nGenerando 07b_scatter_disciplinar.png ...")

# Cargar clustering disciplinar
labels_disc_path = DATA_DIR / "labels_disciplinar.npy"
if not labels_disc_path.exists():
    print("  ⚠ No se encontró labels_disciplinar.npy — ejecuta primero 07_disciplinar.py")
else:
    labels_disc = np.load(labels_disc_path)

    # Cargar perfil disciplinar
    disc_csv = TABLE_DIR / "07_perfil_disciplinar.csv"
    if not disc_csv.exists():
        print("  ⚠ No se encontró 07_perfil_disciplinar.csv — ejecuta primero 07_disciplinar.py")
    else:
        df_disc = pd.read_csv(disc_csv)
        df_disc["cluster_disc"] = labels_disc + 1

        # PCA 2D sobre el perfil disciplinar
        AREAS_EN = [c for c in df_disc.columns if c not in
                    ("universidad", "cluster_disc", "cluster_main",
                     "is_public", "es_publico")]
        X_disc = df_disc[AREAS_EN].fillna(0).values
        X_s    = StandardScaler().fit_transform(X_disc)
        pca2   = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d   = pca2.fit_transform(X_s)

        best_k = int(df_disc["cluster_disc"].max())
        DISC_COLORS = ["#E07B3F", "#4C72B0", "#55A868", "#C44E52",
                       "#8172B2", "#937860"][:best_k]

        # Etiquetas por clúster dominante
        disc_labels = {}
        for cl in range(1, best_k + 1):
            sub = df_disc[df_disc["cluster_disc"] == cl][AREAS_EN]
            dom = sub.mean().idxmax()
            disc_labels[cl] = dom

        fig, ax = plt.subplots(figsize=(9, 7))
        for cl in range(1, best_k + 1):
            mask = df_disc["cluster_disc"] == cl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=DISC_COLORS[cl - 1], s=65, alpha=0.82,
                       edgecolors="white", linewidths=0.4,
                       label=f"C{cl}: {disc_labels[cl]}")

        for i, uni in enumerate(df_disc["universidad"]):
            if abs(X_2d[i, 0]) > 1.8 or abs(X_2d[i, 1]) > 1.8:
                ax.annotate(
                    get_sigla(uni),
                    (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=6.5, xytext=(4, 4),
                    textcoords="offset points", alpha=0.85,
                )

        ax.axhline(0, color="#B4B2A9", linewidth=0.5)
        ax.axvline(0, color="#B4B2A9", linewidth=0.5)
        ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax.set_title("HEI disciplinary clusters — PCA space", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "07b_scatter_disciplinar.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  ✓ Guardado: outputs/figures/07b_scatter_disciplinar.png")

print("\n✅ Listo. Los 3 gráficos han sido actualizados con las siglas.")
