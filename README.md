# Clustering de Universidades Peruanas por Perfil Académico-Investigador

Análisis de clustering sobre datos públicos del sistema universitario peruano (SUNEDU + CONCYTEC).  
**Periodo:** 2024-II, 2025-I, 2025-II | **Universo:** 99 IES | **Features:** 25 variables

---

## Estructura del proyecto

```
clustering-universidades-peru/
├── data/
│   ├── matriz_maestra.csv      ← valores originales + metadata
│   └── matriz_escalada.csv     ← estandarizada, lista para ML
├── src/
│   ├── config.py               ← rutas, parámetros globales
│   ├── 01_pca.py               ← análisis de componentes principales
│   ├── 02_k_optimo.py          ← elbow + silhouette + gap statistic
│   ├── 03_clustering.py        ← K-Means, Ward, DBSCAN, GMM
│   ├── 04_perfilado.py         ← radar charts + Kruskal-Wallis + tabla
│   └── 05_longitudinal.py      ← análisis temporal 3 periodos
├── outputs/
│   ├── figures/                ← PNG/PDF de todas las figuras
│   └── tables/                 ← CSV de tablas para el paper
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/clustering-universidades-peru.git
cd clustering-universidades-peru

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

Ejecutar los scripts en orden desde la raíz del proyecto:

```bash
python src/01_pca.py
python src/02_k_optimo.py
python src/03_clustering.py
python src/04_perfilado.py
python src/05_longitudinal.py
```

Todas las figuras se guardan en `outputs/figures/` y las tablas en `outputs/tables/`.

---

## Datos

Los archivos CSV en `data/` son la salida del pipeline de preprocesamiento que integra:
- **SUNEDU:** docente (2024-II, 2025-I, 2025-II), matriculado (2025-I, 2025-II), egresado (2025)
- **CONCYTEC:** Reporte de investigadores RENACYT activos (CTI Vitae, abril 2026)

---

## Cita

> [En elaboración] — Clustering del sistema universitario peruano usando datos administrativos SUNEDU y CONCYTEC (2024–2025).
