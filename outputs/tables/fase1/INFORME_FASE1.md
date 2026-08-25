# Fase 1 — Resultados y decisiones para la reescritura

Análisis ejecutado sobre el repositorio `etocto/clustering-universidades-peru`
(datos y pipeline originales). Script: `10_fase1_sensibilidad.py`.
Log completo: `fase1_resumen.txt`.

---

## 0. La solución publicada se reproduce exactamente

| Métrica | Valor |
|---|---|
| Componentes retenidos (90 %) | 13 (90.82 % de varianza) |
| PC1 / PC2 | 22.56 % / 15.09 % |
| Tamaños C1–C4 | 46 / 36 / 3 / 14 |
| ASW en *k* = 4 | 0.227 |
| ARI contra `labels_final.npy` | 1.000 |

Todos los números del manuscrito son correctos. Nada de lo que sigue cuestiona
la ejecución; cuestiona qué se puede afirmar a partir de ella.

### Dos discrepancias entre el código y el texto (corregir antes de reenviar)

1. **Inicialización del clustering.** Methods describe Ward → K-means (Punj y
   Stewart 1983). `03_clustering.py` usa K-means++ directo con `n_init=50`.
   Corrí ambas: ARI entre ellas = 0.936, y la solución Ward da 46/37/3/13 con
   ASW 0.230 en lugar de 46/36/3/14 con ASW 0.227. Es decir, la partición
   publicada **no es** la que describe el manuscrito. Hay que alinear texto y
   código, o reportar ambas.
2. **Réplica longitudinal.** Methods afirman que "el pipeline completo de
   estandarización, PCA y K-means se replicó independientemente en tres olas".
   `05_longitudinal.py` usa **9 variables docentes, sin PCA**. Esto es
   defendible como diseño (aísla el eje laboral), pero hay que decirlo así.
   Un revisor que abra el repositorio lo verá.

---

## 1. La objeción de R2 sobre el C4 es correcta, y el hallazgo cambia de forma

Confirmado: **7 variables** (`pct_renacyt_doc`, `puntaje_medio`, `nivel_medio`,
`pct_prod_rec`, `antiguedad_med`, `n_areas_ocde`, `pct_fem_renacyt`) valen 0 en
las 15 IES sin investigadores registrados. Esas instituciones ocupan un vértice
del espacio y su agrupamiento está en parte garantizado por construcción.

| Especificación | n | vars | ASW | ARI vs base | ¿las 15 sin RENACYT siguen juntas? |
|---|---|---|---|---|---|
| S0 baseline (25 vars, ceros) | 99 | 25 | 0.227 | 1.000 | sí: 93 % en un cluster puro |
| S1 binaria `has_renacyt` | 99 | 19 | **0.238** | 0.882 | **no: solo 60 %** |
| S2 binaria + densidad | 99 | 20 | 0.234 | 0.848 | **no: solo 60 %** |
| S3 submuestra con RENACYT | 84 | 25 | **0.130** | 0.634 | n/a |
| S4 solo organizacional | 99 | 18 | 0.171 | 0.325 | no: 67 %, cluster impuro (22 %) |

**Lectura.** El C4 no desaparece pero se parte: con una codificación honesta,
el 60 % de las IES sin RENACYT forma un grupo pequeño y puro (9 instituciones)
y el resto se reparte. La estructura general aguanta (ARI 0.85–0.88), pero
**"cuatro perfiles con un C4 de 14" no es una afirmación robusta**; lo robusto
es que existe un segmento sin acoplamiento formal a RENACYT cuyo tamaño depende
de la especificación.

Nota adicional: S3 (n = 84) baja el ASW a 0.130. Una vez que se quitan los
ceros estructurales, la diferenciación restante es débil. Hay que decirlo.

### El C4 es, en gran medida, una categoría legal redescubierta

| | no universidad | universidad |
|---|---|---|
| C1 | 0 | 46 |
| C2 | 0 | 36 |
| C3 | 3 | 0 |
| C4 | **11** | 3 |

De las 15 IES sin RENACYT, **12 no son universidades** (escuelas de arte,
conservatorios, escuelas de posgrado). De las 14 no-universidades, 12 no tienen
RENACYT. Jaccard = 0.71.

Esto es una **oportunidad, no un problema**: refuerza exactamente la tesis que
R1 pidió (la reforma comprimida metió tipos legales incompatibles bajo un mismo
régimen de licenciamiento) y elimina la circularidad que R1 señaló en el
hallazgo C4. La formulación honesta es: *el algoritmo redescubre la frontera
legal que la Ley 30220 borró administrativamente pero no organizacionalmente.*

---

## 2. La tesis central del paper sobrevive (esto es la buena noticia)

Prueba limpia, sin clustering, correlaciones de Spearman:

| | pct_renacyt_doc (n=99) | pct_renacyt_doc (n=84, solo con RENACYT) |
|---|---|---|
| `pct_ordinario` | +0.50 * | **+0.34 ​*** |
| `pct_contratado` | −0.17 | **−0.36 ​*** |
| `pct_tc` | +0.39 * | +0.24 * |

\* p < 0.05

La asociación entre régimen de empleo y densidad de investigadores **se mantiene
al excluir los ceros estructurales**. Es decir, no la producen las 15
instituciones sin RENACYT. Esta es la evidencia que hay que poner en el
manuscrito para blindar la tesis, y es más fuerte que el argumento actual
basado en Kruskal-Wallis sobre los clusters.

**Advertencia.** Si se quita del clustering toda la dimensión de investigación
(S4), los perfiles organizacionales resultantes ya **no** difieren
significativamente en RENACYT (χ² presencia p = 0.18; densidad p = 0.10). Por
tanto no conviene ir por la Opción 1C analítica pura: el eje se sostiene como
asociación bivariada, no como validación externa del clustering.

---

## 3. C3 sobrevive a leave-one-out pero depende de una sola variable

| Prueba | ARI vs base | ASW | ¿las otras 2 quedan juntas? |
|---|---|---|---|
| Sin Gerens | 0.971 | 0.221 | sí |
| Sin Newman | 1.000 | 0.232 | sí |
| Sin Guerra Naval | 0.935 | 0.225 | sí |
| **Sin `pct_posgrado` y `pct_posgrado_egr`** | 0.885 | 0.238 | **no (67 %)** |

Siluetas individuales: Gerens +0.428, Guerra Naval +0.271, Newman **−0.027**.

R2 tiene razón en la sustancia: C3 existe porque una variable (100 % de
matrícula de posgrado) lo define, y esa variable **no es un hallazgo**, es su
licencia. Newman tiene silueta negativa.

### La salida limpia: sacar C3 del clustering

| Diseño | n | k | ASW | ARI vs base |
|---|---|---|---|---|
| Escuelas de posgrado como estrato legal a priori | 96 | 3 | 0.216 | **1.000** |

Con las 3 escuelas de posgrado fuera y *k* = 3 sobre las 96 restantes se obtiene
**exactamente la misma partición** (46 / 36 / 14, ARI = 1.000). Esto:

- elimina la contradicción de R2.6 (ya no hay que defender un cluster de 3
  instituciones invocando "tamaño suficiente para ser categoría regulatoria");
- convierte una debilidad en una decisión de diseño defendible;
- no cuesta ningún hallazgo.

**Recomendación: adoptarlo.**

---

## 4. Selección de *k*: el argumento actual no se sostiene, pero hay uno mejor

ASW por especificación (`C_seleccion_k.csv`):

| k | S0 base | S1 binaria | S2 bin+dens | S4 organiz. |
|---|---|---|---|---|
| 2 | **0.363** | 0.192 | 0.186 | 0.198 |
| 3 | 0.209 | 0.206 | 0.208 | **0.211** |
| 4 | 0.227 | **0.238** | 0.234 | 0.171 |
| 5 | 0.231 | 0.222 | **0.241** | 0.208 |
| 6 | 0.236 | 0.216 | 0.195 | **0.234** |

Dato útil: bajo las especificaciones honestas (S1, S2) **k = 4 sí es el óptimo
por ASW**, no una concesión a la utilidad de política. El ASW alto de k = 2 en
el baseline es un artefacto de los ceros estructurales: separa "tiene RENACYT"
de "no tiene". Este argumento es mucho más fuerte que el actual y hay que usarlo.

---

## 5. La pérdida de instituciones es aleatoria (R2.9 resuelto a favor)

| Transición | n | salen | χ² (perfil de origen) | p |
|---|---|---|---|---|
| 2024-II → 2025-I | 113 → 99 | 16 | 5.13 (gl 3) | 0.162 |
| 2025-I → 2025-II | 99 → 75 | 26 | 4.17 (gl 3) | 0.244 |
| 2025-I → 2025-II, tipología publicada C1–C4 | | | 0.94 (gl 3) | **0.815** |

La salida no se concentra en ningún perfil. Es una respuesta directa y
favorable a R2.9; basta reportar la tabla.

### Pero el panel balanceado cambia los números longitudinales

| Par de olas | manuscrito | panel balanceado (n = 72) |
|---|---|---|
| 2024-II → 2025-I | ARI 0.893, τ 39.2 % | ARI 0.900, **τ 4.2 %** |
| 2025-I → 2025-II | ARI 0.513, τ 42.5 % | ARI 0.510, **τ 27.8 %** |
| Año completo | 21 reclasificadas | ARI 0.579, τ 25.0 %, **18 reclasificadas** |

Los ARI coinciden; las tasas de transición **no**. La τ de 39.2 % reportada para
la primera transición cae a 4.2 % sobre el panel balanceado con etiquetas
alineadas por el algoritmo húngaro. Sospecho que la τ publicada se computó sin
alinear etiquetas entre olas. **Hay que verificar y corregir esta cifra antes de
reenviar**: es un error numérico que un revisor puede reproducir en minutos.

---

## 6. Decisiones que quedan tomadas para la Fase 2

1. **Especificación principal: S2** (18 organizacionales + binaria + densidad).
   ASW 0.234, ARI 0.848 contra la publicada, y sin ceros estructurales que
   fabriquen distancias. Reportar S0, S1, S3 como tabla de sensibilidad.
2. **Escuelas de posgrado fuera del clustering**, como estrato legal a priori.
   Reportar el resultado *k* = 3 sobre 96 con ARI 1.000.
3. **El C4 se reformula**: de "cuarto perfil institucional" a "segmento sin
   acoplamiento formal al sistema nacional de ciencia, que coincide en un 71 %
   con la categoría legal de instituciones no universitarias incorporadas por
   la Ley 30220". Más honesto y más interesante.
4. **La tesis central se defiende con las correlaciones bivariadas**, no con el
   Kruskal-Wallis sobre clusters.
5. **Justificación de k = 4** basada en que es el óptimo por ASW bajo la
   especificación honesta, no en utilidad de política.
6. **Corregir** la τ longitudinal, la descripción de la inicialización del
   clustering y la descripción de la réplica longitudinal.

El hallazgo estrella cambia de forma pero no se pierde: el paper deja de decir
"cuatro perfiles empíricos" y pasa a decir "dos perfiles organizacionales
robustos, más dos categorías legales que la reforma metió en el mismo régimen y
que la evidencia organizacional no disuelve". Eso encaja mejor con el
reposicionamiento hacia *Higher Education* y responde a R1 y R2 a la vez.

---

## Archivos generados

| Archivo | Contenido |
|---|---|
| `10_fase1_sensibilidad.py` | Script completo, reproducible con semilla 42 |
| `fase1_resumen.txt` | Log íntegro de la corrida |
| `A_especificaciones.csv` | Resumen comparativo de las 5 especificaciones |
| `A_crosswalk_*.csv` | Tablas cruzadas baseline × especificación |
| `A_asignaciones.csv` | Etiqueta de cada IES bajo cada especificación |
| `A_categoria_legal.csv` | Cluster × condición de universidad |
| `A_bivariado_empleo_investigacion.csv` | Correlaciones de Spearman |
| `A_validacion_externa_S4.csv` | Perfiles organizacionales × RENACYT |
| `B_c3_leave_one_out.csv` | Diagnóstico de C3 |
| `C_seleccion_k.csv` | ASW, gap e inercia para k = 2…8 |
| `D_attrition.csv` | Composición de las IES que salen |
| `D_panel_balanceado.csv` | Panel de 72 IES con etiquetas alineadas |

Para reproducir: colocar el script en `src/` del repositorio y ejecutar
`python src/10_fase1_sensibilidad.py`.
