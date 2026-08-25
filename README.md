# Clustering de Universidades Peruanas por Perfil Organizacional

Tipologia organizacional del sistema universitario peruano tras la Ley 30220,
sobre registros administrativos publicos (SUNEDU + CONCYTEC RENACYT).

Periodo: 2024-II, 2025-I, 2025-II
Universo: 99 IES licenciadas — 96 en el clustering + 3 escuelas de posgrado como estrato a priori
Indicadores: 20 (8 de regimen de empleo, 10 de composicion de matricula, 2 de acoplamiento formal al sistema nacional de investigacion)
Solucion: k = 3, ASW = 0.221, semilla fija random_state = 42

## Especificacion v7

Reemplaza la especificacion anterior (25 indicadores, k = 4, 99 IES).

| | Anterior | Actual |
|---|---|---|
| Universo del clustering | 99 | 96 (+3 estrato a priori) |
| Indicadores | 25 | 20 |
| Variables condicionales de RENACYT | 5, codificadas como 0 | excluidas; presencia binaria + densidad |
| Densidad RENACYT | sin acotar | winsorizada al 100 % |
| k | 4 | 3 |

Codificar como cero cinco variables definidas solo para instituciones con
investigadores situaba a las 15 IES sin RENACYT en un vertice del espacio de
distancias y generaba un cluster por construccion. Las tres escuelas de posgrado
formaban grupo por una variable determinada por su licencia, no por similitud
organizacional.

La estructura resiste el cambio: ARI = 0.842 contra la particion anterior, y la
asociacion entre regimen de empleo y densidad de investigadores se mantiene al
excluir los ceros estructurales (rho = +0.34 ordinarios, rho = -0.36 contratados,
p < 0.05, n = 84).

## Perfiles

| | n | % contratado | % ordinario | Densidad RENACYT | % sin RENACYT | % publico | % universidad |
|---|---|---|---|---|---|---|---|
| P1 No universitario, sin acoplamiento | 9 | 49.6 | 8.3 | 0.0 | 100.0 | 100.0 | 0.0 |
| P2 Flexible-contractual | 49 | 81.7 | 13.5 | 15.5 | 8.2 | 10.2 | 98.0 |
| P3 Estable-consolidado | 38 | 28.0 | 63.2 | 36.7 | 2.6 | 84.2 | 97.4 |
| Estrato de posgrado (a priori) | 3 | 46.8 | 19.9 | 16.4 | 33.3 | 33.3 | 0.0 |

## Reproducir

Instalar dependencias con: pip install -r requirements.txt

Luego ejecutar en este orden:

- python src/13_regenerar_v7.py — particion, figuras F1-F7, tablas
- python src/14_figuras_suplemento_v7.py — figuras S1-S5 (requiere el anterior)
- python src/15_figura_overall.py — Figura 1 del manuscrito
- python src/10_fase1_sensibilidad.py — sensibilidad de especificacion S0-S4
- python src/11_especificacion_final.py — verificacion de la especificacion
- python src/05_longitudinal.py — replica en tres olas

Comprobacion de entorno: las primeras lineas de 13_regenerar_v7.py deben mostrar
"Componentes retenidos: 12 (90.8% de varianza)" y
"ASW = 0.221   tamanos = {'P1': 9, 'P2': 49, 'P3': 38}".

## Correcciones de reproducibilidad

Tres discrepancias entre el codigo y descripciones anteriores del metodo,
documentadas por transparencia:

1. Inicializacion. El pipeline usa k-means++ con 50 reinicios, no inicializacion
   jerarquica de Ward. Ward se calcula como contraste independiente (ARI = 0.735).
2. Replica longitudinal. Usa las nueve variables de empleo docente sin PCA.
   Produce configuraciones de empleo E1-E3, que no son los perfiles P1-P3.
3. Tasa de transicion tau. Las etiquetas de olas ajustadas por separado llevan
   numeracion arbitraria. El script 05_longitudinal.py aplica el algoritmo hungaro
   antes de contar transiciones e imprime ambos valores para que la diferencia sea
   verificable.

## Calidad de dato

Tres instituciones (UNJ, UJCM, ULADECH) reportan mas investigadores RENACYT que
docentes totales, lo que indica registros SUNEDU truncados. La densidad se acota
al 100 %. La particion es identica con y sin la correccion (ARI = 1.000).

## Robustez

| Prueba | Resultado |
|---|---|
| Criterio de retencion de componentes (Kaiser a 95 %) | ARI = 1.000 en los cinco criterios |
| Winsorizacion | ARI = 1.000 |
| Especificaciones alternativas del acoplamiento (S0-S4) | ARI >= 0.85 sin ceros estructurales |
| Contraste algoritmico | ARI = 0.825 vs GMM; 0.735 vs Ward |
| Attrition entre olas | chi2 = 2.19, gl = 2, p = 0.334 |
| Estabilidad longitudinal (panel n = 72) | ARI 0.828 y 0.612; tau 5.6 % y 13.9 % |

## Datos

Los datos brutos proceden de portales publicos de acceso abierto: SUNEDU
(docentes, matricula, egresados) y CONCYTEC CTI Vitae (padron RENACYT,
extraccion de abril de 2026). Los archivos derivados en data/ permiten
reproducir todos los analisis sin volver a descargarlos.
