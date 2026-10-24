# Apartado B.4 — Búsqueda y comparación de candidatos

## 1. Objetivo

Explorar tres configuraciones adicionales por pregunta (P1, P2 y P3),
compararlas con los modelos de referencia y registrar los resultados
en el experimento MLflow `mibici-equipo26`.

Se conserva la partición temporal del proyecto:

- Entrenamiento: agosto de 2025 a julio de 2026.
- Validación: agosto de 2026.
- Prueba: septiembre de 2026, reservada para evaluación final.

Los candidatos se comparan sin utilizar el conjunto de prueba para
seleccionar hiperparámetros.

## 2. Experimentos y trazabilidad

Todas las corridas se lanzan desde el MLflow Project. Las combinaciones viven en
`config.BUSQUEDAS` y se ejecutan con el punto de entrada `search`:

```bash
RUN="mlflow run . --env-manager uv --experiment-name mibici-equipo26"
$RUN -e search -P question=P1                     # HGB: learning_rate y max_iter
$RUN -e search -P question=P2 -P model=logreg     # C
$RUN -e search -P question=P2 -P model=hgb        # learning_rate y max_leaf_nodes
$RUN -e search -P question=P3                     # k = 2, 4, 5
```

Cada búsqueda abre un run padre (`stage_of_work=search`) y un run hijo por
combinación. Además de las búsquedas, cada pregunta tiene candidatos de otra
familia de modelo lanzados con `train`: P1 `glm` (Poisson lineal) y P2 `hgb`
(gradient boosting). Todos los hijos registran métricas de entrenamiento,
validación y prueba, `fit_time_s` y los artefactos completos (modelo con firma,
manifiestos, gráficas y configuración).

La tabla comparativa por pregunta se genera con
`python -m mibici.modeling.comparar` a partir de los runs del servidor.

**Pendiente:** actualizar las tablas de las secciones 3 a 5 y los IDs de run
con la corrida final en un solo servidor. Los valores de abajo vienen de la
búsqueda exploratoria inicial (solo validación) y se conservan como referencia.

## 3. P1 — Predicción de demanda

Modelo: `HistGradientBoostingRegressor`.

Métrica de selección: MAE de validación (menor es mejor).

| Configuración | learning_rate | max_iter | MAE validación |
|---|---:|---:|---:|
| Referencia | 0.05 | 300 | **4.851352** |
| Candidato 1 | 0.03 | 400 | 4.915204 |
| Candidato 2 | 0.10 | 200 | 4.936871 |
| Candidato 3 | 0.07 | 350 | 4.878019 |

**Decisión preliminar:** conservar la configuración de referencia.
Ninguna de las tres configuraciones adicionales reduce el MAE
de validación.

## 4. P2 — Clasificación del riesgo

Modelo: `LogisticRegression`.

Métrica de selección: F1 macro de validación (mayor es mejor).

| Configuración | C | F1 macro validación |
|---|---:|---:|
| Referencia | 1.0 | 0.722594 |
| Candidato 1 | 0.1 | 0.722698 |
| Candidato 2 | 0.5 | 0.722596 |
| Candidato 3 | 5.0 | **0.722702** |

**Decisión preliminar:** el candidato C=5 obtiene el mayor
F1 macro de validación. Sin embargo, la mejora frente a la
referencia es de aproximadamente 0.000108 puntos, por lo que
su relevancia práctica es limitada.

## 5. P3 — Agrupación de estaciones

Modelo: `KMeans`, con los mismos perfiles de 336 variables
y la misma cohorte de entrenamiento.

| k | Silhouette entrenamiento | ARI validación | Fracción estable | Silhouette validación |
|---:|---:|---:|---:|---:|
| 2 | 0.132305 | **0.689818** | **0.915531** | **0.050108** |
| 3 (referencia) | **0.143329** | 0.684891 | 0.896458 | 0.048935 |
| 4 | 0.079173 | 0.611742 | 0.841962 | -0.006209 |
| 5 | 0.067184 | 0.596464 | 0.822888 | -0.017547 |

La validación utiliza agosto de 2026, con 367 estaciones
evaluables y dos excluidas. La asignación de validación
se realiza sin reajustar los centroides.

**Interpretación:** k=3 ofrece mayor silhouette en entrenamiento,
mientras que k=2 mejora la estabilidad y ligeramente la silhouette
de validación. La decisión final debe considerar el propósito
operativo de la segmentación y el valor de disponer de tres
grupos interpretables frente a dos grupos más estables.

Los runs de P3 registran las métricas `train_*` y `val_*`.
La configuración congelada de referencia en
`references/p3_frozen.json` no se modifica.

## 6. Reproducibilidad y pruebas

La construcción de perfiles de validación P3 se realiza
automáticamente a partir de los datos procesados, sin
depender de archivos temporales de `reports/`.

Se verificó independientemente el candidato k=2:
369 estaciones de entrenamiento, 367 evaluables,
ARI 0.689818, fracción estable 0.915531 y
silhouette de validación 0.050108, coincidentes con MLflow.

Las pruebas automatizadas incluyen cuatro casos nuevos
para k=2, 3, 4 y 5 en `tests/test_evaluacion_p3.py`.

**Resultado de pruebas:** 49 aprobadas (`python -m pytest -q`).

## 7. Alcance y limitaciones

Los resultados corresponden a una búsqueda exploratoria
limitada de hiperparámetros, no a una optimización exhaustiva.
Las diferencias pequeñas entre configuraciones no deben
interpretarse automáticamente como mejoras relevantes
sin análisis adicional.

La selección de candidatos se fundamenta en entrenamiento
y validación. Septiembre se reserva para la evaluación final.
