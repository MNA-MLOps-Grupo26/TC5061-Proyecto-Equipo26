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

Los scripts de búsqueda están en:

- `experiments/busqueda_p1.py`
- `experiments/busqueda_p2.py`
- `experiments/busqueda_p3.py`

La evaluación temporal generalizada de P3 está en
`experiments/evaluacion_p3.py`.

Las búsquedas utilizan runs anidados de MLflow: un run padre por
pregunta y tres runs hijos, uno por configuración candidata.

| Pregunta | Run padre |
|---|---|
| P1 | `c75206dd2a584d67939aa708b41bbb9b` |
| P2 | `90ef1aa7d7a141a4b46e68100a20f464` |
| P3 | `35500c48c9474138ab9170bec686104e` |

Los runs hijos incluyen las etiquetas `question`, `model_family`
y `stage_of_work=search`.

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
