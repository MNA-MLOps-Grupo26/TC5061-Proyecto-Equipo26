# A/B pareado v1 frente a v3 en octubre: corrida local del 15 de noviembre de 2026

> **Corrección (15 de noviembre de 2026, posterior):** la diferencia de v1 queda explicada y el
> reporte se reproduce desde cero. Ver [`../coordenadas-20261115`](../coordenadas-20261115/README.md).
> La sección «Discrepancia de v1» de abajo se conserva como registro de lo que se sabía entonces.
> El primer punto de divergencia es la conversión de texto a float de la latitud del catálogo:
> con FMA o sin FMA cambian 15 estaciones. Esa conversión sí explica el resultado.

Corrida nueva y local de `mibici/ab_testing.py`. **No reemplaza** la evidencia histórica de
[`../comparison.json`](../comparison.json) ni ninguna otra; solo la compara. No se reentrenó,
registró ni promovió ningún modelo. Tampoco se cambiaron aliases, el registro MLflow ni los datos.
Las referencias, la semilla y los bloques se dejaron como estaban.

## Contenido

| Archivo | Qué es |
|---|---|
| `ab_results.json` | Salida completa de `compare`: métricas, diferencias v3 − v1, intervalos, parámetros del bootstrap, procedencia (SHA-256) y comparación contra `../comparison.json` |
| `ab_table.md` | Tabla Markdown generada por el mismo comando |
| `diagnostico_v1.py` | Script de diagnóstico de la diferencia de v1. No produce resultados del A/B |
| `diagnostico-v1.json` | Salida de ese script |

No se incluyen datos, predicciones por fila, modelos ni bases MLflow. `predictions.parquet` se
regenera con los comandos de [`../../week9.md`](../../week9.md#ab-pareado-v1-frente-a-v3).

## Entradas verificadas

- Octubre: `datos_abiertos_2026_10.csv`, descargado de la URL registrada en `drift-summary.json`.
  Su SHA-256 `0cdfb052…b900` es igual al de la evidencia.
- Histórico: 14 Parquet de viajes y `estaciones.parquet`, con hashes iguales a
  `manifests/mibici_processed.json` (`data_version=cd917f9aa025`), y el CSV del catálogo verificado.
  Los archivos ya estaban en local, así que no fue necesaria la corrección de descarga del PR #26.
  Esa corrección sí haría falta para regenerarlos desde cero, porque la URL original de
  `nomenclatura_2026_09.csv` devuelve 404.
- v1: exportación `models/p1` con run `51ef51ad5b534386a3c020692887c6bc` y `model.skops`
  `2cac5ef2…9fae`. Son el mismo run y hash que usan `reports/accuracy-p1` y `reports/challenger-p1`.
  `registry.json` comprueba los hashes de todos los archivos.
- v3: `models/week9-v3/residual_estimator.zip`. El SHA-256 del `.skops` (`278b7a9e…2224`) es igual
  al de `manifest.json`. Se carga con la lista fija de tipos `TreePredictor` y `numpy.dtype`, y se
  rechaza cualquier otro. Se extrae en un directorio temporal, nunca dentro del árbol de Git.
  La demanda se obtiene con `predict_residual_demand`.
- Evaluación: las mismas 14,618 filas estación-día, de 473 estaciones y 31 fechas consecutivas.
  Están alineadas por `estacion_id` y `fecha`, y se verificó que los valores reales coinciden en
  ambas entradas.

## Resultado calculado

| Métrica | Original v1 | Candidata v3 | v3 − v1 | IC 95% de v3 − v1 |
|---|---:|---:|---:|---|
| MAE | 8.0343 | 7.7664 | -0.2680 | [-0.8082, 0.4692] |
| RMSE | 15.7012 | 13.8618 | -1.8394 | [-3.2805, 0.2367] |
| WAPE | 18.4802% | 17.8638% | -0.6164 pp | [-1.7973, 1.1226] pp |
| Sesgo (predicción − real) | -3.8848 | 2.3314 | 6.2162 | No calculado |

## Comparación con el reporte (`../comparison.json`)

Se usó una tolerancia absoluta de 1e-9 (WAPE en proporción). El detalle está en
`ab_results.json → reference_comparison`.

| Valor | Calculado | Reporte | Diferencia | ¿Dentro de tolerancia? |
|---|---:|---:|---:|:---:|
| v3 MAE | 7.766359766419323 | 7.766359766419322 | +8.9e-16 | Sí |
| v3 RMSE | 13.861818201448893 | 13.861818201448893 | 0 | Sí |
| v3 WAPE | 0.17863757848317166 | 0.17863757848317163 | +2.8e-17 | Sí |
| v3 sesgo | 2.3314146316266644 | 2.3314146316266644 | 0 | Sí |
| v1 MAE | 8.034345051707092 | 8.054406442839335 | -0.0201 | **No** |
| v1 RMSE | 15.701217812163023 | 15.891006470864005 | -0.1898 | **No** |
| v1 WAPE | 0.1848016300945742 | 0.18526307128976108 | -0.00046 | **No** |
| v1 sesgo | -3.884778825552489 | -3.9076860550677495 | +0.0229 | **No** |
| IC MAE | [-0.8082, 0.4692] | [-0.8293, 0.4524] | +0.021 / +0.017 | **No** |
| IC RMSE | [-3.2805, 0.2367] | [-3.5421, 0.0322] | +0.262 / +0.204 | **No** |
| IC WAPE (pp) | [-1.7973, 1.1226] | [-1.8566, 1.0911] | +0.059 / +0.032 | **No** |

**v3 se reproduce exactamente; v1 no.** Como los intervalos dependen de las predicciones de v1,
tampoco coinciden. El signo y la lectura no cambian: los puntos favorecen a v3 en MAE, RMSE y WAPE,
pero los tres intervalos incluyen el cero. En RMSE, el límite superior pasa de 0.03 a 0.24.

## Discrepancia de v1: lo que se sabe y lo que no

1. No es un problema de datos, del modelo ni del método. Los hashes de los datos, de octubre y de
   v1 son iguales a los registrados, y los valores reales son idénticos (media 43.4755). La misma
   diferencia aparece en septiembre en la corrida de `accuracy_monitoring`, que no usa
   `ab_testing`: MAE 7.4770 frente a 7.4880.
2. No es la versión de pandas. Al reconstruir el panel con pandas 2.3.3 (entorno de
   `requirements-monitoring.txt`) se obtiene exactamente el mismo MAE de v1 (8.034345051707092)
   que con pandas 3.0.6.
3. La causa probable es la sensibilidad de v1 a `latitud` a nivel de un ulp (`diagnostico-v1.json`).
   152 estaciones de octubre tienen una latitud **exactamente igual** a un umbral de división de v1.
   Al mover la latitud un ulp hacia arriba cambian 1,141 predicciones, y v1 queda en MAE 8.0561 y
   RMSE 15.8904, muy cerca del reporte. El PR #26 ya documentó que Windows y macOS representan
   `latitud` de forma distinta tras procesar el mismo CSV. Kim ejecutó en Windows y esta corrida es
   de macOS ARM64.
4. Esto no es una reproducción exacta. El desplazamiento uniforme no da los valores publicados.
   Tampoco los dan los analizadores `legacy`/`round_trip` de pandas ni un desplazamiento limitado
   a las 2 estaciones que quedan un ulp bajo umbrales de v3. v3 también cambia ante un ulp (654
   filas), pero coincide exactamente con el reporte. Por eso la diferencia de representación en
   Windows, si es la causa, afecta solo a estaciones que importan para v1 y no para v3. Eso no se
   pudo verificar sin el entorno de Kim.

**Bloqueo para cerrar la discrepancia:** falta una de dos cosas. Una es el
`october_predictions.parquet` (o el `predictions.parquet`) de la corrida de Kim, para compararlo
fila por fila. La otra es una corrida de `accuracy_monitoring` + `ab_testing` en Windows con las
mismas versiones. Ninguno de esos archivos se publicó, por decisión correcta de no subir
predicciones por fila. No se sustituyó v1 por otro modelo ni se alteró la latitud para «cuadrar»
el resultado: el diagnóstico solo mide sensibilidad, en memoria.

## Entorno

macOS ARM64 y Python 3.13.9, con `requirements-accuracy.txt` (MLflow, mlflow-skinny y
mlflow-tracing 3.17.2, pandas 3.0.6, scikit-learn 1.9.1, skops 0.16.0) salvo `pywin32==312`, que
no tiene wheel para macOS. uv agregó `gunicorn==26.2.0` como dependencia de MLflow fuera de Windows.
La lista congelada es de Windows; no se modificó.

## Limitaciones

- Es una evaluación retrospectiva de un solo mes (octubre de 2026). Octubre ya se había
  inspeccionado en el trabajo de drift y precisión, así que no es un holdout intacto.
- Los intervalos son exploratorios: hay unos cuatro o cinco bloques semanales independientes,
  dependencia entre estaciones y fechas, y no estacionariedad. No son una prueba formal de
  significancia.
- v1 se entrenó hasta julio y v3 hasta septiembre, así que los efectos del algoritmo y de la
  historia de entrenamiento están confundidos.
- Se trata de una historia observada de un paso diario, no de un pronóstico del mes completo.
- No se calcula intervalo para el sesgo porque no forma parte del método publicado.
