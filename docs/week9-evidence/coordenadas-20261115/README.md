# Coordenadas del catálogo y reproducción del A/B desde cero (15 de noviembre de 2026)

Esta evidencia es nueva y local. **Corrige** la explicación de
[`../ab-v1-v3-20261115`](../ab-v1-v3-20261115/README.md), donde la latitud se dejó como «causa
probable» sin cerrar, y la conclusión posterior, que la descartó a partir de una corrida en
Windows con un v1 reentrenado. No reemplaza la evidencia histórica, ni cambia modelos, registros,
referencias o datos.

## Dos resultados distintos

| | Estado |
|---|---|
| Reporte reproducido usando el panel de Kim | **Sí.** El v1 oficial y el v3 portable, aplicados a su `current.parquet`, dan todo `comparison.json` |
| Reporte reproducido desde cero, a partir de los CSV originales | **Sí**, con `--coordenadas sin_fma` en un clon limpio. Los paneles `current` y `reference` resultan idénticos en valores a los de Kim |

## Primer punto de divergencia: el CSV del catálogo convertido a float

Rastreé `latitud` desde el texto de `nomenclatura_2026_09.csv` hasta el panel final.

1. Entre el panel de Kim y el local solo difiere `latitud`: 465 filas de **15 estaciones**, con
   una diferencia máxima de 7.1e-15 (1–2 ulp). Las demás columnas son iguales; `fecha` cambia solo
   de tipo (`datetime64[ns]` en el de Kim, `[us]` en el local).
2. El código del proyecto no hace aritmética con `latitud`: el único paso que produce sus valores
   es la conversión de texto a número en `read_csv` / `to_numeric`. Ninguna variante de pandas
   reproduce el panel de Kim en este Mac (`float_precision` `high`/`legacy`/`round_trip`, motores
   `python`/`pyarrow`, pandas 3.0.6 y 2.3.3). Tampoco lo reproduce el redondeo correcto de
   `float()`, que coincide en 449 de 470.
3. pandas convierte con `precise_xstrtod`, que acumula hasta 17 dígitos con
   `number = number * 10. + digit` (bucle idéntico en pandas 2.3.3 y 3.0.6). Con 17 dígitos el
   acumulado supera 2^53, así que el resultado depende de si esa operación se fusiona en una FMA
   (un redondeo) o no (dos redondeos). Simulé el algoritmo en Python puro sobre las 470 estaciones
   con latitud:

   | Conversión | = panel de Kim | = panel local |
   |---|---:|---:|
   | `precise_xstrtod` sin FMA | **470/470** | 455/470 |
   | `precise_xstrtod` con FMA | 455/470 | **470/470** |
   | `read_csv` en este Mac (ARM64) | 455/470 | 470/470 |

   Las dos variantes difieren exactamente en las 15 estaciones.

**Alcance:** está demostrado que el panel de Kim corresponde a la conversión sin FMA y el local a
la conversión con FMA. Que el entorno de Kim compile pandas sin contracción FMA (por ejemplo, en
x86-64) es una inferencia coherente, pero **no se comprobó en su equipo**. Faltan su sistema
operativo, su arquitectura y su `pip freeze`.

## Por qué cambia v1 y no v3

- El **v1 oficial** tiene umbrales de `latitud` exactamente en los valores **con FMA** de 6 de esas
  estaciones (21, 51, 92, 342, 352 y 400) y en ninguno de los valores sin FMA. Es decir, se
  entrenó con la representación con FMA.
- **v3** tiene umbrales en los valores **sin FMA** de 2 estaciones (59 y 342) y en ninguno de los
  valores con FMA: se entrenó con el panel de Kim.
- Con el panel sin FMA cambian 31 predicciones de v1, y ninguna de v3. Al sustituir solo
  `latitud` en el panel local se obtienen exactamente las predicciones del reporte.

Por eso el MAE de 8.0544 del reporte evalúa v1 con coordenadas que difieren en 1–2 ulp de las
que usó en su entrenamiento. Con su propia representación, v1 da 8.0343. Ninguna representación
es la «correcta» para ambos modelos a la vez. La conclusión del A/B es la misma con las dos.

## Cambio implementado

- `mibici/coordenadas.py` reproduce el bucle de `precise_xstrtod` con la contracción fija, en
  Python puro, de modo que da el mismo resultado en cualquier plataforma. `sin_fma` usa dos
  redondeos; `con_fma` usa uno, con `math.fma`.
- `preprocess.leer_estaciones`, `monitoring.prepare` y `accuracy_monitoring` aceptan
  `--coordenadas {pandas,sin_fma,con_fma}`. Con un modo determinista, el catálogo se reconstruye
  desde el CSV verificado por hash, y se exige que `estacion_id` y `capacidad` sean idénticos al
  Parquet. También se registran las coordenadas cambiadas y una huella de la latitud.
- **El valor por defecto sigue siendo `pandas`.** No cambia el comportamiento, ni `data_version`,
  ni el manifiesto, ni los modelos. Elegir un modo canónico es una decisión del equipo.
- No se ajustan coordenadas por estación, no se redondean y no se modifican referencias.

## Verificación desde cero (`cadena-limpia.json`)

- **Entorno:** macOS 26.7.1 ARM64, Python 3.13.9, pandas 3.0.6, scikit-learn 1.9.1 y MLflow 3.17.2,
  con `requirements-accuracy.txt` salvo `pywin32`, y 2 hilos. El clon era nuevo, sin `data/`.
- **Datos:** ingesta verificada de `data_version=cd917f9aa025`, con `estaciones.parquet` igual al
  manifiesto, y octubre verificado por SHA-256.
- **v1:** exportación oficial con `model.skops` `2cac5ef2…`. **v3:** `models/week9-v3`, `278b7a9e…`.

| Modo | v1 MAE sept. | v1 MAE oct. | ¿`comparison.json` dentro de 1e-9? |
|---|---:|---:|---|
| `sin_fma` | 7.487992599450225 | 8.054406442839335 | **Sí** (diferencia máxima 1.8e-15), con métricas, sesgo y los tres intervalos |
| `con_fma` (= `pandas` en este Mac) | 7.477021536287801 | 8.034345051707092 | No (v1 e intervalos difieren; v3 es igual) |

Con `sin_fma`, los septiembres de Kim (`reports/accuracy-p1/accuracy.json`) también coinciden.

## Pruebas

- `tests/test_coordenadas.py` toma como valores esperados los observados en el panel del reporte y
  en el `estaciones.parquet` del manifiesto, no valores calculados por este código. Cubre:
  - los decimales cortos, que son exactos;
  - los dígitos extra;
  - el rechazo de formatos no soportados, como la notación científica;
  - la equivalencia de `pandas` con el comportamiento original;
  - la protección de `estacion_id` y `capacidad`;
  - la verificación del hash del CSV.
- No se exige que el modelo sea insensible a perturbaciones: ya sabemos que v1 no lo es.

## Archivos

| Archivo | Contenido |
|---|---|
| `diagnostico_coordenadas.py` | Rastreo reproducible; necesita los Parquet de Kim, que no están en Git |
| `diagnostico-coordenadas.json` | Su salida: columnas distintas, conversiones, umbrales y efecto en predicciones |
| `cadena-limpia.json` | Entorno, datos, modelos y resultados de los dos modos en el clon limpio |

Los Parquet de Kim se identifican por hash en `diagnostico-coordenadas.json` (`current`
`f9f2af4d…`) y no se publican.

## Pendientes para el equipo

1. Confirmar el entorno de Kim (sistema operativo, arquitectura, `pip freeze` y el comando que
   generó `current.parquet`) para cerrar la inferencia sobre FMA.
2. Elegir una representación canónica de coordenadas para el pipeline:
   - `con_fma` coincide con el manifiesto y con el entrenamiento de v1;
   - `sin_fma` coincide con el reporte y con el entrenamiento de v3.

   Hasta decidirlo, el valor por defecto sigue dependiendo de la plataforma.
3. Para futuros entrenamientos, decidir por separado si conviene otra política de precisión.
   Ninguna elimina por sí sola la sensibilidad de los árboles a valores exactos.
