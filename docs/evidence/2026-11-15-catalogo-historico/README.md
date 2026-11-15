# Recuperación del catálogo histórico y reproducción de P1

El 15 de noviembre de 2026 se probaron dos clones nuevos de `main`, base
`2298aa6e4844dedb782400f0c993ce1f018d9235`, con el cambio local de descarga
alternativa. Ambos comenzaron sin CSV ni Parquet; los datos se descargaron por
HTTP. Las pruebas se ejecutaron en macOS ARM64, Python 3.13.5 y dependencias
de `requirements.txt`. El segundo clon también reentrenó P1 y lo sirvió en Docker.

## Recuperación de los datos

MiBici devolvió HTTP 404 para `nomenclatura_2026_09.csv`. La ingesta recuperó
la [captura de Wayback del 9 de octubre de 2026 a las 13:15:25 UTC](https://web.archive.org/web/20261009131525id_/https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv).
El catálogo tiene 40,841 bytes, 484 registros y SHA-256
`206e45fe633922b634d858142fa21e6fd9f4c421b6435c02980ef290cc2df90d`,
idéntico al manifiesto original. Los catorce CSV mensuales se obtuvieron de MiBici.

| Verificación | Resultado |
|---|---|
| CSV descargados | 15; 455,541,584 bytes en total |
| SHA-256 de los CSV contra la base | 15 de 15 idénticos |
| SHA-256 de los Parquet regenerados contra la base | 15 de 15 idénticos |
| Viajes normalizados | 5,248,635 |
| `data_version` | `cd917f9aa025`, sin cambios |
| Manifiesto procesado | Idéntico al original |
| Manifiesto de descargas | Agrega `downloaded_from`; conserva fuentes y hashes originales |

Los datos y modelos permanecen fuera de Git. Esta carpeta contiene metadatos y
salidas de pruebas, sin datasets. Se sustituyeron rutas personales por marcadores
y se retiraron colores ANSI de los logs; los resultados numéricos se conservan.

Evidencia de la primera ingesta: [estado inicial](ingesta/baseline.json),
[comparación de hashes y entorno](ingesta/resultado.json),
[log](ingesta/ingesta.txt) y [identidad del código probado](ingesta/codigo-verificado.json).

## Pruebas automatizadas

Las ocho pruebas de `tests/test_dataset_download.py` aprobaron. Cubren fuente
primaria válida, HTTP 404, HTML, CSV alterado, agotamiento de fuentes, validación
de archivos locales, ausencia de hashes y conservación de procedencia sin
consultar el portal.

La [suite inicial](ingesta/pytest-sandbox.txt) terminó con 202 aprobadas,
3 omitidas y 11 fallidas porque el sandbox bloqueó `/dev/fd/62` usado por Bash.
Se repitió el archivo completo `test_probar_contenedor.py` fuera del sandbox:
[29 aprobadas](ingesta/pytest-bash.txt). El resultado agregado es 213 aprobadas
y 3 omitidas; no representa una única ejecución completa sin fallas.
Las omisiones eran DeepChecks y dos pruebas de API con modelo real, ejecutadas
después en la reproducción de P1. [Ruff check](ingesta/ruff-check.txt) y
[format](ingesta/ruff-format.txt) aprobaron.

## Reentrenamiento y servicio de P1

El segundo clon usó un almacén MLflow nuevo y exclusivo para esta prueba.
Se ejecutaron MLflow Projects `ingest` y `train`, registro, exportación y pruebas
de API. [Resultado y comandos](p1/resultado.json) y [resumen](p1/resumen-final.json).

| Etapa | Resultado |
|---|---|
| Descarga y preparación | 46.12 segundos; mismos datos y hashes |
| Entrenamiento P1 | 57.14 segundos, incluida preparación y registro de artefactos |
| Comparación histórica | 16 parámetros y 9 métricas de desempeño idénticos; diferencia máxima 0.0 |
| Model Registry aislado | `mibici-equipo26-p1-demand`, versión 1, alias `champion` |
| Exportación | 8 archivos verificados más `registry.json` |
| API con modelo real | 2 pruebas aprobadas |
| DeepChecks con datos reales | 51 chequeos; 0 bloqueantes fallidos y 0 avisos |
| Pruebas del módulo DeepChecks | 19 aprobadas |
| Docker | Build, arranque y 7 verificaciones aprobados; HTTP 200/422 y usuario no root |

Las nueve métricas coinciden con el run histórico
`9d55326449364961b1cbc585ac28cefe` de
[octubre](../2026-10-24-final/runs.json). El tiempo de ajuste no se considera
una métrica de desempeño que deba coincidir.

| Conjunto | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Entrenamiento | 4.930429960868 | 7.995721422713 | 15.5462161309% |
| Validación | 4.851352276717 | 8.171478370762 | 17.8129455805% |
| Prueba | 7.477021536288 | 13.673625827542 | 21.5754576609% |

Run candidato: `b77adcb1769d41839eef31f7220522af`; logged model:
`m-f99d519f75b64004a62dd05666d13037`. Son identificadores del almacén local
de la prueba, no del registro compartido del equipo. MLflow y el contenedor
temporal se detuvieron al finalizar. Docker reutilizó capas de dependencias
existentes e incorporó el modelo recién entrenado.

Logs: [ingesta](p1/01-ingesta.txt), [entrenamiento](p1/02-entrenamiento.txt),
[registro](p1/04-registro.txt), [exportación](p1/05-exportacion.txt),
[API real](p1/06-api-modelo-real.txt), [DeepChecks](p1/10-deepchecks-real.txt),
[detalle de chequeos](p1/deepchecks-resumen.json),
[pruebas DeepChecks](p1/deepchecks-tests.txt), [build Docker](p1/08-docker-build.txt),
[servicio Docker](p1/09-docker-api.txt) y [verificaciones Docker](p1/docker-resultado.json).

## Repetir la descarga

En otro clon con este cambio, sin datos locales:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m mibici.dataset
.venv/bin/python -m pytest tests/test_dataset_download.py -q
```

En Windows, sustituir `.venv/bin/python` por `.venv\Scripts\python.exe`.
Una segunda ingesta en el mismo clon valida y reutiliza los CSV locales;
ya no prueba la descarga desde cero. Para entrenar, registrar y exportar,
seguir el [README del proyecto](../../../README.md); los comandos concretos
de esta ejecución están en [resultado.json](p1/resultado.json).

## Límites y pendiente histórico

- Internet Archive es una fuente externa cuya disponibilidad futura no está
  garantizada. Si no existe una copia local válida y ninguna fuente entrega
  los bytes esperados, la ingesta se detiene antes de actualizar manifiestos
  o procesar datos. El hash verifica integridad, no disponibilidad.
- El digest de entrenamiento de MLflow es `e3b4786b`, frente a `10f3651e`
  del registro original de octubre. La diferencia sigue sin explicación;
  la igualdad de métricas y datos crudos no demuestra identidad binaria del
  modelo original ni resuelve ese pendiente.
- El alcance es P1 histórico. No reproduce P2/P3, el challenger de semana 9,
  el drift de octubre ni los intervalos bootstrap del A/B testing.
- Las pruebas preceden al commit del PR: los runs registraron correctamente
  `git_dirty=true`. Los [SHA-256 del código](p1/codigo-verificado.json) permiten
  identificar el estado probado; los archivos de código coinciden con esos hashes.
