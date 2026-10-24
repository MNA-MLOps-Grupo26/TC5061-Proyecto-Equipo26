# Verificación integrada final — 24 de octubre de 2026

Contribución de Kimberly Ruiz: ejecución independiente del flujo integrado y publicación de evidencia. La autoría de los modelos y de su integración corresponde a los integrantes que los desarrollaron.

Este directorio es una **exportación de corridas locales**, no un servidor MLflow compartido ni una importación ejecutada en otro servidor. El README del proyecto documenta `127.0.0.1:5000`; `mibici/config.py` usa SQLite local por defecto. No se encontró un destino compartido configurado y autorizado. Los identificadores de runs/modelos conservados aquí son referencias al almacén local original: no son enlaces descargables ni versiones de un registro compartido.

## Registro posterior

Los tres modelos seleccionados ya tienen versión 1 y alias `champion` en el registro **local aislado**, ver [registro y comprobaciones](REGISTRATION.md). No es un registry compartido. Las auditorías siguientes describen el snapshot anterior a esta actualización.

## Resultado y alcance

- Fuente ejecutada: `8c7c709feb1a118f1a87a931e4a362492fa3f744`, árbol `56c84ae92f4aba9f20580ff3a5f7e3e5af912ccf`.
- Datos: `cd917f9aa025`; entrenamiento agosto 2025–julio 2026, validación agosto 2026, evaluación septiembre 2026. La comparación y las recomendaciones usan validación; septiembre no se utilizó para ajustar alternativas.
- 12 comandos MLflow Projects (ingest, siete train y cuatro search), más comparación, terminaron con código 0. Los 35 runs terminaron `FINISHED`; las cuatro búsquedas tienen tres candidatos cada una.
- Los 19 runs con métricas reprodujeron la ejecución integrada anterior, con tolerancias `rtol=1e-10`, `atol=1e-12`, y los mismos digests de entrenamiento.
- 17 modelos registrados como **logged models** locales estaban `READY`: se comprobaron firma, ejemplo, carga y predicción. **No se completó el Model Registry**, selección definitiva ni promoción. Esta publicación tampoco los completa.
- La auditoría conserva cuatro fallos de artefactos: faltan las carpetas `tables/` prometidas para la referencia y los tres candidatos HGB de P2. Los otros artefactos comprobados están presentes. No se corrigió código para ocultar este resultado.
- Windows 11, Python 3.13.16, MLflow 3.17.1, scikit-learn 1.9.1; versiones restantes en `environment.json`. Servidor local aislado, ya detenido.

## Lectura rápida

| Pregunta / configuración | Validación | Run original |
|---|---|---|
| P1 promedio semanal | MAE 5.574144; WAPE 0.204669 | ver índice, `p1-baseline-promedio-semanal` |
| P1 HGB lr=0.05, 300 iteraciones | MAE 4.851352; WAPE 0.178129 | `9d55326449364961b1cbc585ac28cefe` |
| P2 baseline semana anterior | F1 macro 0.654202; Recall@30 0.209993 | ver índice, baseline P2 |
| P2 HGB lr=0.10, 31 hojas | F1 macro 0.720582; Recall@30 0.221677 | `394c9aafc6c742a2a3baf23dc61d1cf5` |
| P3 KMeans k=3 | ARI 0.684891; estabilidad 0.896458; silhouette 0.048935 | `b42dcd8af94e4f1bb6571f16318ad427` |
| P3 KMeans k=2 | ARI 0.689818; estabilidad 0.915531; silhouette 0.050108 | `62d0608983ac479a91e243cd6c2f5c2f` |

La métrica declarada de P1 es WAPE (MAE también se muestra); P2 prioriza Recall@30 y no solo F1. P3 exige interpretar estabilidad y utilidad además de separación; esta tabla no finaliza la selección.

- [Índice de 35 runs](run-index.csv): jerarquía, parámetros principales y métricas de validación.
- [Exportación completa de métricas y parámetros](runs.json): todos los runs, sin rutas personales ni datos de entrada.
- [Comparación generada por el proyecto](comparison.md): incluye también métricas de test como evaluación, no como criterio de selección.
- [Índice de modelos](model-run-index.json), [comprobaciones de carga](final-model-reloads.json), [auditoría de artefactos](integrated-audit.json).
- [Linaje](final-lineage-audit.json), [reproducción](reproduction-comparison.json), [igualdad de datos](data-content-equality.json).
- [Comandos y comienzo](commands-start.jsonl), [salidas y término](commands-end.jsonl). Fechas con offset `-07:00`, no UTC implícito.

## Caveat del manifiesto en Windows

El clon estaba limpio antes de ingest. Después cambió únicamente `stations.sha256` en `manifests/mibici_processed.json`; los runs de modelos registran honestamente `git_dirty=true`. No se debe presentar como ejecución con árbol totalmente limpio.

- Hash Parquet comprometido: `80ef17e6566e9a1821896e9739a19fc765c8590f27660488267606ed5a56afce`.
- Hash regenerado en Windows: `bcfd7d47b881edd6159cec56c68325a805d4354417f46ab600f54eec77014eaa`.
- Los 15 CSV originales se verificaron por hash. Los 14 Parquet de viajes coinciden byte a byte con el manifiesto comprometido. Las 484 filas decodificadas de estaciones coinciden exactamente (orden, valores, nulos y tipos) con la normalización del CSV verificado y con la ejecución Windows anterior.
- El manifiesto regenerado pasa la comprobación de integridad. Usar el hash anterior con el nuevo Parquet falla esa comprobación: la diferencia no es meramente cosmética. No están disponibles los bytes del Parquet original de estaciones; no se afirma igualdad decodificada contra ese archivo ausente. El `data_version` basado en CSV no cambió.

Se conserva el [diff exacto](ingest-manifest.diff). No se restauró el manifiesto para esconder el estado dirty.

## Usar la evidencia o reproducir

Los JSON/CSV/Markdown se leen directamente después de clonar esta rama; no requieren MLflow ni acceso al computador original. No contienen datasets, SQLite, credenciales ni binarios de modelos. `models:/m-...` identifica un logged model del almacén original y no puede cargarse desde este export. Para obtener modelos utilizables hay que reproducirlos o acordar posteriormente un destino compartido; no se han inventado runs nuevos para simular una importación.

Para una reproducción real, crear **otro clon** fijado al SHA anterior, instalar las dependencias documentadas y ejecutar los comandos del README principal. En PowerShell usar ` $env:PYTHONUTF8='1' `, ` $env:MLFLOW_TRACKING_URI='http://127.0.0.1:5000' ` y ejecutar `mlflow run` directamente (las continuaciones bash `\` y `export` no son sintaxis PowerShell). El servidor debe ejecutarse en otra terminal, con su propio directorio/base y puerto disponible:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Con el entorno activo, en el clon del SHA ejecutado:

```powershell
$env:PYTHONUTF8='1'
$env:MLFLOW_TRACKING_URI='http://127.0.0.1:5000'
mlflow run . -e ingest --env-manager uv --experiment-name mibici-equipo26
foreach ($q in @('P1','P2')) {
  $models = if ($q -eq 'P1') { @('baseline','hgb','glm') } else { @('baseline','logreg','hgb') }
  foreach ($m in $models) { mlflow run . -e train --env-manager uv --experiment-name mibici-equipo26 -P question=$q -P model=$m }
}
mlflow run . -e train --env-manager uv --experiment-name mibici-equipo26 -P question=P3 -P model=default
mlflow run . -e search --env-manager uv --experiment-name mibici-equipo26 -P question=P1 -P model=hgb
mlflow run . -e search --env-manager uv --experiment-name mibici-equipo26 -P question=P2 -P model=logreg
mlflow run . -e search --env-manager uv --experiment-name mibici-equipo26 -P question=P2 -P model=hgb
mlflow run . -e search --env-manager uv --experiment-name mibici-equipo26 -P question=P3 -P model=kmeans
python -m mibici.modeling.comparar
```

Detenerse si un comando falla; conservar el diff del manifiesto y los nuevos IDs. Los tiempos dependen del equipo; la ejecución original limitó OMP/MKL/OPENBLAS a dos hilos. El caché de datos públicos evitó descargas repetidas, verificando hashes antes de usarlo. No sobrescribir bases o datos de otro integrante.
