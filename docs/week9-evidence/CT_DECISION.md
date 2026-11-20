# Continuous Training (CT): controlador de decisiones

## Objetivo
Automatizar decisiones de monitoreo y evaluación del modelo P1 usando las evidencias existentes de la semana 9.

## Entradas
- evidently.json: detección de drift.
- ab-v1-v3-20261115/ab_results.json: métricas e intervalos de confianza A/B.

## Ejecución
Desde la raíz del repositorio: python3 scripts/ct_decision.py
Pruebas: python3 -m pytest -q tests/test_ct_decision.py

## Resultados de octubre de 2026
- Drift detectado en 14 de 19 variables (73.68 %).
- Se recomienda revisar la necesidad de reentrenamiento.
- Los intervalos A/B de MAE, RMSE y WAPE incluyen cero.
- Se conserva v1; no se recomienda promover v3.

## Alcance
El controlador lee evidencias existentes y genera decisiones reproducibles.
No descarga datos, no entrena modelos, no modifica MLflow ni promueve versiones.
La detección de drift no implica automáticamente reentrenamiento o promoción.
La evaluación A/B es retrospectiva y requiere validación temporal independiente.

## Pruebas automatizadas
- Evidencia real: detecta drift y conserva v1.
- Ausencia de drift: continúa el monitoreo.
- Intervalos inválidos: rechaza la evidencia.
- Intervalos favorables simulados con limitaciones: conserva v1.
