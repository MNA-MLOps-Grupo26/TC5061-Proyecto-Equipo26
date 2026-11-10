# P1: comparacion champion contra un modelo alternativo

## Diseno fijado antes del ajuste

Alternativa: Pipeline con SimpleImputer(median), StandardScaler y PoissonRegressor(alpha=1.0,max_iter=1000), valores predeterminados GLM ya definidos en el proyecto. Se ajusta con 165857 estacion-dias del 2 agosto 2025 al 30 septiembre 2026. El original se mantiene sin cambios y fue entrenado hasta julio 2026. Ambos se evaluan sobre exactamente las mismas 14618 filas de octubre, 473 estaciones, 31 fechas, con las mismas 18 entradas y orden.

Imputacion, escalamiento y estimador se ajustan SOLO en entrenamiento. Octubre nunca participa en fit ni ajuste de hiperparametros. Se usa historia observada anterior al dia objetivo para rezagos; no se promete un pronostico mensual de una sola vez. No se hicieron busquedas de candidatos hasta encontrar uno ganador.

Advertencia: octubre ya habia sido inspeccionado en los diagnosticos anteriores, por lo que es un holdout de ajuste, no una confirmacion totalmente nueva. Ademas, cambian tanto familia del modelo como ventana de entrenamiento; el experimento compara dos versiones operativas, no aisla causalmente el efecto de cada cambio. Para confirmacion posterior debe reservarse un periodo futuro antes de explorarlo.

## Resultados en octubre (menor es mejor)

| Metrica | Original HGB | Alternativa Poisson |
|---|---:|---:|
| MAE | 8.0544 | 27.4741 |
| RMSE | 15.8910 | 208.3560 |
| WAPE | 18.5263% | 63.1945% |
| Sesgo predicho-real | -3.9077 | 7.5873 |

La alternativa converge en 69 iteraciones, pero sus errores, especialmente los grandes, son claramente peores. Conservar el original para este comparativo. Registrar un challenger no significa promoverlo ni cambiar el champion. No justificar reentrenamiento automatico solo porque existe drift: exigir mejora validada y controles de errores extremos antes de desplegar otra version.

Bootstrap exploratorio pareado por bloques moviles de 7 dias, todas las estaciones de cada fecha juntas, 2000 replicas, semilla 42. Intervalos 95% de alternativa menos original: MAE [16.7759,21.8345], RMSE [156.8010,223.4526], WAPE [39.4609,49.4303] puntos porcentuales. Evidencia descriptiva consistente con peor desempeno de esta alternativa; un solo mes, dependencia y no estacionariedad limitan generalizacion.

## MLflow y alcance del registro

Se registra el original exportado como version 1 y la alternativa como version 2 del mismo nombre mibici-equipo26-p1-demand en un ALMACEN LOCAL AISLADO de comparacion. NO es el almacen canonico del equipo mlflow-p1-store-20261107 alojado por el companero. No se modifican su base de datos ni alias champion. comparison.json conserva los run_id y versiones locales verificadas. Para que aparezcan en el servidor canonico del equipo es necesario acceso a ese servidor y ejecutar alli una publicacion controlada; no afirmar que ya sucedio.

Almacen local: Documents/TC5061-mlflow-comparison-20261110-final. La primera prueba de serializacion no registro versiones; una segunda registro solamente el original antes de detenerse. Se inspecciono el estado y se reanudo solo la version faltante, sin crear duplicados. Se conservan esos runs fallidos para auditoria. Se permiten explicitamente solo TreePredictor y numpy.dtype conocidos en el original verificado, y numpy.dtype en el pipeline construido localmente; no se confian automaticamente todos los tipos reportados por skops.

## Reproducir

Usar requirements-accuracy.txt y Python 3.13.16. El destino debe ser un almacen NUEVO:

    python -m mibici.challenger_monitoring --source RUTA_DATA_PROCESSED --october data/monitoring/datos_abiertos_2026_10.csv --model RUTA_EXPORTADO_P1 --store RUTA_NUEVO_ALMACEN

No usar el servidor canonico con este flujo de reproduccion local. --resume-original-only requiere exactamente una version 1 etiquetada original_export_copy y el run origen esperado; no reutiliza automaticamente registros desconocidos. Los datos, predicciones y binarios quedan locales, no en GitHub. El script, pruebas y resultados agregados se agregan al PR de monitoreo sin merge.

Trabajo pendiente para la entrega: revisar requerimiento de almacen compartido con el equipo, integrar graficas y conclusiones en PDF de 3-5 paginas, revisar contribuciones y entregar. Este experimento no equivale a la entrega completa.

Verificacion final: ambas versiones registradas se recargaron y reprodujeron todas las predicciones de octubre exactamente (diferencia maxima 0.0). Sin aliases locales. La prediccion extrema del candidato fue 7837.61 para estacion 51 el 21 octubre, frente a 472 viajes reales; original 300.38. Esto explica parte de su RMSE elevado y refuerza no promoverlo. 21 pruebas focalizadas aprobadas tras los ultimos cambios; Ruff/formato y diff check aprobados. No se repitio la suite general para este ultimo commit (el commit anterior obtuvo 190 passed,30 skipped).
