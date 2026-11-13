# Semana 9: drift y comparación de versiones de P1

Esta entrega agrega código y evidencia sin reemplazar las entregas anteriores. No modifica la API, modelos desplegados ni aliases. El PDF de tres páginas contiene la interpretación y la recomendación.

## Resultado

Referencia: entrenamiento original agosto 2025–julio 2026 (137290 estación-días). Actual: octubre 2026 (14618 estación-días, 473 estaciones). Evidently detectó drift en 14/19 columnas: 13/18 entradas y el objetivo. Se usan distancias Wasserstein normalizada y Jensen–Shannon, umbral 0.1. El calendario y la antigüedad explican parte del cambio: drift no demuestra por sí solo pérdida de precisión.

Se conservan faltantes compatibles con HGB: 12 filas con historia semanal incompleta y 72 con atributos faltantes del catálogo fijo (78 filas distintas en total). No hay objetivos faltantes; se verificaron fechas, identificadores y duplicados. Revisar el catálogo y estaciones sin eventos antes de decisiones operativas, sin alterar retroactivamente la referencia. El Parquet del catálogo difiere en hash; se verificó el CSV original y la igualdad numérica. Los ceros pueden incluir cierres o captura incompleta.

En las mismas 14618 filas, original HGB Poisson frente a alternativa HGB residual: MAE 8.05440644 → 7.76635977; RMSE 15.89100647 → 13.86181820; WAPE 18.52630713% → 17.86375785%. La alternativa aprende la corrección de la media de siete días previos y suma esa base, recortando a cero. Selección en septiembre; ajuste final hasta el 30 de septiembre, 165857 filas. Octubre ya había sido inspeccionado: comparación retrospectiva diaria de un paso, no holdout intacto ni pronóstico de todo el mes desde septiembre. Se recomienda conservar el original vigente hasta validar la alternativa en un periodo posterior no utilizado; no promover automáticamente por drift o por esta única comparación. Continuous Training debe combinar calidad, error observado, reentrenamiento y evaluación temporal antes de una decisión de promoción.

## Registro y acceso a modelos

Mismo nombre `mibici-equipo26-p1-demand`: versión 1 copia del original, versión 2 alternativa previa conservada, versión 3 residual, run `c9973a366f47463c91f80a509bd074ca`. Se verificó carga pyfunc fuera del repo y coincidencia exacta de las 14618 predicciones. El wrapper es indispensable: el estimador solo devuelve residuos.

El registro es SQLite LOCAL AISLADO, no el almacén del modelo servido ni un servicio compartido. GitHub contiene código, recibos y, desde el 13 de noviembre, una [copia portable del estimador v3 con guía de carga](../models/week9-v3/README.md), publicada por solicitud de Kimberly para el equipo. Sus bytes coinciden con el artefacto registrado y el wrapper reproduce las predicciones. No se publican DB ni datos; no se afirma acceso del instructor/equipo al almacén MLflow original. La copia permite inferencia desde esta rama sin ese almacén. `registry-verification-20261111.json` omite la ruta personal. `comparison.json` es evidencia histórica conservada: sus banderas `no_github_write` y `no_registry_write` describen el momento del experimento, no el registro/publicación posteriores.

## Reproducción (desde la raíz)

Los entornos de Evidently y precisión están separados para no sustituir `requirements.txt`: Evidently requiere pandas 2.x; el paquete residual se verificó con pandas 3.x. Las listas congeladas corresponden a Windows/Python 3.13.

1. Crear un entorno de monitoreo e instalar `requirements-monitoring.txt`. Ejecutar `python -m mibici.monitoring --source RUTA_PROCESSED --october OCTUBRE.csv --out reports/drift-p1`. La carpeta hermana raw debe conservar el CSV original del catálogo. Produce paneles y reporte Evidently HTML/JSON.
2. En un entorno separado con `requirements-accuracy.txt`, proporcionar la exportación original verificada (incluye registry.json): `python -m mibici.accuracy_monitoring --source RUTA_PROCESSED --october OCTUBRE.csv --model MODELO_ORIGINAL --out reports/accuracy-p1`. Produce current.parquet y october_predictions.parquet. No ajusta el original.
3. Solo para reproducir el entrenamiento, con directorio de salida NUEVO: `python -m scripts.select_local_residual --source RUTA_PROCESSED --current reports/accuracy-p1/current.parquet --original-predictions reports/accuracy-p1/october_predictions.parquet --out reports/improved-p1-local-residual`. El directorio de datos debe ser el histórico verificado agosto 2025–septiembre 2026. Conserva selección temporal y parámetros originales; no amplía la búsqueda.
4. `scripts/register_local_residual.py` documenta el registro de la versión congelada. Solo acepta el almacén aislado llamado TC5061-mlflow-comparison-20261110-final con versiones previas 1 y 2; requiere `--store`, `--current` y opcional `--base`. Se niega a repetir sobre v3 existente. No es un bootstrap de un registro vacío ni se debe ejecutar contra el almacén servido.

No se reentrenó ni re-registró durante esta integración. Las pruebas usan fixtures sintéticos y no escriben en MLflow. El análisis completo existente queda local; solo se integra la cadena necesaria para esta entrega, sin modelos experimentales descartados.

## Feedback anterior y contribuciones

La rama principal ya actualizó las dependencias vulnerables del proyecto y documenta dos excepciones de setuptools del entorno DeepChecks. Se preservan esas correcciones del equipo. Evidently complementa, no sustituye ni vuelve a afirmar una nueva ejecución de DeepChecks sobre datos reales. El resultado de las verificaciones actuales se registra en [week9-evidence/verification.md](week9-evidence/verification.md). La receta nueva usa MLflow 3.17.2 para corregir CVE-2026-53995; el artefacto v3 histórico conserva su metadato 3.17.1. Se debe cargar en el entorno parcheado, no instalar automáticamente ese pin histórico. No se alteró el artefacto ni se creó otra versión.

Los commits de esta rama muestran la integración real bajo la identidad Git configurada, sin atribuir trabajo previo del equipo ni fabricar aportaciones individuales. Cada integrante debe conservar sus contribuciones auténticas en sus propios commits/ramas. No se fusionó ni entregó en Canvas.
