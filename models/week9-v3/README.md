# Modelo residual P1, copia portable de la versión 3

Este archivo contiene los bytes exactos del estimador de la versión 3 del registro local `mibici-equipo26-p1-demand`. **No necesita reentrenamiento ni acceso a la computadora de Kimberly para inferencia.** El registro MLflow original no fue modificado, publicado ni convertido en un servicio compartido. No se cambió el modelo vigente/champion.

## Uso desde un clon de esta rama

Python 3.13; crear un entorno separado e instalar `requirements-accuracy.txt` de la raíz (MLflow 3.17.2, no el pin histórico 3.17.1). Instalar el proyecto con `python -m pip install --no-deps -e .`. Descomprimir `models/week9-v3/residual_estimator.zip` dentro de la misma carpeta para obtener `residual_estimator.skops`. Desde la raíz:

```python
import hashlib
import json
from pathlib import Path
import pandas as pd
import skops.io
from mibici.residual_model import predict_residual_demand

folder = Path("models/week9-v3")
artifact = folder / "residual_estimator.skops"
manifest = json.loads((folder / "manifest.json").read_text())
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["sha256"]
estimator = skops.io.load(
    artifact,
    trusted=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "numpy.dtype"],
)
# Datos propios preparados con las columnas VARIABLES_P1 del proyecto.
frame = pd.read_parquet("reports/accuracy-p1/current.parquet")
prediction = predict_residual_demand(estimator, frame)
```

El estimador solo devuelve residuos: **no usar `estimator.predict` como demanda final**. El wrapper suma `salidas_media7` y recorta a cero, igual que la versión MLflow verificada. Los datos no se incluyen: generar las entradas siguiendo `docs/week9.md`; el artefacto no reemplaza la preparación temporal correcta.

`manifest.json` identifica versión/run originales y checksum. `verification.json` documenta la nueva recarga de esta copia y la comparación de las 14,618 predicciones contra la evidencia congelada: diferencia máxima 0.0. Esto no es una nueva evaluación en datos nunca vistos ni una nueva versión registrada.

## Contenido y seguridad

Solo se publica el estimador `.skops` comprimido en ZIP, su procedencia, esta guía y el recibo de verificación. No se publican SQLite, datos de viajes, predicciones por fila, credenciales ni rutas personales. No se copian los metadatos históricos de entorno de MLflow. La lista de tipos permitidos es fija; no se debe reemplazar por confianza automática en cualquier tipo. El modelo sigue siendo una alternativa exploratoria: conservar el original hasta validar en un periodo posterior no utilizado.
