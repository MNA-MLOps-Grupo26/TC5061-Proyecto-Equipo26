# Registro local de los modelos seleccionados

Actualización posterior a la verificación: se registraron los **modelos ya entrenados** en el almacén aislado de la ejecución final. No hubo reentrenamiento ni nuevas corridas de experimentación. El almacén original de otro directorio no se modificó. El registro es **local**, no un servicio compartido ni un despliegue. La evidencia publicada aquí sí es accesible desde GitHub.

| Pregunta | Nombre registrado | Versión / alias | Run fuente |
|---|---|---|---|
| P1 | `mibici-equipo26-p1-demand` | `1` / `champion` | `9d55326449364961b1cbc585ac28cefe` |
| P2 | `mibici-equipo26-p2-risk` | `1` / `champion` | `394c9aafc6c742a2a3baf23dc61d1cf5` |
| P3 | `mibici-equipo26-p3-profiles` | `1` / `champion` | `b42dcd8af94e4f1bb6571f16318ad427` |

Todos conservan experimento, run, logged model, commit `8c7c709feb1a118f1a87a931e4a362492fa3f744`, data_version `cd917f9aa025` y caveat `git_dirty=true` en descripciones/tags. Se inspeccionaron versiones y aliases antes de crear: no había registros al inicio; no se reemplazaron aliases ajenos. El intento inicial del harness completó P1 y registró P2 antes de fallar en una comprobación numérica aplicada a etiquetas de texto. Se inspeccionó el estado y se reanudó reutilizando ambas versiones 1; no se crearon duplicados. El error original se conserva localmente.

## Elección de ciclo de vida: alias `champion`

El alias `champion` identifica la versión seleccionada y permite cargarla por una referencia estable, `models:/<nombre>@champion`, sin acoplar el consumidor al número de versión. Si en una revisión futura se autoriza otra selección, el alias puede apuntar a esa versión conservando la misma referencia de carga. La versión numérica y el run fuente siguen siendo la evidencia inmutable para auditoría; el alias no reemplaza ese linaje. En esta entrega los tres aliases apuntan a la versión 1 del registro **local aislado**. Esta elección expresa selección verificada, no despliegue, disponibilidad de un servicio compartido ni operación real en producción.

## Decisión y comprobación

- **P1:** HGB Poisson original, learning_rate 0.05, max_iter 300. MAE de validación 4.851352 y WAPE 0.178129; mejora sobre el promedio semanal.
- **P2:** HGB learning_rate 0.10, max_leaf_nodes 31, max_iter 300. Se eligió por Recall@30 de validación 0.22167667783739053; F1 macro 0.7205824428770148. No es el candidato con mayor F1.
- **P3:** KMeans original k=3, sin PCA. Conserva el patrón dominical de 23 estaciones de entrenamiento: 23.4%/28.1% de la masa normalizada de salidas/llegadas en domingo, frente a aproximadamente 9% en los otros grupos. k=2 reparte esas estaciones 12/11 y diluye la distinción. Se acepta estabilidad 0.896458 frente a 0.915531 de k=2 para conservar esa interpretación. Silhouette de validación 0.048935: separación débil, grupos solapados; no implica zonas exclusivas ni causas de uso del suelo.

Cada alias se resolvió a la versión y run esperados. Se cargó con `models:/<nombre>@champion` y se compararon sus predicciones exactamente con el modelo guardado sobre su ejemplo original (cinco entradas por modelo). P1/P3 produjeron resultados numéricos finitos; P2 etiquetas válidas no nulas. Ver [recibo completo](registration-evidence.json), sin filas de entrada ni binarios. El aviso de sklearn sobre nombres de columnas de KMeans no impidió igualdad de predicciones.

## Reproducir el registro en otro almacén

Primero reproducir las corridas según el README de esta carpeta; **los IDs locales no existen automáticamente en otro almacén**. Identificar los nuevos runs equivalentes por parámetros, métricas, commit y data_version. Configurar las URI de tracking y registry hacia el almacén elegido y autorizado; inspeccionar `MlflowClient().search_registered_models()` y `search_model_versions()` antes de crear versiones o cambiar aliases. No sobrescribir un `champion` ajeno.

Para cada modelo equivalente, usar su nuevo URI de logged model:

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()
# Sustituir únicamente después de verificar el modelo/run en ese almacén.
version = mlflow.register_model("models:/<logged-model-id>", "<nombre-registrado>")
client.set_registered_model_alias(version.name, "champion", version.version)
model = mlflow.pyfunc.load_model(f"models:/{version.name}@champion")
# Comparar model.predict(ejemplo) con el modelo fuente sobre el mismo ejemplo.
```

Añadir descripciones y tags de linaje como en el recibo. La publicación no contiene una base SQLite portable ni modelos descargables; no equivale a un registro remoto compartido. Los JSON de auditoría anteriores son snapshots **previos al registro** y conservan honestamente la lista vacía de ese momento. Esta actualización documenta el estado posterior.
