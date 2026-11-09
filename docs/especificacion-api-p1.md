# Especificación de la API del modelo P1 (Actividad 8)

Este documento define lo que falta implementar: el servicio FastAPI en `api/`. El
modelo ya está recuperado del Model Registry y exportado, y la infraestructura del
contenedor está lista (ver la sección "Servicio de predicción P1" del README). Aquí
solo hay requisitos; el código lo escribe cada responsable.

Corte interno propuesto: domingo 8 de noviembre de 2026, 18:00 (la entrega cierra a
las 23:59).

## 1. Qué ya existe y qué se usa como contrato

| Pieza | Qué espera de la API |
|---|---|
| `Containerfile` | Copia `api/` a `/app/api/`, instala `api/requirements.txt` y ejecuta `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Define `MODEL_PATH=/app/modelo` |
| `models/p1/` (no se versiona) | Modelo MLflow descargado de `models:/mibici-equipo26-p1-demand/1` más `registry.json` con linaje y SHA-256 de cada archivo |
| `scripts/probar_contenedor.sh` | Prueba de aceptación: `/health` debe responder 200, `/predict` debe devolver `viajes_esperados` y una petición incoherente debe dar 422 |
| `docs/evidencia-api/peticion-predict.json`, `peticion-batch.json` | Peticiones de ejemplo con filas reales del panel de P1 |
| `docs/evidencia-api/prediccion-referencia.json` | Predicciones del modelo para esas peticiones, calculadas sin API. La API debe devolver exactamente esos valores |

Para obtener `models/p1/`, pidan la carpeta comprimida a Francisco o expórtenla
desde el almacén con `scripts/exportar_modelo.py` (README, pasos 1 y 2).

## 2. Estructura y arranque

- Paquete `api/` con `__init__.py` y `main.py`, que expone un objeto `app` de FastAPI.
- `api/` debe ser **autocontenido**: la imagen solo copia `api/` y el modelo, así que
  el código de la API no puede importar `mibici` ni otros módulos del repositorio. La
  lista de las 18 variables se define dentro de `api/`; una prueba (que corre fuera de
  la imagen) la compara con `mibici.features.VARIABLES_P1`.
- El directorio del modelo se lee de la variable de entorno `MODEL_PATH`.
- El modelo se carga **una sola vez al arrancar** (por ejemplo, en el `lifespan` de
  FastAPI) con `mlflow.pyfunc.load_model`.
- Antes de aceptar peticiones, el servicio verifica y **no arranca** si algo falla:
  1. **Hashes.** `registry.json` tiene un mapa `archivos` con la ruta relativa de
     cada archivo del modelo y su SHA-256 (`registry.json` no se lista a sí mismo).
     Los archivos del directorio deben ser exactamente esos, sin faltantes, sin
     sobrantes y con el mismo hash.
  2. **Firma.** Los nombres de entrada de la firma del modelo
     (`metadata.get_input_schema().input_names()`) deben ser exactamente las 18
     variables de la sección 4, en el mismo orden.
- Las pruebas deben poder inyectar un modelo falso sin MLflow (por ejemplo, con una
  función que cree la app recibiendo cómo cargar el modelo).

## 3. Endpoints

### `GET /health`

200 con el linaje tomado de `registry.json["modelo"]`:

```json
{
  "estado": "ok",
  "modelo": {"nombre": "...", "version": "1", "alias_consultado": "champion",
             "run_id": "...", "logged_model_id": "...", "source_sha": "...",
             "data_version": "..."},
  "verificacion": {"hashes": "ok", "firma": "ok", "archivos_verificados": 8}
}
```

### `POST /predict`

Cuerpo: un objeto con las 18 variables y, opcionalmente, `estacion_id` (entero ≥ 1)
y `fecha` (texto `AAAA-MM-DD`), que no entran al modelo y se devuelven en la respuesta.

```json
{"viajes_esperados": 122.55957713166208, "estacion_id": 21, "fecha": "2026-09-15",
 "modelo": {"nombre": "mibici-equipo26-p1-demand", "version": "1", "run_id": "..."}}
```

### `POST /predict/batch`

Cuerpo: `{"registros": [...]}` con 1 a 500 objetos iguales a los de `/predict`. Si un
registro es inválido, se rechaza todo el lote con 422.

```json
{"n": 3,
 "predicciones": [{"indice": 0, "viajes_esperados": 122.55957713166208,
                   "estacion_id": 21, "fecha": "2026-09-15"}, "..."],
 "modelo": {"nombre": "...", "version": "1", "run_id": "..."}}
```

Las predicciones conservan el orden de entrada (`indice`), y su número debe ser igual
al de registros.

### `GET /docs`

La documentación automática de FastAPI, con un ejemplo válido (pueden usar
`peticion-predict.json`).

## 4. Contrato de entrada

Las 18 claves son **requeridas**: deben venir siempre, aunque algunas acepten `null`.
Cualquier campo no listado se rechaza (`extra="forbid"`). Los números se validan en
modo estricto: no se aceptan textos (`"49"`), booleanos (`true`), NaN ni infinitos.
Los enteros no aceptan decimales.

Los rangos de la columna "Observado" salen del panel real de P1 (165,857 filas
estación-día, `data_version` `cd917f9aa025`).

| Variable | Tipo | ¿Admite null? | Regla | Observado |
|---|---|---|---|---|
| `salidas_lag1` | número | No | ≥ 0, entero | 0 a 561 |
| `llegadas_lag1` | número | No | ≥ 0, entero | 0 a 568 |
| `saldo_lag1` | número | No | entero, puede ser negativo | −137 a 170 |
| `salidas_lag7` | número | Sí | ≥ 0, entero | 2,826 nulos |
| `salidas_media7` | número | No | ≥ 0 | |
| `salidas_media28` | número | No | ≥ 0 | |
| `llegadas_media7` | número | No | ≥ 0 | |
| `llegadas_media28` | número | No | ≥ 0 | |
| `salidas_mismo_dia_media4` | número | Sí | ≥ 0 | 2,826 nulos |
| `dias_historia` | entero | No | ≥ 1 | 1 a 425 |
| `dias_desde_ultimo_viaje` | entero | No | ≥ 1 | 1 a 133 |
| `dia_semana` | entero | No | 0 (lunes) a 6 (domingo) | |
| `es_fin_semana` | entero | No | 0 o 1 | |
| `mes` | entero | No | 1 a 12 | |
| `dia_mes` | entero | No | 1 a 31 | |
| `capacidad` | número | Sí | > 0 y ≤ 100 | 11 a 39; 225 nulos |
| `latitud` | número | Sí | 20.55 a 20.80 | 20.636 a 20.740; 225 nulos |
| `longitud` | número | Sí | −103.50 a −103.20 | −103.433 a −103.297; 225 nulos |

Reglas cruzadas (todas se cumplen en las 165,857 filas):

1. `es_fin_semana` vale 1 exactamente cuando `dia_semana` ≥ 5.
2. `dia_mes` existe en `mes` (el 29 de febrero se acepta porque no hay año).
3. Si viene `fecha`, su mes, día y día de la semana coinciden con `mes`, `dia_mes` y
   `dia_semana`.
4. `saldo_lag1 = salidas_lag1 − llegadas_lag1`.
5. `dias_desde_ultimo_viaje ≤ dias_historia`.
6. `salidas_lag7` y `salidas_mismo_dia_media4` son nulas juntas o ninguna lo es.
7. `capacidad`, `latitud` y `longitud` son nulas juntas o ninguna lo es.

Cualquier violación responde **422**. Ojo: el manejador 422 por defecto de FastAPI
falla al serializar un NaN recibido y termina en 500; una petición con NaN debe
responder 422.

## 5. Llamada al modelo y salida

- Se arma un DataFrame con las 18 columnas en el orden de la firma, en `float64`, con
  `null` convertido a NaN.
- Antes de responder se valida la salida: misma longitud que la entrada, valores
  finitos y no negativos. Si no se cumple, se responde **500** con un mensaje
  controlado, sin truncar ni corregir valores.

## 6. Dependencias

`api/requirements.in` con las dependencias directas del servicio (FastAPI, Uvicorn,
Pydantic, `mlflow==3.17.2`, scikit-learn, skops, pandas, numpy, PyYAML).
`api/requirements.txt` se genera con las mismas versiones del repositorio:

```bash
uv pip compile api/requirements.in -c constraints/base.txt \
  -o api/requirements.txt --python-version 3.13 --universal
```

No modifiquen los metadatos ni el `requirements.txt` del modelo exportado. Al
arrancar, MLflow avisará que el modelo declara `mlflow==3.17.1`; es esperado.

## 7. Pruebas requeridas

Con un modelo falso (no requieren MLflow ni datos):

- `/health` con el linaje.
- Petición válida (200), petición con nulos admitidos (200) y lote válido con orden
  y longitud correctos.
- Lista de 422: clave requerida ausente, `null` en un campo no nulo, fuera de rango,
  campo desconocido, texto o booleano en lugar de número, decimal en un entero,
  NaN e infinito, y cada una de las siete reglas cruzadas.
- Lote vacío y de 501 registros (422).
- Salida negativa, NaN, infinita o de longitud distinta (500 controlado).
- Hashes alterados, archivo sobrante y firma incompatible impiden arrancar.
- Las 18 variables definidas en `api/` coinciden con `mibici.features.VARIABLES_P1`
  (la prueba importa `mibici`; el código de `api/` no).

Con el modelo real (que se omitan si no está definida una variable, por ejemplo
`MIBICI_MODELO_EXPORTADO`): las respuestas de la API son idénticas a
`prediccion-referencia.json` y a `mlflow.pyfunc.load_model(models/p1)`, incluidos los
casos con nulos.

`ruff check` y `ruff format --check` deben pasar (el `pyproject.toml` ya incluye `api/`).

## 8. Criterios de aceptación

1. `python -m pytest tests` pasa, sin romper las pruebas existentes.
2. `bash scripts/probar_contenedor.sh` termina con código de salida 0 y la línea
   `ACEPTACIÓN: OK` (en Windows, los pasos 3 a 6 del README a mano, con `curl.exe`).
3. La sección del README sobre la API documenta el contrato y un ejemplo **real** de
   petición y respuesta obtenido desde el contenedor, y el estado "pendiente" se
   actualiza con el resultado.
4. `docs/evidencia-api/` contiene `contenedor-docker.txt`, el log del build y
   `captura-contenedor.png` (con `docker ps` y la respuesta de `/predict` visibles).

## 9. Reparto sugerido

Corte interno: domingo 8 de noviembre de 2026, 18:00 (hora de Ciudad de México).

| Tarea | Contenido |
|---|---|
| A | Contrato de entrada y salida (sección 4 y salida de la sección 5) y sus pruebas de 422 |
| B | `main.py`, carga y verificación del modelo, endpoints (secciones 2, 3 y 5), dependencias (sección 6) y pruebas de arranque y de 500 |
| C | Integración modelo real → API → contenedor, ejecución de `scripts/probar_contenedor.sh`, captura, sección del README con ejemplo real (sección 8) y revisión independiente del exportador |

Las pruebas de `scripts/exportar_modelo.py` forman parte de la entrega del modelo;
la revisión independiente de C complementa esas pruebas.

Cada quien trabaja en su rama, abre su PR hacia `main` y revisa el de otro
integrante, según el flujo de Git del README.
