# Proyecto MiBici Equipo 26

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador de TC5061 Operaciones de Aprendizaje Automático. Usamos los
datos abiertos de MiBici, el sistema de bicicleta pública del Área Metropolitana
de Guadalajara, para estimar la demanda por estación, aproximar el riesgo de
quedarse sin bicicletas o sin anclajes y agrupar estaciones con patrones de uso
parecidos.

## Preguntas del proyecto

| # | Pregunta | Tarea | Usuario |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en cada estación el día siguiente? | Regresión | Operador del sistema, para planear la flota. |
| P2 | ¿Qué estaciones están en riesgo de quedarse sin bicicletas o sin anclajes el día siguiente? | Clasificación | Equipo de rebalanceo, para decidir qué estaciones revisar primero. |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión del sistema. |

Los datos no incluyen el inventario por estación. El riesgo de P2 se aproxima con
el saldo de salidas y llegadas y la capacidad de cada estación. La formulación
completa está en el [ML Canvas](references/ML_Canvas_MiBici.pdf) y la arquitectura en
[references/arquitectura.md](references/arquitectura.md).

## Datos

- **Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/):
  viajes mensuales (`datos_abiertos_AAAA_MM.csv`) y nomenclatura de estaciones
  (`nomenclatura_2026_09.csv`).
- **Periodo:** agosto de 2025 a septiembre de 2026.
- **Partición temporal:** entrenamiento de agosto de 2025 a julio de 2026,
  validación en agosto de 2026 y prueba en septiembre de 2026. Las fechas viven
  en `mibici/config.py`. Las variables solo usan información disponible antes de
  cada pronóstico.
- **Datos personales:** no se usan `Usuario_Id`, `Genero` ni `Año_de_nacimiento`
  como variables, ni se registran en MLflow.

- **Tamaño:** 14 meses, 5,248,635 viajes después de limpieza y entre 367 y 468
estaciones activas por mes (`reports/mibici_resumen_mensual.csv`).

Los datos no se versionan en Git. `mlflow run . -e ingest` los reconstruye y
actualiza dos manifiestos versionados en `manifests/`:

- `mibici_downloads.json`: URL, encoding, columnas, registros y SHA-256 de cada
CSV original, incluida la nomenclatura.
- `mibici_processed.json`: registros limpios, duplicados e inválidos eliminados,
SHA-256 de cada Parquet y el `data_version`, un identificador de 12 caracteres
derivado de los hashes de los CSV originales. Ese valor se registra en cada run
de MLflow; si MiBici modifica un archivo, cambia la versión.

## Estructura del repositorio

```
├── MLproject          <- MLflow Project: puntos de entrada ingest, train y search
├── python_env.yaml    <- Entorno del MLflow Project (Python + requirements.txt)
├── Makefile           <- Comandos de conveniencia (make requirements, make lint)
├── README.md
├── pyproject.toml     <- Metadatos del paquete y configuración de ruff
├── requirements.in    <- Dependencias directas (se editan aquí)
├── requirements.txt   <- Versiones exactas de todas las dependencias
├── .python-version    <- Versión de Python del proyecto (3.13)
├── data               <- No se versiona
│   ├── external       <- Datos de terceros
│   ├── interim        <- Datos intermedios
│   ├── processed      <- Datos limpios en Parquet
│   └── raw            <- CSV originales de MiBici
├── docs
├── models             <- Modelos serializados (no se versionan)
├── notebooks          <- Solo exploración: 1.0-iniciales-descripcion.ipynb
├── references         <- ML Canvas, arquitectura y opciones de ccds
├── reports
│   └── figures        <- Gráficas y mapas
├── tests              <- Pruebas con pytest
└── mibici                 <- Código del proyecto
    ├── config.py          <- Rutas, fechas, semilla, umbrales y MLflow en un solo lugar
    ├── dataset.py         <- Ingesta, versionado y lectura verificada contra el manifiesto
    ├── preprocess.py      <- Normalización de CSV a Parquet
    ├── features.py        <- Variables de P1, P2 y P3 (panel estación-día y perfiles de uso)
    ├── evaluation.py      <- Comparación y evaluación de P3 (selección de k, validación, reproducción)
    ├── plots.py           <- Gráficas y mapa de P3
    ├── tracking.py        <- Registro común de runs y artefactos de candidatos en MLflow
    ├── modeling
    │   ├── train.py       <- Punto de entrada `train`: un modelo de una pregunta
    │   ├── search.py      <- Punto de entrada `search`: búsqueda con nested runs
    │   ├── comparar.py    <- Tabla comparativa por pregunta desde MLflow (reporte)
    │   ├── p1.py          <- Demanda (P1): baseline, HGB Poisson y GLM Poisson
    │   ├── p2.py          <- Riesgo (P2): baseline, regresión logística y HGB
    │   ├── p3.py          <- KMeans de perfiles de estaciones (P3) y auditoría congelada
    │   └── predict.py     <- Inferencia (plantilla)
```

La estructura se generó con Cookiecutter Data Science v2.3.0 (`ccds`, con
`include_code_scaffold` en Yes). Las opciones usadas están en
[references/ccds-options.json](references/ccds-options.json).

## Instalación

Requisitos: Git, Python 3.13 y acceso al repositorio del equipo.
Usamos [uv](https://docs.astral.sh/uv/) para crear el entorno y resolver las
dependencias. Las versiones están fijadas en `requirements.txt`; también se
pueden instalar con pip. Esta modalidad utiliza `requirements.in` y
`requirements.txt`, sin `uv.lock` ni `uv sync`.

```bash
git clone https://github.com/MNA-MLOps-Grupo26/TC5061-Proyecto-Equipo26.git
cd TC5061-Proyecto-Equipo26
```

Con uv (flujo del equipo):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

En Windows, sustituye la activación por `.venv\Scripts\Activate.ps1` en
PowerShell o `.venv\Scripts\activate.bat` en CMD. Los ejemplos con `source`
corresponden a macOS/Linux.

Alternativa con pip (requiere Python 3.13 instalado):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, crea el entorno con `py -3.13 -m venv .venv` y actívalo como arriba.

Si tienes Make y uv, puedes usar `make create_environment`, activar `.venv`
y ejecutar `make requirements`.

### Verificar la instalación

Desde la raíz del repositorio, con `.venv` activo:

```bash
python -m pytest tests
```

Las pruebas cubren la normalización de datos, la partición temporal, los rezagos
sin fuga de información, la etiqueta de riesgo de P2, las métricas de P1 y la
preparación y reproducción de P3. No necesitan los datos de MiBici.
La revisión de estilo es opcional:

```bash
ruff check
ruff format --check
```

### Agregar o quitar una dependencia

1. Agreguen o quiten el paquete en `requirements.in` (sin versión).
2. Regeneren `requirements.txt` con las versiones exactas:

   ```bash
   uv pip compile requirements.in -o requirements.txt --python-version 3.13
   ```

   Sin uv, el equivalente es `pip-compile` de pip-tools.
3. Instalen de nuevo: `uv pip install -r requirements.txt` (o `python -m pip install -r requirements.txt`).
4. Suban `requirements.in` y `requirements.txt` en el mismo commit, por ejemplo
   `build(deps): add pyarrow for Parquet output`.

Reglas:

- No editen `requirements.txt` a mano; siempre se genera desde `requirements.in`.
- Después de hacer pull, vuelvan a correr `uv pip install -r requirements.txt` si
  cambió.
- Si hay conflicto en `requirements.txt`, resuelvan `requirements.in` y vuelvan
  a generarlo.

## Comandos del proyecto

Todas las corridas de la entrega se lanzan desde el MLflow Project (`MLproject`),
no desde notebooks ni scripts sueltos. El entorno está en `python_env.yaml`
(Python 3.13.16 y las versiones fijas de `requirements.txt`).

| Punto de entrada | Qué hace | Parámetros |
|---|---|---|
| `ingest` | Descarga, verifica y normaliza los datos; genera manifiestos con `data_version` | Ninguno |
| `train` | Entrena un modelo (o baseline) de una pregunta | `question`, `model`, `learning_rate`, `max_iter` |
| `search` | Búsqueda de hiperparámetros con nested runs (un padre y un hijo por combinación) | `question`, `model` |

### 1. Servidor de tracking (terminal 1)

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

`mlflow.db` y `mlartifacts/` están en `.gitignore`. En macOS, si el puerto 5000
está ocupado (AirPlay Receiver), usen `--port 5001` y ajusten la URI.

### 2. Reproducir desde un clon limpio (terminal 2)

```bash
git clone https://github.com/MNA-MLOps-Grupo26/TC5061-Proyecto-Equipo26.git
cd TC5061-Proyecto-Equipo26
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
RUN="mlflow run . --env-manager uv --experiment-name mibici-equipo26"

# Ingesta
$RUN -e ingest

# Baselines y candidatos
$RUN -e train -P question=P1 -P model=baseline
$RUN -e train -P question=P1 -P model=hgb
$RUN -e train -P question=P1 -P model=glm
$RUN -e train -P question=P2 -P model=baseline
$RUN -e train -P question=P2 -P model=logreg
$RUN -e train -P question=P2 -P model=hgb
$RUN -e train -P question=P3                      # k=3 congelado (baseline de P3)

# Búsquedas con nested runs (combinaciones en config.BUSQUEDAS)
$RUN -e search -P question=P1                     # hgb: learning_rate y max_iter
$RUN -e search -P question=P2 -P model=logreg     # C
$RUN -e search -P question=P2 -P model=hgb        # learning_rate y max_leaf_nodes
$RUN -e search -P question=P3                     # k = 2, 4, 5

# Tabla comparativa por pregunta para el reporte (solo lee MLflow)
python -m mibici.modeling.comparar
```

- `--env-manager uv` crea el entorno de `python_env.yaml` desde cero y requiere
  `uv` actualizado: se verificó con **uv 0.13.5** y **Python 3.13.16**. Para
  iterar más rápido con `.venv` activo, usen `--env-manager local`.
- `--experiment-name mibici-equipo26` es obligatorio: `mlflow run` crea el run
  del proyecto en ese experimento antes de ejecutar el código.
- Agreguen `--run-name <nombre>` para nombrar el run del proyecto; los runs de
  modelos ya tienen nombre descriptivo.

| Pregunta | `model` en `train` | Hiperparámetros por `-P` | `model` en `search` |
|---|---|---|---|
| P1 | `default` (baseline + hgb), `baseline`, `hgb`, `glm` | hgb: `learning_rate`, `max_iter`; glm: `max_iter` | `hgb` |
| P2 | `default` (baseline + logreg), `baseline`, `logreg`, `hgb` | hgb: `learning_rate`, `max_iter`; logreg: `max_iter` | `logreg`, `hgb` |
| P3 | `default` (k=3) | Ninguno | `kmeans` |

Los valores por defecto viven en `mibici/config.py` (`P1_HIPERPARAMETROS`,
`P1_GLM_HIPERPARAMETROS`, `P2_HIPERPARAMETROS`, `P2_HGB_HIPERPARAMETROS`,
`P3_HIPERPARAMETROS` y `BUSQUEDAS`).

La ingesta reutiliza los CSV que ya estén en `data/raw` y solo consulta el
portal si falta alguno (~600 MB en una computadora nueva). Si cambian los
manifiestos, súbanlos en un commit antes de entrenar: cada run registra el
commit y marca `git_dirty=true` si había cambios de código sin commit.

Pruebas y estilo: `python -m pytest tests`, `ruff check` y `ruff format --check`.

## Experimentos en MLflow

- Experimento único: `mibici-equipo26`. Todos los runs pasan por
  `mibici/tracking.py`, que registra los mismos parámetros y etiquetas.
- Jerarquía: cada `mlflow run` crea un run de proyecto (`stage_of_work=project`);
  cada modelo es su hijo. En `search`, el padre de la búsqueda cuelga del run
  de proyecto y cada combinación es hija del padre de la búsqueda.
- Parámetros: rango de fechas, `data_version`, semilla, modelo, hiperparámetros
  efectivos y `selection_metric`.
- Etiquetas: `team`, `question`, `model`, `model_family`, `stage_of_work`
  (`project`, `baseline`, `search`, `candidate`), `git_commit` y `git_dirty`.
  Ejemplo de filtro en la interfaz: `tags.question = 'P2' and tags.stage_of_work != 'project'`.
- Métricas con el conjunto en el nombre (`train_`, `val_`, `test_`), más
  `fit_time_s` como costo de entrenamiento.
- Artefactos de cada candidato:

  | Carpeta | Contenido |
  |---|---|
  | `model` (Logged model) | Modelo sklearn con firma y ejemplo de entrada |
  | `data/` | Manifiestos de la ingesta; el entrenamiento se registra como input del run (`log_input`: esquema, perfil y digest, sin filas) |
  | `config/` | `run_config.json` (hiperparámetros efectivos, partición, variables) y `config.py` |
  | `plots/` | P1: real vs. predicho, residuales y serie diaria; P2: matriz de confusión y curva precisión-recall; P3: mapa y perfiles de grupos |
  | `tables/` | P1: predicciones por estación-día; P2 logística: coeficientes; P3: asignaciones, perfiles y resultados agregados. Los candidatos HGB de P2 no generan esta carpeta en la implementación actual. Sin datos personales. |

- Selección (validación elige, prueba solo se reporta):

  | Pregunta | Métrica de negocio | Apoyo |
  |---|---|---|
  | P1 | `val_wape` (menor) | `val_mae`, `val_rmse` |
  | P2 | `val_recall_at_30` (mayor) | `val_precision_at_30`, `val_f1_macro` |
  | P3 | `val_ari_vs_train` (mayor) | `val_fraccion_estable`, `train_silhouette`, interpretabilidad |

  `python -m mibici.modeling.comparar` arma esta comparación con la mejora
  respecto al baseline, la brecha validación-prueba (estabilidad) y
  `fit_time_s` (costo). La metodología y la decisión están en
  [references/BUSQUEDA_CANDIDATOS.md](references/BUSQUEDA_CANDIDATOS.md).
- Sin credenciales ni datos personales en parámetros, etiquetas o artefactos.
- P3 fija su configuración en `references/p3_frozen.json` (elegida sin ver
  prueba). El run agrega la etiqueta `reproduce_configuracion_congelada=true`
  cuando sus métricas coinciden con las registradas al congelarla.
- La secuencia con la que se eligió esa configuración (revisión, comparación de
  k, sensibilidad a la historia y validación en agosto) está en
  `notebooks/2.0-fvep-p3-seleccion.ipynb`, y la auditoría de la evaluación
  congelada en `notebooks/3.0-fvep-p3-auditoria.ipynb`. El protocolo está en
  [references/P3_REPRODUCIBILIDAD.md](references/P3_REPRODUCIBILIDAD.md).

**Limitación conocida:** el hash de `data/processed/estaciones.parquet` puede
variar entre sistemas operativos aunque el contenido sea el mismo. Si la ingesta
modifica ese valor en `manifests/mibici_processed.json`, el run queda con
`git_dirty=true`. `data_version` no cambia, porque se calcula con los CSV
originales.

## Flujo de trabajo con Git

Seguimos [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. `main` está protegida: los cambios posteriores al commit inicial entran mediante PR.
2. Cada cambio se hace en una rama nueva creada desde `main`.
3. Al terminar se abre un Pull Request hacia `main`.
4. Otro integrante lo revisa y lo aprueba antes de hacer merge.

La protección requiere al menos una aprobación, descarta aprobaciones anteriores
cuando se agregan commits y exige resolver las conversaciones. También se aplica
a administradores; no se permiten force pushes ni eliminar `main`.

Para comenzar un cambio:

```bash
git switch main
git pull --ff-only origin main
git switch -c chore/base-i1
```

Sustituyan `chore/base-i1` por el nombre de su propia rama. Después de guardar
sus commits, suban esa rama con `git push -u origin <nombre-de-rama>` y abran el
PR hacia `main`. Conserven los enlaces a PR y revisiones como evidencia de la
contribución de cada integrante.

**Ramas:** `tipo/descripcion-corta`, por ejemplo `feat/ingesta-parquet`,
`fix/codificacion-enero`, `docs/readme-instalacion`.

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/)
con mensajes en inglés y el formato `type(scope): description`; el alcance es
opcional. Tipos: `feat`, `fix`, `docs`,
`refactor`, `test`, `build`, `chore`. Ejemplo:
`feat(dataset): normalize CSV encoding`.

Cada integrante abre al menos un Pull Request y revisa al menos uno de otro
integrante.

## Buenas prácticas

- **Configuración en un solo lugar:** rutas, fechas y semillas solo en
  `mibici/config.py`.
- **Notebooks solo para exploración**, con el formato `1.0-iniciales-descripcion.ipynb`;
  la lógica final vive en el módulo `mibici`.
- **Dependencias fijas:** `requirements.txt` permite reconstruir el mismo entorno
  desde un clon limpio.
- **Nada de datos en Git:** el `.gitignore` excluye `data/`, CSV, Parquet, `mlruns/`,
  `mlartifacts/`, `mlflow.db`, modelos serializados y archivos `.env`.

## Integrantes y roles

Integrantes del equipo 26:

- Francisco Vicente Eugenio Ponce
- Dan Michael García Rios
- Kimberly Ruiz
- Eduardo Daniel Sauza Machado

| Integrante | Rol | Contribuciones principales |
|---|---|---|
| Francisco Vicente Eugenio Ponce | Infraestructura e integración | Estructura con ccds, configuración, dependencias, arquitectura, README, protección de `main`, revisión e integración de PRs, ejecución desde un clon limpio |
| Eduardo Daniel Sauza Machado | Datos y exploración | Notebook de exploración, descarga y normalización inicial, manifiesto procesado, partición temporal y utilidades de features |
| Dan Michael García Rios | Ingesta, tracking y P2 | Ingesta en un comando con nomenclatura, helper común de MLflow, panel estación-día sin fuga, indicador de riesgo y modelos de P2 |
| Kimberly Ruiz | P1 y P3 | Baseline de P1 con evidencias, integración de P1 y P2, perfiles de estaciones, agrupación, mapa y reproducibilidad de P3 |

## Evidencia integrada publicada

[Verificación final del 24 de octubre de 2026](docs/evidence/2026-10-24-final/README.md): exportación compacta de corridas locales, métricas, linaje y reproducción. Incluye el registro local verificado de los tres modelos seleccionados; no constituye un servidor MLflow ni un Model Registry compartido.

El alias `champion` identifica cada versión seleccionada y permite cargarla con
`models:/<nombre>@champion` sin fijar el número de versión en el consumidor.
Los tres aliases actuales apuntan a la versión 1 del registro local; no implican
un despliegue ni un servicio compartido. La versión y el run conservan el linaje
auditable. Véase la [justificación del ciclo de vida](docs/evidence/2026-10-24-final/REGISTRATION.md).
