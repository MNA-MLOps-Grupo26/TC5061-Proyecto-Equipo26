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

Los datos no se versionan en Git. `python -m mibici.dataset` los reconstruye y
actualiza dos manifiestos versionados en `manifests/`:

- `mibici_downloads.json`: URL, encoding, columnas, registros y SHA-256 de cada
CSV original, incluida la nomenclatura.
- `mibici_processed.json`: registros limpios, duplicados e inválidos eliminados,
SHA-256 de cada Parquet y el `data_version`, un identificador de 12 caracteres
derivado de los hashes de los CSV originales. Ese valor se registra en cada run
de MLflow; si MiBici modifica un archivo, cambia la versión.

## Estructura del repositorio

```
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
    ├── tracking.py        <- Registro común de runs en MLflow
    ├── modeling
    │   ├── train.py       <- Entrada única: corre P1, P2 y P3 y los registra en MLflow
    │   ├── p1.py          <- Baseline de demanda (P1)
    │   ├── p2.py          <- Baseline y regresión logística de riesgo (P2)
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

Ejecuten los comandos desde la raíz del repositorio con `.venv` activo.

| Paso | Comando | Qué hace |
|---|---|---|
| 1. Ingesta y versionado | `python -m mibici.dataset` | Descarga los 14 meses y la nomenclatura, verifica, normaliza a Parquet y genera manifiestos con `data_version` |
| 2. Experimento base | `python -m mibici.modeling.train` | Construye variables y registra los 5 runs de P1, P2 y P3 en MLflow (`--pregunta P1`, `P2` o `P3` para una sola) |
| 3. Interfaz de MLflow | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Muestra runs, parámetros, métricas y artefactos en `http://127.0.0.1:5000` |
| Pruebas | `python -m pytest tests` | Ingesta, partición, rezagos sin fuga, indicador de riesgo, métricas y P3 |

La ingesta reutiliza los CSV que ya estén en `data/raw` y solo consulta el
portal si falta alguno. En una computadora nueva la ingesta tarda unos minutos
(descarga de ~600 MB) y el experimento completo, alrededor de un minuto.
Las gráficas, tablas y el mapa de P3 se guardan en `reports/figures/` (no se
versionan) y se registran como artefactos de cada run. Si cambian los manifiestos, súbanlos en un commit antes
de correr el experimento: cada run registra el commit y marca `git_dirty=true`
si había cambios de código sin commit.

## Experimentos en MLflow

- Experimento único: `mibici-equipo26`. Todos los runs pasan por
`mibici/tracking.py`, que registra los mismos parámetros y etiquetas.
- Runs con nombre descriptivo, nunca el nombre aleatorio:

  | Pregunta | Run | Modelo |
  |---|---|---|
  | P1 | `p1-baseline-promedio-semanal` | Baseline: promedio del mismo día de la semana en las 4 semanas previas |
  | P1 | `p1-hgb-poisson` | Candidato: `HistGradientBoostingRegressor` con pérdida Poisson |
  | P2 | `p2-baseline-semana-anterior` | Baseline: categoría de riesgo de la semana anterior |
  | P2 | `p2-logreg-multiclase` | Candidato: regresión logística multiclase balanceada |
  | P3 | `p3-kmeans-k3-historia180` | KMeans k=3 sobre perfiles semanales de salidas y llegadas, con mapa |

- Parámetros: rango de fechas, versión de los datos, semilla, modelo e
  hiperparámetros.
- Etiquetas: equipo, pregunta (P1, P2 o P3) y commit de Git.
- Métricas con el conjunto en el nombre, por ejemplo `test_mae`.
- Sin credenciales ni datos personales en parámetros, etiquetas o artefactos.
- P3 fija su configuración en `references/p3_frozen.json` (elegida sin ver
  prueba). El run agrega la etiqueta `reproduce_configuracion_congelada=true`
  cuando sus métricas coinciden con las registradas al congelarla.
- La secuencia con la que se eligió esa configuración (revisión, comparación de
  k, sensibilidad a la historia y validación en agosto) está en
  `notebooks/2.0-fvep-p3-seleccion.ipynb`, y la auditoría de la evaluación
  congelada en `notebooks/3.0-fvep-p3-auditoria.ipynb`. Ambos llaman funciones
  de `dataset.py`, `features.py`, `evaluation.py`, `plots.py` y
  `modeling/p3.py`. El protocolo está en
  [references/P3_REPRODUCIBILIDAD.md](references/P3_REPRODUCIBILIDAD.md).

**Limitación conocida:** el hash de `data/processed/estaciones.parquet` puede
variar entre sistemas operativos aunque el contenido sea el mismo. Si la ingesta
modifica ese valor en `manifests/mibici_processed.json`, el run queda con
`git_dirty=true`. `data_version` no cambia, porque se calcula con los CSV
originales.


### MLflow Projects: ejecución reproducible

El proyecto utiliza **MLflow Projects** para ejecutar la ingesta y el entrenamiento mediante comandos estandarizados. El archivo `MLproject` define los puntos de entrada `ingest` y `train`, mientras que `python_env.yaml` y `requirements.txt` especifican el entorno de Python y las dependencias.

#### 1. Preparación del entorno

Desde una copia del repositorio, seguir las instrucciones de instalación descritas anteriormente en este README. Activar el entorno virtual e instalar las dependencias antes de continuar.

#### 2. Iniciar el servidor de MLflow

En una terminal, desde la raíz del repositorio:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

El servidor estará disponible en http://127.0.0.1:5000.

El archivo `mlflow.db` almacena los metadatos de seguimiento y está excluido del control de versiones mediante `.gitignore`.

#### 3. Configurar el servidor de seguimiento

En una segunda terminal, desde la raíz del repositorio:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Esta variable permite que los experimentos registren sus resultados en el servidor local.

#### 4. Ejecutar la ingesta

```bash
mlflow run . -e ingest --env-manager uv --experiment-name mibici-equipo26
```

La ingesta descarga o reutiliza los datos necesarios, realiza su procesamiento y genera los manifiestos correspondientes.

#### 5. Ejecutar los entrenamientos

**P1 — Predicción de demanda:**

```bash
mlflow run . -e train --env-manager uv --experiment-name mibici-equipo26 -P question=P1 -P model=hgb
```

**P2 — Clasificación de riesgo:**

```bash
mlflow run . -e train --env-manager uv --experiment-name mibici-equipo26 -P question=P2 -P model=logreg
```

**P3 — Agrupamiento de estaciones:**

```bash
mlflow run . -e train --env-manager uv --experiment-name mibici-equipo26 -P question=P3
```

Los hiperparámetros predeterminados se encuentran centralizados en `mibici/config.py`.

Para P1 y P2, los parámetros opcionales `learning_rate` y `max_iter` permiten modificar determinados valores desde `MLproject`. Los valores realmente utilizados se registran en MLflow como `effective_learning_rate` y `effective_max_iter`, cuando corresponda.

#### 6. Consultar resultados

Abrir http://127.0.0.1:5000 y seleccionar el experimento `mibici-equipo26`.

Cada ejecución permite consultar parámetros, métricas, etiquetas y artefactos generados durante el entrenamiento.

**Reproducibilidad:** para repetir el procedimiento desde una copia limpia del repositorio, instalar las dependencias según la sección «Instalación», iniciar el servidor MLflow, configurar `MLFLOW_TRACKING_URI`, ejecutar la ingesta y posteriormente los entrenamientos. Los datos utilizados deben corresponder a la versión indicada en los manifiestos.




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
