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
└── mibici             <- Código del proyecto
    ├── config.py      <- Rutas, fechas, semilla y MLflow en un solo lugar
    ├── dataset.py     <- Ingesta y versionado de datos
    ├── features.py    <- Variables por pregunta
    ├── plots.py       <- Visualizaciones
    └── modeling
        ├── train.py   <- Entrenamiento y registro en MLflow
        └── predict.py <- Inferencia
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

Estas pruebas comprueban la configuración temporal y el nombre del experimento.
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
| 2. Experimento base | `python -m mibici.modeling.train` | Construye variables y registra los runs en MLflow (`--pregunta P2` para una sola) |
| 3. Interfaz de MLflow | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Muestra runs, parámetros, métricas y artefactos en `http://127.0.0.1:5000` |
| Pruebas | `python -m pytest tests` | Partición temporal, rezagos sin fuga e indicador de riesgo |

La ingesta reutiliza los CSV que ya estén en `data/raw` y solo consulta el
portal si falta alguno. Si cambian los manifiestos, súbanlos en un commit antes
de correr el experimento: cada run registra el commit y marca `git_dirty=true`
si había cambios de código sin commit.

## Experimentos en MLflow

- Experimento único: `mibici-equipo26`. Todos los runs pasan por
`mibici/tracking.py`, que registra los mismos parámetros y etiquetas.
- Un run por pregunta con nombre descriptivo, nunca el nombre aleatorio.
- Parámetros: rango de fechas, versión de los datos, semilla, modelo e
  hiperparámetros.
- Etiquetas: equipo, pregunta (P1, P2 o P3) y commit de Git.
- Métricas con el conjunto en el nombre, por ejemplo `test_mae`.
- Sin credenciales ni datos personales en parámetros, etiquetas o artefactos.

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

Roles y responsabilidades propuestos:

| Rol | Responsabilidades |
|---|---|
| I1 · Infraestructura e integración | Estructura, configuración, dependencias, arquitectura, README y validación |
| I2 · Datos y P1 | `dataset.py`, manifiesto, features y baseline de P1 |
| I3 · P2 | Indicador de riesgo, features y baseline de P2 |
| I4 · P3 | Features, agrupación y visualizaciones de P3 |

El equipo acordará la asignación de estos roles y actualizará este apartado
mediante un Pull Request.
