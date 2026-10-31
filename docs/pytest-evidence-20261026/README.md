# Evidencia PyTest — 26 de octubre de 2026

## Estado actual para revisión

**Verificación después del formato: 83 passed en 25.53 s, salida 0**, sin fallos, omisiones ni advertencias de pytest; proceso completo: 29.5014993 s. La corrida anterior de 20.81 s y todos sus logs se conservan y se identifican por separado más abajo.

Se agregaron 18 casos con datos sintéticos para poder calcular de antemano las respuestas esperadas. Se preservaron los 65 casos existentes. Estos ejemplos comprueban comportamiento y casos límite; no son una nueva evaluación de precisión de los modelos con datos reales ni una demostración exhaustiva de corrección.

Después se ejecutó `.\.venv\Scripts\ruff.exe format --no-cache mibici/dataset.py mibici/features.py`. El diff revisado contiene solo distribución de líneas, paréntesis y líneas en blanco: `dataset.py` +6/-16, `features.py` +16/-33 (22 inserciones y 49 eliminaciones en total). El árbol de sintaxis abstracta de cada archivo es idéntico al de la fuente `9faa91b3810fd1492a0f83bd987915fe74712e0a`, ignorando posiciones de líneas. El hash del nuevo archivo de pruebas tampoco cambió. No se alteraron algoritmos, fechas, modelos, hiperparámetros ni evidencia congelada.

| Verificación posterior al formato | Resultado | Evidencia |
|---|---|---|
| `.\.venv\Scripts\python.exe -m pytest tests -v --durations=10` | 83 passed en 25.53 s; salida 0 | [cleanup-suite.txt](published/cleanup-suite.txt) |
| `.\.venv\Scripts\ruff.exe check --no-cache` | Aprobado; salida 0 | [cleanup-ruff-check.txt](published/cleanup-ruff-check.txt) |
| `.\.venv\Scripts\ruff.exe format --check --no-cache` | 16 archivos ya formateados; salida 0 | [cleanup-ruff-format.txt](published/cleanup-ruff-format.txt) |
| Comparación AST contra la fuente | Iguales en ambos archivos; salida 0 | [cleanup-ast.txt](published/cleanup-ast.txt) |
| `git diff --check` | Aprobado; salida 0 | [cleanup-diff-check.txt](published/cleanup-diff-check.txt) |
| Aplicación de formato a los dos archivos | 2 archivos reformateados; salida 0 | [cleanup-formatter.txt](published/cleanup-formatter.txt) |

Se usaron el mismo intérprete, versiones y variables de entorno que se detallan abajo. Las verificaciones de esta tabla corresponden al código final de las pruebas y al formato actual; la preparación de publicación solo actualiza documentación y copias de logs.

Las copias de `published/` proceden de la salida real guardada en cada corrida. Únicamente se sustituyó la ruta absoluta personal por `<PROJECT_ROOT>` y se normalizaron saltos de línea para compartirlas. Se inspeccionaron para evitar publicar rutas personales o credenciales. Los logs originales y los estados de Git con archivos ajenos permanecen locales, sin incluirse en el PR. No se capturaron screenshots. Los archivos nuevos y el formato estaban sin commit durante las pruebas; la fuente base figura arriba y el commit de publicación queda en el historial de Git.

El alcance para revisión comprende pruebas, esta evidencia y el formato de los dos archivos. DeepChecks, seguridad e integración del equipo quedan fuera. No se afirma que toda la rúbrica esté completa. No hay fallos pendientes en los chequeos ejecutados; las limitaciones de cobertura de la matriz siguen vigentes.

## Historial: etapa inicial y cobertura

**Corrida inicial: 83 pruebas aprobadas en 20.81 s; salida 0.** Son 65 casos existentes y 18 casos nuevos (incluidas parametrizaciones). No hubo fallos, omisiones ni advertencias de pytest. Tiempo total del proceso: 23.9710252 s. Evidencia textual real: [suite.txt](published/suite.txt). No se capturaron screenshots.

## Etapa inicial de PyTest: qué cambió y cómo leerlo

Se agregó solamente `tests/test_casos_limite.py`: ejemplos pequeños, datos sintéticos, aserciones directas y comentarios que explican el riesgo. En esta etapa inicial no se cambió código de producción, modelos, fechas, configuración ni pruebas existentes. Más adelante se aplicó solamente formato a dos archivos, como se explica en el estado actual. No se instalaron dependencias. Se revisaron los archivos existentes antes de decidir dónde agregar casos; P3 ya tenía cobertura amplia y no se duplicó su prueba de reproducibilidad.

Pasar estas pruebas significa que las funciones producen los resultados calculados en ejemplos controlados. Por ejemplo, un cero válido del baseline no se reemplaza por otra media; sin volumen, WAPE devuelve NaN (no un cero engañoso); los rezagos no mezclan estaciones ni usan viajes del mismo día; top-K resuelve empates por orden de entrada. No demuestra que el modelo tenga buena precisión con datos reales.

## Matriz de cobertura basada en riesgos

| Área / funciones clave | Pruebas existentes conservadas | Aporte nuevo | Límite pendiente |
|---|---|---|---|
| `particionar_dataframe`, `validar_columnas_viajes`, `agregar_variables_temporales` | `test_data.py`: fechas de corte, exclusión del fin, columnas y calendario | Partición vacía, columna ausente, fecha inválida | Zonas horarias y todas las formas de entrada inválida |
| `asignar_dia`, `contar_por_estacion_dia`, `construir_panel`, `agregar_rezagos` | `test_features.py`: corte a las 05:00, ceros, rezago sin futuro | Dos estaciones intercaladas; lag1/lag7, medias, historia previa, entrada intacta; perturbación del presente/futuro | Grandes volúmenes, fechas duplicadas y panel no diario |
| `dataset_p1`, `dataset_p2` (integración de preparación) | Componentes compartidos probados por separado | Primer día excluido; validación/test; un viaje del día objetivo cambia etiqueta P2 pero no variables históricas de P1/P2 | No recorre el flujo completo con datos reales |
| P1 `calcular_metricas`, `predecir_baseline` | `test_p1.py`: predicción perfecta, error conocido, WAPE ponderado | Volumen cero con/sin error; prioridad semanal → media7 → cero; respeta cero válido, índice y entrada | No prueba ajuste/predicción de candidatos HGB/GLM |
| P2 `calcular_indicador_riesgo` | `test_features.py`: etiquetas de salidas/llegadas | Debajo/en umbral; capacidad desconocida excluida; primer cruce conservado aun al cruzar después el opuesto | Capacidad cero/negativa, ocupación no predeterminada, eventos simultáneos |
| P2 `metricas_top_k`, `calcular_metricas` | No había pruebas directas de estas métricas | Media diaria; día sin riesgo; K mayor que población; techo de recall; empates con índice no ordenado; clasificación perfecta en tres clases | Contrato para K no positivo y puntajes ausentes/invalidos |
| P3 `preparar_perfiles_p3`, `perfiles_agosto`, `matriz_forma`, `cohorte_historia` | `test_features_p3.py`, `test_validate_p3.py`, `test_compare_p3.py`, `test_history_p3.py`, `test_p3.py`: cortes, cobertura, normalización, probabilidades inválidas, metadatos, cohortes, septiembre | Se conserva esa cobertura sin duplicarla | No demuestra representatividad de datos reales |
| P3 `comparar`, `asignar_fijo`, `resumir_fijo`, `verificar_reproduccion` | Comparación repetida con métricas/asignaciones idénticas; centroides sin refit; varios k; rechazo de evidencia alterada | `comparar` rechaza seis estaciones (requiere más que el mayor k) | `comparar` es integración, no una prueba unitaria de una función aritmética |
| Entrada CLI, búsqueda, catálogo y reproducción | `test_train.py`, `test_search.py`, `test_modeling_p3.py`, `test_p3.py`: rechazo antes de cargar datos, checksums, asignaciones/metricas congeladas, mapa | Sin cambios | Descargas, servicios externos y registro MLflow quedan fuera |

Esta matriz no es un porcentaje de cobertura de líneas: no se instaló ni ejecutó un complemento de cobertura. El mínimo de tres funciones con casos normales y límites está respaldado por las filas anteriores. La calidad se evalúa por riesgos cubiertos, no por el número de pruebas; quedan límites explícitos y no se afirma cobertura exhaustiva de todas las funciones.

## Reproducibilidad con semilla fija

La prueba existente `tests/test_compare_p3.py::test_comparacion_acotada_y_representantes_miembros` genera 18 perfiles sintéticos con `np.random.default_rng(17)` y llama dos veces a `comparar`. KMeans usa `RANDOM_SEED=42` de `mibici/config.py`; Ward es determinista. La prueba exige igualdad exacta de la tabla de métricas y de todas las asignaciones de cluster, además de comprobar representantes y silhouette. Pasó en esta suite. Para P3 las asignaciones son las predicciones del agrupamiento. Es una comprobación de integración reproducible dentro del mismo entorno, no garantía entre todas las versiones/plataformas. Los ajustes sintéticos pertenecen a las pruebas existentes; no se lanzaron entrenamientos de proyecto ni runs de MLflow.

## Entorno y comandos exactos

- Checkout local del proyecto (ruta personal omitida en la publicación).
- Fuente: `9faa91b3810fd1492a0f83bd987915fe74712e0a`.
- Se inició en `main`, igual a la referencia local `origin/main`, sin cambios rastreados y con reportes/referencias sin seguimiento. No se hizo fetch.
- Rama nueva: `feat/pytest-casos-limite`; no existía antes. Al cerrar la corrida inicial aún no había stage, commit, push ni PR; posteriormente se autorizó preparar la publicación para revisión.
- Intérprete: `.venv\Scripts\python.exe`.
- Python 3.13.16; pytest 9.1.1; numpy 2.5.4; pandas 3.0.6; scikit-learn 1.9.1; scipy 1.18.1; mlflow 3.17.1; pyarrow 25.0.1; ruff 0.17.1. Salida real: [environment.txt](published/environment.txt).
- Árbol durante la ejecución: sucio por archivos nuevos sin seguimiento; fuente rastreada intacta. Se preservaron todas las referencias y reportes previos. Estado de esa corrida: `git-status.txt` (se conserva solo local; incluye nombres de archivos ajenos al cambio).
- No se encontraron AGENTS.md aplicables en las ubicaciones inspeccionadas ni `.agents` del proyecto. No se requirieron skills específicos para editar estas pruebas Python.

Preparación de entorno **solo del proceso PowerShell**, para evitar cachés y limitar paralelismo numérico; no modifica archivos de configuración:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_ADDOPTS = '-p no:cacheprovider'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

Comando único para ejecutar toda la suite usando el entorno existente:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v --durations=10
```

El README y Makefile ya documentan `python -m pytest tests` con el entorno activado; no fue necesario alterar la integración/README del equipo.

| Ejecución | Resultado real | Duración pytest / proceso | Evidencia |
|---|---|---|---|
| `python -m pytest tests/test_casos_limite.py -v` (versión final, con el intérprete arriba) | 18 passed; salida 0 | 15.72 s / 19.2051427 s | [focused-final.txt](published/focused-final.txt) |
| `python -m pytest tests -v --durations=10` (mismo intérprete) | 83 passed; salida 0 | 20.81 s / 23.9710252 s | [suite.txt](published/suite.txt) |
| `.\.venv\Scripts\ruff.exe check --no-cache` | Aprobado; salida 0 | No cronometrado | [ruff-check.txt](published/ruff-check.txt) |
| `.\.venv\Scripts\ruff.exe format --check --no-cache` | Salida 1: 2 archivos requieren formato, 14 ya formateados | No cronometrado | [ruff-format.txt](published/ruff-format.txt) |
| `.\.venv\Scripts\ruff.exe check --no-cache tests/test_casos_limite.py` | Aprobado; salida 0 | No cronometrado | [ruff-new-tests.txt](published/ruff-new-tests.txt) |
| `.\.venv\Scripts\ruff.exe format --check --no-cache tests/test_casos_limite.py` | Aprobado; salida 0 | No cronometrado | [ruff-new-tests.txt](published/ruff-new-tests.txt) |

En la corrida inicial, el formato global pendiente afectaba `mibici/dataset.py` y `mibici/features.py`, ambos idénticos al commit fuente. En esa etapa todavía no se había aplicado el formato; la limpieza posterior y su verificación están documentadas en el estado actual. El descubrimiento de Ruff en pyproject.toml se centra en mibici; por eso se revisó explícitamente también el archivo nuevo.

En desarrollo hubo una corrida inicial de 14 casos (14 passed, 37.03 s, salida 0, salida conservada en la conversación) y otra de 16 (16 passed, 19.24 s, salida 0: [focused.txt](published/focused.txt)). Las dos últimas adiciones justifican la corrida final de 18. No se debilitaron aserciones para obtener resultados verdes ni se encontraron bugs que exigieran cambios de producción.

## Evidencia anterior preservada

Antes de este alcance ampliado se ejecutó solo `python -m pytest tests/test_data.py::test_particion_temporal_sin_traslapes -v`, con el mismo intérprete y commit, en main. Resultado anterior: **1 passed in 8.32s**, salida 0, proceso 12.9974939 s. Esta es una transcripción de la evidencia ya registrada en la conversación, no una nueva ejecución ni un screenshot. Esa prueba vuelve a aparecer aprobada en la nueva suite, pero ambos resultados se mantienen separados.

## Cierre histórico de la etapa inicial

Revisar el archivo nuevo y esta carpeta de evidencia. No incluir los reportes/referencias previos sin seguimiento. En ese momento el chequeo global de formato estaba pendiente por diferencias preexistentes. Después se autorizó y completó esa limpieza, sin cambios de lógica. DeepChecks, seguridad, integración de entrega y Canvas/Drive no se tocaron. No hay bloqueadores para revisar estas pruebas; las limitaciones de cobertura anteriores siguen vigentes.
