# Diagnóstico de P1 en Windows frente a macOS

La prueba se ejecutó inicialmente en una rama de diagnóstico y su evidencia se
integra al PR #26. No modifica el modelo compartido, los datos históricos ni
el registro MLflow del equipo. El workflow incluido se activa manualmente; no
se ejecuta automáticamente al abrir un PR ni al hacer push.

Se ejecuta `scripts/diagnosticar_entorno_p1.py` en Windows Server 2025 x64
(GitHub Actions) y en macOS ARM64 (equipo local), con Python 3.13.5, las mismas
versiones de librerías y dos hilos. Windows descarga desde cero mediante la
ingesta del proyecto. En macOS se reutilizan los archivos ya verificados de la
reproducción del 15 de noviembre, se regenera el catálogo en un directorio
temporal y se vuelven a construir las variables y entrenar P1.

El diagnóstico registra SHA-256 de entradas, Parquet y código, huellas por
columna del DataFrame, índice y orden de variables, digest de MLflow, nueve
métricas y predicciones de las peticiones de ejemplo. Las huellas de valores
redondeados son solo diagnósticas; no sustituyen las comparaciones exactas ni
modifican las entradas del modelo.

Para repetir en Windows con Python 3.13.5, desde un clon que incluya este cambio, sin datos locales:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
python -m pip install -r requirements.txt
python scripts/diagnosticar_entorno_p1.py --ingest --output reports/diagnostico-entorno/windows.json
```

El workflow configura los mismos límites de dos hilos. El JSON contiene hashes y resultados,
no filas del dataset. GitHub Actions conserva únicamente ese JSON como artefacto.

## Resultado observado

La [ejecución en Windows](https://github.com/MNA-MLOps-Grupo26/TC5061-Proyecto-Equipo26/actions/runs/43743086810)
terminó correctamente sobre el commit `4d60046e4f5a4d7ecd29d6b69c14407819c6c889`.
Resultados: [macos.json](macos.json), [windows.json](windows.json) y
[comparacion-entornos.json](comparacion-entornos.json). Los JSON conservan los valores obtenidos;
no se ajustaron datos ni métricas para hacerlos coincidir. Solo se normalizaron
los finales de línea del JSON Windows a LF para versionarlo en Git.

| Verificación | macOS ARM64 frente a Windows x64 |
|---|---|
| Python y cinco librerías principales | Versiones idénticas |
| Quince CSV originales | Todos los SHA-256 idénticos |
| `data_version` | `cd917f9aa025` en ambos |
| Catorce Parquet de viajes | SHA-256 idénticos |
| Parquet de estaciones | SHA-256 diferente |
| Filas, índice, orden de columnas, tipos y nulos | Coinciden |
| Valores del catálogo y del entrenamiento | Solo difiere la huella de `latitud` |
| Latitud redondeada a diez decimales (solo diagnóstico) | Huellas idénticas |
| Digest MLflow de macOS | `e3b4786b` |
| Digest MLflow de Windows | `10f3651e`, igual al registro histórico de octubre |
| Nueve métricas de P1 | Exactamente iguales; diferencia máxima 0.0 |
| Tres predicciones de ejemplo del modelo reentrenado | Exactamente iguales |
| Huellas de los vectores completos de predicciones | No todas coinciden; no se afirma identidad bit a bit |

Windows regeneró el catálogo con SHA-256
`bcfd7d47b881edd6159cec56c68325a805d4354417f46ab600f54eec77014eaa`,
el mismo registrado en la revisión Windows del 8 de noviembre. macOS generó
`80ef17e6566e9a1821896e9739a19fc765c8590f27660488267606ed5a56afce`,
el del manifiesto compartido.

La diferencia aparece en la representación numérica de `latitud` tras procesar
el mismo CSV. Esa columna llega al DataFrame de entrenamiento y su diferencia
de precisión cambia el digest. Se reproduce así el patrón histórico sin
cambiar la fuente de datos, el código funcional ni las versiones de librerías.
La comparación descarta diferencias de índice, tipos u orden de columnas en
estas dos ejecuciones. La etapa exacta de conversión decimal y el efecto
individual del sistema operativo frente a la arquitectura no se aislaron.

Los SHA-256 de los archivos de código difieren por finales de línea: convertir
los LF de la copia macOS a CRLF reproduce exactamente los seis hashes observados
en Windows. La función de cálculo del digest de MLflow tiene el mismo hash.

## Alcance de la conclusión

Este experimento cambia tanto el sistema operativo como la arquitectura:
permite detectar diferencias entre esos dos entornos, pero no atribuirlas solo
a Windows. Tampoco reconstruye la computadora de Kim ni el entorno histórico
de octubre. Reproducir su digest en Windows respalda una explicación de precisión
numérica entre entornos, pero no demuestra identidad binaria del modelo original.
La prueba de predicciones usa modelos recién entrenados; no reproduce la
diferencia del último decimal al cargar el mismo artefacto exportado en la API.
Aunque las métricas agregadas coinciden, los hashes de los vectores completos
de predicciones muestran diferencias; esta prueba no cuantifica su magnitud
por fila y no demuestra que todas las predicciones sean exactamente iguales.

El redondeo a diez decimales se usa solo para localizar la diferencia: no se
aplicó al entrenamiento ni se propone actualizar hashes para ocultarla. Adoptar
una conversión decimal uniforme requeriría un cambio separado, validación en
ambos entornos y una decisión del equipo sobre los artefactos derivados.
