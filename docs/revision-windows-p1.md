# Revisión local de P1 en Windows (Actividad 8)

Revisión del 8 de noviembre de 2026 sobre el PR #19, apilado sobre el #18,
commit `fba69f199ba9ac46d917883254df35062aeab476`. Incluye el ajuste local de
comparación numérica, las pruebas de la API y una nueva corrida de DeepChecks.
Se propone como documentación para revisión del equipo, sobre la rama del PR #19.

## 1. Resultados

| Verificación | Resultado |
|---|---|
| Suite del proyecto en Windows, con modelo real | 178 pasaron; 30 omitidas |
| Pruebas del script Bash, en Linux sin permisos de root | 29 pasaron |
| DeepChecks con datos sintéticos, en su entorno aislado | 19 pasaron |
| DeepChecks con datos reales, nueva corrida | 51 pasaron; 0 fallas bloqueantes; 0 avisos |
| Ruff | Sin errores; 49 archivos con formato correcto |
| Contenedor con modelo real | Construcción y aceptación correctas |

Las 30 omisiones de Windows corresponden a las 29 pruebas que requieren Bash y
al módulo de DeepChecks. Se ejecutaron por separado en sus entornos compatibles;
no se cuentan las omisiones como pruebas aprobadas. Las 11 pruebas de seguridad
pasaron y forman parte de la suite, no son 11 pruebas adicionales.

El contenedor respondió 200 en `/health`, `/predict` y `/predict/batch`, y 422
ante una petición incoherente y otra sin campos requeridos. También se comprobó
el usuario no root y que el servicio no puede escribir en el código ni el modelo.

## 2. Ajuste de comparación probado localmente

En x86, la petición de referencia devuelve `122.55957713166207`; la referencia
calculada en ARM contiene `122.55957713166208`. La comparación exacta del script
marcaba como distinta esa diferencia de redondeo.

En la copia local, `scripts/probar_contenedor.sh` usa `math.isclose(..., rel_tol=1e-12)`, igual
que `tests/test_api_modelo_real.py`, en la predicción individual, el lote y la
predicción final para la captura. Se siguen rechazando NaN e infinito. El tamaño
y el orden del lote se comparan de forma exacta.

**El ajuste de código y sus nuevos tests no están incluidos en este PR de documentación.**
Los resultados siguientes corresponden a esa copia local modificada.

No se cambió el modelo, la respuesta de la API ni la referencia para hacer pasar
la prueba. Los casos de regresión comprueban que el redondeo pasa y que una
predicción alterada en una unidad, un lote incompleto o una salida no finita
siguen fallando. El script completo terminó con aceptación correcta en Windows
mediante Git Bash y Docker, usando el modelo real.

## 3. Datos y alcance de la corrida

Se validaron 5,248,635 viajes. Los 14 archivos mensuales Parquet coinciden con los
SHA-256 del manifiesto del PR. Las particiones de viajes contienen 4,365,813 /
393,524 / 489,298 filas; las de P1, 137,290 / 14,448 / 14,119, en orden de
entrenamiento, validación y prueba.

**Límite de procedencia del catálogo.** El Parquet local de estaciones no
coincidía con el hash del manifiesto. Se regeneró con el código del PR desde el
CSV cuyo hash sí coincide con la fuente registrada. Tiene 484 estaciones y
4 capacidades faltantes. Sus valores y tipos coinciden con el catálogo local
anterior, pero sus bytes no coinciden con el Parquet registrado por el equipo.
No se modificó el manifiesto ni se afirma identidad de ese archivo. Antes de
presentar la corrida como reproducción binaria completa, falta obtener el
Parquet registrado o explicar y documentar esa diferencia.

Los ocho archivos del modelo exportado sí pasan sus hashes; el hash de
`registry.json` coincide con el de la referencia del PR. Esto verifica el
artefacto compartido, no su identidad con el modelo histórico original, cuya
limitación ya está indicada en el recibo del equipo.

## 4. Qué muestran los chequeos y qué hacer si fallan

Los 51 resultados de la nueva corrida coinciden con el resumen ya versionado.
No hubo duplicados de llave, nulos prohibidos, tipos incorrectos ni solapamiento
de fechas entre entrenamiento y evaluación. Los chequeos bloqueantes revisan
todas las filas; los de drift usan muestreo.

En P1, el máximo drift numérico fue 0.11 en validación y 0.08 en prueba; el KS
del objetivo fue 0.09 y 0.04, por debajo del umbral 0.2. `mes`, `dia_mes` y
`dias_historia` están excluidos de drift por su cambio esperado con el calendario.
Las estaciones nuevas representan 0.63% y 1.11% de las filas, por debajo del 5%.

Los nulos de `salidas_lag7` y `salidas_mismo_dia_media4` fueron 1.97%, 0.8% y
0.04% en las tres particiones, por debajo del 10% permitido. Pasar estos límites
no demuestra que no exista drift ni garantiza calidad predictiva futura.

El código actual termina con error ante fallas bloqueantes. Drift y los demás
chequeos informativos generan avisos, no detienen por sí solos la ejecución.
La siguiente respuesta queda propuesta para revisión del equipo; no se agregó
una regla automática ni se cambiaron los umbrales.

| Si aparece | Acción antes de seguir |
|---|---|
| Hash distinto, tipo incorrecto, llave duplicada o nulo prohibido | Separar el archivo afectado, revisar descarga y transformación, corregir la causa y repetir la exportación y los chequeos. No actualizar el hash solo para aceptar un archivo distinto. |
| Solapamiento temporal o indicio de fuga | Revisar fechas de corte y cálculo de rezagos. Rehacer las variables con información disponible antes del pronóstico y repetir entrenamiento/evaluación si la fuga afectó esos resultados. |
| Más nulos en rezagos o catálogo | Distinguir falta de historia de un cruce de estaciones incompleto. Revisar las filas afectadas y cómo responde el modelo; no rellenar automáticamente con cero. |
| Drift de variables u objetivo fuera del límite | Revisar variable, periodo y estaciones afectadas; comprobar cambios de fuente y estacionalidad. Medir desempeño en datos recientes antes de decidir reentrenar. No aumentar el umbral para ocultar el aviso. |
| Más estaciones nuevas | Revisar el catálogo, cobertura e historial de esas estaciones. Evaluar ese subconjunto por separado antes de confiar en el promedio general. |

Para cerrar una incidencia, conservar el reporte que falló, la causa, el cambio
aplicado y la nueva corrida. Un aviso revisado puede ser aceptable, pero necesita
una explicación; no equivale automáticamente a una falla del modelo.

## 5. Auditoría de seguridad

`pip-audit` no reportó vulnerabilidades conocidas en las 173 dependencias del
entorno principal ni en las 89 de la API al momento de la consulta. Bandit revisó
`mibici`, `validacion_datos`, `api` y `scripts`: 0 hallazgos medios/altos y 5 bajos
relacionados con subprocess y búsqueda de ejecutables en PATH. Las llamadas usan
listas de argumentos, no `shell=True`; sigue siendo necesario usar ejecutables
y rutas de confianza. Estos resultados no prueban ausencia total de riesgos.

En el entorno aislado de DeepChecks, la consulta a PyPI agotó el tiempo de espera.
La consulta posterior a OSV terminó y encontró dos avisos únicos en
`setuptools==81.0.0`:

| Aviso | Riesgo descrito | Versión corregida |
|---|---|---|
| `PYSEC-2026-3447` / `CVE-2026-59890` | Omisión de exclusiones de `MANIFEST.in` por normalización Unicode al construir distribuciones fuente en macOS | 83.0.0 |
| `GHSA-xmj3-9gf7-h52w` | Escape de rutas durante instalación legacy de wheels en Windows | 84.1.0 |

OSV devolvió el primer aviso dos veces; no son tres vulnerabilidades distintas.
Los dos ya figuran como excepciones en el Makefile, pero **siguen sin corregirse**.
No aparecieron en el análisis del entorno de la API.

La compatibilidad con `pkg_resources` impide tratar la actualización como un
cambio directo: las versiones nuevas de setuptools lo retiraron y DeepChecks
lo utiliza. Por ahora, mantener este entorno solo para validación, sin construir
paquetes ni instalar wheels legacy de origen desconocido. La siguiente prueba
sería evaluar una combinación compatible y corregida en otro entorno, repetir
los 19 tests sintéticos y los chequeos reales, y retirar las excepciones solo
cuando se confirme que funciona. No se actualizó ninguna dependencia en esta
revisión.

## 6. Evidencia para revisar

Se incluyen los resultados de la revisión en `docs/evidencia-revision-windows/`.
Los reportes del equipo en `reports/deepchecks/` se conservaron sin cambios.
En los logs se sustituyeron rutas personales por `<entorno-python>` o
`<usuario-local>`; no se cambiaron los resultados.

- [Resumen JSON de la nueva corrida de DeepChecks](evidencia-revision-windows/deepchecks-resumen.json).
- [Procedencia de datos](evidencia-revision-windows/procedencia-datos.json),
  [regeneración de estaciones](evidencia-revision-windows/regeneracion-estaciones.txt)
  y [comparación de valores](evidencia-revision-windows/comparacion-estaciones.txt).
- [Aceptación con tolerancia](evidencia-revision-windows/aceptacion-tolerancia.txt),
  [regresiones Bash](evidencia-revision-windows/regresiones-bash.txt)
  y [suite final](evidencia-revision-windows/suite-windows.txt).
- Auditorías de [API](evidencia-revision-windows/auditoria-api.json),
  [entorno principal](evidencia-revision-windows/auditoria-principal.json),
  [DeepChecks](evidencia-revision-windows/auditoria-deepchecks.json)
  y [Bandit](evidencia-revision-windows/bandit.json).

## 7. Pendientes

Revisar por separado el ajuste de tolerancia y decidir si se incorpora al código.
Obtener el Parquet de estaciones registrado o explicar la diferencia de bytes
antes de afirmar reproducción binaria completa. Las excepciones de setuptools
siguen abiertas; esta documentación no las resuelve.
