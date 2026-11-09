# Revisión independiente en Ubuntu — MiBici P1

## Alcance

Validación independiente sobre el commit base 4ff9484 de main, integrado mediante el PR #23, utilizando Ubuntu/WSL.

## Resultados

| Verificación | Resultado |
|---|---|
| Suite general pytest | 205 aprobadas, 3 omitidas |
| API con modelo real | 2 aprobadas |
| DeepChecks (Python 3.11) | 19 aprobadas |
| Ruff | Sin errores |
| Bandit (severidad media o superior) | Sin hallazgos |
| Auditoría de dependencias principales | Sin vulnerabilidades conocidas detectadas |
| Auditoría DeepChecks | Sin vulnerabilidades no exceptuadas |
| Pruebas de seguridad | 11 aprobadas |

Las pruebas del modelo real y DeepChecks se ejecutaron por separado. Las 11 pruebas de seguridad ya estaban incluidas en la suite general.

## API y contenedor

Se construyó y ejecutó el contenedor P1 en Ubuntu/WSL. Se verificaron las rutas /health, /predict y /predict/batch con HTTP 200; una entrada inconsistente devolvió HTTP 422. También se comprobaron los hashes de los ocho archivos del modelo exportado y la correspondencia de las predicciones con las referencias. Estas verificaciones precedieron a la corrección del Makefile.

## Corrección de portabilidad

Los comandos make test y make seguridad fallaban con error 127 porque no existía un ejecutable global llamado python. Se modificó PYTHON_INTERPRETER para utilizar uv con Python 3.13 y requirements.txt, y make test pasó a utilizar esa variable. Tras el cambio, make test completó 205 pruebas aprobadas y make seguridad terminó correctamente con 11 pruebas aprobadas.

## Limitaciones

Se observaron advertencias de Starlette/httpx, SQLAlchemy/MLflow y pkg_resources en DeepChecks. Bandit informó cinco hallazgos de severidad baja, fuera del umbral configurado. La auditoría de DeepChecks aplicó excepciones de vulnerabilidades configuradas; la auditoría principal utilizó --no-deps y no identificó la versión del requisito editable. El modelo local models/p1 no se incorporará al repositorio. La validación no equivale a una reproducción completa desde cero en otro equipo.

## Conclusión

Las pruebas ejecutadas respaldan el funcionamiento de MiBici P1 en Ubuntu/WSL y confirman la corrección de un problema de portabilidad en los comandos oficiales del proyecto.
