# Verificación de integración — 11 noviembre 2026

- Base: main 2298aa6e4844dedb782400f0c993ce1f018d9235. Se conservan las entregas anteriores.
- Suite del código integrado con el entorno original: 193 passed, 32 skipped, 2 warnings. Los skips corresponden a 2 pruebas API que requieren el modelo exportado, 29 de Bash en Windows y 1 del entorno aislado DeepChecks.
- Ruff check y format --check: aprobados (58 archivos); git diff --cached --check: aprobado.
- Bandit sobre mibici y validacion_datos: 0 problemas medios/altos, 5 bajos preexistentes.
- pip-audit 2.10.1: requirements.txt y requirements-monitoring.txt sin vulnerabilidades conocidas. La entrada editable local del proyecto se omite por no tener versión resoluble.
- requirements-accuracy.txt detectó CVE-2026-53995 en MLflow 3.17.1. Se actualizó la receta a 3.17.2 (incluyendo skinny/tracing) y se repitió el análisis: sin vulnerabilidades conocidas.
- Suite completa con MLflow 3.17.2: 193 passed, 32 skipped, 2 warnings, 234.33 s. Recarga pyfunc del artefacto v3 congelado con 3.17.2: 14618 predicciones exactamente iguales, diferencia máxima 0.0. Solo aparece la advertencia esperada de metadatos 3.17.1 frente al entorno 3.17.2.
- No se ejecutó el agregado make calidad: DeepChecks real y aceptación Bash/contenedor no se volvieron a ejecutar en esta integración Windows. No se presenta como una nueva validación de esos componentes.
- El artefacto v3 histórico conserva su dependencia embebida MLflow 3.17.1. No se modificó ni registró otra versión: usar el entorno parcheado explícito para cargarlo, sin reinstalar ese pin histórico.
- No se reentrenó ningún modelo ni se modificó el registro/aliases/despliegue. El PDF se entrega por separado.
