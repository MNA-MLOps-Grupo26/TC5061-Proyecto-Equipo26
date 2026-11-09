# Servicio REST del modelo P1 de MiBici (FastAPI) con el modelo del registry empaquetado.
# Compatible con el estándar OCI: docker build -f Containerfile (principal) o podman build.
#
# Requisito previo: exportar el modelo del Model Registry a un directorio nuevo
# (por defecto models/p1) con scripts/exportar_modelo.py. Ver README.
#
#   docker build -f Containerfile -t mibici-p1-api:1 .
#   docker run -d --name mibici-p1-api -p 127.0.0.1:8000:8000 mibici-p1-api:1

ARG PYTHON_IMAGE=python:3.13.16-slim
FROM ${PYTHON_IMAGE}

# Directorio del modelo exportado dentro del contexto de build.
ARG MODEL_DIR=models/p1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/modelo

WORKDIR /app

RUN useradd --system --create-home --uid 10001 --user-group app

# Dependencias primero para aprovechar la caché de capas.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api/ /app/api/
COPY ${MODEL_DIR}/ /app/modelo/

# El código y el modelo quedan de solo lectura para el usuario del servicio:
# si alguien altera un archivo del modelo, la verificación de hashes lo detecta al arrancar.
RUN chmod -R a+rX,go-w /app

USER app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
