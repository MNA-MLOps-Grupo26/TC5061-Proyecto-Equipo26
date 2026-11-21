FROM python:3.13.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY docker/mlflow/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "1"]
