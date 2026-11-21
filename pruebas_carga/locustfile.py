"""Prueba de carga de la API P1 de MiBici (Semana 10).

Cada usuario simulado pide predicciones de demanda a /predict con la petición de
ejemplo del repositorio y, en menor proporción (1 de cada 11), consulta /health.
La validación exige HTTP 200 y el campo ``viajes_esperados``.

Ejecución recomendada: ``python scripts/prueba_carga.py --config base``.
"""

import json
from pathlib import Path

from locust import HttpUser, between, task

RAIZ = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads((RAIZ / "docs" / "evidencia-api" / "peticion-predict.json").read_text())


class UsuarioMiBici(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predecir(self):
        with self.client.post("/predict", json=PAYLOAD, name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "viajes_esperados" not in r.json():
                r.failure("Respuesta sin viajes_esperados")

    @task(1)
    def salud(self):
        self.client.get("/health", name="/health")
