"""API REST del modelo P1 de MiBici.

Arranque: ``uvicorn api.main:app --host 0.0.0.0 --port 8000`` con ``MODEL_PATH``
apuntando al modelo exportado. El modelo se carga y verifica una sola vez.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

from api.esquemas import (
    VARIABLES_P1,
    LinajeModelo,
    LoteP1,
    PrediccionLote,
    RegistroP1,
    RespuestaLote,
    RespuestaPrediccion,
    RespuestaSalud,
    Verificacion,
)
from api.modelo import ModeloServido, cargar_modelo

logger = logging.getLogger("api.p1")


class ErrorSalidaModelo(RuntimeError):
    """La salida del modelo no es válida (longitud, NaN, infinito o negativa)."""


def _cargador_por_defecto() -> ModeloServido:
    ruta = os.environ.get("MODEL_PATH")
    if not ruta:
        raise RuntimeError("Defina la variable de entorno MODEL_PATH con el modelo exportado")
    return cargar_modelo(Path(ruta))


def _seguro_json(valor: Any) -> Any:
    """Convierte NaN, infinitos y objetos no serializables a texto para el 422."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, dict):
        return {str(k): _seguro_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_seguro_json(v) for v in valor]
    if valor is None or isinstance(valor, (str, int, float, bool)):
        return valor
    return str(valor)


def _predecir(modelo: ModeloServido, registros: list[RegistroP1]) -> list[float]:
    datos = pd.DataFrame([r.a_fila() for r in registros], columns=VARIABLES_P1).astype("float64")
    salida = np.asarray(modelo.predictor.predict(datos), dtype="float64").reshape(-1)
    if salida.shape[0] != len(registros):
        raise ErrorSalidaModelo(
            f"el modelo devolvió {salida.shape[0]} valores para {len(registros)} registros"
        )
    if not np.all(np.isfinite(salida)):
        raise ErrorSalidaModelo("el modelo devolvió valores NaN o infinitos")
    if np.any(salida < 0):
        raise ErrorSalidaModelo("el modelo devolvió valores negativos")
    return [float(v) for v in salida]


def crear_app(cargador: Callable[[], ModeloServido] = _cargador_por_defecto) -> FastAPI:
    """Crea la app. Las pruebas inyectan ``cargador`` con un modelo falso."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.modelo = cargador()  # Si falla, el servicio no arranca.
        logger.info("Modelo cargado: %s", app.state.modelo.info)
        yield

    app = FastAPI(
        title="MiBici P1 - demanda por estación",
        description=(
            "Predice las salidas esperadas por estación para un día con el modelo "
            "`mibici-equipo26-p1-demand` (HistGradientBoosting) del Model Registry de MLflow."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def _error_validacion(_: Request, exc: RequestValidationError) -> JSONResponse:
        # El manejador por defecto falla al serializar un NaN recibido (termina en 500).
        errores = [
            {
                "loc": list(e.get("loc", ())),
                "msg": e.get("msg"),
                "type": e.get("type"),
                "input": _seguro_json(e.get("input")),
            }
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errores})

    @app.exception_handler(ErrorSalidaModelo)
    async def _error_salida(_: Request, exc: ErrorSalidaModelo) -> JSONResponse:
        logger.error("Salida inválida del modelo: %s", exc)
        return JSONResponse(
            status_code=500, content={"detail": f"Salida inválida del modelo: {exc}"}
        )

    def _modelo(request: Request) -> ModeloServido:
        return request.app.state.modelo

    @app.get("/health", response_model=RespuestaSalud, tags=["servicio"])
    def health(request: Request) -> RespuestaSalud:
        modelo = _modelo(request)
        return RespuestaSalud(
            estado="ok",
            modelo=LinajeModelo(
                **{k: str(v) for k, v in modelo.linaje.items() if k in LinajeModelo.model_fields}
            ),
            verificacion=Verificacion(
                hashes="ok", firma="ok", archivos_verificados=modelo.archivos_verificados
            ),
        )

    @app.post("/predict", response_model=RespuestaPrediccion, tags=["prediccion"])
    def predict(registro: RegistroP1, request: Request) -> RespuestaPrediccion:
        modelo = _modelo(request)
        (valor,) = _predecir(modelo, [registro])
        return RespuestaPrediccion(
            viajes_esperados=valor,
            estacion_id=registro.estacion_id,
            fecha=registro.fecha,
            modelo=modelo.info,
        )

    @app.post("/predict/batch", response_model=RespuestaLote, tags=["prediccion"])
    def predict_batch(lote: LoteP1, request: Request) -> RespuestaLote:
        modelo = _modelo(request)
        valores = _predecir(modelo, lote.registros)
        return RespuestaLote(
            n=len(valores),
            predicciones=[
                PrediccionLote(
                    indice=i, viajes_esperados=v, estacion_id=r.estacion_id, fecha=r.fecha
                )
                for i, (r, v) in enumerate(zip(lote.registros, valores, strict=True))
            ],
            modelo=modelo.info,
        )

    return app


app = crear_app()
