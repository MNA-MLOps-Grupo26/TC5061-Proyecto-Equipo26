"""Carga y verificación del modelo P1 exportado del Model Registry.

El servicio no arranca si los archivos del modelo no coinciden con los SHA-256 de
``registry.json`` o si la firma del modelo no son las 18 variables de P1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from api.esquemas import VARIABLES_P1

ARCHIVO_REGISTRY = "registry.json"


class ErrorModelo(RuntimeError):
    """El modelo exportado no pasa la verificación; el servicio no debe arrancar."""


class Predictor(Protocol):
    def predict(self, datos: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class ModeloServido:
    """Modelo verificado listo para predecir, con su linaje."""

    predictor: Predictor
    linaje: dict[str, Any]
    archivos_verificados: int

    @property
    def info(self) -> dict[str, str]:
        return {k: str(self.linaje[k]) for k in ("nombre", "version", "run_id")}


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_hashes(directorio: Path) -> dict[str, Any]:
    """Compara los archivos del directorio con el mapa ``archivos`` de registry.json.

    Deben ser exactamente esos archivos (sin faltantes ni sobrantes) y con el mismo
    SHA-256. Devuelve el contenido de registry.json.
    """
    ruta_registry = directorio / ARCHIVO_REGISTRY
    if not ruta_registry.is_file():
        raise ErrorModelo(f"No existe {ruta_registry}")
    registry = json.loads(ruta_registry.read_text(encoding="utf-8"))
    esperados: dict[str, str] = registry.get("archivos") or {}
    if not esperados:
        raise ErrorModelo("registry.json no tiene el mapa 'archivos'")

    presentes = {
        p.relative_to(directorio).as_posix()
        for p in directorio.rglob("*")
        if p.is_file() and p != ruta_registry
    }
    faltantes = sorted(set(esperados) - presentes)
    sobrantes = sorted(presentes - set(esperados))
    if faltantes or sobrantes:
        raise ErrorModelo(f"Archivos del modelo distintos: faltan {faltantes}, sobran {sobrantes}")
    distintos = sorted(r for r, h in esperados.items() if _sha256(directorio / r) != h)
    if distintos:
        raise ErrorModelo(f"SHA-256 distinto en: {distintos}")
    if "modelo" not in registry:
        raise ErrorModelo("registry.json no tiene la sección 'modelo' con el linaje")
    return registry


def verificar_firma(nombres_entrada: Sequence[str] | None) -> None:
    """La firma debe ser exactamente las 18 variables de P1, en el mismo orden."""
    if list(nombres_entrada or []) != VARIABLES_P1:
        raise ErrorModelo(
            f"La firma del modelo no coincide con VARIABLES_P1: {list(nombres_entrada or [])}"
        )


def _cargar_pyfunc(directorio: Path) -> tuple[Predictor, list[str]]:
    import mlflow.pyfunc  # Import diferido: las pruebas con modelo falso no requieren MLflow.

    modelo = mlflow.pyfunc.load_model(str(directorio))
    esquema = modelo.metadata.get_input_schema()
    return modelo, (esquema.input_names() if esquema is not None else [])


def cargar_modelo(
    directorio: Path,
    cargar_pyfunc: Callable[[Path], tuple[Predictor, list[str]]] = _cargar_pyfunc,
) -> ModeloServido:
    """Verifica hashes, carga el modelo y verifica la firma, en ese orden."""
    directorio = Path(directorio)
    registry = verificar_hashes(directorio)
    predictor, nombres = cargar_pyfunc(directorio)
    verificar_firma(nombres)
    return ModeloServido(
        predictor=predictor,
        linaje=registry["modelo"],
        archivos_verificados=len(registry["archivos"]),
    )
