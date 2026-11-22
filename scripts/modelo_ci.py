"""Genera un modelo sintético P1 para el smoke test de Docker Compose en CI.

El modelo real (``models/p1``) no se versiona en Git, así que el CI no puede
levantar la API con él. Este script crea un modelo MLflow pyfunc con la misma
firma (las 18 variables de ``VARIABLES_P1``) y un ``registry.json`` con los
SHA-256 de sus archivos, para que la API pase la verificación de arranque.

Solo sirve para validar que Compose levanta los servicios y que /health y
/predict responden. Sus predicciones no tienen valor de negocio.

    python scripts/modelo_ci.py --destino models/ci-smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from api.esquemas import VARIABLES_P1  # noqa: E402

ARCHIVO_REGISTRY = "registry.json"
# Mismo tipo que declara mibici/tracking.py para el modelo real: lo generamos nosotros.
TIPOS_SKOPS_CONFIABLES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def generar(destino: Path, semilla: int = 26) -> dict:
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from sklearn.ensemble import HistGradientBoostingRegressor

    if destino.exists() and any(destino.iterdir()):
        raise SystemExit(f"{destino} ya existe y no está vacío")

    rng = np.random.default_rng(semilla)
    datos = pd.DataFrame(
        rng.uniform(0, 100, size=(500, len(VARIABLES_P1))), columns=VARIABLES_P1
    ).astype("float64")
    objetivo = datos[VARIABLES_P1[0]] * 0.8 + rng.uniform(0, 5, size=len(datos))

    modelo = HistGradientBoostingRegressor(max_iter=20, random_state=semilla)
    modelo.fit(datos, objetivo)

    firma = infer_signature(datos, modelo.predict(datos))
    mlflow.sklearn.save_model(
        modelo, str(destino), signature=firma, skops_trusted_types=TIPOS_SKOPS_CONFIABLES
    )

    archivos = {
        p.relative_to(destino).as_posix(): _sha256(p)
        for p in sorted(destino.rglob("*"))
        if p.is_file()
    }
    registro = {
        "proposito": "Modelo sintético para smoke test de CI. No usar para predicciones reales.",
        "modelo": {"nombre": "mibici-ci-smoke", "version": "0", "run_id": "ci-smoke"},
        "archivos": archivos,
    }
    (destino / ARCHIVO_REGISTRY).write_text(
        json.dumps(registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return registro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--destino", type=Path, default=RAIZ / "models" / "ci-smoke")
    args = parser.parse_args(argv)
    registro = generar(args.destino)
    print(f"Modelo sintético en {args.destino} ({len(registro['archivos'])} archivos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
