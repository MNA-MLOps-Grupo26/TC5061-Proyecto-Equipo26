"""Descarga del Model Registry el modelo P1 y lo deja listo para empaquetar (modo B).

Pasos:

1. Verifica el registry esperado: el modelo registrado existe y el alias se
   resuelve **una sola vez** a una versión numérica en estado READY.
2. Comprueba que esa versión corresponde al run, logged model y experimento
   esperados, y lee su linaje (commit, data_version, referencias históricas).
3. Descarga ``models:/<nombre>/<versión>`` (la versión numérica, no el alias) a
   un directorio temporal y lo copia a ``--destino``, que **no debe existir**.
4. Escribe ``registry.json`` (creación exclusiva) con la identidad del almacén,
   el linaje y el SHA-256 de cada archivo exportado, sin incluirse a sí mismo.
5. Carga el modelo exportado y comprueba firma y predicción del input_example.

Los IDs esperados por defecto son los de la entrega del 2026-11-07. Para otro
almacén, pasen los suyos con ``--esperar-*``.

Ejemplo:

    python scripts/exportar_modelo.py \\
        --tracking-uri http://127.0.0.1:5055 \\
        --destino models/p1 \\
        --recibo-almacen docs/evidencia-api/recibo-registro-p1.json
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

NOMBRE = "mibici-equipo26-p1-demand"
ALIAS = "champion"
EXPERIMENTO = "mibici-equipo26"
VERSION_ESPERADA = "1"
RUN_ESPERADO = "51ef51ad5b534386a3c020692887c6bc"
LOGGED_MODEL_ESPERADO = "m-8917791035574d139539ba27ae896ad4"
ARCHIVO_REGISTRO = "registry.json"
VARIABLES_P1 = [
    "salidas_lag1",
    "llegadas_lag1",
    "saldo_lag1",
    "salidas_lag7",
    "salidas_media7",
    "salidas_media28",
    "llegadas_media7",
    "llegadas_media28",
    "salidas_mismo_dia_media4",
    "dias_historia",
    "dias_desde_ultimo_viaje",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "dia_mes",
    "capacidad",
    "latitud",
    "longitud",
]


class ErrorExportacion(RuntimeError):
    pass


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _hashes(directorio: Path) -> dict[str, str]:
    return {
        p.relative_to(directorio).as_posix(): _sha256(p)
        for p in sorted(directorio.rglob("*"))
        if p.is_file() and p.relative_to(directorio).as_posix() != ARCHIVO_REGISTRO
    }


def _exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise ErrorExportacion(mensaje)


def _leer_mlmodel(directorio: Path) -> dict:
    import yaml

    return yaml.safe_load((directorio / "MLmodel").read_text(encoding="utf-8"))


def exportar(args: argparse.Namespace) -> dict:
    import mlflow
    from mlflow import MlflowClient

    destino = Path(args.destino).resolve()
    _exigir(not destino.exists(), f"{destino} ya existe; elijan otro destino (no se sobrescribe)")
    _exigir(destino.parent.is_dir(), f"No existe el directorio padre {destino.parent}")

    registry_uri = args.registry_uri or args.tracking_uri
    cliente = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=registry_uri)

    # 1. Registry esperado y resolución única del alias.
    registrado = cliente.get_registered_model(args.nombre)
    version = cliente.get_model_version_by_alias(args.nombre, args.alias)
    _exigir(version.status == "READY", f"La versión {version.version} no está READY")
    numero = str(version.version)
    print(f"Alias {args.nombre}@{args.alias} -> versión {numero} (run {version.run_id})")

    # 2. Identidad de la versión resuelta.
    if args.esperar_version:
        _exigir(numero == args.esperar_version, f"versión {numero} != {args.esperar_version}")
    if args.esperar_run_id:
        _exigir(
            version.run_id == args.esperar_run_id,
            f"run_id {version.run_id} != {args.esperar_run_id}",
        )
    logged_model_id = getattr(version, "model_id", None) or version.source.removeprefix("models:/")
    if args.esperar_logged_model_id:
        _exigir(
            logged_model_id == args.esperar_logged_model_id,
            f"logged model {logged_model_id} != {args.esperar_logged_model_id}",
        )
    run = cliente.get_run(version.run_id)
    _exigir(run.info.status == "FINISHED", f"El run {run.info.run_id} no terminó")
    experimento = cliente.get_experiment(run.info.experiment_id)
    _exigir(
        experimento.name == args.esperar_experimento,
        f"experimento {experimento.name} != {args.esperar_experimento}",
    )
    etiquetas = dict(version.tags)

    # 3. Descarga de la versión numérica y copia a un destino nuevo.
    uri_modelo = f"models:/{args.nombre}/{numero}"
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    with tempfile.TemporaryDirectory(prefix="exportar-p1-") as temporal:
        descargado = Path(mlflow.artifacts.download_artifacts(uri_modelo, dst_path=temporal))
        mlmodel = _leer_mlmodel(descargado)
        _exigir(
            mlmodel.get("model_id") == logged_model_id,
            f"MLmodel.model_id {mlmodel.get('model_id')} != {logged_model_id}",
        )
        _exigir(mlmodel.get("run_id") == version.run_id, "MLmodel.run_id no coincide")
        shutil.copytree(descargado, destino)  # falla si el destino apareció mientras tanto

    # 4. registry.json (creación exclusiva).
    recibo_almacen = None
    if args.recibo_almacen:
        recibo_almacen = json.loads(Path(args.recibo_almacen).read_text(encoding="utf-8")).get(
            "almacen"
        )
    registro = {
        "almacen": {
            "tracking_uri": args.tracking_uri,
            "registry_uri": registry_uri,
            "experimento": {"id": experimento.experiment_id, "nombre": experimento.name},
            "identidad_declarada": recibo_almacen,
        },
        "modelo": {
            "nombre": registrado.name,
            "version": numero,
            "alias_consultado": args.alias,
            "uri_descargada": uri_modelo,
            "run_id": version.run_id,
            "logged_model_id": logged_model_id,
            "source": version.source,
            "source_sha": etiquetas.get("source_sha", run.data.tags.get("git_commit")),
            "git_dirty": etiquetas.get("git_dirty", run.data.tags.get("git_dirty")),
            "data_version": etiquetas.get("data_version", run.data.params.get("data_version")),
            "mlflow_version_del_modelo": mlmodel.get("mlflow_version"),
        },
        "historico": {
            k: v for k, v in etiquetas.items() if k.startswith(("reproduction_", "selection_"))
        },
        "exportado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "exportado_con": {
            "python": platform.python_version(),
            "mlflow": mlflow.__version__,
            "plataforma": platform.platform(),
        },
        "archivos": _hashes(destino),
    }
    with (destino / ARCHIVO_REGISTRO).open("x", encoding="utf-8") as archivo:
        json.dump(registro, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")

    # 5. Comprobación de la copia exportada.
    import numpy as np
    import pandas as pd

    _exigir(_hashes(destino) == registro["archivos"], "Los hashes cambiaron tras escribir")
    cargado = mlflow.pyfunc.load_model(str(destino))
    nombres = cargado.metadata.get_input_schema().input_names()
    _exigir(nombres == VARIABLES_P1, f"Firma inesperada: {nombres}")
    ejemplo = pd.DataFrame(cargado.input_example).astype("float64")
    prediccion = np.asarray(cargado.predict(ejemplo), dtype="float64")
    _exigir(np.isfinite(prediccion).all() and (prediccion >= 0).all(), "Predicción inválida")
    print(f"Exportado a {destino} ({len(registro['archivos'])} archivos + {ARCHIVO_REGISTRO})")
    print(f"Predicción del input_example: {np.round(prediccion, 4).tolist()}")
    return registro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", required=True, help="URI explícita del tracking")
    parser.add_argument("--registry-uri", help="URI del registry (por defecto, la de tracking)")
    parser.add_argument("--destino", required=True, help="Directorio nuevo; no debe existir")
    parser.add_argument("--nombre", default=NOMBRE)
    parser.add_argument("--alias", default=ALIAS)
    parser.add_argument("--esperar-version", default=VERSION_ESPERADA)
    parser.add_argument("--esperar-run-id", default=RUN_ESPERADO)
    parser.add_argument("--esperar-logged-model-id", default=LOGGED_MODEL_ESPERADO)
    parser.add_argument("--esperar-experimento", default=EXPERIMENTO)
    parser.add_argument("--recibo-almacen", help="JSON con un bloque 'almacen' (identidad)")
    args = parser.parse_args(argv)
    from mlflow.exceptions import MlflowException

    try:
        exportar(args)
    except (ErrorExportacion, MlflowException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
