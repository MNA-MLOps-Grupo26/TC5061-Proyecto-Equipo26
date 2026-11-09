"""Registra el candidato P1 en un almacén nuevo, para reproducir la entrega desde cero.

No se usa sobre el registro ya entregado (versión 1 del almacén
``mlflow-p1-store-20261107``). Está pensado para quien reproduzca el
entrenamiento en un clon y un almacén independientes.

Reglas de seguridad:

- El run se indica explícitamente con ``--run-id``; nunca se elige por nombre.
- Antes de tocar el registry se verifica el run concreto: experimento, estado,
  los 16 parámetros publicados, las tres métricas de validación (atol 1e-9) y
  que tenga exactamente un logged model.
- Si ya existe una versión de ese logged model, se reutiliza sin cambiar sus tags.
- Ante cualquier conflicto (otra versión ya registrada, ``champion`` apuntando a
  otra versión, varias coincidencias) se detiene sin cambios.
- Solo "no existe" (``RESOURCE_DOES_NOT_EXIST``) se interpreta como ausencia. Un
  error de conexión, permisos o servidor detiene el script antes de escribir.
- Las métricas deben ser finitas; NaN o infinito no pasan la verificación.
- Sin ``--aplicar`` solo muestra el plan.

Ejemplo:

    python scripts/registrar_p1.py --tracking-uri http://127.0.0.1:5056 --run-id <id>
    python scripts/registrar_p1.py --tracking-uri http://127.0.0.1:5056 --run-id <id> --aplicar
"""

from __future__ import annotations

import argparse
import math
import sys

NOMBRE = "mibici-equipo26-p1-demand"
ALIAS = "champion"
EXPERIMENTO = "mibici-equipo26"

# Publicados en docs/evidence/2026-10-24-final/registration-evidence.json.
PARAMETROS_ESPERADOS = {
    "train_start": "2025-08-01",
    "validation_start": "2026-08-01",
    "test_start": "2026-09-01",
    "data_end_exclusive": "2026-10-01",
    "data_version": "cd917f9aa025",
    "random_seed": "42",
    "loss": "poisson",
    "learning_rate": "0.05",
    "max_iter": "300",
    "max_leaf_nodes": "31",
    "min_samples_leaf": "50",
    "early_stopping": "False",
    "horizonte_dias": "1",
    "n_variables": "18",
    "fit": "train_only",
    "selection_metric": "val_wape",
}
METRICAS_ESPERADAS = {
    "val_mae": 4.851352276716691,
    "val_rmse": 8.171478370762053,
    "val_wape": 0.17812945580458703,
}
TOLERANCIA = 1e-9
HISTORICO = {
    "reproduction_of_run_id": "9d55326449364961b1cbc585ac28cefe",
    "reproduction_of_logged_model_id": "m-a1ba41ed76654e6fa8eb2fd611b2bd88",
}


class Conflicto(RuntimeError):
    """El estado del registry o del run impide continuar sin cambios."""


def verificar_run(cliente, run_id: str, experimento: str):
    run = cliente.get_run(run_id)
    errores = []
    exp = cliente.get_experiment(run.info.experiment_id)
    if exp.name != experimento:
        errores.append(f"experimento {exp.name!r} != {experimento!r}")
    if run.info.status != "FINISHED":
        errores.append(f"estado {run.info.status}")
    for clave, valor in PARAMETROS_ESPERADOS.items():
        if run.data.params.get(clave) != valor:
            errores.append(f"param {clave}={run.data.params.get(clave)!r} (esperado {valor!r})")
    for clave, valor in METRICAS_ESPERADAS.items():
        actual = run.data.metrics.get(clave)
        if actual is None or not math.isfinite(actual) or abs(actual - valor) > TOLERANCIA:
            errores.append(f"métrica {clave}={actual!r} (esperado {valor!r}, atol {TOLERANCIA})")
    modelos = [
        m
        for m in cliente.search_logged_models(experiment_ids=[run.info.experiment_id])
        if m.source_run_id == run_id
    ]
    if len(modelos) != 1:
        errores.append(f"se esperaba 1 logged model del run y hay {len(modelos)}")
    if errores:
        raise Conflicto("El run no coincide con la entrega publicada: " + "; ".join(errores))
    return run, modelos[0]


def planear(cliente, nombre: str, alias: str, run, logged_model) -> list[tuple[str, object]]:
    """Devuelve las acciones necesarias o lanza Conflicto sin modificar nada."""
    from mlflow.exceptions import MlflowException

    acciones: list[tuple[str, object]] = []
    try:
        registrado = cliente.get_registered_model(nombre)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise  # conexión, permisos o servidor: main se detiene sin escribir
        return [("crear_modelo", nombre), ("crear_version", None), ("fijar_alias", None)]
    # El alias se lee del modelo registrado, sin interpretar excepciones como ausencia.
    alias_registrado = dict(registrado.aliases or {}).get(alias)

    versiones = cliente.search_model_versions(f"name='{nombre}'")
    coinciden = [
        v
        for v in versiones
        if v.run_id == run.info.run_id
        and (getattr(v, "model_id", None) or v.source.removeprefix("models:/"))
        == logged_model.model_id
    ]
    if len(coinciden) > 1:
        raise Conflicto(f"{len(coinciden)} versiones coinciden con el run; revisen a mano")
    if not coinciden:
        otras = sorted(v.version for v in versiones)
        if otras:
            raise Conflicto(
                f"{nombre} ya tiene versiones {otras} de otros runs; no se crea otra versión"
            )
        acciones.append(("crear_version", None))
    else:
        acciones.append(("reutilizar_version", coinciden[0].version))
    actual_alias = alias_registrado
    if actual_alias is None:
        acciones.append(("fijar_alias", None))
    elif coinciden and str(actual_alias) == str(coinciden[0].version):
        acciones.append(("alias_correcto", actual_alias))
    else:
        raise Conflicto(f"@{alias} apunta a la versión {actual_alias}; no se reemplaza")
    return acciones


def aplicar(cliente, nombre: str, alias: str, run, logged_model, acciones) -> str:
    import mlflow

    version = None
    for accion, valor in acciones:
        if accion == "crear_modelo":
            cliente.create_registered_model(nombre)
        elif accion == "crear_version":
            mv = mlflow.register_model(f"models:/{logged_model.model_id}", nombre)
            version = str(mv.version)
            etiquetas = {
                "question": "P1",
                "source_sha": run.data.tags.get("git_commit", ""),
                "git_dirty": run.data.tags.get("git_dirty", ""),
                "data_version": run.data.params["data_version"],
                "source_run_id": run.info.run_id,
                "source_logged_model_id": logged_model.model_id,
                "experiment_id": run.info.experiment_id,
                "selection_scope": "validation; no September tuning",
                **HISTORICO,
            }
            for clave, valor_tag in etiquetas.items():
                cliente.set_model_version_tag(nombre, version, clave, valor_tag)
        elif accion == "reutilizar_version":
            version = str(valor)
        elif accion == "fijar_alias":
            # Se vuelve a leer justo antes de escribir para no reemplazar un alias nuevo.
            vigente = dict(cliente.get_registered_model(nombre).aliases or {}).get(alias)
            if vigente is not None and str(vigente) != str(version):
                raise Conflicto(f"@{alias} apareció apuntando a la versión {vigente}")
            if vigente is None:
                cliente.set_registered_model_alias(nombre, alias, version)
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--registry-uri")
    parser.add_argument("--run-id", required=True, help="Run concreto a registrar")
    parser.add_argument("--nombre", default=NOMBRE)
    parser.add_argument("--alias", default=ALIAS)
    parser.add_argument("--experimento", default=EXPERIMENTO)
    parser.add_argument("--aplicar", action="store_true", help="Ejecuta el plan")
    args = parser.parse_args(argv)

    import mlflow
    from mlflow import MlflowClient

    registry_uri = args.registry_uri or args.tracking_uri
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    cliente = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=registry_uri)
    try:
        run, logged_model = verificar_run(cliente, args.run_id, args.experimento)
        acciones = planear(cliente, args.nombre, args.alias, run, logged_model)
    except Conflicto as error:
        print(f"DETENIDO sin cambios: {error}", file=sys.stderr)
        return 2
    # Intencionalmente amplio: cualquier fallo de consulta (conexión, permisos, servidor,
    # run inexistente) detiene el script antes de escribir.
    except Exception as error:  # noqa: BLE001
        print(
            f"DETENIDO sin cambios: no se pudo consultar MLflow ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        return 3
    print(f"Run {run.info.run_id} verificado; logged model {logged_model.model_id}")
    for accion, valor in acciones:
        print(f"  plan: {accion}" + (f" ({valor})" if valor is not None else ""))
    if not args.aplicar:
        print("Modo de revisión: no se hicieron cambios (usen --aplicar).")
        return 0
    try:
        version = aplicar(cliente, args.nombre, args.alias, run, logged_model, acciones)
    except Exception as error:  # noqa: BLE001  (se informa y se termina sin reintentar)
        print(
            f"ERROR al aplicar ({type(error).__name__}: {error}). Revisen el registry antes "
            "de repetir: el script reutiliza lo que ya coincida y no reemplaza aliases.",
            file=sys.stderr,
        )
        return 4
    print(f"Listo: {args.nombre} versión {version}, alias @{args.alias}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
