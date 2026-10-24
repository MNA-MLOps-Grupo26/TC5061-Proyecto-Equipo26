"""Experimento base: ejecuta los runs de P1, P2 y P3 y los registra en MLflow.

Uso, desde la raíz del repositorio con el entorno activo y después de la ingesta:

    python -m mibici.modeling.train              # todas las preguntas disponibles
    python -m mibici.modeling.train --pregunta P2
    python -m mibici.modeling.train --pregunta P1 --modelo hgb --max-iter 500

Para la entrega se lanza con ``mlflow run . -e train`` (ver README). Modelos por
pregunta: P1 default, baseline, hgb, glm; P2 default, baseline, logreg, hgb;
P3 default o kmeans.

Cada pregunta vive en su módulo (p1.py, p2.py, p3.py) y expone
``ejecutar(viajes, estaciones)``; P1 y P2 aceptan además el modelo y sus
hiperparámetros. Para agregar una, regístrenla en EJECUTORES y en
``modelos_permitidos``.
"""

from typing import Annotated

from loguru import logger
import typer

from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.modeling import p1, p2, p3

app = typer.Typer()

EJECUTORES = {"P1": p1.ejecutar, "P2": p2.ejecutar, "P3": p3.ejecutar}


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="P1, P2, P3 o todas.")] = "todas",
    modelo: Annotated[str, typer.Option(help="Algoritmo a utilizar.")] = "default",
    learning_rate: Annotated[
        str | None, typer.Option(help="Tasa de aprendizaje opcional.")
    ] = None,
    max_iter: Annotated[
        str | None, typer.Option(help="Número máximo de iteraciones opcional.")
    ] = None,
):
    """Corre el experimento base para una pregunta o para todas."""
    pregunta = pregunta.upper()
    if pregunta == "TODAS":
        seleccion = list(EJECUTORES)
    elif pregunta in EJECUTORES:
        seleccion = [pregunta]
    else:
        raise typer.BadParameter(
            f"{pregunta} no está disponible. Opciones: {', '.join(EJECUTORES)} o todas."
        )

    modelos_permitidos = {
        "P1": set(p1.MODELOS),
        "P2": set(p2.MODELOS),
        "P3": {"default", "kmeans"},
    }

    for clave in seleccion:
        if modelo not in modelos_permitidos[clave]:
            opciones = ", ".join(sorted(modelos_permitidos[clave]))
            raise typer.BadParameter(
                f"El modelo '{modelo}' no está disponible para {clave}. Opciones: {opciones}.",
                param_hint="--modelo",
            )

    # Convertir los parámetros opcionales
    learning_rate = None if learning_rate in (None, "None") else float(learning_rate)
    max_iter = None if max_iter in (None, "None") else int(max_iter)

    # Evitar ignorar hiperparámetros cuando el modelo no los usa
    if modelo in ("default", "baseline") and (learning_rate is not None or max_iter is not None):
        raise typer.BadParameter(
            f"Con modelo='{modelo}' no se permiten learning_rate ni max_iter. "
            "Selecciona un modelo específico para configurar sus hiperparámetros."
        )
    # learning_rate solo existe en los modelos de gradient boosting
    if learning_rate is not None and modelo != "hgb":
        raise typer.BadParameter(
            "learning_rate solo aplica a model=hgb (P1 o P2); los demás modelos usan max_iter."
        )

    # P3 no admite los hiperparámetros de P1 y P2
    if "P3" in seleccion and (learning_rate is not None or max_iter is not None):
        raise typer.BadParameter("P3 no admite learning_rate ni max_iter.")

    viajes = cargar_viajes() if any(p != "P3" for p in seleccion) else None
    estaciones = cargar_estaciones()
    logger.info(f"{len(estaciones)} estaciones cargadas; P3 verifica sus fuentes por periodo")

    for clave in seleccion:
        logger.info(f"Ejecutando {clave}")

        if clave == "P1":
            p1.ejecutar(
                viajes,
                estaciones,
                modelo=modelo,
                learning_rate=learning_rate,
                max_iter=max_iter,
            )
        elif clave == "P2":
            p2.ejecutar(
                viajes,
                estaciones,
                modelo=modelo,
                learning_rate=learning_rate,
                max_iter=max_iter,
            )
        else:
            EJECUTORES[clave](viajes, estaciones, modelo=modelo)

    logger.success(
        "Experimento terminado. Consulten el servidor de MLflow (MLFLOW_TRACKING_URI), "
        "por ejemplo http://127.0.0.1:5000"
    )


if __name__ == "__main__":
    app()
