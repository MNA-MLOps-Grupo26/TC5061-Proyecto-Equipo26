"""Experimento base: ejecuta los runs de P1, P2 y P3 y los registra en MLflow.

Uso, desde la raíz del repositorio con el entorno activo y después de la ingesta:

    python -m mibici.modeling.train              # todas las preguntas disponibles
    python -m mibici.modeling.train --pregunta P2

Cada pregunta vive en su módulo (p1.py, p2.py, p3.py) y expone
``ejecutar(viajes, estaciones)``. Para agregar una, regístrenla en EJECUTORES.
"""

from loguru import logger
import typer
from typing import Annotated
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

    # Convertir los parámetros opcionales
    learning_rate = None if learning_rate in (None, "None") else float(learning_rate)
    max_iter = None if max_iter in (None, "None") else int(max_iter)

    # Evitar ignorar hiperparámetros cuando se usa el modelo predeterminado
    if modelo == "default" and (learning_rate is not None or max_iter is not None):
        raise typer.BadParameter(
            "Con modelo='default' no se permiten learning_rate ni max_iter. "
            "Selecciona un modelo específico para configurar sus hiperparámetros."
        )
    # P2 utiliza max_iter, pero no learning_rate
    if "P2" in seleccion and learning_rate is not None:
        raise typer.BadParameter(
            "P2 no admite learning_rate. Utiliza únicamente max_iter con modelo='logreg'."
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
                max_iter=max_iter,
            )
        else:
            EJECUTORES[clave](viajes, estaciones)

    logger.success(
        "Experimento base terminado. Consulten: mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )


if __name__ == "__main__":
    app()
