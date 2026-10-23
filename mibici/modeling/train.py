"""Experimento base: ejecuta los runs de P1, P2 y P3 y los registra en MLflow.

Uso, desde la raíz del repositorio con el entorno activo y después de la ingesta:

    python -m mibici.modeling.train              # todas las preguntas disponibles
    python -m mibici.modeling.train --pregunta P2

Cada pregunta vive en su módulo (p1.py, p2.py, p3.py) y expone
``ejecutar(viajes, estaciones)``. Para agregar una, regístrenla en EJECUTORES.
"""

from loguru import logger
import typer

from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.modeling import p1, p2, p3

app = typer.Typer()

EJECUTORES = {"P1": p1.ejecutar, "P2": p2.ejecutar, "P3": p3.ejecutar}


@app.command()
def main(
    pregunta: str = typer.Option("todas", help="P1, P2, P3 o todas."),
    modelo: str = typer.Option("default", help="Algoritmo a utilizar."),
    learning_rate: str = typer.Option("None", help="Tasa de aprendizaje opcional."),
    max_iter: str = typer.Option("None", help="Número máximo de iteraciones opcional."),

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

    viajes = cargar_viajes() if any(p != "P3" for p in seleccion) else None
    estaciones = cargar_estaciones()
    logger.info(f"{len(estaciones)} estaciones cargadas; P3 verifica sus fuentes por periodo")

    learning_rate = None if learning_rate == "None" else float(learning_rate)
    max_iter = None if max_iter == "None" else int(max_iter)

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
            p2.ejecutar(viajes,estaciones,modelo=modelo,max_iter=max_iter,)
        else:
            EJECUTORES[clave](viajes, estaciones)


    logger.success(
        "Experimento base terminado. Consulten: mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )


if __name__ == "__main__":
    app()
