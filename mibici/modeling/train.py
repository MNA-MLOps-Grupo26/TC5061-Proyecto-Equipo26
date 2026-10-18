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

app = typer.Typer()

# Cada pregunta se agrega aquí al integrar su módulo, por ejemplo:
#   from mibici.modeling import p1
#   EJECUTORES = {"P1": p1.ejecutar}
EJECUTORES = {}


@app.command()
def main(
    pregunta: str = typer.Option("todas", help="P1, P2, P3 o todas."),
):
    """Corre el experimento base para una pregunta o para todas."""
    if not EJECUTORES:
        raise typer.Exit("Todavía no hay preguntas registradas en EJECUTORES.")
    pregunta = pregunta.upper()
    if pregunta == "TODAS":
        seleccion = list(EJECUTORES)
    elif pregunta in EJECUTORES:
        seleccion = [pregunta]
    else:
        raise typer.BadParameter(
            f"{pregunta} no está disponible. Opciones: {', '.join(EJECUTORES)} o todas."
        )

    viajes = cargar_viajes()
    estaciones = cargar_estaciones()
    logger.info(f"{len(viajes):,} viajes y {len(estaciones)} estaciones cargados")

    for clave in seleccion:
        logger.info(f"Ejecutando {clave}")
        EJECUTORES[clave](viajes, estaciones)

    logger.success(
        "Experimento base terminado. Consulten: mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )


if __name__ == "__main__":
    app()
