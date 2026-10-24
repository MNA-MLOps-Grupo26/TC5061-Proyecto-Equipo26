"""Búsqueda de hiperparámetros con nested runs (punto de entrada ``search``).

Uso, con el servidor de tracking activo (ver README):

    mlflow run . -e search --env-manager uv --experiment-name mibici-equipo26 -P question=P1
    mlflow run . -e search --env-manager uv --experiment-name mibici-equipo26 -P question=P2 -P model=hgb

Cada búsqueda abre un run padre (``stage_of_work=search``) y un run hijo por
combinación de ``config.BUSQUEDAS``. Los hijos registran métricas de
entrenamiento, validación y prueba y los mismos artefactos que un candidato.
La selección se hace con validación; prueba solo se reporta.
"""

from typing import Annotated

from loguru import logger
import typer

from mibici.config import BUSQUEDAS
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.modeling import p1, p2, p3

app = typer.Typer()

BUSCADORES = {"P1": p1.buscar, "P2": p2.buscar, "P3": p3.buscar}
MODELO_POR_DEFECTO = {"P1": "hgb", "P2": "logreg", "P3": "kmeans"}


@app.command()
def main(
    pregunta: Annotated[str, typer.Option(help="P1, P2 o P3.")] = "P1",
    modelo: Annotated[
        str, typer.Option(help="Modelo a buscar; 'default' usa el de referencia.")
    ] = "default",
):
    """Corre la búsqueda definida en config.BUSQUEDAS para una pregunta y un modelo."""
    pregunta = pregunta.upper()
    if pregunta not in BUSCADORES:
        raise typer.BadParameter(f"{pregunta} no está disponible. Opciones: P1, P2, P3.")
    modelo = MODELO_POR_DEFECTO[pregunta] if modelo == "default" else modelo
    if (pregunta, modelo) not in BUSQUEDAS:
        opciones = ", ".join(m for p, m in BUSQUEDAS if p == pregunta)
        raise typer.BadParameter(
            f"No hay búsqueda para {pregunta}/{modelo}. Opciones: {opciones}.",
            param_hint="--modelo",
        )

    viajes = cargar_viajes() if pregunta != "P3" else None
    estaciones = cargar_estaciones() if pregunta != "P3" else None
    combinaciones = BUSQUEDAS[(pregunta, modelo)]
    logger.info(f"Búsqueda {pregunta}/{modelo}: {len(combinaciones)} combinaciones")
    BUSCADORES[pregunta](viajes, estaciones, modelo=modelo)
    logger.success("Búsqueda terminada. Consulten el servidor de MLflow (MLFLOW_TRACKING_URI).")


if __name__ == "__main__":
    app()
