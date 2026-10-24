"""Tabla comparativa por pregunta a partir de los runs de MLflow (para el reporte).

    python -m mibici.modeling.comparar            # P1, P2 y P3
    python -m mibici.modeling.comparar --pregunta P2

Solo lee el servidor de tracking; no crea runs. Escribe una tabla CSV y otra en
Markdown por pregunta en ``reports/figures/`` (no se versionan) e imprime la
segunda. Para cada run con métricas de validación muestra modelo, etapa,
hiperparámetros clave, la métrica de selección en validación y prueba, la
mejora respecto al baseline, la brecha validación-prueba (estabilidad) y el
tiempo de entrenamiento (costo).
"""

from typing import Annotated

import mlflow
import pandas as pd
import typer

from mibici.config import EXPERIMENT_NAME, FIGURES_DIR, P2_TOP_K
from mibici.tracking import configurar_mlflow

app = typer.Typer()

# Métrica ligada a la decisión de negocio de cada pregunta y si mayor es mejor.
CRITERIOS = {
    "P1": {"metrica": "wape", "mayor_mejor": False, "otras": ["mae", "rmse"]},
    "P2": {
        "metrica": f"recall_at_{P2_TOP_K}",
        "mayor_mejor": True,
        "otras": [f"precision_at_{P2_TOP_K}", "f1_macro"],
    },
    "P3": {
        "metrica": "ari_vs_train",
        "mayor_mejor": True,
        "otras": ["fraccion_estable", "silhouette_asignacion_fija"],
    },
}
HIPERPARAMETROS_CLAVE = ["learning_rate", "max_iter", "max_leaf_nodes", "alpha", "C", "k"]


def tabla_pregunta(pregunta: str) -> pd.DataFrame:
    """Runs de una pregunta con métricas de validación, ordenados por la métrica de selección."""
    criterio = CRITERIOS[pregunta]
    metrica = criterio["metrica"]
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.question = '{pregunta}' and metrics.val_{metrica} > -1e9",
    )
    if runs.empty:
        return runs

    columnas = {
        "run": runs["tags.mlflow.runName"],
        "etapa": runs["tags.stage_of_work"],
        "modelo": runs["tags.model"],
    }
    for clave in HIPERPARAMETROS_CLAVE:
        if f"params.{clave}" in runs:
            columnas[clave] = runs[f"params.{clave}"]
    for nombre in [metrica, *criterio["otras"]]:
        for conjunto in ("val", "test"):
            if f"metrics.{conjunto}_{nombre}" in runs:
                columnas[f"{conjunto}_{nombre}"] = runs[f"metrics.{conjunto}_{nombre}"]
    if "metrics.train_silhouette" in runs:
        columnas["train_silhouette"] = runs["metrics.train_silhouette"]
    columnas["fit_time_s"] = runs.get("metrics.fit_time_s")
    columnas["run_id"] = runs["run_id"]
    tabla = pd.DataFrame(columnas)

    baseline = tabla.loc[tabla["etapa"] == "baseline", f"val_{metrica}"]
    if not baseline.empty:
        referencia = baseline.iloc[0]
        signo = 1 if criterio["mayor_mejor"] else -1
        tabla["mejora_vs_baseline_%"] = (
            100 * signo * (tabla[f"val_{metrica}"] - referencia) / abs(referencia)
        )
    tabla["brecha_val_test"] = tabla[f"test_{metrica}"] - tabla[f"val_{metrica}"]
    return tabla.sort_values(f"val_{metrica}", ascending=not criterio["mayor_mejor"])


def a_markdown(tabla: pd.DataFrame) -> str:
    """Markdown simple (sin dependencias extra) con 4 decimales."""
    tabla = tabla.drop(columns="run_id").fillna("")
    filas = [
        [f"{v:.4f}" if isinstance(v, float) else str(v) for v in fila]
        for fila in tabla.itertuples(index=False)
    ]
    encabezado = "| " + " | ".join(tabla.columns) + " |"
    separador = "|" + "---|" * len(tabla.columns)
    return "\n".join([encabezado, separador, *("| " + " | ".join(f) + " |" for f in filas)])


@app.command()
def main(pregunta: Annotated[str, typer.Option(help="P1, P2, P3 o todas.")] = "todas"):
    """Imprime y guarda la tabla comparativa de cada pregunta."""
    configurar_mlflow()
    preguntas = list(CRITERIOS) if pregunta.lower() == "todas" else [pregunta.upper()]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for clave in preguntas:
        tabla = tabla_pregunta(clave)
        if tabla.empty:
            typer.echo(f"\n{clave}: sin runs con métricas de validación.")
            continue
        tabla.to_csv(FIGURES_DIR / f"comparacion_{clave.lower()}.csv", index=False)
        markdown = a_markdown(tabla)
        (FIGURES_DIR / f"comparacion_{clave.lower()}.md").write_text(markdown, encoding="utf-8")
        criterio = CRITERIOS[clave]
        orden = "mayor" if criterio["mayor_mejor"] else "menor"
        typer.echo(f"\n## {clave} (selección: val_{criterio['metrica']}, {orden} es mejor)\n")
        typer.echo(markdown)


if __name__ == "__main__":
    app()
