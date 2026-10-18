"""Normalización de los archivos mensuales de MiBici."""

from pathlib import Path

from loguru import logger
import pandas as pd

from mibici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

COLUMNAS = {
    "Viaje_Id": "viaje_id",
    "Inicio_del_viaje": "inicio_viaje",
    "Fin_del_viaje": "fin_viaje",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}


def normalizar_archivo(archivo: Path) -> dict:
    """Normaliza un CSV mensual y lo guarda como Parquet."""

    encoding = (
        "cp1252"
        if "2026_01" in archivo.name
        else "utf-8-sig"
    )

    df = pd.read_csv(
        archivo,
        encoding=encoding,
        usecols=list(COLUMNAS),
    )

    registros_originales = len(df)

    # Estandarizar nombres
    df = df.rename(columns=COLUMNAS)

    # Convertir fechas
    for columna in ["inicio_viaje", "fin_viaje"]:
        df[columna] = pd.to_datetime(
            df[columna],
            errors="raise",
        )

    # Estandarizar identificadores
    for columna in ["viaje_id", "origen_id", "destino_id"]:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="raise",
        ).astype("int64")

    # Eliminar duplicados exactos en las columnas conservadas
    df = df.drop_duplicates()

    # Ordenar para obtener resultados reproducibles
    df = df.sort_values(
        ["inicio_viaje", "viaje_id"]
    ).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    salida = (
        PROCESSED_DATA_DIR
        / f"{archivo.stem}.parquet"
    )

    df.to_parquet(salida, index=False)

    return {
        "archivo": archivo.name,
        "originales": registros_originales,
        "normalizados": len(df),
        "duplicados_eliminados": registros_originales - len(df),
    }


def main() -> None:
    """Normaliza todos los archivos mensuales disponibles."""

    archivos = sorted(
        RAW_DATA_DIR.glob("datos_abiertos_*.csv")
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron CSV en {RAW_DATA_DIR}"
        )

    total_originales = 0
    total_normalizados = 0

    for archivo in archivos:
        resultado = normalizar_archivo(archivo)

        total_originales += resultado["originales"]
        total_normalizados += resultado["normalizados"]

        logger.info(
            f"{resultado['archivo']}: "
            f"{resultado['normalizados']:,} viajes, "
            f"{resultado['duplicados_eliminados']} duplicados eliminados"
        )

    logger.success(
        f"Normalización completada: "
        f"{total_originales:,} registros originales → "
        f"{total_normalizados:,} registros normalizados"
    )


if __name__ == "__main__":
    main()
