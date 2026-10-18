
"""Partición temporal de los datos procesados de MiBici."""


from loguru import logger
import pandas as pd

from mibici.config import (
    DATA_END_EXCLUSIVE,
    PROCESSED_DATA_DIR,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)

SPLITS_DIR = PROCESSED_DATA_DIR / "splits"


COLUMNAS_VIAJES = {
    "viaje_id",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
}


def validar_columnas_viajes(df: pd.DataFrame) -> None:
    """Comprueba que un DataFrame tenga las columnas necesarias de viajes.

    Raises:
        ValueError: si faltan una o más columnas requeridas.
    """
    faltantes = COLUMNAS_VIAJES - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )


def agregar_variables_temporales(
    df: pd.DataFrame,
    columna_fecha: str,
) -> pd.DataFrame:
    """Agrega variables de calendario a partir de una columna temporal.

    Las variables generadas son fecha, hora, día de la semana, mes,
    año e indicador de fin de semana.

    La columna temporal debe representar un instante conocido en el
    momento de la predicción. La función no crea rezagos ni objetivos.

    Returns:
        Copia del DataFrame con las variables temporales agregadas.
    """
    if columna_fecha not in df.columns:
        raise ValueError(
            f"No existe la columna temporal: {columna_fecha}"
        )

    resultado = df.copy()
    fechas = pd.to_datetime(
        resultado[columna_fecha],
        errors="raise",
    )

    resultado["fecha"] = fechas.dt.date
    resultado["hora"] = fechas.dt.hour
    resultado["dia_semana"] = fechas.dt.dayofweek
    resultado["mes"] = fechas.dt.month
    resultado["anio"] = fechas.dt.year
    resultado["es_fin_semana"] = (
        fechas.dt.dayofweek >= 5
    ).astype("int8")

    return resultado



def particionar_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Divide los viajes en entrenamiento, validación y prueba."""

    if "inicio_viaje" not in df.columns:
        raise ValueError("Falta la columna inicio_viaje.")

    fechas = pd.to_datetime(df["inicio_viaje"], errors="raise")

    inicio = pd.Timestamp(TRAIN_START)
    validacion = pd.Timestamp(VALIDATION_START)
    prueba = pd.Timestamp(TEST_START)
    fin = pd.Timestamp(DATA_END_EXCLUSIVE)

    return {
        "train": df.loc[
            (fechas >= inicio) & (fechas < validacion)
        ].copy(),
        "validation": df.loc[
            (fechas >= validacion) & (fechas < prueba)
        ].copy(),
        "test": df.loc[
            (fechas >= prueba) & (fechas < fin)
        ].copy(),
    }


def main() -> None:
    """Genera las particiones temporales a partir de los Parquet mensuales."""

    archivos = sorted(
        PROCESSED_DATA_DIR.glob("datos_abiertos_*.parquet")
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos Parquet en {PROCESSED_DATA_DIR}"
        )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    totales = {"train": 0, "validation": 0, "test": 0}
    fuera_periodo = 0

    for archivo in archivos:
        df = pd.read_parquet(archivo)
        particiones = particionar_dataframe(df)

        fechas = pd.to_datetime(df["inicio_viaje"], errors="raise")
        inicio = pd.Timestamp(TRAIN_START)
        fin = pd.Timestamp(DATA_END_EXCLUSIVE)

        fuera = ~((fechas >= inicio) & (fechas < fin))
        fuera_periodo += int(fuera.sum())

        for nombre, subconjunto in particiones.items():
            if subconjunto.empty:
                continue

            directorio = SPLITS_DIR / nombre
            directorio.mkdir(parents=True, exist_ok=True)

            destino = directorio / archivo.name
            subconjunto.to_parquet(destino, index=False)
            totales[nombre] += len(subconjunto)

        logger.info(f"Procesado: {archivo.name}")

    logger.success(
        f"Entrenamiento: {totales['train']:,} | "
        f"Validación: {totales['validation']:,} | "
        f"Prueba: {totales['test']:,}"
    )

    if fuera_periodo:
        logger.warning(
            f"Registros fuera del periodo configurado: {fuera_periodo:,}"
        )

    logger.success(f"Particiones guardadas en: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
