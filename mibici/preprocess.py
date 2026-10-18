"""Normalización de los archivos de MiBici y manifiesto de datos procesados.

Se ejecuta desde ``python -m mibici.dataset`` después de la descarga. También
puede correrse sola (``python -m mibici.preprocess``) si los CSV ya existen en
``data/raw`` y su manifiesto de descarga está actualizado.
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
import pandas as pd

from mibici.config import (
    DOWNLOADS_MANIFEST,
    ESTACIONES_PARQUET,
    MANIFESTS_DIR,
    NOMENCLATURA_FILENAME,
    PROCESSED_DATA_DIR,
    PROCESSED_MANIFEST,
    RAW_DATA_DIR,
)

# Solo se conservan columnas operativas. Usuario_Id, Genero y
# Año_de_nacimiento se descartan aquí y nunca llegan a data/processed.
COLUMNAS = {
    "Viaje_Id": "viaje_id",
    "Inicio_del_viaje": "inicio_viaje",
    "Fin_del_viaje": "fin_viaje",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}

COLUMNAS_ESTACIONES = {
    "id": "estacion_id",
    "name": "nombre",
    "latitude": "latitud",
    "longitude": "longitud",
    "dpcapacity": "capacidad",
}


def normalizar_archivo(archivo: Path, encoding: str = "utf-8-sig") -> dict:
    """Normaliza un CSV mensual de viajes y lo guarda como Parquet.

    El encoding viene del manifiesto de descarga (``dataset.verify_csv`` lo
    detecta leyendo el archivo completo), así no depende del nombre del mes.
    """
    df = pd.read_csv(archivo, encoding=encoding, usecols=list(COLUMNAS))
    registros_originales = len(df)

    df = df.rename(columns=COLUMNAS)

    for columna in ["inicio_viaje", "fin_viaje"]:
        df[columna] = pd.to_datetime(df[columna], errors="raise")

    for columna in ["viaje_id", "origen_id", "destino_id"]:
        df[columna] = pd.to_numeric(df[columna], errors="raise").astype("int64")

    # Duplicados exactos en las columnas conservadas
    df = df.drop_duplicates()
    duplicados = registros_originales - len(df)

    # Viajes con fin anterior al inicio no son físicamente posibles
    invalidos = int((df["fin_viaje"] < df["inicio_viaje"]).sum())
    df = df.loc[df["fin_viaje"] >= df["inicio_viaje"]]

    df = df.sort_values(["inicio_viaje", "viaje_id"]).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_DIR / f"{archivo.stem}.parquet", index=False)

    return {
        "archivo": archivo.name,
        "originales": registros_originales,
        "normalizados": len(df),
        "duplicados_eliminados": duplicados,
        "fin_antes_de_inicio": invalidos,
    }


def normalizar_estaciones(archivo: Path, encoding: str = "utf-8-sig") -> dict:
    """Normaliza el catálogo de estaciones y lo guarda como Parquet.

    Capacidad 0 o nula se guarda como nula: no permite calcular el indicador
    de riesgo de P2 y no debe confundirse con una estación sin anclajes.
    """
    df = pd.read_csv(archivo, encoding=encoding, usecols=list(COLUMNAS_ESTACIONES))
    df = df.rename(columns=COLUMNAS_ESTACIONES)

    df["estacion_id"] = pd.to_numeric(df["estacion_id"], errors="raise").astype("int64")
    for columna in ["latitud", "longitud", "capacidad"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df["nombre"] = df["nombre"].astype("string").str.strip()
    df.loc[df["capacidad"] <= 0, "capacidad"] = pd.NA

    df = df.drop_duplicates("estacion_id").sort_values("estacion_id").reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_DIR / ESTACIONES_PARQUET, index=False)

    return {
        "archivo": archivo.name,
        "estaciones": len(df),
        "sin_capacidad": int(df["capacidad"].isna().sum()),
    }


def calcular_sha256(archivo: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    sha256 = hashlib.sha256()
    with archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(bloque)
    return sha256.hexdigest()


def calcular_data_version(descargas: list[dict]) -> str:
    """Identificador corto y determinista de la versión de los datos.

    Se deriva solo de los SHA-256 de los archivos originales, por lo que no
    cambia entre computadoras ni entre corridas si MiBici no modifica sus CSV.
    Es el valor que se registra en MLflow como ``data_version``.
    """
    huella = sorted((item["filename"], item["sha256"]) for item in descargas)
    return hashlib.sha256(json.dumps(huella).encode("utf-8")).hexdigest()[:12]


def generar_manifiesto(descargas: list[dict], viajes: list[dict], estaciones: dict) -> dict:
    """Genera el manifiesto de los Parquet procesados con su data_version."""
    hashes_origen = {item["filename"]: item["sha256"] for item in descargas}

    archivos = []
    for resultado in viajes:
        nombre_csv = resultado["archivo"]
        ruta_parquet = PROCESSED_DATA_DIR / Path(nombre_csv).with_suffix(".parquet").name
        archivos.append(
            {
                "filename": ruta_parquet.name,
                "sha256": calcular_sha256(ruta_parquet),
                "records": resultado["normalizados"],
                "duplicates_removed": resultado["duplicados_eliminados"],
                "invalid_removed": resultado["fin_antes_de_inicio"],
                "source_filename": nombre_csv,
                "source_sha256": hashes_origen[nombre_csv],
            }
        )

    ruta_estaciones = PROCESSED_DATA_DIR / ESTACIONES_PARQUET
    manifiesto = {
        "data_version": calcular_data_version(descargas),
        "total_files": len(archivos),
        "total_records": sum(item["records"] for item in archivos),
        "files": archivos,
        "stations": {
            "filename": ruta_estaciones.name,
            "sha256": calcular_sha256(ruta_estaciones),
            "records": estaciones["estaciones"],
            "without_capacity": estaciones["sin_capacidad"],
            "source_filename": estaciones["archivo"],
            "source_sha256": hashes_origen[estaciones["archivo"]],
        },
    }

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    destino = MANIFESTS_DIR / PROCESSED_MANIFEST
    destino.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.success(f"Manifiesto generado: {destino} (data_version={manifiesto['data_version']})")
    return manifiesto


def procesar(descargas: list[dict]) -> dict:
    """Normaliza viajes y estaciones listados en el manifiesto de descarga."""
    resultados = []
    estaciones = None

    for info in descargas:
        archivo = RAW_DATA_DIR / info["filename"]
        if info["filename"] == NOMENCLATURA_FILENAME:
            estaciones = normalizar_estaciones(archivo, info["encoding"])
            logger.info(
                f"{archivo.name}: {estaciones['estaciones']} estaciones, "
                f"{estaciones['sin_capacidad']} sin capacidad"
            )
            continue

        resultado = normalizar_archivo(archivo, info["encoding"])
        resultados.append(resultado)
        logger.info(
            f"{resultado['archivo']}: {resultado['normalizados']:,} viajes, "
            f"{resultado['duplicados_eliminados']} duplicados, "
            f"{resultado['fin_antes_de_inicio']} con fin antes de inicio"
        )

    if estaciones is None:
        raise FileNotFoundError(f"El manifiesto no incluye {NOMENCLATURA_FILENAME}")

    total_originales = sum(r["originales"] for r in resultados)
    total_normalizados = sum(r["normalizados"] for r in resultados)
    logger.success(
        f"Normalización completada: {total_originales:,} registros originales -> "
        f"{total_normalizados:,} registros normalizados"
    )
    return generar_manifiesto(descargas, resultados, estaciones)


def main() -> None:
    """Normaliza los archivos listados en el manifiesto de descarga existente."""
    origen = MANIFESTS_DIR / DOWNLOADS_MANIFEST
    descargas = json.loads(origen.read_text(encoding="utf-8"))
    procesar(descargas)


if __name__ == "__main__":
    main()
