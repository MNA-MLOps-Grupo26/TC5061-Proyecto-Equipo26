"""Normalización de los archivos mensuales de MiBici."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from loguru import logger
import pandas as pd

from mibici.config import MANIFESTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

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

def calcular_sha256(archivo: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    sha256 = hashlib.sha256()

    with archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


def generar_manifiesto(resultados: list[dict]) -> None:
    """Genera el manifiesto de los archivos Parquet procesados."""

    origen = MANIFESTS_DIR / "mibici_downloads.json"

    with origen.open(encoding="utf-8") as f:
        descargas = json.load(f)

    hashes_origen = {
        item["filename"]: item["sha256"]
        for item in descargas
    }

    archivos = []

    for resultado in resultados:
        nombre_csv = resultado["archivo"]
        nombre_parquet = Path(nombre_csv).with_suffix(".parquet").name
        ruta_parquet = PROCESSED_DATA_DIR / nombre_parquet

        archivos.append({
            "filename": nombre_parquet,
            "sha256": calcular_sha256(ruta_parquet),
            "records": resultado["normalizados"],
            "source_filename": nombre_csv,
            "source_sha256": hashes_origen[nombre_csv],
        })

    manifiesto = {
        "version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "total_files": len(archivos),
        "total_records": sum(item["records"] for item in archivos),
        "files": archivos,
    }

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    destino = MANIFESTS_DIR / "mibici_processed.json"

    with destino.open("w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    logger.success(f"Manifiesto generado: {destino}")

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
    resultados=[] 
    for archivo in archivos:
        resultado = normalizar_archivo(archivo)
        resultados.append(resultado)
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
    generar_manifiesto(resultados)

if __name__ == "__main__":
    main()
