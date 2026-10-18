"""Ingesta reproducible de MiBici: descarga, verificación, normalización y versión.

Un solo comando, sin pasos manuales:

    python -m mibici.dataset

1. Descarga los CSV mensuales del periodo de config.py y la nomenclatura de
   estaciones (si ya existen en data/raw, los reutiliza).
2. Verifica columnas, encoding, número de registros y SHA-256
   (manifests/mibici_downloads.json).
3. Normaliza viajes y estaciones a Parquet en data/processed y genera
   manifests/mibici_processed.json con el data_version que se registra en MLflow.

Este módulo también expone los cargadores que usan features.py y el modelado.
"""

import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    DATA_END_EXCLUSIVE,
    DOWNLOADS_MANIFEST,
    ESTACIONES_PARQUET,
    MANIFESTS_DIR,
    MIBICI_PORTAL,
    NOMENCLATURA_FILENAME,
    PROCESSED_DATA_DIR,
    PROCESSED_MANIFEST,
    RAW_DATA_DIR,
    TRAIN_START,
    VIAJES_PARQUET_GLOB,
)
from mibici.preprocess import procesar

app = typer.Typer()

REQUIRED_COLUMNS = {
    "Viaje_Id",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
}

REQUIRED_STATION_COLUMNS = {"id", "name", "latitude", "longitude", "dpcapacity"}

# Captura viajes (datos_abiertos_AAAA_MM.csv) y catálogo (nomenclatura_AAAA_MM.csv)
FILE_PATTERN = re.compile(
    r"/site/assets/files/\d+/((?:datos_abiertos|nomenclatura)_\d{4}_\d{2}\.csv)"
)


def run_curl(url: str, output: Path | None = None) -> str:
    """Consulta una URL o descarga su contenido mediante curl."""
    command = [
        "curl",
        "-fLsS",
        "--retry",
        "3",
        "--max-time",
        "300",
    ]

    if output is None:
        command.append(url)
    else:
        command.extend(["-o", str(output), url])

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=output is None,
    )

    return result.stdout if output is None else ""


def expected_months():
    """Genera los meses del periodo definido en config.py."""
    current = TRAIN_START.replace(day=1)

    while current < DATA_END_EXCLUSIVE:
        yield current.strftime("%Y_%m")

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def discover_files() -> dict[str, str]:
    """Identifica los enlaces de descarga publicados por MiBici."""
    html = run_curl(MIBICI_PORTAL)
    links = {}

    for match in FILE_PATTERN.finditer(html):
        links[match.group(1)] = urljoin(MIBICI_PORTAL, match.group(0))

    return links


def detect_encoding(path: Path) -> str:
    """Detecta UTF-8 y utiliza CP1252 cuando sea necesario."""
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            for _ in file:
                pass

        return "utf-8-sig"

    except UnicodeDecodeError:
        return "cp1252"


def verify_csv(path: Path, required: set[str] = REQUIRED_COLUMNS) -> dict:
    """Valida estructura y registra metadatos del CSV."""
    digest = hashlib.sha256()
    size = 0

    # Calcular hash SHA-256 y tamaño del archivo
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
            size += len(chunk)

    # Detectar codificación del archivo completo
    encoding = detect_encoding(path)

    # Leer y verificar el contenido
    with path.open(
        "r",
        encoding=encoding,
        newline="",
    ) as file:
        reader = csv.reader(file)
        columns = next(reader, [])

        missing = required - set(columns)

        if missing:
            raise ValueError(
                f"{path.name}: faltan columnas {sorted(missing)}"
            )

        records = sum(1 for _ in reader)

    if records == 0:
        raise ValueError(
            f"{path.name}: CSV sin registros"
        )

    return {
        "filename": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "records": records,
        "encoding": encoding,
        "columns": columns,
    }

def generate_monthly_report(manifest: list[dict]) -> None:
    """Genera un reporte mensual de viajes y estaciones distintas."""
    import pandas as pd

    from mibici.config import PROJ_ROOT

    results = []
    all_stations = set()

    for info in manifest:
        if info["filename"] == NOMENCLATURA_FILENAME:
            continue
        path = RAW_DATA_DIR / info["filename"]

        df = pd.read_csv(
            path,
            usecols=["Origen_Id", "Destino_Id"],
            encoding=info["encoding"],
        )

        stations = (
            set(df["Origen_Id"].dropna())
            | set(df["Destino_Id"].dropna())
        )

        all_stations.update(stations)

        results.append(
            {
                "mes": info["filename"]
                .replace("datos_abiertos_", "")
                .replace(".csv", ""),
                "viajes": info["records"],
                "estaciones_distintas": len(stations),
            }
        )

    output = PROJ_ROOT / "reports" / "mibici_resumen_mensual.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output, index=False)

    logger.success(
        f"Reporte mensual generado: {output} "
        f"({len(results)} meses, "
        f"{len(all_stations)} estaciones distintas)"
    )




def expected_files() -> list[str]:
    """Archivos que requiere el periodo de config.py, más la nomenclatura."""
    return [f"datos_abiertos_{month}.csv" for month in expected_months()] + [
        NOMENCLATURA_FILENAME
    ]


def previous_urls() -> dict[str, str]:
    """URLs del manifiesto anterior, para no consultar el portal sin necesidad."""
    path = MANIFESTS_DIR / DOWNLOADS_MANIFEST
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["filename"]: item.get("source_url", "") for item in data}


def download(filename: str, url: str) -> Path:
    """Descarga un archivo a data/raw de forma atómica (.part -> nombre final)."""
    destination = RAW_DATA_DIR / filename
    temporary = destination.with_suffix(".csv.part")
    logger.info(f"Descargando {filename}")
    try:
        run_curl(url, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


# ---------------------------------------------------------------------------
# Cargadores para features.py y modelado
# ---------------------------------------------------------------------------


def cargar_viajes(columnas: list[str] | None = None) -> pd.DataFrame:
    """Lee los viajes normalizados del periodo de config.py.

    Filtra por inicio de viaje en [TRAIN_START, DATA_END_EXCLUSIVE). No incluye
    columnas de usuario porque preprocess.py ya las descartó.
    """
    archivos = sorted(PROCESSED_DATA_DIR.glob(VIAJES_PARQUET_GLOB))
    if not archivos:
        raise FileNotFoundError(
            f"No hay Parquet en {PROCESSED_DATA_DIR}. Corran: python -m mibici.dataset"
        )
    df = pd.concat([pd.read_parquet(a, columns=columnas) for a in archivos], ignore_index=True)
    if "inicio_viaje" in df.columns:
        inicio = pd.Timestamp(TRAIN_START)
        fin = pd.Timestamp(DATA_END_EXCLUSIVE)
        df = df.loc[(df["inicio_viaje"] >= inicio) & (df["inicio_viaje"] < fin)]
    return df.reset_index(drop=True)


def cargar_estaciones() -> pd.DataFrame:
    """Lee el catálogo normalizado (estacion_id, nombre, latitud, longitud, capacidad)."""
    ruta = PROCESSED_DATA_DIR / ESTACIONES_PARQUET
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corran: python -m mibici.dataset")
    return pd.read_parquet(ruta)


def obtener_data_version() -> str:
    """Devuelve el data_version del manifiesto procesado para registrarlo en MLflow."""
    ruta = MANIFESTS_DIR / PROCESSED_MANIFEST
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corran: python -m mibici.dataset")
    version = json.loads(ruta.read_text(encoding="utf-8")).get("data_version")
    if not version:
        raise ValueError(
            f"{ruta.name} no tiene data_version (formato anterior). Corran: python -m mibici.dataset"
        )
    return version


@app.command()
def main():
    """Descarga, verifica y normaliza los datos de MiBici en un solo paso."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    files = expected_files()
    urls = previous_urls()
    missing_locally = [f for f in files if not (RAW_DATA_DIR / f).exists()]

    # Solo se consulta el portal si falta algún archivo o su URL
    if missing_locally or any(not urls.get(f) for f in files):
        links = discover_files()
        not_published = [f for f in files if f not in links]
        if not_published:
            raise RuntimeError("No se encontraron enlaces para: " + ", ".join(not_published))
        urls.update({f: links[f] for f in files})

    manifest = []
    for filename in files:
        destination = RAW_DATA_DIR / filename
        if not destination.exists():
            download(filename, urls[filename])
        else:
            logger.info(f"Archivo existente: {filename}")

        required = (
            REQUIRED_STATION_COLUMNS if filename == NOMENCLATURA_FILENAME else REQUIRED_COLUMNS
        )
        info = verify_csv(destination, required)
        info["source_url"] = urls[filename]
        manifest.append(info)
        logger.success(f"{filename}: {info['records']:,} registros ({info['encoding']})")

    output = MANIFESTS_DIR / DOWNLOADS_MANIFEST
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.success(f"Manifiesto generado: {output}")

    generate_monthly_report(manifest)
    procesar(manifest)


if __name__ == "__main__":
    app()
