
"""Descarga y verificación reproducible de los CSV mensuales de MiBici."""

import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin

from loguru import logger
import typer

from mibici.config import (
    DATA_END_EXCLUSIVE,
    MANIFESTS_DIR,
    MIBICI_PORTAL,
    RAW_DATA_DIR,
    TRAIN_START,
)

app = typer.Typer()

REQUIRED_COLUMNS = {
    "Viaje_Id",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
}

FILE_PATTERN = re.compile(
    r"/site/assets/files/\d+/datos_abiertos_(\d{4})_(\d{2})\.csv"
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
        year, month = match.groups()
        key = f"{year}_{month}"
        links[key] = urljoin(MIBICI_PORTAL, match.group(0))

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


def verify_csv(path: Path) -> dict:
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

        missing = REQUIRED_COLUMNS - set(columns)

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




@app.command()
def main():
    """Descarga los CSV mensuales y genera su manifiesto."""
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFESTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    links = discover_files()
    months = list(expected_months())

    missing = [
        month
        for month in months
        if month not in links
    ]

    if missing:
        raise RuntimeError(
            "No se encontraron enlaces para: "
            + ", ".join(missing)
        )

    manifest = []

    for month in months:
        filename = f"datos_abiertos_{month}.csv"
        destination = RAW_DATA_DIR / filename
        url = links[month]

        if not destination.exists():
            logger.info(f"Descargando {filename}")

            temporary = destination.with_suffix(
                ".csv.part"
            )

            try:
                run_curl(url, temporary)
                temporary.replace(destination)

            finally:
                temporary.unlink(missing_ok=True)

        else:
            logger.info(
                f"Archivo existente: {filename}"
            )

        info = verify_csv(destination)
        info["source_url"] = url

        manifest.append(info)

        logger.success(
            f"{filename}: "
            f"{info['records']:,} registros "
            f"({info['encoding']})"
        )

    output = MANIFESTS_DIR / "mibici_downloads.json"

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    logger.success(
        f"Manifiesto generado: {output}"
    )
    generate_monthly_report(manifest)

if __name__ == "__main__":
    app()
