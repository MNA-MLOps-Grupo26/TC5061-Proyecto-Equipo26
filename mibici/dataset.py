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
    VALIDATION_START,
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


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def leer_verificado(nombre, records):
    path = PROCESSED_DATA_DIR / nombre
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if sha != records[nombre]["sha256"]:
        raise ValueError(f"SHA no coincide: {nombre}")
    frame = pd.read_parquet(
        path, columns=["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )
    if len(frame) != records[nombre]["records"]:
        raise ValueError(f"Conteo no coincide: {nombre}")
    return frame, {"archivo": nombre, "registros": len(frame), "sha256": sha}


def leer_catalogo_verificado(ruta, registro: dict) -> pd.DataFrame:
    """Lee el catálogo de estaciones solo si su hash y su conteo coinciden con el manifiesto.

    Las coordenadas del mapa salen de este archivo; un catálogo distinto al de la
    ingesta registrada cambiaría el mapa sin que la versión de datos lo reflejara.
    """
    if sha256(ruta) != registro["sha256"]:
        raise ValueError(
            f"SHA-256 de {ruta.name} no coincide con el manifiesto. "
            "Vuelvan a correr: python -m mibici.dataset"
        )
    estaciones = pd.read_parquet(ruta)
    if len(estaciones) != registro["records"]:
        raise ValueError(
            f"{ruta.name} tiene {len(estaciones)} registros; el manifiesto indica "
            f"{registro['records']}."
        )
    return estaciones


def cargar_entrenamiento_revision():
    """Lee los meses de entrenamiento verificando SHA-256 y conteo contra el manifiesto.

    Devuelve los viajes, la cobertura verificada por archivo y el manifiesto.
    """
    manifest = json.loads((MANIFESTS_DIR / "mibici_processed.json").read_text(encoding="utf-8"))
    records = {f["filename"]: f for f in manifest["files"]}
    frames, coverage = [], []
    columns = ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    for month in pd.period_range(
        TRAIN_START, pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=1), freq="M"
    ):
        name = f"datos_abiertos_{month.year}_{month.month:02d}.parquet"
        path = PROCESSED_DATA_DIR / name
        record = records.get(name)
        if not path.is_file() or record is None:
            raise ValueError(f"Fuente de entrenamiento ausente: {name}")
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"SHA-256 no coincide: {name}")
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != record["records"]:
            raise ValueError(f"Conteo no coincide: {name}")
        frames.append(frame)
        coverage.append(
            {
                "archivo": name,
                "registros": len(frame),
                "sha256_verificado": True,
                "cobertura_diaria_confirmada": False,
                "disponibilidad_estacion_confirmada": False,
            }
        )
    return pd.concat(frames, ignore_index=True), coverage, manifest


def exportar_revision(destino: Path, perfiles, coverage, manifest) -> None:
    """Guarda la revisión de la preparación de P3: perfiles, integridad, excluidas y resumen."""
    if destino.exists():
        raise FileExistsError(f"No se sobrescribe {destino}")
    destino.mkdir(parents=True, exist_ok=False)
    perfiles.to_parquet(destino / "perfiles_entrenamiento.parquet", index=False)
    perfiles.to_csv(destino / "perfiles_entrenamiento.csv", index=False)
    pd.DataFrame(coverage).to_csv(destino / "integridad_fuentes.csv", index=False)
    compact = [
        "estacion_id",
        "nombre",
        "primer_evento",
        "dias_denominador_proxy",
        "dias_con_eventos",
        "total_salidas",
        "total_llegadas",
        "total_movimientos",
        "mapeable",
        "elegible",
        "motivo_exclusion",
    ]
    for tipo in ("salidas", "llegadas"):
        cols = [f"{tipo}_d{d}_h{h:02d}" for d in range(7) for h in range(24)]
        perfiles[f"pico_{tipo}"] = perfiles[cols].apply(
            lambda r: r.idxmax() if r.notna().any() else "sin_perfil", axis=1
        )
        perfiles[f"porcentaje_pico_{tipo}"] = perfiles[cols].max(axis=1) * 100
        compact += [f"pico_{tipo}", f"porcentaje_pico_{tipo}"]
    muestra = pd.concat(
        [perfiles.loc[perfiles.elegible].head(8), perfiles.loc[~perfiles.elegible].head(4)]
    )
    muestra[compact].to_csv(destino / "vista_previa.csv", index=False)
    excluidos = perfiles.loc[~perfiles.elegible, compact]
    excluidos.to_csv(destino / "estaciones_excluidas.csv", index=False)
    razones = (
        perfiles.loc[~perfiles.elegible, "motivo_exclusion"]
        .str.split(";")
        .explode()
        .value_counts()
    )
    resumen = {
        "periodo_inicio": str(TRAIN_START),
        "periodo_fin_exclusivo": str(VALIDATION_START),
        "data_version": manifest["data_version"],
        "estaciones": len(perfiles),
        "elegibles_provisionales": int(perfiles.elegible.sum()),
        "excluidas": len(excluidos),
        "sin_mapa": int((~perfiles.mapeable).sum()),
        "motivos_no_exclusivos": razones.to_dict(),
        "archivos_mensuales_verificados": len(coverage),
        "variables_forma": 336,
        "cobertura_diaria": "desconocida",
        "disponibilidad_estaciones": "desconocida",
    }
    (destino / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    preview = muestra[compact].to_string(index=False)
    texto = f"""REVISION P3: PREPARACION, SIN CLUSTERING
Periodo: {TRAIN_START} <= timestamp de cada evento < {VALIDATION_START}.
{json.dumps(resumen, ensure_ascii=False, indent=2)}

Cada bloque de 168 valores es una proporcion (multiplicar por 100 para porcentaje).
d0=lunes, d6=domingo; horas 00-23 locales. Volumen queda separado.
El primer dia con eventos se excluye como historia parcial; sus eventos se cuentan
en eventos_primer_dia_excluidos. Los totales del perfil excluyen ese dia.
Se verificaron los 12 archivos y sus SHA-256/conteos frente al manifiesto.
Esto NO demuestra cobertura diaria completa ni disponibilidad de cada estacion.
Los promedios usan un proxy calendario desde el dia posterior al primer evento.
Ceros solo indican ausencia de registros en esa celda; NO demanda cero comprobada.
No se infieren cierres por inactividad. La elegibilidad es PROVISIONAL:
28 dias con eventos, 100 movimientos, todos los weekdays presentes y ambas direcciones.
Estaciones sin coordenadas pueden tener perfil pero no mapa. No se usa capacidad
como filtro, ni coordenadas/volumen como variables de forma.
No se leyeron meses de validacion ni prueba. Llegadas se cortan por fin_viaje;
salidas por inicio_viaje. Puede faltar flujo entrante de viajes iniciados antes del
primer archivo disponible; no se reconstruye ni inventa ese limite izquierdo.
No se ajustaron modelos ni se crearon grupos.

VISTA PREVIA REAL
{preview}

Lista completa de excluidas: estaciones_excluidas.csv.
Los conteos por motivo pueden sumar mas que las estaciones excluidas.
"""
    (destino / "REVISION.txt").write_text(texto, encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(preview)


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
