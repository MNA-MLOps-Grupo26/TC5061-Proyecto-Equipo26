"""Exportacion de diagnosticos P3 sin ajustar modelos ni registrar runs."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from mibici.config import MANIFESTS_DIR, PROCESSED_DATA_DIR, TRAIN_START, VALIDATION_START
from mibici.dataset import cargar_estaciones
from mibici.features_p3 import preparar_perfiles_p3


def generar_revision(destino: Path) -> None:
    """Verifica integridad mensual, prepara solo entrenamiento y exporta revision."""
    if destino.exists():
        raise FileExistsError(f"No se sobrescribe {destino}")
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
    perfiles = preparar_perfiles_p3(pd.concat(frames, ignore_index=True), cargar_estaciones())
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


if __name__ == "__main__":
    import sys

    generar_revision(Path(sys.argv[1]))
