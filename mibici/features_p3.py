"""Perfiles descriptivos P3; no ajusta agrupaciones ni inventa disponibilidad.

El denominador es un proxy calendario desde el dia posterior al primer evento
hasta el corte, no dias de operacion comprobados. La cobertura diaria de la
fuente y la disponibilidad de cada estacion siguen siendo desconocidas.
"""

import numpy as np
import pandas as pd

from mibici.config import P3_MIN_DIAS_ACTIVOS, P3_MIN_MOVIMIENTOS, TRAIN_START, VALIDATION_START
from mibici.features import validar_columnas_viajes


def columnas_perfil_p3() -> list[str]:
    """336 variables de forma; volumen, ubicacion y diagnosticos quedan aparte."""
    return [
        f"{tipo}_d{dia}_h{hora:02d}"
        for tipo in ("salidas", "llegadas")
        for dia in range(7)
        for hora in range(24)
    ]


def preparar_perfiles_p3(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    *,
    inicio=TRAIN_START,
    fin=VALIDATION_START,
    min_dias_activos: int = P3_MIN_DIAS_ACTIVOS,
    min_movimientos: int = P3_MIN_MOVIMIENTOS,
) -> pd.DataFrame:
    """Una fila por estacion (catalogo union eventos del periodo).

    Intervalo [inicio, fin), aplicado INDEPENDIENTEMENTE al inicio y fin de viaje.
    Las fechas son locales sin zona, como los Parquet existentes. No acepta zonas
    ambiguas ni horas invalidas. Excluye el primer dia observado como salvaguarda
    de historia parcial. Los promedios horarios usan dias calendario por weekday
    desde el dia siguiente, explicitamente un proxy NO disponibilidad observada.
    Normaliza salidas y llegadas por separado (cada bloque suma uno). Los ceros
    significan ausencia de eventos registrados, nunca demanda cero confirmada.
    Elegibilidad provisional: >=28 dias con eventos, >=100 movimientos, eventos
    en los siete weekdays y volumen positivo en ambas direcciones.
    """
    validar_columnas_viajes(viajes)
    if "estacion_id" not in estaciones or estaciones["estacion_id"].isna().any():
        raise ValueError("Catalogo requiere estacion_id sin nulos")
    if estaciones["estacion_id"].duplicated().any():
        raise ValueError("Catalogo con estacion_id duplicado")
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    if inicio >= fin or inicio != inicio.normalize() or fin != fin.normalize():
        raise ValueError("Periodo debe contener dias completos [inicio, fin)")
    if inicio < pd.Timestamp(TRAIN_START) or fin > pd.Timestamp(VALIDATION_START):
        raise ValueError("Esta preparacion solo admite fechas de entrenamiento")
    if min_dias_activos < 1 or min_movimientos < 1:
        raise ValueError("Umbrales deben ser positivos")

    eventos = []
    for tipo, estacion, momento in [
        ("salidas", "origen_id", "inicio_viaje"),
        ("llegadas", "destino_id", "fin_viaje"),
    ]:
        fechas = pd.to_datetime(viajes[momento], errors="raise")
        if fechas.isna().any() or fechas.dt.tz is not None:
            raise ValueError("Se requieren timestamps locales validos sin zona")
        mask = fechas.ge(inicio) & fechas.lt(fin)
        ids = viajes.loc[mask, estacion]
        if ids.isna().any():
            raise ValueError("Evento del periodo sin estacion_id")
        eventos.append(
            pd.DataFrame(
                {
                    "estacion_id": ids.to_numpy(),
                    "momento": fechas.loc[mask].to_numpy(),
                    "tipo": tipo,
                }
            )
        )
    eventos = pd.concat(eventos, ignore_index=True)
    eventos["fecha"] = eventos["momento"].dt.normalize()
    ids = pd.Index(estaciones["estacion_id"]).union(pd.Index(eventos["estacion_id"].unique()))
    meta = estaciones.set_index("estacion_id")
    grupos = dict(tuple(eventos.groupby("estacion_id", sort=False)))
    filas = []
    for estacion_id in ids:
        ev = grupos.get(estacion_id, eventos.iloc[:0])
        primera = ev["fecha"].min()
        ultima = ev["fecha"].max()
        desde = primera + pd.Timedelta(days=1) if pd.notna(primera) else pd.NaT
        util = ev.loc[ev["fecha"].ge(desde)] if pd.notna(desde) else ev
        calendario = (
            pd.date_range(desde, fin - pd.Timedelta(days=1), freq="D")
            if pd.notna(desde) and desde < fin
            else pd.DatetimeIndex([])
        )
        denominadores = np.bincount(calendario.dayofweek, minlength=7)
        activos = util["fecha"].drop_duplicates()
        dias_activos = np.bincount(activos.dt.dayofweek.to_numpy(dtype=int), minlength=7)
        row = {
            "estacion_id": estacion_id,
            "conjunto": "train",
            "periodo_inicio": inicio.date().isoformat(),
            "periodo_fin_exclusivo": fin.date().isoformat(),
            "primer_evento": primera,
            "ultimo_evento": ultima,
            "inicio_denominador_proxy": desde,
            "dias_denominador_proxy": len(calendario),
            "dias_con_eventos": len(activos),
            "eventos_primer_dia_excluidos": len(ev) - len(util),
            "cobertura_fuente_diaria": "desconocida",
            "disponibilidad_estacion": "desconocida",
            "tipo_denominador": "proxy_calendario_desde_dia_posterior_primer_evento",
            "interpretacion_ceros": "sin_eventos_registrados_no_demanda_cero_confirmada",
            "sin_catalogo": estacion_id not in meta.index,
        }
        for campo in ("nombre", "latitud", "longitud", "capacidad"):
            row[campo] = (
                meta.at[estacion_id, campo]
                if estacion_id in meta.index and campo in meta
                else np.nan
            )
        row["mapeable"] = bool(
            pd.notna(row["latitud"])
            and pd.notna(row["longitud"])
            and -90 <= row["latitud"] <= 90
            and -180 <= row["longitud"] <= 180
        )
        for dia in range(7):
            row[f"dias_proxy_d{dia}"] = int(denominadores[dia])
            row[f"dias_con_eventos_d{dia}"] = int(dias_activos[dia])
        razones = []
        if ev.empty:
            razones.append("sin_eventos_entrenamiento")
        if len(activos) < min_dias_activos:
            razones.append("pocos_dias_con_eventos")
        if len(util) < min_movimientos:
            razones.append("pocos_movimientos")
        if (dias_activos == 0).any():
            razones.append("weekday_sin_eventos")
        for tipo in ("salidas", "llegadas"):
            sub = util.loc[util["tipo"].eq(tipo)]
            row[f"total_{tipo}"] = len(sub)
            row[f"{tipo}_media_dia_proxy"] = (
                len(sub) / len(calendario) if len(calendario) else np.nan
            )
            counts = sub.groupby([sub["momento"].dt.dayofweek, sub["momento"].dt.hour]).size()
            matrix = np.zeros((7, 24), dtype=float)
            for (dia, hora), n in counts.items():
                matrix[int(dia), int(hora)] = n
            medias = np.divide(
                matrix,
                denominadores[:, None],
                out=np.zeros_like(matrix),
                where=denominadores[:, None] > 0,
            )
            forma = medias / medias.sum() if medias.sum() > 0 else np.full((7, 24), np.nan)
            for dia in range(7):
                for hora in range(24):
                    row[f"{tipo}_d{dia}_h{hora:02d}"] = forma[dia, hora]
            if sub.empty:
                razones.append(f"sin_{tipo}")
        row["total_movimientos"] = len(util)
        row["elegible"] = not razones
        row["motivo_exclusion"] = ";".join(razones)
        row["advertencia"] = "perfil_provisional_cobertura_y_disponibilidad_no_confirmadas"
        filas.append(row)
    if not filas:
        return pd.DataFrame(
            columns=["estacion_id", "elegible", "motivo_exclusion"] + columnas_perfil_p3()
        )
    return pd.DataFrame(filas).sort_values("estacion_id").reset_index(drop=True)
