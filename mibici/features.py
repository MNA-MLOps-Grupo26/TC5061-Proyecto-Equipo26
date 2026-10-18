
"""
Módulo de funciones reutilizables para la ingeniería de variables de MiBici.

Funciones disponibles
--------------------

1. particionar_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]
   Divide los viajes en tres conjuntos temporales:
   - train: entrenamiento.
   - validation: validación.
   - test: prueba.

   Utiliza la columna inicio_viaje y las fechas definidas en config.py.
   Los intervalos son cerrados por la izquierda y abiertos por la derecha,
   evitando traslapes entre conjuntos.

2. validar_columnas_viajes(df: pd.DataFrame) -> None
   Comprueba que el DataFrame contenga las columnas requeridas:
   - viaje_id
   - inicio_viaje
   - fin_viaje
   - origen_id
   - destino_id

   Genera un ValueError si falta alguna columna.

3. agregar_variables_temporales(
       df: pd.DataFrame,
       columna_fecha: str
   ) -> pd.DataFrame
   Genera variables de calendario a partir de una columna temporal:
   - fecha: fecha del registro.
   - hora: hora del día (0-23).
   - dia_semana: día de la semana (0=lunes, 6=domingo).
   - mes: mes del año (1-12).
   - anio: año del registro.
   - es_fin_semana: 1 si es sábado o domingo; 0 en caso contrario.

   Devuelve una copia del DataFrame original con las nuevas variables,
   sin modificar el DataFrame de entrada.

4. Panel estación-día para P1 y P2 (sin fuga temporal)
   contar_por_estacion_dia -> construir_panel -> agregar_rezagos ->
   agregar_calendario -> agregar_estaciones -> asignar_conjunto.
   Cada fila es (estación, día objetivo) y solo usa viajes de días anteriores.

5. Etiqueta aproximada de riesgo para P2
   calcular_indicador_riesgo: saldo acumulado (salidas - llegadas) desde las
   05:00, dividido entre la capacidad, y primera categoría alcanzada en el día.

6. Constructores de conjuntos: dataset_p1 y dataset_p2.

Consideraciones
---------------
- Las rutas, fechas y parámetros temporales se obtienen de config.py.
- Las funciones son reutilizables para la preparación de datos de P1, P2 y P3.
- Las variables temporales deben construirse únicamente con información
  disponible en el momento de la predicción para evitar fuga de información.
- Este módulo prepara y particiona los datos; no entrena modelos.
"""


from loguru import logger
import numpy as np
import pandas as pd

from mibici.config import (
    DATA_END_EXCLUSIVE,
    P2_CLASES,
    P2_HORA_INICIO_DIA,
    P2_OCUPACION_INICIAL,
    PROCESSED_DATA_DIR,
    SEMANAS_BASELINE_P1,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
    VENTANAS_MEDIA,
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


# ---------------------------------------------------------------------------
# Panel estación-día (P1 y P2)
# ---------------------------------------------------------------------------


def asignar_dia(fechas: pd.Series, hora_inicio: int = 0) -> pd.Series:
    """Asigna cada instante a su día operativo.

    Con hora_inicio=0 el día va de medianoche a medianoche (P1). Con
    hora_inicio=5 un viaje a las 02:00 pertenece al día anterior, porque el
    sistema reinicia a las 05:00 (P2).
    """
    return (fechas - pd.Timedelta(hours=hora_inicio)).dt.normalize()


def contar_por_estacion_dia(viajes: pd.DataFrame, hora_inicio: int = 0) -> pd.DataFrame:
    """Cuenta salidas (por origen e inicio) y llegadas (por destino y fin) por estación y día.

    Pertinencia: P1 predice salidas; las llegadas y el saldo alimentan a P2,
    porque una estación con más salidas que llegadas tiende a vaciarse.
    """
    validar_columnas_viajes(viajes)
    salidas = (
        viajes.assign(fecha=asignar_dia(viajes["inicio_viaje"], hora_inicio))
        .groupby(["origen_id", "fecha"])
        .size()
        .rename("salidas")
    )
    salidas.index.names = ["estacion_id", "fecha"]
    llegadas = (
        viajes.assign(fecha=asignar_dia(viajes["fin_viaje"], hora_inicio))
        .groupby(["destino_id", "fecha"])
        .size()
        .rename("llegadas")
    )
    llegadas.index.names = ["estacion_id", "fecha"]
    conteos = pd.concat([salidas, llegadas], axis=1).fillna(0).astype("int64")
    return conteos.reset_index()


def construir_panel(conteos: pd.DataFrame) -> pd.DataFrame:
    """Completa la rejilla estación-día con ceros donde hay cobertura.

    Una estación entra al panel el día siguiente a su primer viaje observado:
    antes de eso no sabíamos que existía, así que no se inventan ceros. A partir
    de ahí, un día sin viajes es un cero real. No se eliminan días por una
    inactividad que solo se conocería después (eso sería información futura);
    en su lugar, ``dias_desde_ultimo_viaje`` deja que el modelo la detecte.
    """
    inicio = pd.Timestamp(TRAIN_START)
    fin = pd.Timestamp(DATA_END_EXCLUSIVE) - pd.Timedelta(days=1)
    conteos = conteos.loc[(conteos["fecha"] >= inicio) & (conteos["fecha"] <= fin)]

    primer_dia = conteos.groupby("estacion_id")["fecha"].min()
    filas = [
        pd.DataFrame(
            {
                "estacion_id": estacion,
                # Desde el primer día observado: se conserva para calcular rezagos,
                # y se marca para excluirlo como objetivo.
                "fecha": pd.date_range(dia, fin, freq="D"),
            }
        )
        for estacion, dia in primer_dia.items()
    ]
    panel = pd.concat(filas, ignore_index=True)
    panel = panel.merge(conteos, on=["estacion_id", "fecha"], how="left")
    panel[["salidas", "llegadas"]] = panel[["salidas", "llegadas"]].fillna(0).astype("int64")
    panel["es_primer_dia"] = panel["fecha"].eq(panel["estacion_id"].map(primer_dia))
    return panel.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)


def agregar_rezagos(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega rezagos y promedios móviles calculados solo con días anteriores.

    Todas las variables usan ``shift(1)``: para el día t solo se ven t-1 y antes.
    El panel es diario y completo por estación, así que shift(k) es k días.

    - salidas_lag1, llegadas_lag1, saldo_lag1: último día conocido.
    - salidas_lag7: mismo día de la semana anterior (estacionalidad semanal).
    - salidas_media{7,28}, llegadas_media{7,28}: nivel reciente de demanda.
    - salidas_mismo_dia_media4: promedio de t-7, t-14, t-21 y t-28 (baseline P1).
    - dias_historia: días desde el primer viaje (estaciones nuevas tienen poca).
    - dias_desde_ultimo_viaje: detecta cierres o estaciones sin uso reciente.
    """
    resultado = panel.sort_values(["estacion_id", "fecha"]).copy()
    grupo = resultado.groupby("estacion_id", sort=False)

    resultado["saldo"] = resultado["salidas"] - resultado["llegadas"]
    for columna in ["salidas", "llegadas", "saldo"]:
        resultado[f"{columna}_lag1"] = grupo[columna].shift(1)
    resultado["salidas_lag7"] = grupo["salidas"].shift(7)

    for ventana in VENTANAS_MEDIA:
        for columna in ["salidas", "llegadas"]:
            resultado[f"{columna}_media{ventana}"] = grupo[columna].transform(
                lambda s, w=ventana: s.shift(1).rolling(w, min_periods=1).mean()
            )

    semanas = [grupo["salidas"].shift(7 * k) for k in range(1, SEMANAS_BASELINE_P1 + 1)]
    resultado[f"salidas_mismo_dia_media{SEMANAS_BASELINE_P1}"] = pd.concat(
        semanas, axis=1
    ).mean(axis=1)

    primer_dia = grupo["fecha"].transform("min")
    resultado["dias_historia"] = (resultado["fecha"] - primer_dia).dt.days

    activo = (resultado["salidas"] + resultado["llegadas"]) > 0
    ultimo_activo = resultado["fecha"].where(activo)
    ultimo_activo = ultimo_activo.groupby(resultado["estacion_id"]).transform(
        lambda s: s.ffill().shift(1)
    )
    resultado["dias_desde_ultimo_viaje"] = (resultado["fecha"] - ultimo_activo).dt.days

    return resultado.drop(columns="saldo")


def agregar_calendario(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega el calendario del día objetivo, que sí se conoce de antemano."""
    resultado = panel.copy()
    resultado["dia_semana"] = resultado["fecha"].dt.dayofweek
    resultado["es_fin_semana"] = (resultado["dia_semana"] >= 5).astype("int8")
    resultado["mes"] = resultado["fecha"].dt.month
    resultado["dia_mes"] = resultado["fecha"].dt.day
    return resultado


def agregar_estaciones(panel: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Une capacidad y coordenadas del catálogo (variables estáticas por estación).

    Supuesto: la capacidad de la nomenclatura más reciente aplica a todo el
    periodo. Estaciones sin catálogo quedan con capacidad nula.
    """
    columnas = ["estacion_id", "capacidad", "latitud", "longitud"]
    return panel.merge(estaciones[columnas], on="estacion_id", how="left")


def asignar_conjunto(panel: pd.DataFrame) -> pd.DataFrame:
    """Asigna train, validation o test según la fecha objetivo (fechas de config.py).

    Se parte por fecha objetivo y no por viaje: los rezagos del primer día de
    validación pueden usar días de julio, que ya ocurrieron al pronosticar.
    """
    resultado = panel.copy()
    condiciones = [
        resultado["fecha"] < pd.Timestamp(VALIDATION_START),
        resultado["fecha"] < pd.Timestamp(TEST_START),
    ]
    resultado["conjunto"] = np.select(condiciones, ["train", "validation"], default="test")
    return resultado


# ---------------------------------------------------------------------------
# Indicador de riesgo (etiqueta de P2)
# ---------------------------------------------------------------------------


def umbrales_riesgo(ocupacion_inicial: float = P2_OCUPACION_INICIAL) -> tuple[float, float]:
    """Umbrales del saldo acumulado / capacidad para una ocupación inicial supuesta.

    Si la estación empieza al ``ocupacion_inicial`` de su capacidad, se vacía
    cuando el saldo neto de salidas llega a esa fracción y se llena cuando las
    llegadas netas cubren el resto. Con 0.5 los umbrales son +0.5 y -0.5.
    """
    return ocupacion_inicial, -(1 - ocupacion_inicial)


def calcular_indicador_riesgo(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    hora_inicio: int = P2_HORA_INICIO_DIA,
    ocupacion_inicial: float = P2_OCUPACION_INICIAL,
) -> pd.DataFrame:
    """Calcula la categoría de riesgo aproximado por estación y día operativo.

    Recorre los eventos de cada estación en orden: una salida suma 1 y una
    llegada resta 1. El saldo acumulado desde las 05:00 se divide entre la
    capacidad. La etiqueta es el primer umbral que se alcanza en el día:
    ``sin_bicis`` (>= umbral superior), ``sin_anclajes`` (<= umbral inferior)
    o ``sin_riesgo``. No reconstruye inventario real ni rebalanceo del operador.

    Returns:
        estacion_id, fecha, categoria, indicador_max, indicador_min.
        Estaciones sin capacidad conocida no reciben etiqueta.
    """
    validar_columnas_viajes(viajes)
    umbral_bicis, umbral_anclajes = umbrales_riesgo(ocupacion_inicial)

    eventos = pd.concat(
        [
            pd.DataFrame(
                {"estacion_id": viajes["origen_id"], "momento": viajes["inicio_viaje"], "delta": 1}
            ),
            pd.DataFrame(
                {"estacion_id": viajes["destino_id"], "momento": viajes["fin_viaje"], "delta": -1}
            ),
        ],
        ignore_index=True,
    )
    eventos["fecha"] = asignar_dia(eventos["momento"], hora_inicio)
    capacidad = estaciones.set_index("estacion_id")["capacidad"]
    eventos["capacidad"] = eventos["estacion_id"].map(capacidad)
    eventos = eventos.dropna(subset=["capacidad"])

    eventos = eventos.sort_values(["estacion_id", "fecha", "momento"], kind="stable")
    eventos["indicador"] = (
        eventos.groupby(["estacion_id", "fecha"], sort=False)["delta"].cumsum()
        / eventos["capacidad"]
    )

    claves = ["estacion_id", "fecha"]
    resumen = eventos.groupby(claves).agg(
        indicador_max=("indicador", "max"), indicador_min=("indicador", "min")
    )

    cruces = eventos.loc[
        (eventos["indicador"] >= umbral_bicis) | (eventos["indicador"] <= umbral_anclajes)
    ]
    primer_cruce = cruces.groupby(claves).head(1).set_index(claves)["indicador"]
    categoria = pd.Series(P2_CLASES[0], index=resumen.index)
    categoria.loc[primer_cruce.index[primer_cruce >= umbral_bicis]] = P2_CLASES[1]
    categoria.loc[primer_cruce.index[primer_cruce <= umbral_anclajes]] = P2_CLASES[2]
    resumen["categoria"] = categoria

    return resumen.reset_index()


# ---------------------------------------------------------------------------
# Constructores de conjuntos por pregunta
# ---------------------------------------------------------------------------

VARIABLES_P1 = [
    "salidas_lag1",
    "llegadas_lag1",
    "saldo_lag1",
    "salidas_lag7",
    "salidas_media7",
    "salidas_media28",
    "llegadas_media7",
    "llegadas_media28",
    f"salidas_mismo_dia_media{SEMANAS_BASELINE_P1}",
    "dias_historia",
    "dias_desde_ultimo_viaje",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "dia_mes",
    "capacidad",
    "latitud",
    "longitud",
]

VARIABLES_P2 = VARIABLES_P1 + [
    "indicador_max_lag1",
    "indicador_min_lag1",
    "frac_sin_bicis_28",
    "frac_sin_anclajes_28",
]


def _panel_base(viajes: pd.DataFrame, estaciones: pd.DataFrame, hora_inicio: int) -> pd.DataFrame:
    """Panel común de P1 y P2 con rezagos, calendario, estaciones y conjunto."""
    panel = construir_panel(contar_por_estacion_dia(viajes, hora_inicio))
    panel = agregar_rezagos(panel)
    panel = agregar_calendario(panel)
    panel = agregar_estaciones(panel, estaciones)
    panel = asignar_conjunto(panel)
    # El primer día de cada estación solo aporta historia; no es objetivo.
    return panel.loc[~panel["es_primer_dia"]].drop(columns="es_primer_dia")


def dataset_p1(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Conjunto de P1: una fila por estación y día; objetivo = salidas del día.

    El día va de medianoche a medianoche y todas las variables vienen de días
    anteriores al pronosticado.
    """
    return _panel_base(viajes, estaciones, hora_inicio=0).reset_index(drop=True)


def dataset_p2(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Conjunto de P2: una fila por estación y día operativo; objetivo = categoría.

    Agrega a las variables de P1 (calculadas con día operativo desde las 05:00)
    los extremos del indicador del día anterior, la frecuencia de cada riesgo
    en los 28 días previos y la categoría de hace 7 días (baseline).
    """
    panel = construir_panel(contar_por_estacion_dia(viajes, P2_HORA_INICIO_DIA))
    indicador = calcular_indicador_riesgo(viajes, estaciones)
    panel = panel.merge(indicador, on=["estacion_id", "fecha"], how="left")

    # Días con cobertura y sin viajes: el saldo no se mueve, no hay riesgo.
    panel["categoria"] = panel["categoria"].fillna(P2_CLASES[0])
    panel[["indicador_max", "indicador_min"]] = panel[["indicador_max", "indicador_min"]].fillna(0)

    grupo = panel.groupby("estacion_id", sort=False)
    panel["indicador_max_lag1"] = grupo["indicador_max"].shift(1)
    panel["indicador_min_lag1"] = grupo["indicador_min"].shift(1)
    panel["categoria_lag7"] = grupo["categoria"].shift(7)
    for clase, nombre in [(P2_CLASES[1], "frac_sin_bicis_28"), (P2_CLASES[2], "frac_sin_anclajes_28")]:
        panel[nombre] = (
            panel["categoria"]
            .eq(clase)
            .astype(float)
            .groupby(panel["estacion_id"])
            .transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
        )

    panel = agregar_rezagos(panel)
    panel = agregar_calendario(panel)
    panel = agregar_estaciones(panel, estaciones)
    panel = asignar_conjunto(panel)
    panel = panel.loc[~panel["es_primer_dia"] & panel["capacidad"].notna()]
    return panel.drop(columns="es_primer_dia").reset_index(drop=True)


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
