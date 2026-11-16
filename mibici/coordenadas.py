"""Conversión determinista de coordenadas del catálogo (texto -> float64).

``pandas.read_csv`` convierte decimales con ``precise_xstrtod`` (tokenizador C), que acumula
dígitos con ``number = number * 10. + digit``. Con 17 dígitos significativos el acumulado
supera 2**53, de modo que el resultado depende de si el compilador fusiona esa operación en una
FMA (un redondeo) o no (dos redondeos). En el catálogo de MiBici eso cambia la latitud de 15
estaciones en 1 a 2 ulp según la plataforma, y v1 tiene umbrales exactamente en esos valores
(ver docs/week9-evidence/coordenadas-20261115).

Este módulo reproduce ese mismo algoritmo con la contracción explícita, en Python puro (cada
operación de ``float`` redondea por separado; ``math.fma`` redondea una sola vez), así que da
el mismo resultado en cualquier sistema:

- ``"sin_fma"``: dos redondeos por dígito. Es la representación del panel del reporte de
  semana 9 y del entrenamiento de v3.
- ``"con_fma"``: un redondeo por dígito. Es la representación de ``read_csv`` en macOS ARM64,
  del manifiesto ``estaciones.parquet`` y del entrenamiento del v1 oficial.
- ``"pandas"``: el comportamiento original, ``pd.to_numeric`` dependiente de la plataforma.
"""

import math
import re

import numpy as np
import pandas as pd

MODOS = ("pandas", "sin_fma", "con_fma")
MAX_DIGITOS = 17  # igual que precise_xstrtod
_DECIMAL = re.compile(r"^\s*([+-]?)(\d*)(?:\.(\d*))?\s*$")


def convertir(texto: str, modo: str) -> float:
    """Convierte un decimal sin exponente como lo hace precise_xstrtod, con contracción fija."""
    if modo not in ("sin_fma", "con_fma"):
        raise ValueError(f"Modo de conversión no determinista o desconocido: {modo}")
    match = _DECIMAL.match(texto)
    if match is None or not (match.group(2) or match.group(3)):
        raise ValueError(f"Coordenada con formato no soportado: {texto!r}")
    signo, entera, fraccion = match.group(1), match.group(2), match.group(3) or ""
    numero, digitos, exponente, decimales = 0.0, 0, 0, 0
    for parte, es_fraccion in ((entera, False), (fraccion, True)):
        for caracter in parte:
            if digitos >= MAX_DIGITOS:
                if not es_fraccion:
                    exponente += 1
                continue
            digito = float(ord(caracter) - 48)
            if modo == "con_fma":
                numero = math.fma(numero, 10.0, digito)
            else:
                numero = numero * 10.0 + digito
            digitos += 1
            decimales += es_fraccion
    exponente -= decimales
    if signo == "-":
        numero = -numero
    if not -22 <= exponente <= 22:  # 10**k exacto en float64 solo hasta k = 22
        raise ValueError(f"Coordenada fuera del rango soportado: {texto!r}")
    if exponente > 0:
        return numero * float(10**exponente)
    return numero / float(10**-exponente)


def a_numerico(serie: pd.Series, modo: str) -> pd.Series:
    """Equivalente a ``pd.to_numeric(serie, errors="coerce")`` con conversión determinista.

    Los valores vacíos o no numéricos quedan como NaN, igual que con ``errors="coerce"``.
    """
    if modo not in MODOS:
        raise ValueError(f"Modo desconocido: {modo}")
    if modo == "pandas":
        return pd.to_numeric(serie, errors="coerce")
    valores = []
    for valor in serie:
        if pd.isna(valor):
            valores.append(np.nan)
            continue
        texto = str(valor)
        if _DECIMAL.match(texto) and re.search(r"\d", texto):
            valores.append(convertir(texto, modo))
        elif np.isnan(pd.to_numeric(pd.Series([texto]), errors="coerce").iloc[0]):
            valores.append(np.nan)
        else:  # p. ej. notación científica: no se adivina, se rechaza
            raise ValueError(f"Coordenada con formato no soportado: {texto!r}")
    return pd.Series(valores, index=serie.index, dtype="float64")
