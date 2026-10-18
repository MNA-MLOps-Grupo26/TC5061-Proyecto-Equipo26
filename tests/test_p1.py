import pandas as pd

from mibici.modeling.p1 import calcular_metricas


def test_predicciones_perfectas():
    datos = pd.DataFrame({"salidas": [10, 20, 30]})
    resultado = calcular_metricas(datos, [10, 20, 30])
    assert resultado["mae"] == 0
    assert resultado["rmse"] == 0


def test_error_conocido():
    datos = pd.DataFrame({"salidas": [10, 20]})
    resultado = calcular_metricas(datos, [8, 22])
    assert resultado["mae"] == 2
    assert resultado["rmse"] == 2


def test_wape_pondera_por_volumen():
    datos = pd.DataFrame({"salidas": [10, 30]})
    resultado = calcular_metricas(datos, [12, 26])
    assert resultado["wape"] == 6 / 40
