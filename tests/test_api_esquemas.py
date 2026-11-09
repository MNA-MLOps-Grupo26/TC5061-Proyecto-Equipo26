"""Contrato de entrada de la API P1: peticiones válidas y todos los casos de 422."""

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from api.esquemas import VARIABLES_P1
from api.main import crear_app
from api.modelo import ModeloServido
from mibici.features import VARIABLES_P1 as VARIABLES_P1_PROYECTO

REPO = Path(__file__).resolve().parents[1]
EVID = REPO / "docs" / "evidencia-api"
VALIDO = json.loads((EVID / "peticion-predict.json").read_text())
LOTE = json.loads((EVID / "peticion-batch.json").read_text())


class ModeloFalso:
    def predict(self, datos):
        assert list(datos.columns) == VARIABLES_P1
        return np.full(len(datos), 10.0)


@pytest.fixture
def cliente():
    modelo = ModeloServido(ModeloFalso(), {"nombre": "falso", "version": "1", "run_id": "r"}, 8)
    with TestClient(crear_app(lambda: modelo)) as c:
        yield c


def con(**cambios):
    cuerpo = copy.deepcopy(VALIDO)
    cuerpo.update(cambios)
    return cuerpo


def sin(clave):
    cuerpo = copy.deepcopy(VALIDO)
    del cuerpo[clave]
    return cuerpo


def test_variables_iguales_al_proyecto():
    assert VARIABLES_P1 == VARIABLES_P1_PROYECTO


def test_peticion_valida(cliente):
    r = cliente.post("/predict", json=VALIDO)
    assert r.status_code == 200
    assert r.json()["viajes_esperados"] == 10.0
    assert r.json()["estacion_id"] == 21


def test_nulos_admitidos(cliente):
    cuerpo = con(
        salidas_lag7=None,
        salidas_mismo_dia_media4=None,
        capacidad=None,
        latitud=None,
        longitud=None,
    )
    assert cliente.post("/predict", json=cuerpo).status_code == 200


def test_sin_estacion_ni_fecha(cliente):
    cuerpo = copy.deepcopy(VALIDO)
    del cuerpo["estacion_id"], cuerpo["fecha"]
    assert cliente.post("/predict", json=cuerpo).status_code == 200


def test_29_de_febrero_sin_fecha(cliente):
    cuerpo = con(mes=2, dia_mes=29)
    del cuerpo["fecha"]
    assert cliente.post("/predict", json=cuerpo).status_code == 200


CASOS_422 = {
    "clave_ausente": sin("salidas_media7"),
    "clave_nullable_ausente": sin("capacidad"),
    "null_no_admitido": con(salidas_lag1=None),
    "fuera_de_rango_negativo": con(salidas_media7=-1.0),
    "fuera_de_rango_mes": con(mes=13),
    "fuera_de_rango_latitud": con(latitud=19.0),
    "capacidad_cero": con(capacidad=0.0),
    "estacion_id_cero": con(estacion_id=0),
    "campo_desconocido": con(temperatura=25.0),
    "texto_en_numero": con(salidas_media7="122.7"),
    "booleano_en_numero": con(salidas_media7=True),
    "booleano_en_entero": con(dia_semana=True),
    "decimal_en_entero": con(dias_historia=410.0),
    "decimal_en_conteo": con(salidas_lag1=146.5, saldo_lag1=5.5),
    "fecha_formato": con(fecha="15/09/2026"),
    "fecha_inexistente": con(fecha="2026-02-30"),
    # Reglas cruzadas 1 a 7.
    "regla1_fin_de_semana": con(es_fin_semana=1),
    "regla2_dia_inexistente": con(mes=4, dia_mes=31, fecha=None),
    "regla3_fecha_distinta": con(fecha="2026-09-16"),
    "regla4_saldo": con(saldo_lag1=4.0),
    "regla5_dias": con(dias_desde_ultimo_viaje=411),
    "regla6_nulos_lag7": con(salidas_lag7=None),
    "regla7_nulos_estacion": con(capacidad=None),
}


@pytest.mark.parametrize("cuerpo", CASOS_422.values(), ids=CASOS_422.keys())
def test_rechazo_422(cliente, cuerpo):
    assert cliente.post("/predict", json=cuerpo).status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nan_e_infinito_dan_422_no_500(cliente, literal):
    texto = json.dumps(VALIDO).replace(
        '"salidas_media7": 122.714286', f'"salidas_media7": {literal}'
    )
    assert literal in texto
    r = cliente.post("/predict", content=texto, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == "salidas_media7"


def test_lote_valido_orden_y_longitud(cliente):
    r = cliente.post("/predict/batch", json=LOTE)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["n"] == len(LOTE["registros"])
    assert [p["indice"] for p in cuerpo["predicciones"]] == list(range(cuerpo["n"]))
    assert [p["estacion_id"] for p in cuerpo["predicciones"]] == [
        reg["estacion_id"] for reg in LOTE["registros"]
    ]


@pytest.mark.parametrize("n", [0, 501])
def test_lote_vacio_o_excedido(cliente, n):
    assert cliente.post("/predict/batch", json={"registros": [VALIDO] * n}).status_code == 422


def test_lote_con_un_invalido_se_rechaza_completo(cliente):
    cuerpo = {"registros": [VALIDO, con(mes=13)]}
    assert cliente.post("/predict/batch", json=cuerpo).status_code == 422


def test_lote_campo_extra(cliente):
    assert cliente.post("/predict/batch", json={**LOTE, "x": 1}).status_code == 422
