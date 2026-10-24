from unittest.mock import Mock

import pytest
import typer

from mibici.modeling import train


@pytest.mark.parametrize(
    "argumentos",
    [
        {"pregunta": "P1", "modelo": "logreg"},
        {"pregunta": "P2", "modelo": "hgb"},
        {"pregunta": "P3", "modelo": "hgb"},
        {"pregunta": "todas", "modelo": "hgb"},
        {"pregunta": "todas", "modelo": "logreg"},
        {"pregunta": "P1", "modelo": "desconocido"},
        {"pregunta": "P1", "modelo": "default", "max_iter": "7"},
        {"pregunta": "P2", "modelo": "logreg", "learning_rate": "0.1"},
        {"pregunta": "P3", "max_iter": "7"},
    ],
)
def test_argumentos_invalidos_no_cargan_datos(monkeypatch, argumentos):
    cargar_viajes = Mock()
    cargar_estaciones = Mock()
    monkeypatch.setattr(train, "cargar_viajes", cargar_viajes)
    monkeypatch.setattr(train, "cargar_estaciones", cargar_estaciones)

    with pytest.raises(typer.BadParameter):
        train.main(**argumentos)

    cargar_viajes.assert_not_called()
    cargar_estaciones.assert_not_called()
