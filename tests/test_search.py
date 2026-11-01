from unittest.mock import Mock

import pytest
import typer

from mibici.config import BUSQUEDAS
from mibici.modeling import search


def test_cada_busqueda_tiene_al_menos_tres_combinaciones():
    assert all(len(combinaciones) >= 3 for combinaciones in BUSQUEDAS.values())
    assert {p for p, _ in BUSQUEDAS} == {"P1", "P2", "P3"}


@pytest.mark.parametrize(
    "argumentos",
    [{"pregunta": "P4"}, {"pregunta": "P1", "modelo": "glm"}, {"pregunta": "P3", "modelo": "hgb"}],
)
def test_busqueda_invalida_no_carga_datos(monkeypatch, argumentos):
    cargar = Mock()
    monkeypatch.setattr(search, "cargar_viajes", cargar)
    monkeypatch.setattr(search, "cargar_estaciones", cargar)
    with pytest.raises(typer.BadParameter):
        search.main(**argumentos)
    cargar.assert_not_called()


def test_busqueda_p3_no_carga_viajes_globales(monkeypatch):
    llamado = []
    monkeypatch.setattr(search, "cargar_viajes", Mock(side_effect=AssertionError))
    monkeypatch.setattr(
        search,
        "BUSCADORES",
        {**search.BUSCADORES, "P3": lambda v, e, modelo: llamado.append(modelo)},
    )
    search.main(pregunta="P3")
    assert llamado == ["kmeans"]
