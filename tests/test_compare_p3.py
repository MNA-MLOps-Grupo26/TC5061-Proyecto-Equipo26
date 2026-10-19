import numpy as np
import pandas as pd
import pytest

from mibici.evaluation import comparar
from mibici.features import columnas_perfil_p3, matriz_forma


def perfiles():
    rng = np.random.default_rng(17)
    x = rng.dirichlet(np.ones(168), size=(18, 2))
    p = pd.DataFrame(x.reshape(18, 336), columns=columnas_perfil_p3())
    p["estacion_id"] = range(18)
    p["latitud"] = 1e9
    p["total_movimientos"] = 100
    return p


def test_geometria_y_metadatos_no_afectan():
    p = perfiles()
    x = matriz_forma(p)
    assert np.allclose((x.reshape(-1, 2, 168) ** 2).sum(axis=2), 0.5)
    p["latitud"] = -1e15
    p["total_movimientos"] = 1e20
    p["estacion_id"] += 300
    assert np.array_equal(x, matriz_forma(p))


@pytest.mark.parametrize("valor", [np.nan, -1, np.inf])
def test_rechaza_probabilidades_invalidas(valor):
    p = perfiles()
    p.iloc[0, 0] = valor
    with pytest.raises(ValueError):
        matriz_forma(p)


def test_rechaza_bloques_sin_normalizar():
    p = perfiles()
    p.iloc[0, :168] *= 3
    with pytest.raises(ValueError):
        matriz_forma(p)


def test_comparacion_acotada_y_representantes_miembros():
    p = perfiles()
    tabla, labels, reps = comparar(p)
    assert len(tabla) == 10
    assert set(tabla.k) == {2, 3, 4, 5, 6}
    assert tabla.silhouette.between(-1, 1).all()
    for nombre, asignacion in labels.items():
        assert len(asignacion) == len(p)
        for row in reps[reps.configuracion.eq(nombre)].itertuples():
            assert asignacion[int(row.estacion_id)] == row.grupo
    tabla2, labels2, _ = comparar(p)
    assert tabla.equals(tabla2)
    assert all(np.array_equal(labels[n], labels2[n]) for n in labels)
