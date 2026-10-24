"""Pruebas de evaluación temporal P3 para distintos valores de k."""

from experiments.evaluacion_p3 import evaluar_validacion
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mibici.features import columnas_perfil_p3, matriz_forma


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_evaluacion_validacion_multicluster(k):
    rng = np.random.default_rng(42)

    # Perfiles sintéticos: 30 estaciones, 336 variables.
    x = rng.dirichlet(np.ones(168), size=(30, 2)).reshape(30, 336)

    train = pd.DataFrame(x, columns=columnas_perfil_p3())
    train["estacion_id"] = np.arange(1, 31)

    modelo = KMeans(
        n_clusters=k,
        n_init=5,
        random_state=42,
        algorithm="lloyd",
    ).fit(matriz_forma(train))

    centroides_antes = modelo.cluster_centers_.copy()

    # Validación con los mismos perfiles y una estación excluida.
    val = train.copy()
    val["evaluable"] = True
    val.loc[val.index[-1], "evaluable"] = False

    metricas = evaluar_validacion(modelo, train, val)

    assert metricas["n_evaluables"] == 29
    assert metricas["n_excluidas"] == 1
    assert 0 <= metricas["fraccion_estable"] <= 1
    assert -1 <= metricas["ari_vs_train"] <= 1
    assert np.isfinite(metricas["distancia_mediana"])
    assert np.isfinite(metricas["distancia_p95"])

    assert sum(metricas[f"n_grupo_{g}"] for g in range(k)) == 29

    for g in range(k):
        assert f"n_grupo_{g}" in metricas

    np.testing.assert_array_equal(
        modelo.cluster_centers_,
        centroides_antes,
    )
