"""resumir_fijo funciona para cualquier k (antes estaba fijo en 3)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mibici.evaluation import resumir_fijo
from mibici.features import columnas_perfil_p3, matriz_forma


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_resumir_fijo_multicluster(k):
    rng = np.random.default_rng(42)
    # Perfiles sintéticos: 30 estaciones, 336 variables.
    x = rng.dirichlet(np.ones(168), size=(30, 2)).reshape(30, 336)
    train = pd.DataFrame(x, columns=columnas_perfil_p3())
    train["estacion_id"] = np.arange(1, 31)
    train["nombre"] = "estacion"
    train["dias_con_eventos"] = 300
    train["total_movimientos"] = 500

    modelo = KMeans(n_clusters=k, n_init=5, random_state=42, algorithm="lloyd").fit(
        matriz_forma(train)
    )
    centroides_antes = modelo.cluster_centers_.copy()

    # Validación con los mismos perfiles y una estación excluida.
    val = train.copy()
    val["evaluable"] = True
    val.loc[val.index[-1], "evaluable"] = False

    _, asignaciones, metricas, grupos = resumir_fijo(modelo, train, val)

    assert metricas["n_evaluables"] == 29
    assert metricas["n_excluidas"] == 1
    assert metricas["fraccion_estable"] == 1
    assert sum(metricas[f"n_grupo_{g}"] for g in range(k)) == 29
    assert len(grupos) == k
    assert asignaciones.estable.all()
    np.testing.assert_array_equal(modelo.cluster_centers_, centroides_antes)
