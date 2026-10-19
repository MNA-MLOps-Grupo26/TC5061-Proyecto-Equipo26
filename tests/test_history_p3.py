import numpy as np
import pandas as pd
import pytest

from mibici.evaluation import eta_historia
from mibici.features import cohorte_historia


def test_cohortes_preservan_salvaguardas_y_limites():
    p = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3, 4, 5],
            "conjunto": "train",
            "periodo_inicio": "2025-08-01",
            "periodo_fin_exclusivo": "2026-08-01",
            "elegible": [True, True, True, False, True],
            "dias_con_eventos": [179, 180, 300, 364, 299],
        }
    )
    assert cohorte_historia(p, 180).estacion_id.tolist() == [2, 3, 5]
    assert cohorte_historia(p, 300).estacion_id.tolist() == [3]
    assert len(p) == 5
    with pytest.raises(ValueError):
        cohorte_historia(p, 200)
    p.loc[0, "conjunto"] = "test"
    with pytest.raises(ValueError):
        cohorte_historia(p, 180)


def test_asociacion_historia_es_invariante_a_etiquetas_y_no_causal():
    dias = np.array([40, 40, 360, 360])
    assert eta_historia(dias, np.array([0, 0, 1, 1])) == 1
    assert eta_historia(dias, np.array([9, 9, 3, 3])) == 1
    assert eta_historia(dias, np.array([0, 1, 0, 1])) == 0
    assert eta_historia(np.ones(4), np.array([0, 0, 1, 1])) == 0
