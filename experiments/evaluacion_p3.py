"""Evaluación temporal P3 compatible con distintos números de grupos."""

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from mibici.evaluation import asignar_fijo
from mibici.features import matriz_forma


def evaluar_validacion(modelo, train, periodo):
    """Reproduce las métricas de resumir_fijo para cualquier k >= 2."""
    q = periodo.loc[periodo.evaluable].sort_values("estacion_id").reset_index(drop=True)

    if len(q) < 2:
        raise ValueError("Muy pocas estaciones evaluables")

    x, labels, dist = asignar_fijo(modelo, q)

    matched = train.set_index("estacion_id").loc[q.estacion_id].reset_index()

    xt = matriz_forma(matched)
    lt = modelo.predict(xt)

    fullx = matriz_forma(train)
    full_labels = modelo.predict(fullx)

    dtrain = modelo.transform(fullx)[np.arange(len(fullx)), full_labels]

    k = modelo.n_clusters

    p95 = np.array([np.quantile(dtrain[full_labels == g], 0.95) for g in range(k)])

    shift = np.linalg.norm(x - xt, axis=1)

    metricas = {
        "n_evaluables": len(q),
        "n_excluidas": len(periodo) - len(q),
        "fraccion_evaluable": len(q) / len(train),
        "fraccion_estable": float((lt == labels).mean()),
        "ari_vs_train": float(adjusted_rand_score(lt, labels)),
        "distancia_mediana": float(np.median(dist)),
        "distancia_p95": float(np.quantile(dist, 0.95)),
        "cambio_perfil_mediana": float(np.median(shift)),
        "cambio_perfil_p95": float(np.quantile(shift, 0.95)),
        "fraccion_sobre_p95_train": float((dist > p95[labels]).mean()),
    }

    if 2 <= len(np.unique(labels)) < len(labels):
        metricas["silhouette_asignacion_fija"] = float(silhouette_score(x, labels))

    for g in range(k):
        metricas[f"n_grupo_{g}"] = int((labels == g).sum())

    return metricas
