"""Regresión de la conversión determinista de coordenadas del catálogo.

Los valores esperados NO salen de este código: son las latitudes observadas en el panel de
octubre del reporte de semana 9 (``sin_fma``) y en ``estaciones.parquet`` del manifiesto
(``con_fma``), para las 15 estaciones cuya latitud difiere entre ambos.
"""

import math

import numpy as np
import pandas as pd
import pytest

from mibici import monitoring
from mibici.coordenadas import a_numerico, convertir
from mibici.preprocess import leer_estaciones

# (estacion_id, texto del CSV, panel del reporte, estaciones.parquet del manifiesto)
CASOS = [
    (21, "20.678813670043535", "0x1.4adc6bb90ea2ap+4", "0x1.4adc6bb90ea2bp+4"),
    (34, "20.677224371110494", "0x1.4ad5e938df946p+4", "0x1.4ad5e938df948p+4"),
    (51, "20.674139733956636", "0x1.4ac946bee4725p+4", "0x1.4ac946bee4724p+4"),
    (59, "20.672706722709474", "0x1.4ac3681fdd627p+4", "0x1.4ac3681fdd626p+4"),
    (92, "20.726322930281754", "0x1.4b9f04cafe521p+4", "0x1.4b9f04cafe520p+4"),
    (164, "20.688598107207913", "0x1.4b047f72e8ba6p+4", "0x1.4b047f72e8ba5p+4"),
    (341, "20.712724319465234", "0x1.4b67519db2ac3p+4", "0x1.4b67519db2ac2p+4"),
    (342, "20.665181516539473", "0x1.4aa4955fb70d4p+4", "0x1.4aa4955fb70d3p+4"),
    (352, "20.672638790185932", "0x1.4ac320e45de9cp+4", "0x1.4ac320e45de9dp+4"),
    (374, "20.672212871825575", "0x1.4ac16248c70eap+4", "0x1.4ac16248c70ebp+4"),
    (375, "20.669601412890255", "0x1.4ab6aff89b812p+4", "0x1.4ab6aff89b813p+4"),
    (377, "20.699865427893354", "0x1.4b32a6174672cp+4", "0x1.4b32a6174672bp+4"),
    (385, "20.691523449502654", "0x1.4b107ae4807efp+4", "0x1.4b107ae4807f1p+4"),
    (389, "20.718487802560414", "0x1.4b7eed10e9266p+4", "0x1.4b7eed10e9267p+4"),
    (400, "20.661973179687156", "0x1.4a977130595e2p+4", "0x1.4a977130595e1p+4"),
]


@pytest.mark.parametrize("estacion, texto, reporte, manifiesto", CASOS)
def test_modos_reproducen_cada_representacion_observada(estacion, texto, reporte, manifiesto):
    assert convertir(texto, "sin_fma") == float.fromhex(reporte)
    assert convertir(texto, "con_fma") == float.fromhex(manifiesto)
    assert convertir(texto, "sin_fma") != convertir(texto, "con_fma")


@pytest.mark.parametrize(
    "texto", ["20.666378", "-103.34882", "0", "15", "-0.5", "20.69", " 20.5 ", "+3.25"]
)
def test_decimales_cortos_son_exactos_e_iguales_en_ambos_modos(texto):
    # Con menos de 16 dígitos el acumulado es exacto: ambos modos dan el redondeo correcto.
    assert convertir(texto, "sin_fma") == convertir(texto, "con_fma") == float(texto)


def test_digitos_extra_se_ignoran_como_en_pandas():
    base = "20.678813670043535"
    assert convertir(base + "999", "sin_fma") == convertir(base, "sin_fma")
    assert convertir("123456789012345678", "con_fma") == 12345678901234567 * 10.0


@pytest.mark.parametrize("texto", ["1e-5", "20,5", "abc", "", "1.2.3", "--1"])
def test_formatos_no_soportados_se_rechazan(texto):
    with pytest.raises(ValueError):
        convertir(texto, "sin_fma")


def test_modo_desconocido_se_rechaza():
    with pytest.raises(ValueError):
        convertir("1.5", "pandas")
    with pytest.raises(ValueError):
        a_numerico(pd.Series(["1.5"]), "otro")


def test_a_numerico_vacios_y_texto_como_coerce_y_rechaza_cientifica():
    s = pd.Series(["20.5", None, "", "sin dato", np.nan])
    for modo in ("pandas", "sin_fma", "con_fma"):
        r = a_numerico(s, modo)
        assert r.iloc[0] == 20.5 and r.iloc[1:].isna().all()
    with pytest.raises(ValueError, match="no soportado"):
        a_numerico(pd.Series(["2e1"]), "sin_fma")


def test_un_redondeo_frente_a_dos_redondeos_calculado_con_enteros_exactos():
    # Cada operación float de Python redondea por separado; math.fma redondea una vez.
    a = 2067881367004353.0  # 16 dígitos: exacto en float64
    exacto = int(a) * 10 + 5  # 20678813670043535 > 2**53: no representable
    assert math.fma(a, 10.0, 5.0) == float(exacto)
    assert a * 10.0 + 5.0 == float(int(float(int(a) * 10)) + 5)
    assert math.fma(a, 10.0, 5.0) != a * 10.0 + 5.0


def catalogo(tmp_path, lat_texto):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "nomenclatura.csv"
    filas = "\n".join(
        f'"{i}","E{i}","{lat}","-103.3","{10 + i}"' for i, lat in enumerate(lat_texto, start=1)
    )
    path.write_text('"id","name","latitude","longitude","dpcapacity"\n' + filas + "\n", "utf-8")
    return path


def test_leer_estaciones_pandas_igual_a_comportamiento_original(tmp_path):
    path = catalogo(tmp_path, [t for _, t, _, _ in CASOS] + ["20.6", "x"])
    nuevo = leer_estaciones(path)
    original = pd.read_csv(path, encoding="utf-8-sig")
    assert np.array_equal(
        nuevo.latitud.to_numpy(),
        pd.to_numeric(original.latitude, errors="coerce").to_numpy(),
        equal_nan=True,
    )


def test_deterministic_catalog_registra_cambios_y_protege_otras_columnas(tmp_path):
    textos = [t for _, t, _, _ in CASOS]
    path = catalogo(tmp_path, textos)
    parquet = leer_estaciones(path, coordenadas="con_fma")
    manifest = {
        "stations": {
            "source_filename": path.name,
            "source_sha256": monitoring.digest(path),
        }
    }
    source = tmp_path / "processed"
    rebuilt, changed = monitoring.deterministic_catalog(source, manifest, parquet, "sin_fma")
    assert changed == {"latitud": 15, "longitud": 0}
    assert [float.hex(v) for v in rebuilt.latitud] == [r for _, _, r, _ in CASOS]
    _, same = monitoring.deterministic_catalog(source, manifest, parquet, "con_fma")
    assert same == {"latitud": 0, "longitud": 0}
    otro = parquet.assign(capacidad=parquet.capacidad + 1)
    with pytest.raises(AssertionError):
        monitoring.deterministic_catalog(source, manifest, otro, "sin_fma")
    manifest["stations"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        monitoring.deterministic_catalog(source, manifest, parquet, "sin_fma")


def test_prepare_rechaza_modo_desconocido(tmp_path):
    with pytest.raises(ValueError, match="coordinate mode"):
        monitoring.prepare(tmp_path, tmp_path / "x.csv", tmp_path, coordenadas="otro")
