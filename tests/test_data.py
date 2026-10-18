from mibici import config


def test_particion_temporal_en_orden():
    """La partición no debe mezclar periodos: entrenamiento < validación < prueba."""
    assert config.TRAIN_START < config.VALIDATION_START < config.TEST_START
    assert config.TEST_START < config.DATA_END_EXCLUSIVE


def test_meses_de_descarga_cubren_la_particion():
    assert config.DATA_START_MONTH == config.TRAIN_START.strftime("%Y-%m")
    assert config.DATA_END_MONTH == "2026-09"


def test_nombre_del_experimento_incluye_equipo():
    assert config.EXPERIMENT_NAME == "mibici-equipo26"
