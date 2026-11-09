"""Contrato de entrada y salida de la API del modelo P1.

Este módulo es autocontenido: no importa ``mibici`` porque la imagen solo copia
``api/`` y el modelo. Una prueba compara ``VARIABLES_P1`` con
``mibici.features.VARIABLES_P1``.
"""

from __future__ import annotations

from datetime import date
import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Las 18 variables del modelo P1, en el orden de la firma registrada en MLflow.
VARIABLES_P1: list[str] = [
    "salidas_lag1",
    "llegadas_lag1",
    "saldo_lag1",
    "salidas_lag7",
    "salidas_media7",
    "salidas_media28",
    "llegadas_media7",
    "llegadas_media28",
    "salidas_mismo_dia_media4",
    "dias_historia",
    "dias_desde_ultimo_viaje",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "dia_mes",
    "capacidad",
    "latitud",
    "longitud",
]

MAX_REGISTROS_LOTE = 500

# Días máximos por mes. Febrero admite 29 porque la entrada no trae año.
_DIAS_POR_MES = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

# Tipos base: modo estricto (sin textos ni booleanos) y sin NaN ni infinitos.
Numero = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NumeroNoNegativo = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
Entero = Annotated[int, Field(strict=True)]

# Peticion de ejemplo para /docs (fila real del panel; misma que peticion-predict.json).
EJEMPLO_REGISTRO = {
    "estacion_id": 21,
    "fecha": "2026-09-15",
    "salidas_lag1": 146.0,
    "llegadas_lag1": 141.0,
    "saldo_lag1": 5.0,
    "salidas_lag7": 110.0,
    "salidas_media7": 122.714286,
    "salidas_media28": 106.285714,
    "llegadas_media7": 117.857143,
    "llegadas_media28": 100.928571,
    "salidas_mismo_dia_media4": 105.5,
    "dias_historia": 410,
    "dias_desde_ultimo_viaje": 1,
    "dia_semana": 1,
    "es_fin_semana": 0,
    "mes": 9,
    "dia_mes": 15,
    "capacidad": 23.0,
    "latitud": 20.678814,
    "longitud": -103.368058,
}


class RegistroP1(BaseModel):
    """Una fila estación-día con las 18 variables del modelo P1.

    Las 18 claves son requeridas aunque algunas acepten ``null``. ``estacion_id`` y
    ``fecha`` son opcionales, no entran al modelo y se devuelven en la respuesta.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_REGISTRO]})

    estacion_id: Entero | None = Field(default=None, ge=1)
    fecha: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    # Conteos: número entero no negativo (se aceptan 146 o 146.0, no 146.5).
    salidas_lag1: NumeroNoNegativo
    llegadas_lag1: NumeroNoNegativo
    saldo_lag1: Numero
    salidas_lag7: NumeroNoNegativo | None = Field(...)
    # Promedios móviles.
    salidas_media7: NumeroNoNegativo
    salidas_media28: NumeroNoNegativo
    llegadas_media7: NumeroNoNegativo
    llegadas_media28: NumeroNoNegativo
    salidas_mismo_dia_media4: NumeroNoNegativo | None = Field(...)
    # Historia y calendario.
    dias_historia: Entero = Field(ge=1)
    dias_desde_ultimo_viaje: Entero = Field(ge=1)
    dia_semana: Entero = Field(ge=0, le=6)
    es_fin_semana: Entero = Field(ge=0, le=1)
    mes: Entero = Field(ge=1, le=12)
    dia_mes: Entero = Field(ge=1, le=31)
    # Estación (nulas juntas si la estación no está en el catálogo).
    capacidad: Numero | None = Field(..., gt=0, le=100)
    latitud: Numero | None = Field(..., ge=20.55, le=20.80)
    longitud: Numero | None = Field(..., ge=-103.50, le=-103.20)

    @field_validator("salidas_lag1", "llegadas_lag1", "saldo_lag1", "salidas_lag7")
    @classmethod
    def _sin_decimales(cls, valor: float | None) -> float | None:
        if valor is not None and not float(valor).is_integer():
            raise ValueError("debe ser un número entero (sin decimales)")
        return valor

    @field_validator("fecha")
    @classmethod
    def _fecha_valida(cls, valor: str | None) -> str | None:
        if valor is not None:
            try:
                date.fromisoformat(valor)
            except ValueError as error:
                raise ValueError("fecha inexistente; use AAAA-MM-DD") from error
        return valor

    @model_validator(mode="after")
    def _reglas_cruzadas(self) -> RegistroP1:
        # 1. es_fin_semana vale 1 exactamente cuando dia_semana >= 5.
        if self.es_fin_semana != int(self.dia_semana >= 5):
            raise ValueError(
                "es_fin_semana no coincide con dia_semana (1 solo si dia_semana >= 5)"
            )
        # 2. dia_mes existe en mes.
        if self.dia_mes > _DIAS_POR_MES[self.mes]:
            raise ValueError(f"dia_mes={self.dia_mes} no existe en mes={self.mes}")
        # 3. fecha coincide con mes, dia_mes y dia_semana.
        if self.fecha is not None:
            f = date.fromisoformat(self.fecha)
            if (f.month, f.day, f.weekday()) != (self.mes, self.dia_mes, self.dia_semana):
                raise ValueError("fecha no coincide con mes, dia_mes o dia_semana")
        # 4. saldo_lag1 = salidas_lag1 - llegadas_lag1.
        if not math.isclose(self.saldo_lag1, self.salidas_lag1 - self.llegadas_lag1, abs_tol=1e-9):
            raise ValueError("saldo_lag1 debe ser salidas_lag1 - llegadas_lag1")
        # 5. dias_desde_ultimo_viaje <= dias_historia.
        if self.dias_desde_ultimo_viaje > self.dias_historia:
            raise ValueError("dias_desde_ultimo_viaje no puede ser mayor que dias_historia")
        # 6. salidas_lag7 y salidas_mismo_dia_media4: nulas juntas o ninguna.
        if (self.salidas_lag7 is None) != (self.salidas_mismo_dia_media4 is None):
            raise ValueError("salidas_lag7 y salidas_mismo_dia_media4 deben ser nulas juntas")
        # 7. capacidad, latitud y longitud: nulas juntas o ninguna.
        nulos_estacion = {v is None for v in (self.capacidad, self.latitud, self.longitud)}
        if len(nulos_estacion) > 1:
            raise ValueError("capacidad, latitud y longitud deben ser nulas juntas")
        return self

    def a_fila(self) -> dict[str, float]:
        """Las 18 variables en el orden de la firma, con ``None`` como NaN."""
        return {
            v: math.nan if getattr(self, v) is None else float(getattr(self, v))
            for v in VARIABLES_P1
        }


class LoteP1(BaseModel):
    """Lote de 1 a 500 registros. Si uno es inválido se rechaza todo el lote."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"registros": [EJEMPLO_REGISTRO]}]}
    )

    registros: list[RegistroP1] = Field(min_length=1, max_length=MAX_REGISTROS_LOTE)


# --- Salida ---------------------------------------------------------------------

ViajesEsperados = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class InfoModelo(BaseModel):
    nombre: str
    version: str
    run_id: str


class LinajeModelo(InfoModelo):
    alias_consultado: str | None = None
    logged_model_id: str | None = None
    source_sha: str | None = None
    data_version: str | None = None


class Verificacion(BaseModel):
    hashes: str
    firma: str
    archivos_verificados: int


class RespuestaSalud(BaseModel):
    estado: str
    modelo: LinajeModelo
    verificacion: Verificacion


class RespuestaPrediccion(BaseModel):
    viajes_esperados: ViajesEsperados
    estacion_id: int | None = None
    fecha: str | None = None
    modelo: InfoModelo


class PrediccionLote(BaseModel):
    indice: int
    viajes_esperados: ViajesEsperados
    estacion_id: int | None = None
    fecha: str | None = None


class RespuestaLote(BaseModel):
    n: int
    predicciones: list[PrediccionLote]
    modelo: InfoModelo
