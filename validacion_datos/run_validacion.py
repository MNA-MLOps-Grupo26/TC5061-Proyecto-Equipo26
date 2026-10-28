"""Corre DeepChecks sobre los datos exportados por ``python -m mibici.validacion``.

Uso (desde la raíz del repo): ``make validar-datos``.

Escribe en ``reports/deepchecks/`` un ``reporte.html`` con todos los chequeos y
``resumen.json``.
Termina con código 1 si falla algún chequeo bloqueante.
"""

import json
from pathlib import Path
import sys

from deepchecks.core.suite import SuiteResult
import pandas as pd

from validacion_datos import checks

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "interim" / "deepchecks"
SALIDA = RAIZ / "reports" / "deepchecks"


def _leer(nombre: str) -> pd.DataFrame:
    ruta = ENTRADA / f"{nombre}.parquet"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Corran primero: python -m mibici.validacion")
    return pd.read_parquet(ruta)


def main() -> int:
    esquema = json.loads((ENTRADA / "esquema.json").read_text(encoding="utf-8"))
    p1 = {c: _leer(f"p1_{c}") for c in ("train", "validation", "test")}
    viajes = {c: _leer(f"viajes_{c}") for c in ("train", "test")}

    resultados = []
    for conjunto, df in p1.items():
        resultados += checks.integridad_panel(df, esquema, f"P1 {conjunto}")
    for conjunto in ("validation", "test"):
        resultados += checks.train_test_panel(
            p1["train"], p1[conjunto], esquema, f"P1 train vs {conjunto}"
        )
    for conjunto, df in viajes.items():
        resultados += checks.integridad_viajes(df, esquema, f"viajes {conjunto}")
    resultados += checks.train_test_viajes(
        viajes["train"], viajes["test"], esquema, "viajes train vs test"
    )

    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = []
    for i, r in enumerate(resultados, start=1):
        resumen.append(
            {
                "n": i,
                "chequeo": r.chequeo,
                "datos": r.datos,
                "tipo": "bloqueante" if r.bloqueante else "informativo",
                "paso": r.paso,
                "detalle": r.detalle,
            }
        )
        estado = "OK  " if r.paso else ("FALLA" if r.bloqueante else "AVISO")
        print(f"[{estado}] {i:02d} {r.chequeo:<32} {r.datos:<28} {r.detalle[:90]}")

    # Un solo HTML con todos los chequeos (plotly se incrusta una vez).
    SuiteResult("Validación de datos MiBici", [r.resultado for r in resultados]).save_as_html(
        str(SALIDA / "reporte.html"), as_widget=False
    )
    (SALIDA / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fallidos = checks.bloqueantes_fallidos(resultados)
    avisos = sum(not r.paso and not r.bloqueante for r in resultados)
    print(
        f"\n{len(resultados)} chequeos | {len(fallidos)} bloqueantes fallidos | "
        f"{avisos} avisos | reportes en {SALIDA.relative_to(RAIZ)}"
    )
    return 1 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
