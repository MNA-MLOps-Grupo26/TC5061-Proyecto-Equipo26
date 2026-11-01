"""El repositorio no versiona credenciales, datos ni artefactos locales.

Complementa a bandit y pip-audit (``make seguridad``): revisa lo que Git
ignora y busca patrones de secretos en los archivos versionados.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

RAIZ = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (RAIZ / ".git").exists(),
    reason="Requiere el repositorio Git",
)

# Patrones de credenciales comunes. Se escriben de modo que este archivo no
# coincida consigo mismo.
SECRETOS = {
    "llave privada": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "llave de AWS": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "token de GitHub": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
    "token de Slack": re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}"),
    "URL con contraseña": re.compile(r"[a-z+]+://[^/\s:@]+:[^/\s@]+@"),
    "asignación de secreto": re.compile(
        r"(?i)\b(password|passwd|secret|api_?key|access_?token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=False, cwd=RAIZ, capture_output=True, text=True)


def _versionados() -> list[str]:
    return _git("ls-files").stdout.splitlines()


@pytest.mark.parametrize(
    "ruta",
    [
        ".env",
        ".env.local",
        "data/raw/datos_abiertos_2026_09.csv",
        "datos_abiertos_2026_09.parquet",
        "mlflow.db",
        "mlruns/0/meta.yaml",
        "models/p1.pkl",
        "reports/deepchecks/reporte.html",
    ],
)
def test_gitignore_excluye_credenciales_datos_y_artefactos(ruta):
    assert _git("check-ignore", "-q", ruta).returncode == 0, f"{ruta} no está ignorado"


def test_plantilla_de_variables_de_entorno_si_se_versiona():
    assert _git("check-ignore", "-q", ".env.example").returncode == 1
    assert "MLFLOW_TRACKING_URI" in (RAIZ / ".env.example").read_text(encoding="utf-8")


def test_no_se_versionan_archivos_de_credenciales():
    sensibles = re.compile(r"(^|/)(\.env(\.(?!example$)[^/]+)?|id_rsa|.*\.pem|.*\.key)$")
    assert [f for f in _versionados() if sensibles.search(f)] == []


def test_archivos_versionados_sin_secretos():
    hallazgos = []
    for nombre in _versionados():
        ruta = RAIZ / nombre
        if not ruta.is_file() or ruta.stat().st_size > 2_000_000:
            continue
        try:
            texto = ruta.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binarios (imágenes)
        for tipo, patron in SECRETOS.items():
            for coincidencia in patron.finditer(texto):
                linea = texto.count("\n", 0, coincidencia.start()) + 1
                hallazgos.append(f"{nombre}:{linea} ({tipo})")
    assert hallazgos == []
