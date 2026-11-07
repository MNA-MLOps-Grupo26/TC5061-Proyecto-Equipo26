"""Aceptación del script con comandos dobles, sin Docker ni modelos reales."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
DOBLE = r"""
import json
import os
from pathlib import Path
import sys

nombre = Path(sys.argv[0]).name
args = sys.argv[1:]
caso = os.environ["CASO_ACEPTACION"]
raiz = Path.cwd()
referencia = json.loads((raiz / "docs/evidencia-api/prediccion-referencia.json").read_text())

if nombre == "git":
    if args == ["rev-parse", "--show-toplevel"]:
        print(raiz)
    elif args == ["rev-parse", "HEAD"]:
        print("commit-doble")
elif nombre == "docker":
    if args[0] == "--version":
        print("Docker doble")
    elif args[0] == "image":
        print("imagen-doble usuario=app arquitectura=arm64")
    elif args[0] == "run":
        print("contenedor-doble")
    elif args[0] == "exec":
        if args[2:] == ["id", "-u"]:
            print("0" if caso == "root" else "10001")
        elif args[2:4] == ["python", "-c"]:
            if caso == "exec_error":
                print("Error simulado de docker exec", file=sys.stderr)
                sys.exit(125)
            # Ejecuta la comprobación real de permisos sobre el árbol temporal.
            codigo = args[4].replace("/app/", str(raiz / "app") + "/")
            exec(compile(codigo, "<permisos-del-script>", "exec"))
        else:
            raise AssertionError(f"docker exec inesperado: {args}")
elif nombre == "curl":
    if "@-" in args:
        sys.stdin.read()
        print("200" if caso == "sin422" else "422", end="")
        sys.exit(0)
    url = next(a for a in args if a.startswith("http://"))
    estado = 200
    if url.endswith("/health"):
        cuerpo = {"estado": "error" if caso == "salud" else "ok"}
        # Las sondas de arranque pasan; se prueba la respuesta de aceptación.
        if "-w" in args and caso in ("health500", "health201"):
            estado = int(caso[-3:])
    elif url.endswith("/predict/batch"):
        valores = referencia["peticion_batch"]["viajes_esperados_en_orden"][:]
        if caso == "lote":
            valores[0] += 1
        indices = list(range(len(valores)))
        if caso == "orden":
            indices.reverse()
        cuerpo = {"n": len(valores), "predicciones": [
            {"indice": i, "viajes_esperados": v} for i, v in zip(indices, valores)
        ]}
        if caso == "batch500":
            estado = 500
    else:
        contador = raiz / "consultas-predict.txt"
        numero = int(contador.read_text()) + 1 if contador.exists() else 1
        contador.write_text(str(numero))
        valor = referencia["peticion_predict"]["viajes_esperados"]
        if caso == "individual" or (caso == "captura_distinta" and numero == 2):
            valor += 1
        cuerpo = {"viajes_esperados": valor}
        if caso in ("predict500", "predict201"):
            estado = int(caso[-3:])
        if caso == "captura500" and numero == 2:
            estado = 500
        if caso == "transporte":
            sys.exit(7)
    print(json.dumps(cuerpo), end="")
    if "-w" in args:
        formato = args[args.index("-w") + 1].replace(r"\n", "\n")
        print(formato.replace("%{http_code}", str(estado)), end="")
elif nombre == "uname":
    print("Darwin")
elif nombre == "lsof":
    sys.exit(1)
elif nombre == "osascript":
    print("123")
elif nombre == "screencapture":
    Path(args[-1]).write_bytes(b"captura simulada")
"""


@pytest.mark.parametrize(
    "caso",
    [
        "ok",
        "health500",
        "health201",
        "predict500",
        "predict201",
        "batch500",
        "individual",
        "lote",
        "orden",
        "sin422",
        "salud",
        "root",
        "modelo_archivo",
        "modelo_directorio",
        "api_archivo",
        "api_directorio",
        "exec_error",
        "captura_distinta",
        "captura500",
        "transporte",
    ],
)
@pytest.mark.skipif(os.name == "nt", reason="El script de aceptación requiere Bash")
@pytest.mark.skipif(
    os.name != "nt" and os.geteuid() == 0,
    reason="Todos los escenarios requieren un árbol de solo lectura para un usuario no root",
)
def test_aceptacion_solo_captura_si_todas_las_comprobaciones_pasan(tmp_path, caso):
    binarios = tmp_path / "bin"
    binarios.mkdir()
    doble = binarios / "doble"
    doble.write_text(f"#!{sys.executable}\n{DOBLE}", encoding="utf-8")
    doble.chmod(0o755)
    for nombre in (
        "git",
        "docker",
        "curl",
        "uname",
        "lsof",
        "osascript",
        "screencapture",
        "sleep",
    ):
        (binarios / nombre).symlink_to(doble)
    (binarios / "python3").symlink_to(sys.executable)
    evidencia = tmp_path / "docs/evidencia-api"
    evidencia.mkdir(parents=True)
    for archivo in ("prediccion-referencia.json", "peticion-predict.json", "peticion-batch.json"):
        shutil.copyfile(REPO / "docs/evidencia-api" / archivo, evidencia / archivo)
    shutil.copyfile(REPO / "scripts/probar_contenedor.sh", tmp_path / "probar.sh")
    (tmp_path / "Containerfile").write_text("# doble\n", encoding="utf-8")
    directorios = [tmp_path / "app" / nombre for nombre in ("modelo", "api")]
    for directorio in directorios:
        directorio.mkdir(parents=True)
        archivo = directorio / "registry.json"
        archivo.write_text("{}", encoding="utf-8")
        archivo.chmod(0o444)
        directorio.chmod(0o555)
    if caso in ("modelo_archivo", "modelo_directorio", "api_archivo", "api_directorio"):
        nombre, tipo = caso.split("_")
        objetivo = tmp_path / "app" / nombre
        if tipo == "archivo":
            objetivo /= "registry.json"
        objetivo.chmod(0o755 if tipo == "directorio" else 0o644)
    entorno = {
        **os.environ,
        "PATH": str(binarios) + os.pathsep + os.environ.get("PATH", ""),
        "CASO_ACEPTACION": caso,
        "MODEL_DIR": "app/modelo",
        "MOTOR": "docker",
        "CAPTURA": "1",
        "TERM_PROGRAM": "Apple_Terminal",
    }
    try:
        resultado = subprocess.run(
            ["/bin/bash", "probar.sh"],
            check=False,
            cwd=tmp_path,
            env=entorno,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        for directorio in directorios:
            directorio.chmod(0o755)
    assert resultado.returncode == (0 if caso == "ok" else 1), resultado.stdout + resultado.stderr
    assert (evidencia / "captura-contenedor.png").exists() == (caso == "ok")
    assert ("ACEPTACIÓN: OK" in resultado.stdout) == (caso == "ok")
    if caso.startswith("captura"):
        assert "==================== CAPTURA:" in resultado.stdout
    elif caso in ("modelo_archivo", "modelo_directorio", "api_archivo", "api_directorio"):
        assert "Rutas escribibles:" in resultado.stdout
    elif caso == "exec_error":
        assert "Error simulado de docker exec" in resultado.stdout
