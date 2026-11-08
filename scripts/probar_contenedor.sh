#!/usr/bin/env bash
# Construye y prueba la imagen del servicio P1 dentro de un contenedor y guarda la evidencia.
#
# Uso (desde cualquier carpeta del repositorio):
#   bash scripts/probar_contenedor.sh
#
# Variables opcionales:
#   MOTOR=docker|podman   (docker por defecto)
#   PUERTO=8000           puerto local publicado en 127.0.0.1
#   NOMBRE=mibici-p1-api  nombre del contenedor (si ya existe, el script no lo toca y se detiene)
#   IMAGEN=mibici-p1-api:1
#   MODEL_DIR=models/p1   modelo exportado con scripts/exportar_modelo.py
#   PYTHON_IMAGE=...      imagen base alternativa (por ejemplo python:3.13-slim)
#   CAPTURA=1             en macOS + Terminal.app guarda la ventana como PNG (0 para omitir)
#
# No sobrescribe archivos: si la evidencia ya existe, se detiene. El contenedor queda
# en ejecución para la captura; al final se imprime el comando para detenerlo.
#
# Criterios de aceptación (el script termina con código distinto de 0 si alguno falla):
# usuario no root, modelo y API de solo lectura, HTTP 200 y /health con estado ok, /predict y
# /predict/batch equivalentes (tolerancia relativa 1e-12) a docs/evidencia-api/prediccion-referencia.json y 422 ante
# una petición incoherente. Solo si todos pasan se toma la captura.
set -euo pipefail

MOTOR="${MOTOR:-docker}"
PUERTO="${PUERTO:-8000}"
NOMBRE="${NOMBRE:-mibici-p1-api}"
IMAGEN="${IMAGEN:-mibici-p1-api:1}"
MODEL_DIR="${MODEL_DIR:-models/p1}"
CAPTURA="${CAPTURA:-1}"

cd "$(git rev-parse --show-toplevel)"
EVID="docs/evidencia-api"
SALIDA="$EVID/contenedor-$MOTOR.txt"
LOG_BUILD="$EVID/contenedor-$MOTOR-build.log"
PNG="$EVID/captura-contenedor.png"
REFERENCIA="$EVID/prediccion-referencia.json"
FALLAS=()
fallo() { echo "FALLA: $*"; FALLAS+=("$*"); }

falla() { echo "DETENIDO: $*" >&2; exit 1; }
json() { local j; j="$(cat)"; printf '%s\n' "$j" | python3 -m json.tool 2>/dev/null || printf '%s\n' "$j"; }

pedir_200() {
  local respuesta codigo
  respuesta="$(curl -sS -w '\n%{http_code}' "$@")" || return 1
  codigo="${respuesta##*$'\n'}"
  if [ "$codigo" != 200 ]; then
    echo "Se esperaba HTTP 200; se recibió $codigo" >&2
    return 1
  fi
  printf '%s' "${respuesta%$'\n'*}"
}

validar_prediccion() {
  python3 -c '
import json, math, sys
with open(sys.argv[1]) as archivo:
    esperado = json.load(archivo)["peticion_predict"]["viajes_esperados"]
valor = json.load(sys.stdin)["viajes_esperados"]
sys.exit(0 if math.isfinite(valor) and math.isclose(valor, esperado, rel_tol=1e-12) else 1)
' "$REFERENCIA"
}

# --- Precondiciones (no se modifica nada hasta pasar todas) ---------------------
[ -f Containerfile ] || falla "no se encontró Containerfile en $(pwd)"
[ -f "$MODEL_DIR/registry.json" ] || falla "falta $MODEL_DIR/registry.json; exporten el modelo (ver README)"
for f in "$SALIDA" "$LOG_BUILD"; do
  [ ! -e "$f" ] || falla "$f ya existe; muévanlo o renómbrenlo (no se sobrescribe)"
done
command -v "$MOTOR" >/dev/null || falla "no se encontró '$MOTOR' en el PATH"
command -v curl >/dev/null || falla "se necesita curl"
command -v python3 >/dev/null || falla "se necesita python3 para comparar las respuestas"
[ -f "$REFERENCIA" ] || falla "falta $REFERENCIA (predicciones de referencia del modelo)"

if ! "$MOTOR" info >/dev/null 2>&1; then
  if [ "$MOTOR" = docker ] && [ "$(uname)" = Darwin ]; then
    echo "Docker no responde; abriendo Docker Desktop y esperando hasta 3 minutos..."
    open -a Docker || true
    for _ in $(seq 1 90); do "$MOTOR" info >/dev/null 2>&1 && break; sleep 2; done
  fi
  "$MOTOR" info >/dev/null 2>&1 || falla "el motor $MOTOR no responde (¿está iniciado?)"
fi
if "$MOTOR" ps -a --format '{{.Names}}' | grep -qx "$NOMBRE"; then
  falla "ya existe un contenedor '$NOMBRE'; no se toca. Usen NOMBRE=<otro> o elimínenlo ustedes"
fi
if command -v lsof >/dev/null && lsof -nP -iTCP:"$PUERTO" -sTCP:LISTEN >/dev/null 2>&1; then
  falla "el puerto $PUERTO ya está en uso; usen PUERTO=<otro>"
fi

# --- Ejecución con registro de evidencia ---------------------------------------
exec > >(tee "$SALIDA") 2>&1
echo "# Prueba del servicio P1 dentro del contenedor ($MOTOR)"
echo "# Fecha UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# Commit: $(git rev-parse HEAD); archivos con cambios sin commit: $(git status --porcelain | wc -l | tr -d ' ')"
echo "# Host: $(uname -sm)"
echo "# Motor: $("$MOTOR" --version)"
echo "# Modelo: $MODEL_DIR (registry.json sha256 $(shasum -a 256 "$MODEL_DIR/registry.json" 2>/dev/null | cut -d' ' -f1 || sha256sum "$MODEL_DIR/registry.json" | cut -d' ' -f1))"
echo

ARGS=(-f Containerfile -t "$IMAGEN" --build-arg "MODEL_DIR=$MODEL_DIR")
[ -n "${PYTHON_IMAGE:-}" ] && ARGS+=(--build-arg "PYTHON_IMAGE=$PYTHON_IMAGE")
echo "\$ $MOTOR build ${ARGS[*]} .   (log completo en $LOG_BUILD)"
"$MOTOR" build "${ARGS[@]}" . >"$LOG_BUILD" 2>&1 || { tail -n 30 "$LOG_BUILD"; falla "falló el build"; }
tail -n 3 "$LOG_BUILD"
echo "Imagen: $("$MOTOR" image inspect "$IMAGEN" --format '{{.Id}} usuario={{.Config.User}} arquitectura={{.Architecture}}')"
echo

echo "\$ $MOTOR run -d --name $NOMBRE -p 127.0.0.1:$PUERTO:8000 $IMAGEN"
"$MOTOR" run -d --name "$NOMBRE" -p "127.0.0.1:$PUERTO:8000" "$IMAGEN"
URL="http://127.0.0.1:$PUERTO"
for _ in $(seq 1 60); do curl -sf "$URL/health" >/dev/null && break; sleep 1; done
curl -sf "$URL/health" >/dev/null || { "$MOTOR" logs "$NOMBRE"; falla "la API no respondió en 60 s"; }
echo

echo "\$ $MOTOR ps --filter name=$NOMBRE"
"$MOTOR" ps --filter "name=$NOMBRE"
echo
echo "\$ $MOTOR exec $NOMBRE id -u   (usuario no root)"
UID_CONT="$("$MOTOR" exec "$NOMBRE" id -u)"
echo "$UID_CONT"
[ "$UID_CONT" != 0 ] || fallo "el contenedor corre como root"
echo "Verificación de archivos y directorios de solo lectura en /app/modelo y /app/api"
if ! "$MOTOR" exec "$NOMBRE" python -c '
import os, sys

def error_recorrido(error):
    raise error

escribibles = []
for raiz in ("/app/modelo", "/app/api"):
    if not os.path.isdir(raiz):
        raise RuntimeError(f"No existe el directorio {raiz}")
    for carpeta, directorios, archivos in os.walk(raiz, onerror=error_recorrido):
        for ruta in [carpeta, *(os.path.join(carpeta, n) for n in directorios + archivos)]:
            if os.access(ruta, os.W_OK):
                escribibles.append(ruta)
if escribibles:
    print("Rutas escribibles:\n" + "\n".join(sorted(set(escribibles))))
    sys.exit(1)
print("Modelo y API de solo lectura")
'; then
  fallo "modelo o API escribibles, o error al verificar sus permisos con $MOTOR exec"
fi
echo

echo "\$ curl -s $URL/health"
SALUD="$(pedir_200 "$URL/health")" || falla "/health no respondió HTTP 200"
printf '%s\n' "$SALUD" | json
printf '%s' "$SALUD" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('estado') == 'ok' else 1)" ||
  fallo "/health no reporta estado ok"
echo
echo "\$ curl -s -X POST $URL/predict -H 'Content-Type: application/json' -d @$EVID/peticion-predict.json"
RESP="$(pedir_200 -X POST "$URL/predict" -H 'Content-Type: application/json' -d @"$EVID/peticion-predict.json")" || falla "/predict no respondió HTTP 200"
printf '%s\n' "$RESP" | json
echo
echo "\$ curl -s -X POST $URL/predict/batch -H 'Content-Type: application/json' -d @$EVID/peticion-batch.json"
LOTE="$(pedir_200 -X POST "$URL/predict/batch" -H 'Content-Type: application/json' -d @"$EVID/peticion-batch.json")" || falla "/predict/batch no respondió HTTP 200"
printf '%s\n' "$LOTE" | json
echo
echo "\$ curl (martes marcado como fin de semana; se espera 422)"
CODIGO="$(sed 's/"es_fin_semana": 0/"es_fin_semana": 1/' "$EVID/peticion-predict.json" |
  curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/predict" -H 'Content-Type: application/json' -d @-)"
echo "HTTP $CODIGO"
[ "$CODIGO" = 422 ] || fallo "la petición incoherente respondió $CODIGO en lugar de 422"
echo

echo "Comparación con $REFERENCIA"
if ! RESP="$RESP" LOTE="$LOTE" python3 - "$REFERENCIA" <<'PY'
import json, math, os, sys
ref = json.load(open(sys.argv[1]))
uno = json.loads(os.environ["RESP"])["viajes_esperados"]
lote = json.loads(os.environ["LOTE"])
valores = [p["viajes_esperados"] for p in lote["predicciones"]]
indices = [p["indice"] for p in lote["predicciones"]]
esperado = ref["peticion_batch"]["viajes_esperados_en_orden"]
# Misma tolerancia que test_api_modelo_real.py: permite redondeo entre ARM y x86.
def coincide(actual, referencia):
    return math.isfinite(actual) and math.isclose(actual, referencia, rel_tol=1e-12)

ok_uno = coincide(uno, ref["peticion_predict"]["viajes_esperados"])
ok_lote = (
    len(valores) == len(esperado)
    and all(coincide(a, e) for a, e in zip(valores, esperado))
    and indices == list(range(len(esperado)))
    and lote["n"] == len(esperado)
)
print(f"/predict: {uno!r}; referencia: {ref['peticion_predict']['viajes_esperados']!r}; coinciden (rel_tol=1e-12): {ok_uno}")
print(f"/predict/batch: {valores!r}; referencia: {esperado!r}; coinciden en orden (rel_tol=1e-12): {ok_lote}")
sys.exit(0 if ok_uno and ok_lote else 1)
PY
then
  fallo "las predicciones no coinciden con la referencia"
fi
echo
echo "\$ $MOTOR logs $NOMBRE"
"$MOTOR" logs "$NOMBRE" 2>&1 | tail -n 15
echo
echo "Evidencia: $SALIDA y $LOG_BUILD"
echo "Para detener: $MOTOR stop $NOMBRE && $MOTOR rm $NOMBRE"
echo
if [ "${#FALLAS[@]}" -gt 0 ]; then
  echo "ACEPTACIÓN: NO PASA (${#FALLAS[@]} fallas). No se toma captura."
  printf '  - %s\n' "${FALLAS[@]}"
  exit 1
fi
# Resumen al final para que la captura muestre el contenedor y una respuesta real.
echo "==================== CAPTURA: contenedor en ejecución y respuesta ===================="
echo "\$ $MOTOR ps --filter name=$NOMBRE --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"
"$MOTOR" ps --filter "name=$NOMBRE" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
echo "\$ curl -s -X POST $URL/predict -H 'Content-Type: application/json' -d @$EVID/peticion-predict.json"
RESP_CAPTURA="$(pedir_200 -X POST "$URL/predict" -H 'Content-Type: application/json' -d @"$EVID/peticion-predict.json")" || falla "la petición de la captura no respondió HTTP 200"
printf '%s\n' "$RESP_CAPTURA" | json
printf '%s' "$RESP_CAPTURA" | validar_prediccion || falla "la predicción de la captura no coincide con la referencia"
echo "ACEPTACIÓN: OK (todas las comprobaciones pasaron)"

# --- Captura de la ventana de Terminal (macOS) ---------------------------------
if [ "$CAPTURA" = 1 ] && [ "$(uname)" = Darwin ] && [ "${TERM_PROGRAM:-}" = Apple_Terminal ]; then
  if [ -e "$PNG" ]; then
    echo "No se tomó captura: $PNG ya existe"
  else
    sleep 1
    VENTANA="$(osascript -e 'tell application "Terminal" to id of front window' 2>/dev/null || true)"
    if [ -n "$VENTANA" ] && screencapture -x -o -l "$VENTANA" "$PNG" 2>/dev/null && [ -s "$PNG" ]; then
      echo "Captura guardada en $PNG (revísenla antes de entregarla)"
    else
      echo "No se pudo capturar la ventana (permiso de Grabación de pantalla para Terminal)."
      echo "Tomen la captura a mano con Cmd+Shift+4, barra espaciadora y clic en esta ventana."
    fi
  fi
else
  echo "Tomen la captura de esta ventana (Cmd+Shift+4, barra espaciadora, clic) y guárdenla como $PNG"
fi
