# Pruebas de carga con Locust (Semana 10)

Prueba de carga del sistema levantado con `docker-compose.yml` (API P1 y MLflow).
Compara la configuración del repositorio con una variante que limita el modelo a un
hilo de OpenMP por predicción (`OMP_NUM_THREADS=1`). No se modifican la API, el modelo
ni el Compose del repositorio: la variante se aplica con un archivo de override solo
durante la prueba.

## Cómo reproducirla

Requisitos: Docker con Compose, el modelo exportado en `models/p1/` (ver README
principal) y un entorno aparte con `requirements-locust.txt`.

```bash
python -m pip install -r requirements-locust.txt
python scripts/prueba_carga.py --config base
python scripts/prueba_carga.py --config omp1
python scripts/graficas_carga.py
```

`prueba_carga.py` levanta Compose, espera `/health`, verifica una predicción de control,
corre Locust sin interfaz y registra `docker stats` al mismo tiempo. Al terminar apaga
Compose (el volumen de MLflow se conserva). Los resultados crudos quedan en
`reports/locust/<config>/` (CSV de Locust, `reporte.html` y `docker-stats.jsonl`) y no se
suben a Git. `graficas_carga.py` genera las gráficas y `resumen-carga.json` en esta carpeta.

## Escenario

| Parámetro | Valor |
|---|---|
| Usuarios | Rampa de 0 a 200 en 240 s (0.83 usuarios/s), luego 60 s con 200 |
| Duración por configuración | 300 s |
| Mezcla de peticiones | `/predict` 10 : `/health` 1 |
| Espera entre peticiones de cada usuario | 0.5 a 1.5 s |
| Petición | `docs/evidencia-api/peticion-predict.json` (estación 21) |
| Validación | HTTP 200 y campo `viajes_esperados` |

La rampa permite ver en una sola corrida dónde empieza a degradarse el servicio.

**Entorno:** MacBook Pro con Apple M3 Pro, macOS 26.7.3 y Docker 29.8.0. La VM de Docker
tiene 11 CPU y 7.7 GiB, y Locust 2.47.0 corre en el mismo equipo. El modelo es
`mibici-equipo26-p1-demand` versión 1 (run `51ef51ad…`). La predicción de control fue
122.5596 en ambas configuraciones.

## Resultados

| Indicador (`/predict`) | Compose base | `OMP_NUM_THREADS=1` |
|---|---:|---:|
| Peticiones atendidas | 15,873 | 29,208 |
| Fallas | 0 | 0 |
| Peticiones por segundo, máximo sostenido | ~73 (101–125 usuarios) | ~146 (200 usuarios, sin saturar) |
| Peticiones por segundo con 200 usuarios | 60.4 | 146.9 |
| Latencia p50 con 200 usuarios | 2,000 ms | 260 ms |
| Latencia p95 con 200 usuarios | 2,200 ms | 400 ms |
| CPU media de la API (100% = 1 núcleo) | 496% | 85% |
| Memoria media de la API | 217 MiB | 205 MiB |
| Memoria media de MLflow | 1,660 MiB | 1,657 MiB |

Detalle por tramo de usuarios (medianas por tramo; fuente: `resumen-carga.json`):

| Usuarios | Base: pet./s | Base: p50 / p95 (ms) | Base: CPU | OMP1: pet./s | OMP1: p50 / p95 (ms) | OMP1: CPU |
|---|---:|---:|---:|---:|---:|---:|
| 1–25 | 7.3 | 69 / 110 | 119% | 8.4 | 19 / 30 | 15% |
| 26–50 | 26.5 | 110 / 140 | 318% | 28.6 | 13 / 35 | 27% |
| 51–75 | 46.6 | 140 / 160 | 465% | 50.4 | 10 / 37 | 36% |
| 76–100 | 64.3 | 160 / 240 | 544% | 73.1 | 10 / 40 | 45% |
| 101–125 | 73.0 | 305 / 570 | 573% | 95.9 | 13 / 57 | 67% |
| 126–150 | 60.1 | 1,100 / 1,300 | 581% | 115.6 | 35 / 225 | 98% |
| 151–175 | 60.0 | 1,400 / 1,500 | 575% | 133.9 | 51 / 245 | 120% |
| 176–200 | 59.6 | 2,000 / 2,200 | 579% | 146.0 | 250 / 390 | 143% |

Gráficas: `latencia_vs_usuarios.png`, `rps_vs_usuarios.png` y `cpu_api_vs_usuarios.png`.

## Interpretación

1. **Configuración base: se satura cerca de 100–125 usuarios.** Hasta unos 100 usuarios
   atiende la carga con un p95 menor de 250 ms. A partir de ahí la CPU de la API se queda
   en unos 5.8 núcleos y el rendimiento **baja** de ~73 a ~60 peticiones por segundo,
   mientras la latencia sube a 1–2 s. El servicio no falla, pero degrada.
2. **Cuello de botella: hilos de OpenMP del modelo.** Por defecto, cada predicción de
   una sola fila usa todos los hilos disponibles del modelo HGB. Con muchas peticiones
   simultáneas, los hilos compiten entre sí: el sistema consume más CPU y atiende menos.
   Esto se ve en que la CPU sube casi en línea recta desde los primeros usuarios.
3. **Con `OMP_NUM_THREADS=1`, la misma predicción cuesta mucho menos.** Atiende el doble
   de peticiones (146 frente a 60 por segundo con 200 usuarios), con latencias 7–8 veces
   menores y unas 6 veces menos CPU. El rendimiento sigue subiendo con la carga, así que
   200 usuarios no alcanzaron su límite. Con p95 por debajo de 250 ms sostiene unas
   **134 peticiones por segundo con ~1.2 núcleos**.
4. **MLflow ocupa ~1.66 GiB sin participar en la inferencia.** Es el 89% de la memoria
   del sistema durante la prueba, y su CPU es casi nula.

## Datos para FinOps

| Configuración | Capacidad con p95 < 250 ms | CPU de la API en ese punto | Peticiones/s por núcleo |
|---|---:|---:|---:|
| Base | ~64 pet./s (76–100 usuarios) | ~5.4 núcleos | ~12 |
| `OMP_NUM_THREADS=1` | ~134 pet./s (151–175 usuarios) | ~1.2 núcleos | ~111 |

Con estos datos el costo por inferencia se calcula como costo por hora de la instancia
entre las inferencias por hora sostenidas. Áreas de mejora identificadas:

1. fijar `OMP_NUM_THREADS=1` en el servicio `api`;
2. separar MLflow del servicio de inferencia o dimensionarlo aparte;
3. medir la capacidad de `OMP_NUM_THREADS=1` con más usuarios o varios workers de uvicorn.

## Limitaciones

- Es una sola corrida por configuración, en una laptop. Locust y los contenedores
  comparten el equipo, así que los valores absolutos dependen del hardware y de la carga
  del momento. Las tendencias (punto de saturación y mejora relativa) son lo comparable.
- Siempre se envía la misma petición. Una mezcla de estaciones y fechas no cambiaría el
  costo del modelo por predicción, pero sería más realista.
- Con 200 usuarios no se alcanzó el límite de `OMP_NUM_THREADS=1`.
- La API corre con un solo proceso de uvicorn, como en el Compose del repositorio.
