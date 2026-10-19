# P3: configuracion congelada y reproduccion

La decision anterior a la primera prueba se conserva en `p3_frozen.json`:
KMeans k=3, 369 estaciones elegibles con >=180 dias registrados en entrenamiento,
semilla42, n_init20, max_iter300, Lloyd. Ajuste solo con entrenamiento
[2025-08-01,2026-08-01). Agosto es validacion y septiembre prueba.
No se seleccionan configuraciones usando septiembre.

Cada direccion tiene168variables weekday/hora normalizadas por separado.
Se divide cada conteo entre ocurrencias calendario del weekday y despues se
normaliza cada bloque; se usa sqrt(p)/sqrt(2) y distancia euclidea. Esto da igual
peso a salidas y llegadas. No entran IDs, volumen ni ubicacion.
En cada mes se conservan las estaciones ya conocidas desde el dia1; se requieren
14dias registrados,100movimientos, ambos sentidos y7weekdays. Los ceros no prueban
demanda cero, cobertura completa ni disponibilidad. Los grupos se solapan y son
descriptivos; no representan funcion de estacion ni disponibilidad real.

## Entradas externas a Git

Datos y artefactos grandes quedan locales, conforme a `.gitignore`:

- `data/processed/datos_abiertos_AAAA_MM.parquet`, `estaciones.parquet` y
  `manifests/mibici_processed.json` de la ingesta existente (version cd917f9aa025).
- `reports/p3-preparacion-20261018/perfiles_entrenamiento.parquet`: preparacion
  revisada, checksum congelado en JSON. Puede regenerarse con
  la sección 1 de `notebooks/2.0-fvep-p3-seleccion.ipynb` y comprobar su checksum antes
  de usarlo; no sobrescribir las evidencias originales.
- `reports/p3-validacion-agosto-tentativa-20261018/perfiles_agosto.parquet` y
  `resumen.json`: evidencia mensual revisada, generada por la sección 4 de `notebooks/2.0-fvep-p3-seleccion.ipynb`.
- Los CSV de sensibilidad y recurrencia previa se conservan en sus carpetas
  `reports/p3-sensibilidad-historia-20261018` y
  `reports/p3-validacion-agosto-tentativa-20261018`, para adjuntarlos a MLflow.

No es un ejecutor de descarga: preparar estos insumos con los modulos P3
versionados y la ingesta del proyecto. `requirements.txt` fija dependencias;
el JSON registra versiones usadas, checksum de centroides y resultados esperados.
La comparacion byte a byte de CSV depende de ese entorno y serializacion.
Una discrepancia detiene la reproduccion; nunca habilita tuning automatico.

## Comandos

Desde raiz del repositorio, con el entorno del proyecto activo:

```powershell
python -m pytest -q
python -m ruff check mibici tests
python -m mibici.modeling.p3 --reproducir
```

El código de P3 está organizado por responsabilidad: lectura verificada y revisión
en `mibici/dataset.py`, perfiles y cohortes en `mibici/features.py`, comparación y
evaluación en `mibici/evaluation.py`, gráficas y mapa en `mibici/plots.py`, y el
experimento base y la auditoría en `mibici/modeling/p3.py`. Los antiguos scripts se
sustituyen así:

| Antes | Ahora |
|---|---|
| `python -m mibici.review_p3 <salida>` | Sección 1 de `notebooks/2.0-fvep-p3-seleccion.ipynb` |
| `python -m mibici.compare_p3 <entrada> <salida>` | Sección 2 del mismo notebook |
| `python -m mibici.history_p3 <entrada> <anterior> <salida>` | Sección 3 del mismo notebook |
| `python -m mibici.validate_p3 <preparacion> <anterior> <salida>` | Sección 4 del mismo notebook (requiere `PROTOCOLO_PREVIO.txt` escrito a mano) |
| `python -m mibici.modeling.p3 [--solo-congelar \| --reproducir]` | Sin cambio; también desde `notebooks/3.0-fvep-p3-auditoria.ipynb` |

La reproduccion exige Git versionado limpio. Crea una salida NUEVA
`reports/p3-reproduccion-limpia-20261018`, reajusta solo el mismo modelo train,
verifica centroides congelados y ejecuta una sola repeticion autorizada de prueba
sin cambiar parametros/cohorte/reglas. Compara todos los metricos numericos con
rtol=atol=1e-12 y SHA-256 de asignaciones train/val/test. Guarda la comprobacion,
modelo, perfiles, mapa y graficas; registra un run nuevo con el commit exacto y
git_dirty real. Un marcador bloquea repeticiones accidentales o sobrescrituras.
El run historico sucio `fbcf2809ea3b4132ba605fe6c31e37fb` permanece intacto.

La primera evaluacion protegida en `reports/p3-evaluacion-congelada-20261018`
se ejecuta con `python -m mibici.modeling.p3`; si ya existe su
marcador, se detiene. La entrada compartida
`python -m mibici.modeling.train --pregunta P3` usa `mibici/modeling/p3.py`, que
reajusta la misma configuracion sin marcadores ni evidencias previas. P3 verifica sus propios archivos por mes.
P1/P2 no se ejecutan al pedir solo P3. No borrar marcadores para repetir prueba.

## Evidencia e interpretacion

El mapa es de puntos usando `nomenclatura_2026_09.csv`, no cartografia base ni
prueba de ubicacion historica. El CSV identifica ubicaciones faltantes; las369
estaciones del modelo revisado tienen coordenadas. Historial largo no repara
cobertura de fuente. Distancias a p95train son descriptivas, no alertas calibradas.
Comparar un mes con un anio mezcla ruido de muestreo y cambios temporales.
No se debe presentar silhouette bajo como grupos bien separados.

Solo codigo, pruebas, este documento y configuracion compacta se versionan.
No agregar Parquet, CSV masivos, modelos, SQLite, capturas ni datos personales.
