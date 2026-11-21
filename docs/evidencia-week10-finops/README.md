# Semana 10 · FinOps de inferencia P1 MiBici

**Conclusión.** Con un objetivo de comparación de 250 ms para la mediana del p95 de `/predict`, la variante `OMP_NUM_THREADS=1` sostuvo 2.08 veces el rendimiento del escenario base en los mejores tramos observados que cumplen ese criterio. Al aplicar la misma tarifa de cómputo a ambos escenarios, el costo estimado por millón de inferencias baja de **USD 1.54 a USD 0.74** (52.0% menos). Esto estima el costo unitario bajo carga sostenida; no demuestra un ahorro en la factura ni un rendimiento equivalente en AWS.

## 1. Evidencia y criterio de comparación

Datos de Francisco en el [PR #31](https://github.com/MNA-MLOps-Grupo26/TC5061-Proyecto-Equipo26/pull/31), fijados al [resumen JSON de Locust](https://github.com/MNA-MLOps-Grupo26/TC5061-Proyecto-Equipo26/blob/4e2133e78534ebd16e30d1e5bfb583d71c7a5588/docs/evidencia-week10-locust/resumen-carga.json). Ambas configuraciones ejecutaron la misma API y modelo P1 localmente en un MacBook Pro M3 Pro, con la misma petición de control y 0 fallas observadas. El resumen separa `/predict` de `/health`; las comprobaciones de salud no cuentan como inferencias. Se hizo una corrida de 300 s por variante: rampa de 0 a 200 usuarios y 60 s a 200 usuarios.

Se tomó el tramo de 30 s con mayor tasa media de `/predict` entre los tramos cuya **mediana de las muestras de p95** es de 250 ms o menos. Ese umbral es un supuesto para esta comparación, no un requisito del curso ni un SLO certificado. No implica que el p95 de todas las peticiones del tramo sea el mismo ni que se cumpla el umbral en cada instante.

| Métrica | Compose base | `OMP_NUM_THREADS=1` |
|---|---:|---:|
| Tramo de usuarios simulados | 76–100 | 151–175 |
| Duración del tramo | 30 s | 30 s |
| Inferencias por segundo, media | 64.29 | 133.90 |
| Latencia p95, mediana de las muestras | 240 ms | 245 ms |
| CPU media de la API en el tramo (100% ≈ 1 núcleo lógico) | 544.2% | 120.4% |
| Inferencias por hora si se mantiene esa tasa | 231,444 | 482,040 |
| USD por inferencia, solo cómputo | 0.0000015425 | 0.0000007406 |
| USD por millón de inferencias, solo cómputo | 1.54 | 0.74 |

Los tramos representan la **mayor capacidad observada a un criterio de latencia común**, no el mismo número de usuarios. Como comprobación a igual tramo (76–100 usuarios), la variante registró 73.08 frente a 64.29 inferencias/s y p95 de 40 frente a 240 ms: aproximadamente USD 1.36 frente a USD 1.54 por millón, una reducción estimada de 12.0%. A 200 usuarios, el p95 mediano fue 2,200 ms frente a 400 ms; ninguna configuración cumple allí el objetivo de 250 ms.

## 2. Tarifa y cálculo reproducible

Instancia de referencia: **Amazon EC2 c7i.2xlarge**, 8 vCPU, 16 GiB, Linux x86-64, tenencia compartida, On-Demand, región **US East (N. Virginia), `us-east-1`**. Tarifa **USD 0.357/h**, consultada el **21 de noviembre de 2026** en el [AWS Price List de EC2 para us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.csv), versión 20261121004816, SKU `D23DFG4YVGYMSX7G`, código de tarifa `D23DFG4YVGYMSX7G.JRTCKXETXF.6YS6EN2CT7`. Véanse también las [especificaciones de EC2](https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html). Se usa una sola tarifa como referencia para aislar el efecto de la configuración; la instancia no se desplegó ni se midió.

```text
inferencias/hora = RPS de /predict × 3,600
USD/inferencia = 0.357 USD/hora ÷ inferencias/hora
USD/millón = USD/inferencia × 1,000,000

Base:  0.357 ÷ (64.29 × 3,600) × 1,000,000 = 1.54249
OMP=1: 0.357 ÷ (133.90 × 3,600) × 1,000,000 = 0.74060
```

La VM costaría **USD 260.61** por 730 horas encendida en cualquiera de las dos configuraciones. A demanda fija y con la misma VM encendida durante el mismo tiempo no baja la factura por aplicar el cambio. El beneficio económico aparece si el rendimiento adicional permite reducir horas facturadas o usar menos capacidad aprovisionada. El cálculo no incluye EBS, red, balanceador, observabilidad ni impuestos; para estimar costo con tráfico real, dividir el costo del período entre las inferencias exitosas observadas en ese período.

## 3. Optimizaciones propuestas

1. **Evaluar `OMP_NUM_THREADS=1` como valor de despliegue para la API.** Redujo CPU y latencia en la prueba y mantuvo la misma predicción de control. Hoy solo se aplicó con un override de pruebas. Comparar al menos varias corridas repetibles en la instancia objetivo y con peticiones representativas antes de cambiar el valor por defecto.
2. **Dimensionar MLflow aparte de la inferencia.** En el benchmark, la memoria media de la API fue 217 y 205 MiB, mientras MLflow usó alrededor de 1,660 y 1,657 MiB y no intervino en la ruta de predicción. Medir memoria pico y CPU de ambos, conservar persistencia y validar una API con menos vCPU y MLflow ajustado a su carga. Separarlos puede añadir costo fijo; comparar la suma de los dos servicios contra una VM compartida antes de decidir.
3. **Medir antes de recortar capacidad.** Repetir pruebas con el generador de carga en otro host, vigilar p95, errores y recursos, y comparar tamaño de instancia y número de workers. Esta prueba única en ARM local, con Locust y contenedores compartiendo máquina, no predice por sí sola resultados en x86-64 ni prueba la capacidad máxima de la variante.
