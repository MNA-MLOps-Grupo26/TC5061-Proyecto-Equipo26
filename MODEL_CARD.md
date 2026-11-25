# Model Card | Proyecto MiBici

**Equipo 26 · TC5061 · Modelo P1**

## 1. Propósito y alcance

El modelo P1 estima el número de viajes que iniciarán al día siguiente en cada estación de MiBici. Se desarrolló para apoyar la planeación de la operación e identificar estaciones con mayor demanda prevista. El pronóstico puede orientar la priorización de estaciones, pero no mide la disponibilidad de bicicletas. Para tomar decisiones de rebalanceo, debe complementarse con el inventario y las llegadas esperadas.

## 2. Método y trazabilidad

Se utilizó **HistGradientBoosting con pérdida Poisson**, un modelo basado en árboles de decisión que estima conteos. El pronóstico incorpora 18 variables, entre ellas el historial de salidas y llegadas, el calendario, la capacidad y la ubicación de cada estación.

- **Registro en MLflow:** `mibici-equipo26-p1-demand`, versión 1, con alias `champion` al momento de la exportación del 7 de noviembre de 2026.
- **Corrida de entrenamiento:** `51ef51ad5b534386a3c020692887c6bc`.
- **Commit de entrenamiento:** `8c7c709`. **Versión de datos:** `cd917f9aa025`.

La [prueba documentada del contenedor](docs/evidencia-api/contenedor-docker.txt) muestra que `/health` reportó esta versión y esta corrida, con los hashes y la firma del modelo verificados. Antes de grabar la demostración final, se debe confirmar que el entorno de grabación sirve el mismo artefacto.

## 3. Evaluación e interpretación de resultados

Se utilizaron datos de agosto de 2025 a julio de 2026 para el entrenamiento, agosto de 2026 para la validación y septiembre de 2026 para la prueba histórica. En la validación, el pronóstico se desvió en promedio **4.85 viajes por estación al día**, independientemente de si sobreestimó o subestimó la demanda (MAE). El RMSE, que da más peso a los errores grandes, fue de **8.17 viajes**. El error absoluto acumulado representó **17.81% del total de viajes observados** (WAPE).

En la revisión retrospectiva de octubre, el error absoluto promedio fue de **8.05 viajes por estación al día**, con un RMSE de **15.89 viajes** y un WAPE de **18.53%**, sobre **14,618 observaciones estación-día**. Como ese periodo ya se había analizado durante el proyecto, los resultados sirven para revisar el desempeño en ese mes, no como evidencia de una prueba independiente.

## 4. Limitaciones y monitoreo

Las predicciones de estaciones nuevas o con poco historial requieren mayor revisión. Un registro de cero viajes puede reflejar baja demanda, un cierre temporal o datos incompletos. Se recomienda monitorear la calidad de los datos y el error por estación, además de utilizar Evidently para detectar cambios en la distribución de las variables de entrada (*drift*). Si se detecta un cambio, conviene revisar su causa y su efecto en el error antes de decidir si se debe reentrenar el modelo.

## 5. Recomendación operativa

La versión candidata v3 obtuvo mejores resultados puntuales en octubre, pero los intervalos de la comparación pareada incluyeron cero. En términos prácticos, la evidencia disponible no permite concluir que la mejora se mantendría en otro periodo. Se recomienda **mantener v1 como modelo de referencia** y volver a evaluar v3 con datos de un periodo que no se haya utilizado para desarrollar ni comparar las versiones.

**Evidencia:** [Registro del modelo](docs/evidencia-api/exportacion-registry.json) · [Resultados de validación](docs/evidencia-api/recibo-registro-p1.json) · [Drift y comparación A/B](docs/week9.md)
