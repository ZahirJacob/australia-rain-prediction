# Inferencia con Docker

## Qué hace

Esta imagen permite ejecutar la inferencia del modelo de predicción de lluvia
en un entorno reproducible. A partir de observaciones meteorológicas en formato
CSV, genera la predicción de lluvia para el día siguiente y su probabilidad.

## Componentes utilizados

La inferencia utiliza:

- el modelo Keras `artifacts/selected_nn_model.keras`;
- el preprocesador `artifacts/selected_nn_preprocessor.joblib`;
- el threshold definido en `artifacts/oof_selection.json`;
- el script de entrada `docker/inferencia.py`.

## Build

La imagen debe construirse desde la raíz del repositorio:

```bash
docker build --file docker/Dockerfile --tag prediccion-lluvia .
```

## Ejecución

Con el archivo de entrada ubicado en `docker/files/input.csv`, ejecutar desde la
raíz del repositorio:

```bash
docker run --rm --mount type=bind,source="${PWD}/docker/files",target=/files prediccion-lluvia
```

## Entrada y salida

El contenedor lee `/files/input.csv` y genera `/files/output.csv`, que queda
disponible como `docker/files/output.csv` en el equipo local.

La salida contiene dos columnas:

- `prediccion`: `Llueve` o `No llueve`;
- `Probabilidad`: probabilidad estimada de lluvia para el día siguiente.

## Validación

La inferencia Docker fue comparada con la inferencia local sobre 512
observaciones del conjunto de desarrollo. Ambas produjeron las mismas etiquetas
y diferencias numéricas despreciables en las probabilidades.

Los detalles técnicos de la comparación están disponibles en
`artifacts/docker_inference_parity.json`.
