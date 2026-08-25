# Inferencia - Prediccion de lluvia en Australia

La imagen utiliza exclusivamente el modelo final `nn_config_5`, el
preprocesador `joblib` y el threshold del manifiesto OOF. No contiene PyCaret ni
artefactos del flujo anterior.

## Bundle incluido

- `artifacts/oof_selection.json`
- `artifacts/selected_nn_preprocessor.joblib`
- `artifacts/selected_nn_model.keras`
- `weather_preprocessing.py`
- `docker/inferencia.py`

La inferencia solo carga componentes congelados, ejecuta
`preprocessor.transform`, obtiene `P(RainTomorrow=Yes)` y aplica el threshold
`0.5957959890365601`. No realiza `fit` ni reentrenamiento.

## Verificacion local

Desde la raiz del repositorio:

```bash
python docker/inferencia.py --input docker/files/input.csv --output docker/files/output_nn.csv --artifacts-dir artifacts
python verify_local_inference_parity.py
```

## Build y ejecucion

El contexto debe ser la raiz del repositorio:

```bash
docker build --file docker/Dockerfile --tag prediccion-lluvia .
docker run --rm --mount type=bind,source="${PWD}/docker/files",target=/files prediccion-lluvia
```

El contenedor lee `/files/input.csv` y escribe `/files/output.csv`, con columnas
`prediccion` y `Probabilidad`.

## Paridad validada

La comprobacion completa se ejecuta desde la raiz:

```bash
python verify_docker_inference_parity.py
```

La verificacion real de Docker quedo completada sobre 512 filas de desarrollo:

- interseccion con test final: 0;
- etiquetas identicas: si;
- diferencia absoluta maxima de probabilidad: `8.772735593520764e-08`;
- tolerancias: `atol=1e-7`, `rtol=1e-6`.

La evidencia y el ID de la imagen estan en
`artifacts/docker_inference_parity.json`.
