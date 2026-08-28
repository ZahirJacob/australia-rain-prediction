# Predicción de lluvia en Australia

Proyecto de Machine Learning para predecir si lloverá al día siguiente en
distintas localidades de Australia. El trabajo abarca análisis exploratorio,
preprocesamiento, comparación de modelos, selección mediante validación
cruzada, evaluación temporal e inferencia reproducible con Docker.

## Problema

El objetivo es estimar la variable binaria `RainTomorrow` utilizando únicamente
información meteorológica disponible durante el día actual. Es un problema
desbalanceado: los días con lluvia representan aproximadamente el 22,4% de las
observaciones, por lo que se utilizó el F1 de la clase positiva como métrica
principal.

Además de F1, se reportan precision, recall, PR-AUC y ROC-AUC para describir el
comportamiento del modelo desde distintas perspectivas.

## Dataset

El proyecto utiliza `data/weatherAUS_2026C1.csv`, con observaciones meteorológicas
diarias de 49 ubicaciones australianas entre noviembre de 2007 y junio de 2017.

- 145.412 registros originales.
- 142.153 observaciones con `RainTomorrow` disponible para modelado.
- Variables de temperatura, humedad, presión, viento, nubosidad, lluvia,
  evaporación, horas de sol, fecha y ubicación.
- Variable objetivo: `RainTomorrow` (`Yes` / `No`). Tanto `RainToday` como
  `RainTomorrow` valen `Yes` cuando la lluvia del día supera 1 mm (en el
  dataset hay unas 1.750 filas con `No` y lluvia entre 0 y 1 mm).

La columna `RainfallTomorrow`, que contiene información del día que se intenta
predecir, se excluye de las variables de entrada para evitar fuga de datos.

## Metodología

Se reservó un 20% de los datos como test final mediante un split aleatorio
estratificado con `random_state=42`. El 80% restante se utilizó como conjunto de
desarrollo.

La selección se realizó con una validación cruzada estratificada de 5 folds y
predicciones out-of-fold (OOF). Cada observación de desarrollo fue evaluada por
un modelo que no la había visto durante su ajuste. Se compararon 23
configuraciones candidatas, incluyendo modelos lineales, árboles, ensembles y
redes neuronales. De ellas, 16 son distintas: las 7 variantes de regresión
logística con `class_weight="balanced"` producen exactamente las mismas
probabilidades que sus pares sin ponderación, porque el balanceo con SMOTE ya
iguala las clases antes del ajuste.

El pipeline incorpora:

- variables estacionales derivadas de la fecha;
- información geográfica de las ubicaciones;
- imputación por mediana (por región y estación), moda y KNN según el tipo de
  variable; la imputación KNN se aplica solo a `Evaporation`, `Cloud9am` y
  `Cloud3pm`, usando esas tres columnas entre sí como vecindad, por lo que en
  las estaciones que nunca reportan las tres (16 no reportan evaporación, 12
  no reportan nubosidad) equivale en la práctica a imputar la media;
- codificación one-hot y escalado;
- balanceo con SMOTE aplicado solo sobre el entrenamiento de cada fold.

Todas las transformaciones que aprenden parámetros se ajustan dentro de cada
fold. Los folds de validación y el test final reciben únicamente las
transformaciones ya aprendidas.

El modelo, sus hiperparámetros y el threshold se seleccionaron maximizando el
F1 positivo sobre las probabilidades OOF. Como evaluación complementaria, se
utilizaron cinco ventanas temporales expansivas para medir el rendimiento sobre
observaciones cronológicamente posteriores.

Sobre la partición principal conviene tener presente que los datos son series
diarias por estación y el split es aleatorio por fila: `RainTomorrow` de un día
coincide con `RainToday` del día siguiente en la misma estación (se cumple en
el 100 % de los pares de días consecutivos), de modo que el conjunto de
desarrollo contiene los días vecinos de cada día de test. Esto no es fuga
directa de la etiqueta (el modelo nunca ve la fila del día siguiente al
predecir), pero hace que el F1 OOF y el del test sean algo optimistas para un
uso de pronóstico real. La evaluación temporal expansiva (F1 0,640) es la
referencia más adecuada para ese uso, y es la que reporta la aplicación web
derivada de este proyecto.

## Modelo y resultados

El modelo seleccionado fue una red neuronal con la siguiente configuración:

- 50 epochs;
- batch size de 128;
- dropout de 0,4;
- learning rate de 0,0005;
- threshold de clasificación de `0.596`.

| Evaluación | F1 positivo | Precision | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| OOF sobre desarrollo | 0.6592 | 0.6251 | 0.6973 | 0.7214 | 0.8819 |
| Temporal expansiva (media) | 0.6395 | 0.6118 | 0.6704 | 0.7075 | 0.8707 |
| **Test final** | **0.6532** | **0.6396** | **0.6672** | **0.7340** | **0.8851** |

El test final contiene 28.431 observaciones. La cercanía entre el F1 de test y
el F1 OOF, junto con la evaluación temporal, permite comparar el rendimiento
del modelo bajo el split principal y frente al paso del tiempo.

Las cuatro mejores configuraciones OOF quedaron dentro de 0,0023 puntos de F1
(`nn_config_5` 0,6592, `nn_config_3` 0,6578, `rf_default` 0,6574,
`nn_config_4` 0,6569), mientras que la desviación entre folds del ganador es de
aproximadamente 0,003. La ventaja de la red neuronal sobre el Random Forest,
por lo tanto, no está estadísticamente separada del ruido de la validación; la
elección se resolvió por el criterio declarado (máximo F1 OOF con desempate
por threshold).

## Inferencia con Docker

La imagen Docker empaqueta el preprocesador, la red neuronal y el threshold
seleccionado para reproducir el mismo flujo de inferencia en un entorno
aislado.

Desde la raíz del repositorio:

```bash
docker build --file docker/Dockerfile --tag prediccion-lluvia .
docker run --rm \
  --mount type=bind,source="${PWD}/docker/files",target=/files \
  prediccion-lluvia
```

El contenedor lee `docker/files/input.csv` y genera
`docker/files/output.csv`. La salida contiene:

- `prediccion`: `Llueve` o `No llueve`;
- `Probabilidad`: probabilidad estimada de `RainTomorrow=Yes`.

Las predicciones del contenedor fueron contrastadas con la inferencia local
sobre 512 observaciones de desarrollo: las etiquetas fueron idénticas y la
diferencia máxima entre probabilidades fue `8.77e-08`.

## Instalación y uso

El dataset y los artefactos binarios se almacenan con Git LFS:

```bash
git lfs install
git lfs pull
```

Para crear el entorno local:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Inferencia local con los artefactos incluidos:

```bash
python docker/inferencia.py \
  --input docker/files/input.csv \
  --output docker/files/output.csv \
  --artifacts-dir artifacts
```

Para ejecutar el notebook completo, con sus visualizaciones y comparaciones
exploratorias, se pueden instalar las dependencias adicionales:

```bash
python -m pip install -r requirements-notebook.txt
```

## Estructura del proyecto

```text
.
├── TP_clasificacion_AA1.ipynb       # Análisis exploratorio y experimentación
├── data/                             # Dataset y coordenadas de ubicaciones
├── src/                              # Preprocesamiento reutilizable
├── scripts/                          # Entrenamiento y evaluaciones
├── verification/                     # Comprobaciones metodológicas y de paridad
├── artifacts/                       # Modelo, preprocesador y resultados
├── docker/                          # Imagen y entrada de inferencia
├── requirements.txt                 # Dependencias del flujo principal
└── requirements-notebook.txt        # Dependencias exploratorias opcionales
```

## Reproducibilidad y auditoría

Las particiones y los algoritmos utilizan una semilla fija (`42`), y las
dependencias principales están versionadas. El repositorio incluye el modelo
Keras seleccionado, el preprocesador serializado, el manifiesto con el
threshold y los resultados OOF, temporales y de test utilizados en este README.

Conviene distinguir dos cosas. Los **artefactos están congelados y auditados
por hash**: los scripts de `verification/` recomputan cada métrica reportada a
partir de las predicciones guardadas (`*.npz`), reconstruyen las particiones
desde el CSV y la semilla, y comparan los SHA-256 del modelo y del
preprocesador con los registrados en los reportes. En cambio, **volver a
entrenar no es reproducible bit a bit**: el entrenamiento de Keras en CPU
(oneDNN) no es determinista y Random Forest y SMOTE solo reproducen sus
resultados bajo versiones idénticas de las librerías. Ejecutar de nuevo
`scripts/oof_model_selection.py` genera artefactos distintos que, por diseño,
no pasan las verificaciones encadenadas.

Los entornos no son idénticos entre sí: `requirements.txt` fija `scipy==1.11.4`
(entorno del notebook, Python 3.11) y `docker/requirements.txt` fija
`scipy==1.17.1` (imagen `python:3.12-slim`). La paridad de inferencia entre
ambos fue verificada (diferencia máxima de probabilidad `8.77e-08`), por lo que
la discrepancia no afecta las predicciones, pero es una diferencia a tener en
cuenta al recrear el entorno.

## Notas posteriores a la entrega

Esta sección documenta observaciones surgidas al revisar el proyecto después
de su entrega. No se modificó ningún artefacto científico ni se reentrenó nada.

- **Hash del manifiesto de selección.** Los reportes `temporal_robustness.json`,
  `final_test_metrics.json`, `local_inference_parity.json` y
  `docker_inference_parity.json` registran para `artifacts/oof_selection.json`
  el SHA-256 `7c03609f…`, mientras que el archivo versionado tiene el SHA-256
  `e26db0a2…`. Los hashes del modelo (`041251e4…`) y del preprocesador
  (`4bca0f30…`) sí coinciden. El manifiesto fue editado después de calcularse
  los hashes (probablemente al agregar las claves `frozen_*` al congelar el
  bundle) y no se conserva la versión original, así que la discrepancia no
  puede resolverse sin reescribir evidencia. Consecuencia práctica: los cuatro
  scripts que comparan esos hashes —`verify_final_test_evaluation.py`,
  `verify_temporal_robustness.py`, `verify_local_inference_parity.py` y
  `verify_docker_inference_parity.py`— fallan en esa comprobación, que ahora
  informa qué hash difiere. `verify_oof_selection.py`,
  `verify_fold_local_preprocessing.py` y `verify_test_isolation.py` no dependen
  de ese hash y pasan.
- **Coordenadas de Richmond.** `data/location_coordinates.json` ubica
  `Richmond` en Richmond, Victoria (-37,81, 144,99); la estación del dataset es
  Richmond RAAF, Nueva Gales del Sur (aprox. -33,60, 150,78). Afecta solo la
  asignación de región climática de esa estación. No se corrige aquí porque el
  archivo es entrada del preprocesador congelado y cambiarlo rompería la cadena
  de hashes.
- **Rutas absolutas en reportes.** Los campos informativos `metrics_artifact`,
  `predictions_artifact` y `winner_source` contenían rutas absolutas de la
  máquina donde se ejecutó el pipeline; se reemplazaron por rutas relativas al
  repositorio y los scripts que los escriben ahora generan rutas relativas.
  Ningún script de verificación lee esos campos.
