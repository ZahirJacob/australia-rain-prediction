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

El proyecto utiliza `weatherAUS_2026C1.csv`, con observaciones meteorológicas
diarias de 49 ubicaciones australianas entre noviembre de 2007 y junio de 2017.

- 145.412 registros originales.
- 142.153 observaciones con `RainTomorrow` disponible para modelado.
- Variables de temperatura, humedad, presión, viento, nubosidad, lluvia,
  evaporación, horas de sol, fecha y ubicación.
- Variable objetivo: `RainTomorrow` (`Yes` / `No`).

La columna `RainfallTomorrow`, que contiene información del día que se intenta
predecir, se excluye de las variables de entrada para evitar fuga de datos.

## Metodología

Se reservó un 20% de los datos como test final mediante un split aleatorio
estratificado con `random_state=42`. El 80% restante se utilizó como conjunto de
desarrollo.

La selección se realizó con una validación cruzada estratificada de 5 folds y
predicciones out-of-fold (OOF). Cada observación de desarrollo fue evaluada por
un modelo que no la había visto durante su ajuste. Se compararon 23 candidatos,
incluyendo modelos lineales, árboles, ensembles y redes neuronales.

El pipeline incorpora:

- variables estacionales derivadas de la fecha;
- información geográfica de las ubicaciones;
- imputación por mediana, moda y KNN según el tipo de variable;
- codificación one-hot y escalado;
- balanceo con SMOTE aplicado solo sobre el entrenamiento de cada fold.

Todas las transformaciones que aprenden parámetros se ajustan dentro de cada
fold. Los folds de validación y el test final reciben únicamente las
transformaciones ya aprendidas.

El modelo, sus hiperparámetros y el threshold se seleccionaron maximizando el
F1 positivo sobre las probabilidades OOF. Como evaluación complementaria, se
utilizaron cinco ventanas temporales expansivas para medir el rendimiento sobre
observaciones cronológicamente posteriores.

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
├── weather_preprocessing.py         # Preprocesamiento reutilizable
├── oof_model_selection.py           # Selección de modelos mediante CV y OOF
├── temporal_robustness_evaluation.py # Evaluación temporal expansiva
├── artifacts/                       # Modelo, preprocesador y resultados
├── docker/                          # Imagen y entrada de inferencia
├── verify_*.py                      # Comprobaciones metodológicas y de paridad
├── requirements.txt                 # Dependencias del flujo principal
└── requirements-notebook.txt        # Dependencias exploratorias opcionales
```

## Reproducibilidad

Las particiones y los algoritmos utilizan una semilla fija (`42`), y las
dependencias principales están versionadas. El repositorio incluye el modelo
Keras seleccionado, el preprocesador serializado, el manifiesto con el
threshold y los resultados OOF, temporales y de test utilizados en este README.

El repositorio también incluye scripts para reproducir y validar el pipeline
experimental, además de comprobar la paridad entre la inferencia local y
Docker.
