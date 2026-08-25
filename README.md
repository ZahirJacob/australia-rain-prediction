# Prediccion de lluvia del dia siguiente en Australia

Proyecto de clasificacion binaria para estimar `RainTomorrow` a partir de
observaciones meteorologicas disponibles al momento de la prediccion. El flujo
oficial conserva un split aleatorio estratificado como protocolo principal y
una evaluacion temporal expansiva como analisis secundario de robustez.

## Estado final validado

- Modelo: red neuronal `nn_config_5`.
- Hiperparametros: 50 epochs, batch size 128, dropout 0.4 y learning rate
  0.0005.
- Threshold congelado: `0.5957959890365601`.
- Criterio principal: F1 de la clase positiva.
- Seleccion: CV estratificada de 5 folds con probabilidades OOF sobre todo el
  conjunto de desarrollo.
- Test final: reservado antes de la seleccion y evaluado una unica vez.

| Evaluacion | F1 positivo | Precision | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| OOF principal | 0.6592 | 0.6251 | 0.6973 | 0.7214 | 0.8819 |
| Temporal, media macro | 0.6395 | 0.6118 | 0.6704 | 0.7075 | 0.8707 |
| Test final | 0.6532 | 0.6396 | 0.6672 | 0.7340 | 0.8851 |

Los resultados temporales y del test son descriptivos: no modificaron modelo,
hiperparametros, preprocessing ni threshold.

## Protocolo y particiones

1. Un 20% estratificado se reserva como test final con `random_state=42`.
2. El 80% restante es desarrollo. La seleccion oficial usa una unica
   `StratifiedKFold` de 5 folds y probabilidades OOF.
3. En cada fold, toda transformacion que aprende parametros y SMOTE se ajustan
   exclusivamente con el subtrain. La validacion recibe solamente `transform`.
4. Modelo, hiperparametros y threshold se eligen por F1 positivo OOF.
5. Cinco ventanas temporales expansivas evaluan robustez sin intervenir en la
   seleccion.
6. El test final se abre una sola vez, despues del congelamiento. No debe
   ejecutarse nuevamente `final_test_evaluation.py`.

`RainfallTomorrow` se elimina antes del modelado porque contiene informacion
del dia objetivo. El dataset, la semilla y los hashes de indices almacenados en
los reportes permiten reconstruir y verificar las particiones.

## Estructura

- `TP_clasificacion_AA1.ipynb`: analisis exploratorio y narrativa experimental.
  Las comparaciones anteriores con PyCaret son historicas; no definen el modelo
  final.
- `weather_preprocessing.py`: transformadores clonables y preprocessing
  fold-local.
- `oof_model_selection.py`: seleccion oficial y congelamiento. Regenera los
  artefactos finales; no es una comprobacion rutinaria.
- `temporal_robustness_evaluation.py`: evaluacion temporal secundaria.
- `final_test_evaluation.py`: evaluacion final de una sola ejecucion. Esta
  cerrada y no debe volver a ejecutarse.
- `verify_*.py`: comprobaciones metodologicas y de paridad.
- `artifacts/`: bundle final y evidencia auditable; consultar
  `artifacts/README.md`.
- `docker/`: inferencia autocontenida con el bundle final validado.

## Entornos y dependencias

El dataset y los artefactos binarios se versionan con Git LFS. Despues de
clonar el repositorio:

```bash
git lfs install
git lfs pull
```

La inferencia Docker esta validada sobre Python 3.12. Para crear el entorno
principal de entrenamiento y verificacion:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las dependencias exploratorias del notebook, incluido PyCaret, son opcionales:

```bash
python -m pip install -r requirements-notebook.txt
```

`requirements.txt` contiene las dependencias directas y versiones del flujo
oficial. `docker/requirements.txt` es deliberadamente mas pequeno porque solo
sirve inferencia. TensorFlow fija semillas en el entrenamiento; aun asi, una
repeticion desde cero puede presentar diferencias numericas menores entre
plataformas. La reproduccion exacta de inferencia se garantiza mediante los
artefactos congelados y sus hashes.

## Verificacion sin reabrir el test

Desde la raiz, estas comprobaciones no entrenan sobre el test ni vuelven a
calcular sus predicciones:

```bash
python verify_test_isolation.py
python verify_fold_local_preprocessing.py
python verify_oof_selection.py
python verify_temporal_robustness.py
python verify_final_test_evaluation.py
```

Las dos comprobaciones siguientes usan exclusivamente 512 filas de desarrollo
y actualizan sus propios archivos de evidencia de paridad:

```bash
python verify_local_inference_parity.py
python verify_docker_inference_parity.py
```

El segundo comando realiza `docker build` y `docker run`. El ultimo resultado
validado obtuvo etiquetas identicas y una diferencia maxima de probabilidad de
`8.77e-08`, con cero filas del test final.

## Artefactos congelados

| Componente | SHA-256 |
|---|---|
| `artifacts/oof_selection.json` | `7c03609f8f206d38f1e474886efa98034c499fcf78cf308f980a178efd6e0a2d` |
| `artifacts/selected_nn_model.keras` | `041251e4eec0cdb5031edaa39a150664424fb5e6cef85156c2cd02a5bd92e077` |
| `artifacts/selected_nn_preprocessor.joblib` | `4bca0f30cfe1d3bde21c0e4da62044d8c3f3a6c19b220d2407c44df3820069ac` |

Estos tres archivos no deben editarse ni regenerarse durante verificaciones.
