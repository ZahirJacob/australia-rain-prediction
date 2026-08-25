FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# El contexto de build es la raiz del repositorio. Solo se incorporan los
# componentes del bundle final validado; los artefactos PyCaret quedan fuera.
COPY docker/inferencia.py /app/docker/inferencia.py
COPY weather_preprocessing.py /app/weather_preprocessing.py
COPY artifacts/oof_selection.json /app/artifacts/oof_selection.json
COPY artifacts/selected_nn_preprocessor.joblib /app/artifacts/selected_nn_preprocessor.joblib
COPY artifacts/selected_nn_model.keras /app/artifacts/selected_nn_model.keras

CMD ["python", "docker/inferencia.py"]
