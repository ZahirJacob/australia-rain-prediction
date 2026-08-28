# Artefactos

Este directorio contiene dos clases de archivos con responsabilidades
diferentes.

## Bundle de inferencia congelado

- `oof_selection.json`: candidato, hiperparametros y threshold seleccionados.
- `selected_nn_preprocessor.joblib`: preprocesador ajustado sobre desarrollo.
- `selected_nn_model.keras`: modelo final `nn_config_5`.

Son los unicos artefactos copiados por Docker. Sus hashes oficiales estan en el
README principal y en los reportes de paridad. No deben regenerarse para
ejecutar verificaciones.

Nota: el SHA-256 registrado en los reportes para `oof_selection.json`
(`7c03609f...`) no coincide con el del archivo versionado (`e26db0a2...`); el
manifiesto fue editado despues de calcularse los hashes y no se conserva la
version original. Los hashes del modelo y del preprocesador si coinciden. Ver
"Notas posteriores a la entrega" en el README principal.

## Evidencia experimental

- `oof_candidate_metrics.csv`, `oof_fold_audit.csv` y `oof_predictions.npz`:
  seleccion OOF y auditoria fold-local.
- `temporal_fold_metrics.csv`, `temporal_predictions.npz` y
  `temporal_robustness.json`: robustez temporal expansiva.
- `final_test_metrics.json`, `final_test_predictions.npz` y
  `final_test_evaluation.completed.json`: apertura unica del test final.
- `local_inference_parity*` y `docker_inference_parity*`: paridad de inferencia
  sobre 512 filas de desarrollo, sin observaciones del test final.

Los archivos NPZ se conservan porque permiten recomputar metricas y comprobar
indices sin volver a entrenar ni repetir la inferencia del test. El reporte
`docker_inference_parity.json` es la fuente autoritativa sobre el estado del
contenedor; el campo `pending_docker_runtime` del reporte local registra el
estado historico anterior a la verificacion Docker.
