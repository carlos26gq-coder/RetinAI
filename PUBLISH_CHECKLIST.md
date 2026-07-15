# Checklist de publicación

- [ ] Exactamente 7 modelos activos.
- [ ] `base_model_names.json` contiene los 7 modelos.
- [ ] `meta_feature_names.json` contiene 106 variables.
- [ ] `optimal_threshold.json` contiene aproximadamente 0.3798.
- [ ] `meta_model.pkl` corresponde al XGBoost final.
- [ ] No existen `backup_*`, `src/src`, scripts patch ni resultados antiguos.
- [ ] No se incluyen imágenes del dataset.
- [ ] No se incluyen `.pth` o `.pkl` en el historial Git.
- [ ] No hay tokens, contraseñas, rutas personales o datos privados.
- [ ] Se ejecutó `pytest -q`.
- [ ] Se ejecutó `python scripts/verify_repository.py --require-models`.
- [ ] Se creó el tag `v1.0.0`.
- [ ] Se publicó el asset de modelos con SHA-256.
