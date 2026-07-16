# RetinAI

Sistema de apoyo para la detección temprana de exudados retinales mediante
siete modelos profundos heterogéneos, 66 características radiomics y
Ensemble Stacking.

## Arquitectura final

- 7 modelos profundos: EfficientNetB0, EfficientNetB3, DenseNet121,
  InceptionV3, U-Net encoder, MobileNetV3 exudate map y ResNet50-CBAM.
- 66 características radiomics.
- Meta-dataset de 106 variables.
- Meta-modelos comparados: regresión logística, Extra Trees y XGBoost.
- Meta-modelo final: XGBoost.
- Umbral final: 0.3798.

## Instalación

Se recomienda Python 3.12.

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Los pesos se distribuyen desde GitHub Releases:

```bash
python scripts/download_models.py --url https://github.com/carlos26gq-coder/RetinAI/releases/download/v1.0.0/retinai_models_v1.0.0.zip --sha256 SHA256_DEL_ASSET
```

Verificación:

```bash
python scripts/verify_repository.py --require-models
python scripts/smoke_test.py --image RUTA_A_UNA_IMAGEN
```

API:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Interfaz:

```bash
python -m streamlit run app.py
```

## Entrenamiento

Estructura esperada:

```text
data/raw/
├── healthy/
└── exudates/
```

```bash
python train_ensemble.py
python compare_stacking_meta_models.py --mode nested --n-jobs -1
python post_training_analysis.py
python analyze_quality_robustness.py
python generate_gradcam_examples.py
python generate_final_report_figures.py
```

## Alcance

RetinAI es un prototipo de investigación y una herramienta de apoyo. No
reemplaza la evaluación de un especialista ni debe utilizarse como único
método de diagnóstico clínico.

## Autores

- Josue Josephie Villegas Murayari
- Washington Carlos Cesar Gutiérrez Quispe
