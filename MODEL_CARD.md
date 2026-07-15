# Model Card — RetinAI v1.0.0

## Descripción

Clasificador binario de imágenes de fondo de ojo: `healthy` y `exudates`.

## Arquitectura

- 7 modelos profundos.
- 66 características radiomics.
- 106 variables en el meta-dataset.
- Meta-modelo final: XGBoost.
- Umbral: 0.3798.

## Métricas en prueba independiente

- n = 643
- Accuracy = 84.76 %
- Sensibilidad = 89.20 %
- Especificidad = 80.25 %
- AUC-ROC = 0.9291
- TP = 289, TN = 256, FP = 63, FN = 35

## Uso previsto

Herramienta de investigación y apoyo al tamizaje.

## Limitaciones

- Requiere validación externa y multicéntrica.
- La calidad y procedencia de la imagen pueden afectar el resultado.
- Grad-CAM ofrece una explicación aproximada, no una segmentación.
- No reemplaza el diagnóstico de un especialista.
