# ==============================================================================
# analyze_quality_robustness.py
# RetinAI_MVP
# Análisis de robustez por calidad de imagen:
# brillo, contraste, nitidez y proporción del campo retinal
# ==============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import build_dataloaders, load_yaml_config
from src.base_models import get_enabled_model_configs


# ==============================================================================
# RUTAS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
META_DIR = BASE_DIR / "models" / "meta"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# UTILIDADES
# ==============================================================================

def clean_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def load_json(path: Path, default=None):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_first_active_model_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    for model_cfg in get_enabled_model_configs(config):
        name = clean_name(model_cfg.get("name", ""))

        if name not in {"retfound", "retclip"}:
            return model_cfg

    raise RuntimeError("No se encontró modelo activo.")


def get_threshold() -> float:
    data = load_json(META_DIR / "optimal_threshold.json", default={})
    return float(data.get("threshold", 0.5))


def get_feature_names(test_df: pd.DataFrame) -> List[str]:
    data = load_json(META_DIR / "meta_feature_names.json", default=None)

    if isinstance(data, dict):
        names = data.get("feature_names", [])
    elif isinstance(data, list):
        names = data
    else:
        names = []

    names = [name for name in names if name in test_df.columns]

    if not names:
        names = [c for c in test_df.columns if c != "label"]

    return names


# ==============================================================================
# CALIDAD DE IMAGEN
# ==============================================================================

def compute_image_quality_metrics(image_path: Path) -> Dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.uint8)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Máscara del campo retinal para no medir demasiado fondo negro
    mask = gray > 12

    if mask.mean() < 0.10:
        mask = np.ones_like(gray, dtype=bool)

    pixels = gray[mask].astype(np.float32)

    brightness = float(np.mean(pixels) / 255.0)
    contrast = float(np.std(pixels) / 255.0)

    # Nitidez con varianza de Laplaciano
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(lap[mask]))

    retinal_area_ratio = float(mask.mean())

    # Saturación promedio
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1][mask]) / 255.0)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "retinal_area_ratio": retinal_area_ratio,
        "saturation": saturation,
    }


def assign_quality_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def tertile(series: pd.Series, low_label: str, mid_label: str, high_label: str):
        q1 = series.quantile(0.33)
        q2 = series.quantile(0.66)

        return np.select(
            [
                series <= q1,
                (series > q1) & (series <= q2),
                series > q2,
            ],
            [
                low_label,
                mid_label,
                high_label,
            ],
            default=mid_label,
        )

    out["brightness_group"] = tertile(
        out["brightness"],
        "oscura",
        "intermedia",
        "clara",
    )

    out["contrast_group"] = tertile(
        out["contrast"],
        "bajo_contraste",
        "contraste_intermedio",
        "alto_contraste",
    )

    out["sharpness_group"] = tertile(
        out["sharpness"],
        "baja_nitidez",
        "nitidez_intermedia",
        "alta_nitidez",
    )

    out["saturation_group"] = tertile(
        out["saturation"],
        "baja_saturacion",
        "saturacion_intermedia",
        "alta_saturacion",
    )

    # Grupo global simple
    bad_quality = (
        (out["brightness_group"] == "oscura")
        | (out["contrast_group"] == "bajo_contraste")
        | (out["sharpness_group"] == "baja_nitidez")
    )

    good_quality = (
        (out["brightness_group"] == "intermedia")
        & (out["contrast_group"] != "bajo_contraste")
        & (out["sharpness_group"] != "baja_nitidez")
    )

    out["quality_group"] = "calidad_intermedia"
    out.loc[bad_quality, "quality_group"] = "calidad_baja_o_desafiante"
    out.loc[good_quality, "quality_group"] = "calidad_mas_favorable"

    return out


# ==============================================================================
# MÉTRICAS POR GRUPO
# ==============================================================================

def compute_group_metrics(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []

    for group_value, group_df in df.groupby(group_col):
        y_true = group_df["label"].values.astype(int)
        y_pred = group_df["pred"].values.astype(int)
        y_prob = group_df["prob_exudates"].values.astype(float)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        n = len(group_df)
        positives = int(np.sum(y_true == 1))
        negatives = int(np.sum(y_true == 0))

        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        except Exception:
            auc = np.nan

        row = {
            "group_type": group_col,
            "group": group_value,
            "n": int(n),
            "positives": positives,
            "negatives": negatives,
            "accuracy": accuracy_score(y_true, y_pred),
            "sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / max(tn + fp, 1),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": auc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "mean_probability_exudates": float(np.mean(y_prob)),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    return result


def save_group_barplot(
    metrics_df: pd.DataFrame,
    group_type: str,
    metric: str,
    filename: str,
    title: str,
) -> None:
    plot_df = metrics_df[metrics_df["group_type"] == group_type].copy()

    if plot_df.empty:
        return

    labels = plot_df["group"].astype(str).tolist()
    values = (plot_df[metric].astype(float) * 100.0).tolist()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel(f"{metric} (%)")
    plt.title(title)
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


def save_fn_barplot(
    metrics_df: pd.DataFrame,
    group_type: str,
    filename: str,
    title: str,
) -> None:
    plot_df = metrics_df[metrics_df["group_type"] == group_type].copy()

    if plot_df.empty:
        return

    labels = plot_df["group"].astype(str).tolist()
    values = plot_df["fn"].astype(int).tolist()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel("Falsos negativos")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 80)
    print("RETINAI — ANÁLISIS DE ROBUSTEZ POR CALIDAD DE IMAGEN")
    print("=" * 80)

    config = load_yaml_config(CONFIG_PATH)

    model_cfg = get_first_active_model_cfg(config)

    print("Modelo usado para reconstruir split:", model_cfg["name"])

    bundle = build_dataloaders(
        config=config,
        model_cfg=model_cfg,
        batch_size=16,
        num_workers=0,
        shuffle_train=False,
    )

    test_split_df = bundle.test_df.reset_index(drop=True)

    meta_test_path = REPORTS_DIR / "meta_features_test.csv"

    if not meta_test_path.exists():
        raise FileNotFoundError(f"No existe: {meta_test_path}")

    meta_test_df = pd.read_csv(meta_test_path).reset_index(drop=True)

    if len(test_split_df) != len(meta_test_df):
        raise RuntimeError(
            f"No coincide test split ({len(test_split_df)}) "
            f"con meta_features_test ({len(meta_test_df)})."
        )

    feature_names = get_feature_names(meta_test_df)
    x_test = meta_test_df[feature_names].values.astype(np.float32)
    y_test = meta_test_df["label"].values.astype(int)

    meta_model_path = META_DIR / "meta_model.pkl"

    if not meta_model_path.exists():
        raise FileNotFoundError(f"No existe: {meta_model_path}")

    meta_model = joblib.load(meta_model_path)

    threshold = get_threshold()

    print("Umbral usado:", threshold)

    prob = meta_model.predict_proba(x_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    rows = []

    print("Extrayendo métricas de calidad de imagen...")

    for idx, row in test_split_df.iterrows():
        image_path = Path(row["image_path"])

        quality = compute_image_quality_metrics(image_path)

        rows.append(
            {
                "image_path": str(image_path),
                "label": int(y_test[idx]),
                "class_name": row.get("class_name", ""),
                "prob_exudates": float(prob[idx]),
                "pred": int(pred[idx]),
                "correct": int(pred[idx] == y_test[idx]),
                **quality,
            }
        )

    quality_df = pd.DataFrame(rows)
    quality_df = assign_quality_groups(quality_df)

    output_quality_path = REPORTS_DIR / "quality_test_predictions.csv"
    quality_df.to_csv(output_quality_path, index=False)

    group_cols = [
        "quality_group",
        "brightness_group",
        "contrast_group",
        "sharpness_group",
        "saturation_group",
    ]

    metrics_parts = []

    for group_col in group_cols:
        metrics_parts.append(compute_group_metrics(quality_df, group_col))

    metrics_df = pd.concat(metrics_parts, ignore_index=True)

    # Porcentajes redondeados
    metrics_export = metrics_df.copy()

    for col in [
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auc_roc",
        "mean_probability_exudates",
    ]:
        metrics_export[col + "_percent"] = (metrics_export[col] * 100.0).round(2)

    metrics_path = REPORTS_DIR / "quality_robustness_metrics.csv"
    metrics_export.to_csv(metrics_path, index=False)

    # Gráficas principales
    save_group_barplot(
        metrics_df,
        group_type="quality_group",
        metric="sensitivity",
        filename="robustness_sensitivity_by_quality.png",
        title="Sensibilidad según calidad global de imagen",
    )

    save_group_barplot(
        metrics_df,
        group_type="brightness_group",
        metric="sensitivity",
        filename="robustness_sensitivity_by_brightness.png",
        title="Sensibilidad según brillo de imagen",
    )

    save_group_barplot(
        metrics_df,
        group_type="contrast_group",
        metric="sensitivity",
        filename="robustness_sensitivity_by_contrast.png",
        title="Sensibilidad según contraste de imagen",
    )

    save_group_barplot(
        metrics_df,
        group_type="sharpness_group",
        metric="sensitivity",
        filename="robustness_sensitivity_by_sharpness.png",
        title="Sensibilidad según nitidez de imagen",
    )

    save_fn_barplot(
        metrics_df,
        group_type="quality_group",
        filename="robustness_false_negatives_by_quality.png",
        title="Falsos negativos según calidad global",
    )

    summary = {
        "threshold": threshold,
        "n_test": int(len(quality_df)),
        "overall_accuracy": float((quality_df["correct"]).mean()),
        "files": {
            "quality_predictions": str(output_quality_path),
            "quality_metrics": str(metrics_path),
            "figures": [
                "reports/figures/robustness_sensitivity_by_quality.png",
                "reports/figures/robustness_sensitivity_by_brightness.png",
                "reports/figures/robustness_sensitivity_by_contrast.png",
                "reports/figures/robustness_sensitivity_by_sharpness.png",
                "reports/figures/robustness_false_negatives_by_quality.png",
            ],
        },
    }

    save_json(summary, REPORTS_DIR / "quality_robustness_summary.json")

    print("\n" + "=" * 80)
    print("MÉTRICAS POR CALIDAD")
    print("=" * 80)
    print(metrics_export.to_string(index=False))

    print("\nArchivos generados:")
    print("-", output_quality_path)
    print("-", metrics_path)
    print("-", REPORTS_DIR / "quality_robustness_summary.json")
    print("- reports/figures/robustness_sensitivity_by_quality.png")
    print("- reports/figures/robustness_sensitivity_by_brightness.png")
    print("- reports/figures/robustness_sensitivity_by_contrast.png")
    print("- reports/figures/robustness_sensitivity_by_sharpness.png")
    print("- reports/figures/robustness_false_negatives_by_quality.png")

    print("\n✅ Análisis de robustez completado.")


if __name__ == "__main__":
    main()