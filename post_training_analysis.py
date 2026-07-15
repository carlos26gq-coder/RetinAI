# ==============================================================================
# post_training_analysis.py
# RetinAI_MVP
# Análisis posterior al entrenamiento:
# - Modelos individuales
# - Promedio simple
# - Votación mayoritaria
# - Ensemble Stacking final
# - Tabla estilo artículo científico
# ==============================================================================

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ==============================================================================
# RUTAS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
META_DIR = BASE_DIR / "models" / "meta"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# UTILIDADES
# ==============================================================================

def load_json(path: Path, default=None):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_model_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def pretty_model_name(name: str) -> str:
    mapping = {
        "efficientnet_b0": "EfficientNetB0",
        "efficientnet_b3": "EfficientNetB3",
        "densenet121": "DenseNet121",
        "inception_v3": "InceptionV3",
        "unet_encoder": "U-Net encoder",
        "mobilenetv3_exudate_map": "MobileNetV3 exudate map",
        "resnet50_cbam": "ResNet50-CBAM",
        "simple_average": "Promedio simple",
        "majority_voting": "Votación mayoritaria",
        "ensemble_stacking": "Ensemble Stacking final",
    }

    return mapping.get(name, name)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0

    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower = (centre - margin) / denom
    upper = (centre + margin) / denom

    return max(0.0, lower), min(1.0, upper)


def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    specificity = 1.0 - fpr
    youden = tpr + specificity - 1.0

    best_idx = int(np.argmax(youden))

    return {
        "threshold": float(thresholds[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(specificity[best_idx]),
        "youden": float(youden[best_idx]),
    }


def compute_metrics(
    model_key: str,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = int(len(y_true))
    correct = int(tp + tn)

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.0

    vpp = tp / max(tp + fp, 1)
    vpn = tn / max(tn + fn, 1)

    acc_low, acc_high = wilson_ci(correct, total)

    return {
        "model_key": model_key,
        "Modelo": model_name,
        "Correctas": correct,
        "Total": total,
        "Correctas/Total": f"{correct}/{total}",
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100,
        "Accuracy IC95%": f"[{acc_low * 100:.2f}, {acc_high * 100:.2f}]",
        "Sensibilidad": sensitivity,
        "Sensibilidad (%)": sensitivity * 100,
        "Especificidad": specificity,
        "Especificidad (%)": specificity * 100,
        "Precisión": precision,
        "Precisión (%)": precision * 100,
        "F1-score": f1,
        "F1-score (%)": f1 * 100,
        "AUC-ROC": auc_roc,
        "AUC-ROC (%)": auc_roc * 100,
        "VPP": vpp,
        "VPP (%)": vpp * 100,
        "VPN": vpn,
        "VPN (%)": vpn * 100,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Umbral usado": float(threshold),
    }


def infer_base_model_names(df: pd.DataFrame) -> List[str]:
    names = []

    for col in df.columns:
        if col.endswith("_prob_exudates") and not col.startswith("group_"):
            name = col.replace("_prob_exudates", "")
            names.append(name)

    return names


def get_base_model_names(df: pd.DataFrame) -> List[str]:
    json_path = META_DIR / "base_model_names.json"
    data = load_json(json_path, default=None)

    names = []

    if isinstance(data, dict):
        names = data.get("base_model_names", data.get("model_names", []))
    elif isinstance(data, list):
        names = data

    names = [clean_model_name(x) for x in names]

    valid_names = []

    for name in names:
        if f"{name}_prob_exudates" in df.columns:
            valid_names.append(name)

    if valid_names:
        return valid_names

    return infer_base_model_names(df)


def load_meta_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_path = REPORTS_DIR / "meta_features_val.csv"
    test_path = REPORTS_DIR / "meta_features_test.csv"

    if not val_path.exists():
        raise FileNotFoundError(f"No existe: {val_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"No existe: {test_path}")

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if "label" not in val_df.columns or "label" not in test_df.columns:
        raise RuntimeError("meta_features_val/test deben tener columna 'label'.")

    return val_df, test_df


def load_meta_feature_names(test_df: pd.DataFrame) -> List[str]:
    path = META_DIR / "meta_feature_names.json"
    data = load_json(path, default=None)

    if isinstance(data, dict):
        names = data.get("feature_names", [])
    elif isinstance(data, list):
        names = data
    else:
        names = []

    valid = [name for name in names if name in test_df.columns]

    if valid:
        return valid

    return [c for c in test_df.columns if c != "label"]


# ==============================================================================
# BASELINES
# ==============================================================================

def evaluate_base_models(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_names: List[str],
) -> List[Dict[str, Any]]:
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    rows = []

    for name in base_names:
        col = f"{name}_prob_exudates"

        if col not in val_df.columns or col not in test_df.columns:
            print(f"Saltando {name}: no existe columna {col}")
            continue

        val_prob = val_df[col].values.astype(float)
        test_prob = test_df[col].values.astype(float)

        threshold_info = find_youden_threshold(y_val, val_prob)
        threshold = threshold_info["threshold"]

        row = compute_metrics(
            model_key=name,
            model_name=pretty_model_name(name),
            y_true=y_test,
            y_prob=test_prob,
            threshold=threshold,
        )

        row["Tipo"] = "Modelo individual"
        row["Umbral origen"] = "Validación / Youden"

        rows.append(row)

    return rows


def evaluate_simple_average(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_names: List[str],
) -> Dict[str, Any]:
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    cols = [f"{name}_prob_exudates" for name in base_names]
    cols = [col for col in cols if col in val_df.columns and col in test_df.columns]

    if not cols:
        raise RuntimeError("No hay columnas de probabilidades base para promedio simple.")

    val_prob = val_df[cols].mean(axis=1).values.astype(float)
    test_prob = test_df[cols].mean(axis=1).values.astype(float)

    threshold_info = find_youden_threshold(y_val, val_prob)
    threshold = threshold_info["threshold"]

    row = compute_metrics(
        model_key="simple_average",
        model_name="Promedio simple",
        y_true=y_test,
        y_prob=test_prob,
        threshold=threshold,
    )

    row["Tipo"] = "Baseline ensemble"
    row["Umbral origen"] = "Validación / Youden"

    return row


def evaluate_majority_voting(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_names: List[str],
) -> Dict[str, Any]:
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    vote_val = []
    vote_test = []

    used_models = []

    for name in base_names:
        col = f"{name}_prob_exudates"

        if col not in val_df.columns or col not in test_df.columns:
            continue

        threshold_info = find_youden_threshold(
            y_val,
            val_df[col].values.astype(float),
        )

        model_threshold = threshold_info["threshold"]

        val_vote = (val_df[col].values.astype(float) >= model_threshold).astype(int)
        test_vote = (test_df[col].values.astype(float) >= model_threshold).astype(int)

        vote_val.append(val_vote)
        vote_test.append(test_vote)
        used_models.append(name)

    if not vote_test:
        raise RuntimeError("No hay votos disponibles para votación mayoritaria.")

    vote_test_matrix = np.vstack(vote_test).T

    # Probabilidad aproximada = proporción de modelos que votan exudado.
    test_prob = vote_test_matrix.mean(axis=1)

    # Votación mayoritaria fija.
    required_votes = int(math.ceil(len(used_models) / 2))
    y_pred = (vote_test_matrix.sum(axis=1) >= required_votes).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = int(len(y_test))
    correct = int(tp + tn)

    try:
        auc_roc = roc_auc_score(y_test, test_prob)
    except Exception:
        auc_roc = 0.0

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    vpp = tp / max(tp + fp, 1)
    vpn = tn / max(tn + fn, 1)

    acc_low, acc_high = wilson_ci(correct, total)

    row = {
        "model_key": "majority_voting",
        "Modelo": "Votación mayoritaria",
        "Correctas": correct,
        "Total": total,
        "Correctas/Total": f"{correct}/{total}",
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100,
        "Accuracy IC95%": f"[{acc_low * 100:.2f}, {acc_high * 100:.2f}]",
        "Sensibilidad": sensitivity,
        "Sensibilidad (%)": sensitivity * 100,
        "Especificidad": specificity,
        "Especificidad (%)": specificity * 100,
        "Precisión": precision,
        "Precisión (%)": precision * 100,
        "F1-score": f1,
        "F1-score (%)": f1 * 100,
        "AUC-ROC": auc_roc,
        "AUC-ROC (%)": auc_roc * 100,
        "VPP": vpp,
        "VPP (%)": vpp * 100,
        "VPN": vpn,
        "VPN (%)": vpn * 100,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Umbral usado": float(required_votes),
        "Tipo": "Baseline ensemble",
        "Umbral origen": f"Mayoría: {required_votes}/{len(used_models)} votos",
    }

    return row


def evaluate_stacking_final(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    meta_model_path = META_DIR / "meta_model.pkl"

    if not meta_model_path.exists():
        raise FileNotFoundError(f"No existe meta-modelo: {meta_model_path}")

    meta_model = joblib.load(meta_model_path)

    feature_names = load_meta_feature_names(test_df)

    x_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)

    y_prob = meta_model.predict_proba(x_test)[:, 1]

    threshold_path = META_DIR / "optimal_threshold.json"
    threshold_data = load_json(threshold_path, default={})

    threshold = float(threshold_data.get("threshold", 0.5))

    row = compute_metrics(
        model_key="ensemble_stacking",
        model_name="Ensemble Stacking final",
        y_true=y_test,
        y_prob=y_prob,
        threshold=threshold,
    )

    row["Tipo"] = "Modelo propuesto"
    row["Umbral origen"] = "OOF validación / Youden"

    return row


# ==============================================================================
# GRÁFICAS
# ==============================================================================

def save_accuracy_auc_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()

    labels = plot_df["Modelo"].tolist()
    accuracy = plot_df["Accuracy (%)"].tolist()
    auc_values = plot_df["AUC-ROC (%)"].tolist()

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, auc_values, width, label="AUC-ROC")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Porcentaje (%)")
    plt.ylim(0, 105)
    plt.title("Comparación de Accuracy y AUC-ROC")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "comparison_accuracy_auc_article_style.png"
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_sensitivity_specificity_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()

    labels = plot_df["Modelo"].tolist()
    sensitivity = plot_df["Sensibilidad (%)"].tolist()
    specificity = plot_df["Especificidad (%)"].tolist()

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, sensitivity, width, label="Sensibilidad")
    plt.bar(x + width / 2, specificity, width, label="Especificidad")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Porcentaje (%)")
    plt.ylim(0, 105)
    plt.title("Comparación de Sensibilidad y Especificidad")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "comparison_sensitivity_specificity_article_style.png"
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_false_negative_plot(df: pd.DataFrame) -> None:
    plot_df = df.copy()

    labels = plot_df["Modelo"].tolist()
    fn_values = plot_df["FN"].tolist()

    plt.figure(figsize=(12, 5))
    plt.bar(labels, fn_values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Falsos negativos")
    plt.title("Comparación de falsos negativos por modelo")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "comparison_false_negatives_article_style.png"
    plt.savefig(out_path, dpi=180)
    plt.close()


# ==============================================================================
# TEXTO PARA ARTÍCULO / TESIS
# ==============================================================================

def generate_article_style_text(df: pd.DataFrame) -> str:
    stack = df[df["model_key"] == "ensemble_stacking"].iloc[0]

    best_acc = df.sort_values("Accuracy (%)", ascending=False).iloc[0]
    best_auc = df.sort_values("AUC-ROC (%)", ascending=False).iloc[0]
    best_sens = df.sort_values("Sensibilidad (%)", ascending=False).iloc[0]
    lowest_fn = df.sort_values("FN", ascending=True).iloc[0]

    text = f"""
RESULTADOS ESTILO ARTÍCULO — RETINAI

El modelo propuesto basado en Ensemble Stacking final obtuvo una exactitud global de {stack['Accuracy (%)']:.2f}%, clasificando correctamente {stack['Correctas']} de {stack['Total']} imágenes del conjunto de prueba. Asimismo, alcanzó una sensibilidad de {stack['Sensibilidad (%)']:.2f}%, una especificidad de {stack['Especificidad (%)']:.2f}%, una precisión de {stack['Precisión (%)']:.2f}%, un F1-score de {stack['F1-score (%)']:.2f}% y un AUC-ROC de {stack['AUC-ROC (%)']:.2f}%.

En términos de matriz de confusión, el modelo identificó correctamente {stack['TP']} imágenes con exudados retinales y {stack['TN']} imágenes sin exudados, mientras que presentó {stack['FP']} falsos positivos y {stack['FN']} falsos negativos. Estos resultados son relevantes para un contexto de apoyo al tamizaje, debido a que la sensibilidad y la reducción de falsos negativos son indicadores críticos en la detección temprana de exudados retinales.

Al comparar los modelos individuales, el mejor accuracy fue obtenido por {best_acc['Modelo']} con {best_acc['Accuracy (%)']:.2f}%. El mayor AUC-ROC fue obtenido por {best_auc['Modelo']} con {best_auc['AUC-ROC (%)']:.2f}%. La mayor sensibilidad fue alcanzada por {best_sens['Modelo']} con {best_sens['Sensibilidad (%)']:.2f}%, mientras que la menor cantidad de falsos negativos fue obtenida por {lowest_fn['Modelo']} con {int(lowest_fn['FN'])} casos.

Estos resultados permiten comparar el aporte del Ensemble Stacking frente a modelos individuales, promedio simple y votación mayoritaria, evitando sustentar el desempeño únicamente en un ejemplo visual o en una métrica aislada.
""".strip()

    return text


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 80)
    print("RETINAI — ANÁLISIS POST-ENTRENAMIENTO ESTILO ARTÍCULO")
    print("=" * 80)

    val_df, test_df = load_meta_features()

    print("meta_features_val:", val_df.shape)
    print("meta_features_test:", test_df.shape)

    base_names = get_base_model_names(test_df)

    print("\nModelos base detectados:")
    for name in base_names:
        print("-", name)

    rows: List[Dict[str, Any]] = []

    rows.extend(
        evaluate_base_models(
            val_df=val_df,
            test_df=test_df,
            base_names=base_names,
        )
    )

    rows.append(
        evaluate_simple_average(
            val_df=val_df,
            test_df=test_df,
            base_names=base_names,
        )
    )

    rows.append(
        evaluate_majority_voting(
            val_df=val_df,
            test_df=test_df,
            base_names=base_names,
        )
    )

    rows.append(
        evaluate_stacking_final(
            val_df=val_df,
            test_df=test_df,
        )
    )

    df = pd.DataFrame(rows)

    order = [
        "efficientnet_b0",
        "efficientnet_b3",
        "densenet121",
        "inception_v3",
        "unet_encoder",
        "mobilenetv3_exudate_map",
        "resnet50_cbam",
        "simple_average",
        "majority_voting",
        "ensemble_stacking",
    ]

    df["order"] = df["model_key"].apply(
        lambda x: order.index(x) if x in order else 999
    )

    df = df.sort_values("order").drop(columns=["order"])

    # Tabla completa con decimales reales
    out_full = REPORTS_DIR / "article_style_model_comparison_full.csv"
    df.to_csv(out_full, index=False)

    # Tabla redondeada para pegar en tesis/artículo
    rounded_df = df.copy()

    percent_cols = [
        "Accuracy (%)",
        "Sensibilidad (%)",
        "Especificidad (%)",
        "Precisión (%)",
        "F1-score (%)",
        "AUC-ROC (%)",
        "VPP (%)",
        "VPN (%)",
    ]

    for col in percent_cols:
        rounded_df[col] = rounded_df[col].round(2)

    rounded_df["Umbral usado"] = rounded_df["Umbral usado"].round(4)

    selected_cols = [
        "Modelo",
        "Tipo",
        "Correctas/Total",
        "Accuracy (%)",
        "Accuracy IC95%",
        "Sensibilidad (%)",
        "Especificidad (%)",
        "Precisión (%)",
        "F1-score (%)",
        "AUC-ROC (%)",
        "TP",
        "TN",
        "FP",
        "FN",
        "Umbral usado",
        "Umbral origen",
    ]

    rounded_df = rounded_df[selected_cols]

    out_article = REPORTS_DIR / "article_style_model_comparison.csv"
    rounded_df.to_csv(out_article, index=False)

    try:
        out_xlsx = REPORTS_DIR / "article_style_model_comparison.xlsx"
        rounded_df.to_excel(out_xlsx, index=False)
        print("Excel guardado:", out_xlsx)
    except Exception as exc:
        print("No se pudo guardar Excel, pero CSV sí fue generado:", exc)

    # Gráficas
    save_accuracy_auc_plot(df)
    save_sensitivity_specificity_plot(df)
    save_false_negative_plot(df)

    # Texto estilo artículo
    text = generate_article_style_text(df)
    text_path = REPORTS_DIR / "article_style_results_text.txt"
    text_path.write_text(text, encoding="utf-8")

    # JSON resumen
    summary = {
        "base_models": base_names,
        "total_models_compared": int(len(df)),
        "files_generated": {
            "full_csv": str(out_full),
            "article_csv": str(out_article),
            "article_text": str(text_path),
            "accuracy_auc_figure": str(FIGURES_DIR / "comparison_accuracy_auc_article_style.png"),
            "sensitivity_specificity_figure": str(FIGURES_DIR / "comparison_sensitivity_specificity_article_style.png"),
            "false_negatives_figure": str(FIGURES_DIR / "comparison_false_negatives_article_style.png"),
        },
    }

    save_json(summary, REPORTS_DIR / "article_style_analysis_summary.json")

    print("\n" + "=" * 80)
    print("TABLA ESTILO ARTÍCULO")
    print("=" * 80)
    print(rounded_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TEXTO GENERADO")
    print("=" * 80)
    print(text)

    print("\nArchivos generados:")
    print("-", out_full)
    print("-", out_article)
    print("-", REPORTS_DIR / "article_style_model_comparison.xlsx")
    print("-", text_path)
    print("-", FIGURES_DIR / "comparison_accuracy_auc_article_style.png")
    print("-", FIGURES_DIR / "comparison_sensitivity_specificity_article_style.png")
    print("-", FIGURES_DIR / "comparison_false_negatives_article_style.png")
    print("\n✅ Análisis post-entrenamiento completado.")


if __name__ == "__main__":
    main()