"""
metrics.py
==========
Módulo de métricas clínicas y científicas para retinopatía diabética.

Métricas implementadas:
  - Sensibilidad (Recall / TPR)
  - Especificidad (TNR)
  - AUC-ROC
  - F1-Score, Precisión
  - Curva ROC
  - Matriz de Confusión
  - Intervalo de Confianza Bootstrap para AUC

Autor: MVP Tesis — Ensemble Stacking Retinopatía Diabética
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────────────────────────────────────
def _as_1d_numpy(x: Any, dtype=None) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _validate_binary_inputs(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
) -> None:
    if y_true.size == 0:
        raise ValueError("y_true está vacío.")

    if y_pred is not None and y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true y y_pred tienen longitudes distintas: {len(y_true)} vs {len(y_pred)}")

    if y_prob is not None and y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"y_true y y_prob tienen longitudes distintas: {len(y_true)} vs {len(y_prob)}")


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def _safe_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Métricas clínicas principales
# ─────────────────────────────────────────────────────────────────────────────
def compute_clinical_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calcula métricas clínicas para clasificación binaria.

    Args:
        y_true: etiquetas reales (0/1)
        y_pred: etiquetas predichas (0/1)
        y_prob: probabilidades de clase positiva
        class_names: nombres de clases
        threshold: umbral usado para la predicción

    Returns:
        Diccionario con métricas clínicas y conteos.
    """
    if class_names is None:
        class_names = ["healthy", "exudates"]

    y_true = _as_1d_numpy(y_true, dtype=int)
    y_pred = _as_1d_numpy(y_pred, dtype=int)
    y_prob = _as_1d_numpy(y_prob, dtype=float)

    _validate_binary_inputs(y_true, y_pred, y_prob)

    if len(class_names) != 2:
        raise ValueError("class_names debe contener exactamente 2 clases para este proyecto binario.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    auc_roc = _safe_auc(y_true, y_prob)
    auc_pr = _safe_auc_pr(y_true, y_prob)

    vpp = precision
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    youden_index = sensibilidad + especificidad - 1.0

    metrics = {
        "sensibilidad": round(float(sensibilidad), 4),
        "especificidad": round(float(especificidad), 4),
        "auc_roc": round(float(auc_roc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "auc_pr": round(float(auc_pr), 4),
        "vpp": round(float(vpp), 4),
        "vpn": round(float(vpn), 4),
        "youden_index": round(float(youden_index), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positivos": int(np.sum(y_true == 1)),
        "n_negativos": int(np.sum(y_true == 0)),
        "class_0_name": class_names[0],
        "class_1_name": class_names[1],
    }

    logger.info("=" * 55)
    logger.info("📊 MÉTRICAS CLÍNICAS")
    logger.info("=" * 55)
    logger.info("  Sensibilidad (Recall):  %.4f  (%d TP / %d positivos)", sensibilidad, tp, tp + fn)
    logger.info("  Especificidad:          %.4f  (%d TN / %d negativos)", especificidad, tn, tn + fp)
    logger.info("  AUC-ROC:                %.4f", auc_roc)
    logger.info("  F1-Score:               %.4f", f1)
    logger.info("  Precisión:              %.4f", precision)
    logger.info("  Accuracy:               %.4f", accuracy)
    logger.info("  Índice de Youden:       %.4f", youden_index)
    logger.info("  VPP:                    %.4f", vpp)
    logger.info("  VPN:                    %.4f", vpn)
    logger.info("=" * 55)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI para AUC
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_auc_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Calcula el intervalo de confianza bootstrap del AUC-ROC.

    Returns:
        (auc_mean, ci_lower, ci_upper)
    """
    y_true = _as_1d_numpy(y_true, dtype=int)
    y_prob = _as_1d_numpy(y_prob, dtype=float)

    _validate_binary_inputs(y_true, y_prob=y_prob)

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "No hay suficientes clases para calcular Bootstrap AUC. "
            "Retornando (0.5, 0.5, 0.5)."
        )
        return 0.5, 0.5, 0.5

    rng = np.random.default_rng(seed)
    aucs: List[float] = []

    max_attempts = max(n_iterations * 10, 2000)
    attempts = 0

    while len(aucs) < n_iterations and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, len(y_true), size=len(y_true))
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            aucs.append(float(roc_auc_score(y_true_boot, y_prob_boot)))
        except Exception:
            continue

    if not aucs:
        logger.warning(
            "No fue posible obtener muestras bootstrap válidas. "
            "Retornando (0.5, 0.5, 0.5)."
        )
        return 0.5, 0.5, 0.5

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(aucs, 100 * alpha / 2))
    ci_upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    auc_mean = float(np.mean(aucs))

    logger.info(
        "AUC-ROC Bootstrap IC%d%%: %.4f [%.4f, %.4f]",
        int(confidence * 100),
        auc_mean,
        ci_lower,
        ci_upper,
    )
    return auc_mean, ci_lower, ci_upper


# ─────────────────────────────────────────────────────────────────────────────
# Umbral óptimo
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    criterion: str = "youden",
) -> Tuple[float, float, float]:
    """
    Encuentra el umbral óptimo a partir de probabilidades.

    criterion:
        - "youden": maximiza sensibilidad + especificidad - 1
        - "f1": maximiza F1
        - cualquier otro: usa criterio Youden por defecto
    """
    y_true = _as_1d_numpy(y_true, dtype=int)
    y_prob = _as_1d_numpy(y_prob, dtype=float)

    _validate_binary_inputs(y_true, y_prob=y_prob)

    if len(np.unique(y_true)) < 2:
        logger.warning("No se puede optimizar el umbral con una sola clase. Retornando 0.5.")
        return 0.5, 0.0, 1.0

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificities = 1.0 - fpr

    # En sklearn moderno, thresholds[0] puede ser inf; lo descartamos para la búsqueda.
    if len(thresholds) > 1 and np.isinf(thresholds[0]):
        thresholds = thresholds[1:]
        tpr = tpr[1:]
        specificities = specificities[1:]

    if len(thresholds) == 0:
        logger.warning("No se generaron umbrales válidos. Retornando 0.5.")
        return 0.5, 0.0, 1.0

    criterion = criterion.lower().strip()

    if criterion == "youden":
        scores = tpr + specificities - 1.0
        best_idx = int(np.argmax(scores))

    elif criterion == "f1":
        f1_scores = [
            f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
            for thr in thresholds
        ]
        best_idx = int(np.argmax(f1_scores))

    else:
        logger.warning("Criterio '%s' no reconocido. Se usará Youden.", criterion)
        scores = tpr + specificities - 1.0
        best_idx = int(np.argmax(scores))

    optimal_threshold = float(thresholds[best_idx])
    sens = float(tpr[best_idx])
    spec = float(specificities[best_idx])

    logger.info(
        "Umbral óptimo (%s): %.4f | Sens=%.4f | Spec=%.4f",
        criterion,
        optimal_threshold,
        sens,
        spec,
    )
    return optimal_threshold, sens, spec


# ─────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    model_name: str = "Ensemble Stacking",
    save_path: Optional[str] = None,
    ci_data: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Grafica la curva ROC.
    Si ci_data se proporciona, se evita recalcular el bootstrap.
    """
    y_true = _as_1d_numpy(y_true, dtype=int)
    y_prob = _as_1d_numpy(y_prob, dtype=float)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = _safe_auc(y_true, y_prob)
        if ci_data is None:
            _, ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_prob)
        else:
            _, ci_lo, ci_hi = ci_data
    except Exception:
        logger.warning("Fallo al calcular datos de ROC. Se omite la gráfica.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        fpr,
        tpr,
        lw=2.5,
        label=f"{model_name}\nAUC = {auc:.4f} (IC95%: {ci_lo:.3f}–{ci_hi:.3f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Clasificador aleatorio (AUC=0.5)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=12)
    ax.set_title("Curva ROC — Detección de Exudados en Retinopatía Diabética", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Curva ROC guardada: %s", save_path)

    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Grafica la matriz de confusión normalizada.
    """
    if class_names is None:
        class_names = ["Sano", "Exudados"]

    if len(class_names) != 2:
        raise ValueError("class_names debe contener exactamente 2 elementos.")

    y_true = _as_1d_numpy(y_true, dtype=int)
    y_pred = _as_1d_numpy(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm[~np.isfinite(cm_norm)] = 0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proporción"},
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    ax.set_ylabel("Etiqueta Real", fontsize=12)
    ax.set_xlabel("Etiqueta Predicha", fontsize=12)
    ax.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Matriz de confusión guardada: %s", save_path)

    plt.close(fig)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    model_name: str = "Modelo",
    save_path: Optional[str] = None,
) -> None:
    """
    Grafica historia de entrenamiento.
    """
    if not (len(train_losses) == len(val_losses) == len(train_accs) == len(val_accs)):
        raise ValueError("Todas las listas de historia de entrenamiento deben tener la misma longitud.")

    if len(train_losses) == 0:
        logger.warning("No hay historial de entrenamiento para graficar.")
        return

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, marker="o", label="Train Loss")
    ax1.plot(epochs, val_losses, marker="o", label="Val Loss")
    ax1.set_title(f"Pérdida de Entrenamiento — {model_name}", fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, marker="o", label="Train Acc")
    ax2.plot(epochs, val_accs, marker="o", label="Val Acc")
    ax2.set_title(f"Accuracy de Entrenamiento — {model_name}", fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Historia de entrenamiento guardada: %s", save_path)

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Exportación
# ─────────────────────────────────────────────────────────────────────────────
def _to_native(obj: Any) -> Any:
    """
    Convierte tipos numpy a tipos nativos de Python para JSON.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_metrics_report(
    metrics: Dict[str, Any],
    model_name: str,
    output_dir: str = "reports",
    ci_data: Optional[Tuple[float, float, float]] = None,
) -> str:
    """
    Exporta métricas a JSON y CSV.
    Devuelve la ruta del CSV para compatibilidad con tu pipeline actual.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_data = {"model": model_name, **{k: _to_native(v) for k, v in metrics.items()}}

    if ci_data is not None:
        report_data["auc_roc_ci_lower"] = round(float(ci_data[1]), 4)
        report_data["auc_roc_ci_upper"] = round(float(ci_data[2]), 4)

    json_path = output_path / f"metrics_{model_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    csv_path = output_path / f"metrics_{model_name}.csv"
    pd.DataFrame([report_data]).to_csv(csv_path, index=False)

    logger.info("Métricas exportadas: %s | %s", json_path, csv_path)
    return str(csv_path)