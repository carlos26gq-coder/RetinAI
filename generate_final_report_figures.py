from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "final_report"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PATH = REPORTS_DIR / "stacking_meta_model_comparison.csv"
OOF_PREDICTIONS_PATH = REPORTS_DIR / "stacking_meta_model_oof_predictions.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "stacking_meta_model_test_predictions.csv"

TRAINING_RESULTS_PATHS = [
    REPORTS_DIR / "base_training_results.json",
    REPORTS_DIR / "training_summary.json",
]

MODEL_ORDER = [
    "xgboost",
    "logistic_regression",
    "extra_trees",
]

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "logistic_regression": "Regresión logística",
    "extra_trees": "Extra Trees",
}

BASE_MODEL_ORDER = [
    "efficientnet_b0",
    "efficientnet_b3",
    "densenet121",
    "inception_v3",
    "unet_encoder",
    "mobilenetv3_exudate_map",
    "resnet50_cbam",
]

BASE_DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNetB0",
    "efficientnet_b3": "EfficientNetB3",
    "densenet121": "DenseNet121",
    "inception_v3": "InceptionV3",
    "unet_encoder": "U-Net encoder",
    "mobilenetv3_exudate_map": "MobileNetV3 exudate map",
    "resnet50_cbam": "ResNet50-CBAM",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido:\n{path}"
        )


def load_training_results() -> List[Dict]:
    for path in TRAINING_RESULTS_PATHS:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        results = data.get("base_training_results", [])
        if results:
            return results

    raise FileNotFoundError(
        "No se encontró base_training_results.json ni un "
        "training_summary.json con los historiales."
    )


def metrics_at_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_prediction = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_prediction,
        labels=[0, 1],
    ).ravel()

    specificity = tn / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_prediction),
        "sensitivity": recall_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "specificity": specificity,
        "precision": precision_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def plot_training_curves(training_results: List[Dict]) -> None:
    by_name = {
        item["model_name"]: item
        for item in training_results
    }

    # Exactitud
    figure, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.ravel()

    for index, model_name in enumerate(BASE_MODEL_ORDER):
        axis = axes[index]
        item = by_name.get(model_name)

        if item is None:
            axis.set_visible(False)
            continue

        history = item["history"]
        train_accuracy = history["train_acc"]
        validation_accuracy = history["val_acc"]
        epochs = np.arange(1, len(train_accuracy) + 1)

        axis.plot(
            epochs,
            train_accuracy,
            marker="o",
            markersize=3,
            label="Entrenamiento",
        )
        axis.plot(
            epochs,
            validation_accuracy,
            marker="o",
            markersize=3,
            label="Validación",
        )

        best_epoch = item.get("best_epoch")
        if best_epoch is not None:
            axis.axvline(
                best_epoch,
                linestyle="--",
                alpha=0.6,
                label="Mejor época",
            )

        axis.set_title(BASE_DISPLAY_NAMES[model_name])
        axis.set_xlabel("Época")
        axis.set_ylabel("Exactitud")
        axis.set_ylim(0.45, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[-1].set_visible(False)

    figure.suptitle(
        "Exactitud de entrenamiento y validación "
        "de los siete modelos profundos",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(
        FIGURES_DIR / "fig07_training_accuracy_7_models.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Pérdida
    figure, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.ravel()

    for index, model_name in enumerate(BASE_MODEL_ORDER):
        axis = axes[index]
        item = by_name.get(model_name)

        if item is None:
            axis.set_visible(False)
            continue

        history = item["history"]
        train_loss = history["train_loss"]
        validation_loss = history["val_loss"]
        epochs = np.arange(1, len(train_loss) + 1)

        axis.plot(
            epochs,
            train_loss,
            marker="o",
            markersize=3,
            label="Entrenamiento",
        )
        axis.plot(
            epochs,
            validation_loss,
            marker="o",
            markersize=3,
            label="Validación",
        )

        best_epoch = item.get("best_epoch")
        if best_epoch is not None:
            axis.axvline(
                best_epoch,
                linestyle="--",
                alpha=0.6,
                label="Mejor época",
            )

        axis.set_title(BASE_DISPLAY_NAMES[model_name])
        axis.set_xlabel("Época")
        axis.set_ylabel("Pérdida")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[-1].set_visible(False)

    figure.suptitle(
        "Pérdida de entrenamiento y validación "
        "de los siete modelos profundos",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(
        FIGURES_DIR / "fig08_training_loss_7_models.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def prepare_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    comparison = comparison.set_index("meta_model")
    comparison = comparison.loc[MODEL_ORDER].reset_index()
    comparison["display_name"] = comparison["meta_model"].map(
        DISPLAY_NAMES
    )
    return comparison


def plot_oof_comparison(comparison: pd.DataFrame) -> None:
    comparison = prepare_comparison(comparison)

    x_positions = np.arange(len(comparison))
    width = 0.24

    figure, axis = plt.subplots(figsize=(10, 6))

    bars_sensitivity = axis.bar(
        x_positions - width,
        comparison["oof_sensitivity_percent"],
        width,
        label="Sensibilidad",
    )
    bars_specificity = axis.bar(
        x_positions,
        comparison["oof_specificity_percent"],
        width,
        label="Especificidad",
    )
    bars_auc = axis.bar(
        x_positions + width,
        comparison["oof_auc_percent"],
        width,
        label="AUC-ROC",
    )

    axis.bar_label(
        bars_sensitivity,
        fmt="%.1f",
        fontsize=8,
        padding=2,
    )
    axis.bar_label(
        bars_specificity,
        fmt="%.1f",
        fontsize=8,
        padding=2,
    )
    axis.bar_label(
        bars_auc,
        fmt="%.1f",
        fontsize=8,
        padding=2,
    )

    axis.set_xticks(
        x_positions,
        comparison["display_name"],
    )
    axis.set_ylabel("Porcentaje (%)")
    axis.set_ylim(0, 100)
    axis.set_title(
        "Comparación de meta-modelos mediante validación OOF"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig09_oof_meta_models_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))

    bars = axis.bar(
        comparison["display_name"],
        comparison["oof_fn"],
    )
    axis.bar_label(bars, fmt="%d", padding=3)

    axis.set_ylabel("Falsos negativos")
    axis.set_title(
        "Falsos negativos en validación fuera de pliegue"
    )
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig10_oof_false_negatives.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_threshold_analysis(
    comparison: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    xgboost_row = comparison.loc[
        comparison["meta_model"] == "xgboost"
    ].iloc[0]

    selected_threshold = float(
        xgboost_row["threshold_oof"]
    )

    y_oof = oof_predictions["label"].to_numpy(dtype=int)
    p_oof = oof_predictions[
        "prob_xgboost"
    ].to_numpy(dtype=float)

    thresholds = np.linspace(0.01, 0.99, 981)

    rows = []
    for threshold in thresholds:
        result = metrics_at_threshold(
            y_oof,
            p_oof,
            float(threshold),
        )
        rows.append(result)

    threshold_frame = pd.DataFrame(rows)
    threshold_frame.to_csv(
        REPORTS_DIR / "xgboost_threshold_sweep_oof.csv",
        index=False,
    )

    selected_metrics = metrics_at_threshold(
        y_oof,
        p_oof,
        selected_threshold,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    axes[0].plot(
        threshold_frame["threshold"],
        threshold_frame["sensitivity"] * 100,
        label="Sensibilidad",
    )
    axes[0].plot(
        threshold_frame["threshold"],
        threshold_frame["specificity"] * 100,
        label="Especificidad",
    )
    axes[0].axhline(
        80,
        linestyle=":",
        label="Especificidad mínima 80 %",
    )
    axes[0].axvline(
        selected_threshold,
        linestyle="--",
        label=f"Umbral seleccionado = {selected_threshold:.4f}",
    )

    axes[0].scatter(
        [selected_threshold],
        [selected_metrics["sensitivity"] * 100],
        zorder=5,
    )
    axes[0].scatter(
        [selected_threshold],
        [selected_metrics["specificity"] * 100],
        zorder=5,
    )

    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_title(
        "Sensibilidad y especificidad según el umbral"
    )
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        threshold_frame["threshold"],
        threshold_frame["fn"],
        label="Falsos negativos",
    )
    axes[1].plot(
        threshold_frame["threshold"],
        threshold_frame["fp"],
        label="Falsos positivos",
    )
    axes[1].axvline(
        selected_threshold,
        linestyle="--",
        label=f"Umbral seleccionado = {selected_threshold:.4f}",
    )

    axes[1].scatter(
        [selected_threshold],
        [selected_metrics["fn"]],
        zorder=5,
    )
    axes[1].scatter(
        [selected_threshold],
        [selected_metrics["fp"]],
        zorder=5,
    )

    axes[1].set_xlabel("Umbral de decisión")
    axes[1].set_ylabel("Cantidad de casos")
    axes[1].set_title(
        "Falsos negativos y falsos positivos según el umbral"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle(
        "Análisis del umbral de XGBoost "
        "sobre predicciones OOF",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(
        FIGURES_DIR / "fig11_xgboost_threshold_oof.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Comparación entre umbral 0.50 y umbral optimizado
    threshold_comparison_rows = []

    datasets = {
        "Validación OOF": (
            y_oof,
            p_oof,
        ),
        "Prueba": (
            test_predictions["label"].to_numpy(dtype=int),
            test_predictions[
                "prob_xgboost"
            ].to_numpy(dtype=float),
        ),
    }

    for dataset_name, (
        y_true,
        probabilities,
    ) in datasets.items():
        for threshold_name, threshold in [
            ("Umbral estándar", 0.50),
            ("Umbral optimizado", selected_threshold),
        ]:
            result = metrics_at_threshold(
                y_true,
                probabilities,
                threshold,
            )

            threshold_comparison_rows.append(
                {
                    "dataset": dataset_name,
                    "configuration": threshold_name,
                    "threshold": threshold,
                    "accuracy_percent":
                        result["accuracy"] * 100,
                    "sensitivity_percent":
                        result["sensitivity"] * 100,
                    "specificity_percent":
                        result["specificity"] * 100,
                    "precision_percent":
                        result["precision"] * 100,
                    "f1_percent":
                        result["f1"] * 100,
                    "tp": result["tp"],
                    "tn": result["tn"],
                    "fp": result["fp"],
                    "fn": result["fn"],
                }
            )

    threshold_comparison = pd.DataFrame(
        threshold_comparison_rows
    )
    threshold_comparison.to_csv(
        REPORTS_DIR / "xgboost_threshold_comparison.csv",
        index=False,
    )

    with (
        REPORTS_DIR / "xgboost_threshold_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_threshold": selected_threshold,
                "oof_selected_metrics": selected_metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def plot_test_roc(
    comparison: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    y_test = test_predictions["label"].to_numpy(dtype=int)

    figure, axis = plt.subplots(figsize=(8, 7))

    for model_name in MODEL_ORDER:
        probabilities = test_predictions[
            f"prob_{model_name}"
        ].to_numpy(dtype=float)

        threshold = float(
            comparison.loc[
                comparison["meta_model"] == model_name,
                "threshold_oof",
            ].iloc[0]
        )

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(y_test, probabilities)

        axis.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{DISPLAY_NAMES[model_name]} "
                f"(AUC={auc_value:.4f})"
            ),
        )

        operating_metrics = metrics_at_threshold(
            y_test,
            probabilities,
            threshold,
        )

        false_positive_rate = (
            operating_metrics["fp"]
            / max(
                operating_metrics["fp"]
                + operating_metrics["tn"],
                1,
            )
        )

        axis.scatter(
            false_positive_rate,
            operating_metrics["sensitivity"],
            s=50,
            zorder=5,
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Clasificación aleatoria",
    )

    axis.set_xlabel("Tasa de falsos positivos")
    axis.set_ylabel("Sensibilidad")
    axis.set_title(
        "Curvas ROC de los meta-modelos "
        "en el conjunto de prueba"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig12_test_roc_meta_models.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_confusion_matrices(
    comparison: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    y_test = test_predictions["label"].to_numpy(dtype=int)

    # Figura combinada para el Word
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    for axis, model_name in zip(axes, MODEL_ORDER):
        probabilities = test_predictions[
            f"prob_{model_name}"
        ].to_numpy(dtype=float)

        threshold = float(
            comparison.loc[
                comparison["meta_model"] == model_name,
                "threshold_oof",
            ].iloc[0]
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Sana", "Exudados"],
        )
        display.plot(
            ax=axis,
            colorbar=False,
            values_format="d",
            cmap="Blues",
        )

        axis.set_title(
            f"{DISPLAY_NAMES[model_name]}\n"
            f"Umbral = {threshold:.4f}"
        )
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Valor real")

    figure.suptitle(
        "Matrices de confusión de los meta-modelos "
        "en el conjunto de prueba",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.91])
    figure.savefig(
        FIGURES_DIR
        / "fig13_test_confusion_matrices_meta_models.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Matrices individuales para PPT o anexos
    for model_name in MODEL_ORDER:
        probabilities = test_predictions[
            f"prob_{model_name}"
        ].to_numpy(dtype=float)

        threshold = float(
            comparison.loc[
                comparison["meta_model"] == model_name,
                "threshold_oof",
            ].iloc[0]
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        figure, axis = plt.subplots(figsize=(6, 5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Sana", "Exudados"],
        )
        display.plot(
            ax=axis,
            colorbar=False,
            values_format="d",
            cmap="Blues",
        )

        axis.set_title(
            f"{DISPLAY_NAMES[model_name]} "
            f"— umbral {threshold:.4f}"
        )
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Valor real")

        figure.tight_layout()
        figure.savefig(
            FIGURES_DIR
            / f"confusion_matrix_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_xgboost_precision_recall(
    comparison: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    y_test = test_predictions["label"].to_numpy(dtype=int)
    probabilities = test_predictions[
        "prob_xgboost"
    ].to_numpy(dtype=float)

    threshold = float(
        comparison.loc[
            comparison["meta_model"] == "xgboost",
            "threshold_oof",
        ].iloc[0]
    )

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )
    average_precision = average_precision_score(
        y_test,
        probabilities,
    )

    operating_metrics = metrics_at_threshold(
        y_test,
        probabilities,
        threshold,
    )

    figure, axis = plt.subplots(figsize=(8, 7))

    axis.plot(
        recall,
        precision,
        linewidth=2,
        label=f"XGBoost (AP={average_precision:.4f})",
    )
    axis.scatter(
        operating_metrics["sensitivity"],
        operating_metrics["precision"],
        s=70,
        label=f"Umbral {threshold:.4f}",
        zorder=5,
    )

    axis.set_xlabel("Sensibilidad / Recall")
    axis.set_ylabel("Precisión")
    axis.set_title(
        "Curva Precision-Recall del stacking final"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR
        / "fig14_xgboost_precision_recall_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def export_final_tables(
    comparison: pd.DataFrame,
) -> None:
    comparison = prepare_comparison(comparison)

    oof_table = comparison[
        [
            "display_name",
            "threshold_oof",
            "oof_accuracy_percent",
            "oof_sensitivity_percent",
            "oof_specificity_percent",
            "oof_f1_percent",
            "oof_auc_percent",
            "oof_fn",
        ]
    ].copy()

    oof_table.columns = [
        "Meta-modelo",
        "Umbral",
        "Accuracy (%)",
        "Sensibilidad (%)",
        "Especificidad (%)",
        "F1 (%)",
        "AUC-ROC (%)",
        "FN",
    ]

    oof_table.round(2).to_csv(
        REPORTS_DIR / "table_v_oof_final.csv",
        index=False,
    )

    test_table = comparison[
        [
            "display_name",
            "test_accuracy_percent",
            "test_sensitivity_percent",
            "test_specificity_percent",
            "test_precision_percent",
            "test_f1_percent",
            "test_auc_percent",
            "test_tp",
            "test_tn",
            "test_fp",
            "test_fn",
        ]
    ].copy()

    test_table.columns = [
        "Meta-modelo",
        "Accuracy (%)",
        "Sensibilidad (%)",
        "Especificidad (%)",
        "Precisión (%)",
        "F1 (%)",
        "AUC-ROC (%)",
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    test_table.round(2).to_csv(
        REPORTS_DIR / "table_test_final.csv",
        index=False,
    )


def main() -> None:
    required_paths = [
        COMPARISON_PATH,
        OOF_PREDICTIONS_PATH,
        TEST_PREDICTIONS_PATH,
    ]

    for path in required_paths:
        require_file(path)

    training_results = load_training_results()
    comparison = pd.read_csv(COMPARISON_PATH)
    oof_predictions = pd.read_csv(
        OOF_PREDICTIONS_PATH
    )
    test_predictions = pd.read_csv(
        TEST_PREDICTIONS_PATH
    )

    required_prediction_columns = {
        "label",
        "prob_xgboost",
        "prob_logistic_regression",
        "prob_extra_trees",
    }

    if not required_prediction_columns.issubset(
        set(oof_predictions.columns)
    ):
        raise RuntimeError(
            "El archivo OOF no contiene todas las "
            "columnas de probabilidades requeridas."
        )

    if not required_prediction_columns.issubset(
        set(test_predictions.columns)
    ):
        raise RuntimeError(
            "El archivo de prueba no contiene todas las "
            "columnas de probabilidades requeridas."
        )

    plot_training_curves(training_results)
    plot_oof_comparison(comparison)
    plot_threshold_analysis(
        comparison,
        oof_predictions,
        test_predictions,
    )
    plot_test_roc(comparison, test_predictions)
    plot_confusion_matrices(
        comparison,
        test_predictions,
    )
    plot_xgboost_precision_recall(
        comparison,
        test_predictions,
    )
    export_final_tables(comparison)

    print("=" * 72)
    print("FIGURAS FINALES GENERADAS")
    print("=" * 72)
    print(f"Carpeta: {FIGURES_DIR}")
    print()
    print("Figuras:")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {path.name}")

    print()
    print("Tablas y análisis:")
    print(" - reports/table_v_oof_final.csv")
    print(" - reports/table_test_final.csv")
    print(" - reports/xgboost_threshold_sweep_oof.csv")
    print(" - reports/xgboost_threshold_comparison.csv")
    print(" - reports/xgboost_threshold_summary.json")


if __name__ == "__main__":
    main()