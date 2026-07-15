# ==============================================================================
# compare_stacking_meta_models.py
# RetinAI_MVP
# Comparación justa de tres meta-modelos para el stacking:
#   1) Regresión logística
#   2) Extra Trees
#   3) XGBoost
#
# No reentrena los 7 modelos profundos. Trabaja con:
#   reports/meta_features_val.csv
#   reports/meta_features_test.csv
#
# Protocolo:
#   - Genera probabilidades OOF de 5 pliegues en validación.
#   - Selecciona el umbral SOLO con OOF, priorizando sensibilidad y exigiendo
#     una especificidad mínima configurable.
#   - Entrena cada meta-modelo con toda la validación.
#   - Evalúa una sola vez en el conjunto de prueba intacto.
#   - El mejor meta-modelo se elige por resultados OOF, no por prueba.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
META_DIR = BASE_DIR / "models" / "meta"
CANDIDATES_DIR = BASE_DIR / "models" / "meta_candidates"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_meta_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    val_path = REPORTS_DIR / "meta_features_val.csv"
    test_path = REPORTS_DIR / "meta_features_test.csv"

    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Faltan reports/meta_features_val.csv o reports/meta_features_test.csv. "
            "Primero ejecuta el entrenamiento/extracción de meta-features."
        )

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if "label" not in val_df.columns or "label" not in test_df.columns:
        raise RuntimeError("Los meta-datasets deben contener la columna 'label'.")

    feature_data = load_json(META_DIR / "meta_feature_names.json", default=None)
    if isinstance(feature_data, dict):
        feature_names = feature_data.get("feature_names", [])
    elif isinstance(feature_data, list):
        feature_names = feature_data
    else:
        feature_names = []

    feature_names = [
        name
        for name in feature_names
        if name in val_df.columns and name in test_df.columns
    ]

    if not feature_names:
        feature_names = [c for c in val_df.columns if c != "label" and c in test_df.columns]

    if not feature_names:
        raise RuntimeError("No se encontraron meta-características compatibles.")

    return val_df, test_df, feature_names


def build_fixed_models(seed: int, n_jobs: int) -> Dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=1,
            reg_lambda=8.0,
            reg_alpha=0.5,
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=seed,
        ),
    }


def build_searches(seed: int, n_jobs: int, inner_cv: StratifiedKFold) -> Dict[str, Any]:
    """Búsqueda pequeña y equivalente para la ejecución rigurosa (modo nested)."""
    fixed = build_fixed_models(seed=seed, n_jobs=n_jobs)

    return {
        "logistic_regression": GridSearchCV(
            estimator=fixed["logistic_regression"],
            param_grid={"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        ),
        "extra_trees": GridSearchCV(
            estimator=fixed["extra_trees"],
            param_grid={
                "max_depth": [4, 6, 8, None],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", 0.5],
            },
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        ),
        "xgboost": GridSearchCV(
            estimator=fixed["xgboost"],
            param_grid={
                "max_depth": [2, 3],
                "n_estimators": [250, 350],
                "learning_rate": [0.02, 0.04],
                "min_child_weight": [1, 3],
            },
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        ),
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0

    p = successes / n
    denominator = 1.0 + (z**2 / n)
    centre = p + (z**2 / (2.0 * n))
    margin = z * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n)

    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: List[float] = []

    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        y_sample = y_true[indices]
        if np.unique(y_sample).size < 2:
            continue
        aucs.append(float(roc_auc_score(y_sample, y_prob[indices])))

    if not aucs:
        return 0.0, 0.0

    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    bootstrap_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)

    sens_low, sens_high = wilson_ci(tp, tp + fn)
    spec_low, spec_high = wilson_ci(tn, tn + fp)
    auc_low, auc_high = bootstrap_auc_ci(
        y_true=y_true,
        y_prob=y_prob,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity_ci95_low": sens_low,
        "sensitivity_ci95_high": sens_high,
        "specificity_ci95_low": spec_low,
        "specificity_ci95_high": spec_high,
        "auc_ci95_low": auc_low,
        "auc_ci95_high": auc_high,
    }


def select_clinical_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Dict[str, Any]:
    """
    Prioridad declarada del proyecto:
      1. Cumplir especificidad mínima en OOF.
      2. Maximizar sensibilidad.
      3. Desempatar por F1, especificidad y precisión.
    """
    _, _, roc_thresholds = roc_curve(y_true, y_prob)
    grid = np.linspace(0.01, 0.99, 981)
    thresholds = np.unique(np.concatenate([roc_thresholds, grid]))
    thresholds = thresholds[np.isfinite(thresholds)]
    thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)

        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    eligible = [row for row in rows if row["specificity"] >= min_specificity]
    if not eligible:
        eligible = rows

    best = max(
        eligible,
        key=lambda row: (
            row["sensitivity"],
            row["f1"],
            row["specificity"],
            row["precision"],
        ),
    )
    best["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    return best


def fit_oof_and_final(
    model_name: str,
    estimator: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    outer_cv: StratifiedKFold,
    mode: str,
    seed: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray, Any, List[Dict[str, Any]]]:
    oof_prob = np.zeros(len(y_val), dtype=np.float64)
    fold_params: List[Dict[str, Any]] = []

    for fold, (train_idx, holdout_idx) in enumerate(outer_cv.split(x_val, y_val), start=1):
        if mode == "nested":
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
            fold_estimator = build_searches(seed, n_jobs, inner_cv)[model_name]
        else:
            fold_estimator = clone(estimator)

        fold_estimator.fit(x_val[train_idx], y_val[train_idx])
        oof_prob[holdout_idx] = fold_estimator.predict_proba(x_val[holdout_idx])[:, 1]

        if hasattr(fold_estimator, "best_params_"):
            fold_params.append(dict(fold_estimator.best_params_))
        else:
            fold_params.append({})

        print(f"  {model_name}: fold {fold}/5 completado")

    if mode == "nested":
        final_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 100)
        final_estimator = build_searches(seed, n_jobs, final_inner)[model_name]
    else:
        final_estimator = clone(estimator)

    final_estimator.fit(x_val, y_val)
    test_prob = final_estimator.predict_proba(x_test)[:, 1]

    if hasattr(final_estimator, "best_estimator_"):
        saved_estimator = final_estimator.best_estimator_
    else:
        saved_estimator = final_estimator

    return oof_prob, test_prob, saved_estimator, fold_params


def save_comparison_figures(results_df: pd.DataFrame, test_probabilities: Dict[str, np.ndarray], y_test: np.ndarray) -> None:
    display_names = {
        "logistic_regression": "Regresión logística",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
    }

    ordered = results_df.copy()
    ordered["display_name"] = ordered["meta_model"].map(display_names)
    x = np.arange(len(ordered))
    width = 0.24

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, ordered["test_sensitivity_percent"], width, label="Sensibilidad")
    plt.bar(x, ordered["test_specificity_percent"], width, label="Especificidad")
    plt.bar(x + width, ordered["test_auc_percent"], width, label="AUC-ROC")
    plt.xticks(x, ordered["display_name"], rotation=10)
    plt.ylabel("Porcentaje (%)")
    plt.ylim(0, 100)
    plt.title("Comparación de meta-modelos del stacking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_meta_models_metrics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(ordered["display_name"], ordered["test_fn"])
    plt.ylabel("Falsos negativos")
    plt.title("Falsos negativos por meta-modelo")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_meta_models_false_negatives.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    for model_name, probabilities in test_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{display_names[model_name]} (AUC={auc_value:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Sensibilidad")
    plt.title("Curvas ROC de los meta-modelos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_meta_models_roc.png", dpi=180)
    plt.close()


def apply_best_model(best_name: str, threshold: float, summary: Dict[str, Any]) -> None:
    source_model = CANDIDATES_DIR / f"{best_name}.pkl"
    target_model = META_DIR / "meta_model.pkl"
    target_threshold = META_DIR / "optimal_threshold.json"

    if not source_model.exists():
        raise FileNotFoundError(f"No existe el candidato: {source_model}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if target_model.exists():
        shutil.copy2(target_model, META_DIR / f"meta_model_backup_{timestamp}.pkl")

    if target_threshold.exists():
        shutil.copy2(
            target_threshold,
            META_DIR / f"optimal_threshold_backup_{timestamp}.json",
        )

    shutil.copy2(source_model, target_model)

    save_json(
        {
            "threshold": float(threshold),
            "source": "stacking_meta_model_comparison_oof",
            "selected_candidate": best_name,
            "selection_rule": (
                "máxima sensibilidad OOF con especificidad OOF mínima, "
                "desempate por AUC-ROC y F1"
            ),
            "clinical_objective": "priorizar_sensibilidad_y_reducir_falsos_negativos",
            "metrics": summary,
        },
        target_threshold,
    )

    save_json(
        {
            "applied": True,
            "selected_meta_model": best_name,
            "threshold": float(threshold),
            "summary": summary,
        },
        REPORTS_DIR / "final_selected_meta_model.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Regresión logística, Extra Trees y XGBoost como meta-modelos del stacking."
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "nested"],
        default="fast",
        help=(
            "fast: configuraciones fijas y OOF de 5 pliegues. "
            "nested: ajuste interno de hiperparámetros dentro de cada pliegue."
        ),
    )
    parser.add_argument("--min-specificity", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Reemplaza models/meta/meta_model.pkl y optimal_threshold.json con backups.",
    )
    args = parser.parse_args()

    val_df, test_df, feature_names = load_meta_data()

    x_val = val_df[feature_names].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    x_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    print("=" * 88)
    print("RETINAI — COMPARACIÓN DE META-MODELOS DEL STACKING")
    print("=" * 88)
    print(f"Modo: {args.mode}")
    print(f"Validación: {x_val.shape} | Prueba: {x_test.shape}")
    print(f"Meta-características: {len(feature_names)}")
    print(f"Especificidad mínima OOF: {args.min_specificity:.2f}")

    fixed_models = build_fixed_models(seed=args.seed, n_jobs=args.n_jobs)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    comparison_rows: List[Dict[str, Any]] = []
    oof_predictions = pd.DataFrame({"label": y_val})
    test_predictions = pd.DataFrame({"label": y_test})
    test_probability_map: Dict[str, np.ndarray] = {}

    for model_name, estimator in fixed_models.items():
        print(f"\nProcesando: {model_name}")

        oof_prob, test_prob, final_model, fold_params = fit_oof_and_final(
            model_name=model_name,
            estimator=estimator,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            outer_cv=outer_cv,
            mode=args.mode,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

        threshold_info = select_clinical_threshold(
            y_true=y_val,
            y_prob=oof_prob,
            min_specificity=args.min_specificity,
        )
        threshold = float(threshold_info["threshold"])

        oof_metrics = evaluate_predictions(
            y_true=y_val,
            y_prob=oof_prob,
            threshold=threshold,
            bootstrap_iterations=args.bootstrap,
            seed=args.seed,
        )
        test_metrics = evaluate_predictions(
            y_true=y_test,
            y_prob=test_prob,
            threshold=threshold,
            bootstrap_iterations=args.bootstrap,
            seed=args.seed + 1,
        )

        joblib.dump(final_model, CANDIDATES_DIR / f"{model_name}.pkl")
        save_json(
            {
                "meta_model": model_name,
                "mode": args.mode,
                "threshold": threshold,
                "min_specificity_oof": args.min_specificity,
                "oof_metrics": oof_metrics,
                "test_metrics": test_metrics,
                "fold_best_params": fold_params,
            },
            CANDIDATES_DIR / f"{model_name}_summary.json",
        )

        oof_predictions[f"prob_{model_name}"] = oof_prob
        test_predictions[f"prob_{model_name}"] = test_prob
        test_probability_map[model_name] = test_prob

        comparison_rows.append(
            {
                "meta_model": model_name,
                "mode": args.mode,
                "threshold_oof": threshold,
                "oof_accuracy_percent": oof_metrics["accuracy"] * 100.0,
                "oof_sensitivity_percent": oof_metrics["sensitivity"] * 100.0,
                "oof_specificity_percent": oof_metrics["specificity"] * 100.0,
                "oof_f1_percent": oof_metrics["f1"] * 100.0,
                "oof_auc_percent": oof_metrics["auc_roc"] * 100.0,
                "oof_fn": oof_metrics["fn"],
                "test_accuracy_percent": test_metrics["accuracy"] * 100.0,
                "test_sensitivity_percent": test_metrics["sensitivity"] * 100.0,
                "test_specificity_percent": test_metrics["specificity"] * 100.0,
                "test_precision_percent": test_metrics["precision"] * 100.0,
                "test_f1_percent": test_metrics["f1"] * 100.0,
                "test_auc_percent": test_metrics["auc_roc"] * 100.0,
                "test_tp": test_metrics["tp"],
                "test_tn": test_metrics["tn"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_sensitivity_ci95": (
                    f"[{test_metrics['sensitivity_ci95_low'] * 100:.2f}, "
                    f"{test_metrics['sensitivity_ci95_high'] * 100:.2f}]"
                ),
                "test_specificity_ci95": (
                    f"[{test_metrics['specificity_ci95_low'] * 100:.2f}, "
                    f"{test_metrics['specificity_ci95_high'] * 100:.2f}]"
                ),
                "test_auc_ci95": (
                    f"[{test_metrics['auc_ci95_low'] * 100:.2f}, "
                    f"{test_metrics['auc_ci95_high'] * 100:.2f}]"
                ),
            }
        )

    results_df = pd.DataFrame(comparison_rows)

    # Selección exclusivamente por OOF.
    results_df = results_df.sort_values(
        ["oof_sensitivity_percent", "oof_auc_percent", "oof_f1_percent"],
        ascending=False,
    ).reset_index(drop=True)

    best_row = results_df.iloc[0].to_dict()
    best_name = str(best_row["meta_model"])
    best_threshold = float(best_row["threshold_oof"])

    results_path = REPORTS_DIR / "stacking_meta_model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    oof_predictions.to_csv(REPORTS_DIR / "stacking_meta_model_oof_predictions.csv", index=False)
    test_predictions.to_csv(REPORTS_DIR / "stacking_meta_model_test_predictions.csv", index=False)

    save_comparison_figures(results_df, test_probability_map, y_test)

    best_summary = {
        "selected_by": "OOF validation only",
        "selection_rule": (
            "mayor sensibilidad OOF cumpliendo especificidad mínima; "
            "desempate por AUC-ROC OOF y F1 OOF"
        ),
        "min_specificity_oof": args.min_specificity,
        "mode": args.mode,
        "best_meta_model": best_name,
        "threshold": best_threshold,
        "metrics": best_row,
    }
    save_json(best_summary, CANDIDATES_DIR / "best_meta_model_summary.json")

    if args.apply_best:
        apply_best_model(
            best_name=best_name,
            threshold=best_threshold,
            summary=best_row,
        )

    display_columns = [
        "meta_model",
        "threshold_oof",
        "oof_sensitivity_percent",
        "oof_specificity_percent",
        "oof_auc_percent",
        "test_accuracy_percent",
        "test_sensitivity_percent",
        "test_specificity_percent",
        "test_auc_percent",
        "test_fn",
        "test_fp",
    ]

    print("\n" + "=" * 88)
    print("RESULTADOS")
    print("=" * 88)
    print(results_df[display_columns].round(4).to_string(index=False))

    print("\nMeta-modelo seleccionado por OOF:", best_name)
    print("Umbral OOF:", round(best_threshold, 6))
    print("Aplicado al predictor:", bool(args.apply_best))
    print("\nArchivos:")
    print("-", results_path)
    print("-", REPORTS_DIR / "stacking_meta_model_oof_predictions.csv")
    print("-", REPORTS_DIR / "stacking_meta_model_test_predictions.csv")
    print("-", FIGURES_DIR / "stacking_meta_models_metrics.png")
    print("-", FIGURES_DIR / "stacking_meta_models_false_negatives.png")
    print("-", FIGURES_DIR / "stacking_meta_models_roc.png")
    print("-", CANDIDATES_DIR / "best_meta_model_summary.json")


if __name__ == "__main__":
    main()
