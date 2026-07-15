# ==============================================================================
# train_ensemble.py
# RetinAI_MVP
# Ensemble Stacking Multimodal para detección temprana de exudados retinales
# ==============================================================================

from __future__ import annotations

import gc
import json
import logging
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    import yaml
except Exception as exc:
    raise ImportError("Falta PyYAML. Instala con: pip install PyYAML") from exc

from src.base_models import (
    build_model_from_config,
    get_enabled_model_configs,
    summarize_model,
)
from src.data_loader import build_dataloaders, load_yaml_config, set_global_seed
from src.preprocessing_branches import extract_radiomics_vector


# ==============================================================================
# LOGGING
# ==============================================================================

logger = logging.getLogger(__name__)


def setup_logging(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_file = reports_dir / "training.log"

    logger.handlers.clear()
    logging.getLogger().handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
    )


# ==============================================================================
# UTILIDADES
# ==============================================================================

def get_device(config: Dict[str, Any]) -> torch.device:
    requested = str(config.get("hardware", {}).get("device", "auto")).lower()

    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)

    logger.info("Device seleccionado: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPUs detectadas: %s", torch.cuda.device_count())
    else:
        logger.warning("GPU no detectada. Entrenando en CPU.")

    return device


def ensure_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    paths_cfg = config.get("paths", {})

    weights_dir = Path(paths_cfg.get("weights_dir", "models/weights"))
    meta_dir = Path(paths_cfg.get("meta_dir", "models/meta"))
    figures_dir = Path(paths_cfg.get("figures_dir", "reports/figures"))
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))

    for path in [weights_dir, meta_dir, figures_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "weights_dir": weights_dir,
        "meta_dir": meta_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
    }


def clean_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def get_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        return output[0]

    if hasattr(output, "logits"):
        return output.logits

    raise TypeError(f"Salida de modelo no soportada: {type(output)}")


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)


# ==============================================================================
# AGRUPACIÓN DE MODELOS
# ==============================================================================

def get_model_group(model_name: str) -> str:
    name = clean_name(model_name)

    if name in {"retfound", "retclip", "efficientnet_b0", "efficientnet_b3", "inception_v3"}:
        return "global_multiscale"

    if name in {"densenet121", "unet_encoder", "mobilenetv3_exudate_map"}:
        return "lesion_focused"

    if name in {"resnet50_cbam", "resnet50"}:
        return "attention_structure"

    return "other"


# ==============================================================================
# ENTRENAMIENTO BASE
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[Any] = None,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                logits = get_logits(outputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            logits = get_logits(outputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        logits = get_logits(outputs)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    return epoch_loss, epoch_acc


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Loss - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"training_loss_{model_name}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Val accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"training_accuracy_{model_name}.png", dpi=160)
    plt.close()


def train_base_model(
    config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: torch.device,
    dirs: Dict[str, Path],
) -> Dict[str, Any]:
    model_name = clean_name(model_cfg["name"])

    logger.info("")
    logger.info("=" * 70)
    logger.info("ENTRENANDO MODELO BASE: %s", model_name.upper())
    logger.info("=" * 70)

    bundle = build_dataloaders(
        config=config,
        model_cfg=model_cfg,
        shuffle_train=True,
    )

    class_weights = bundle.class_weights.to(device)

    model = build_model_from_config(
        model_cfg=model_cfg,
        num_classes=len(bundle.class_names),
    ).to(device)

    summary = summarize_model(model, model_name)
    logger.info("Resumen modelo: %s", summary)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise RuntimeError(f"El modelo {model_name} no tiene parámetros entrenables.")

    train_cfg = config.get("training", {})

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("learning_rate", 0.0002)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_scheduler", {}).get("factor", 0.5)),
        patience=int(train_cfg.get("lr_scheduler", {}).get("patience", 2)),
    )

    epochs = int(train_cfg.get("epochs_base", 15))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.001))

    use_amp = (
        bool(config.get("hardware", {}).get("mixed_precision", False))
        and device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    weights_path = dirs["weights_dir"] / f"{model_name}_best.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_loss, val_acc = evaluate_epoch(
            model=model,
            loader=bundle.val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        logger.info(
            "Época [%s/%s] | Train Loss: %.4f | Acc: %.4f || Val Loss: %.4f | Acc: %.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        improved = val_loss < (best_val_loss - min_delta)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_name": model_name,
                    "model_cfg": model_cfg,
                    "state_dict": model.state_dict(),
                    "class_names": bundle.class_names,
                    "input_mode": bundle.input_mode,
                    "image_size": bundle.image_size,
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                weights_path,
            )

            logger.info("  ✅ Mejor modelo guardado: %s", weights_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                "Early stopping en época %s. Mejor época: %s",
                epoch,
                best_epoch,
            )
            break

    elapsed = time.time() - start_time

    plot_training_history(
        history=history,
        model_name=model_name,
        figures_dir=dirs["figures_dir"],
    )

    logger.info(
        "Modelo %s finalizado en %.2f min. Mejor val_loss=%.4f en época %s",
        model_name,
        elapsed / 60.0,
        best_val_loss,
        best_epoch,
    )

    del model
    empty_cuda_cache()

    return {
        "model_name": model_name,
        "weights_path": str(weights_path),
        "history": history,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "elapsed_minutes": float(elapsed / 60.0),
        "summary": summary,
    }


# ==============================================================================
# EXTRACCIÓN DE PROBABILIDADES
# ==============================================================================

@torch.no_grad()
def extract_probabilities(
    config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    model_name = clean_name(model_cfg["name"])

    logger.info("Extrayendo probabilidades: %s", model_name)

    bundle = build_dataloaders(
        config=config,
        model_cfg=model_cfg,
        shuffle_train=False,
    )

    model = build_model_from_config(
        model_cfg=model_cfg,
        num_classes=len(bundle.class_names),
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def _extract(loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs_all = []
        labels_all = []
        preds_all = []

        for images, labels in tqdm(loader, desc=f"  {model_name}", leave=False):
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            logits = get_logits(outputs)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            probs_all.append(probs.detach().cpu().numpy())
            labels_all.append(labels.numpy())
            preds_all.append(preds.detach().cpu().numpy())

        probs_all = np.concatenate(probs_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)

        return probs_all, labels_all, preds_all

    train_probs, train_y, train_preds = _extract(bundle.train_loader)
    val_probs, val_y, val_preds = _extract(bundle.val_loader)
    test_probs, test_y, test_preds = _extract(bundle.test_loader)

    del model
    empty_cuda_cache()

    return {
        "model_name": model_name,
        "train_probs": train_probs,
        "train_y": train_y,
        "train_preds": train_preds,
        "val_probs": val_probs,
        "val_y": val_y,
        "val_preds": val_preds,
        "test_probs": test_probs,
        "test_y": test_y,
        "test_preds": test_preds,
        "train_df": bundle.train_df,
        "val_df": bundle.val_df,
        "test_df": bundle.test_df,
        "class_names": bundle.class_names,
    }


# ==============================================================================
# RADIOMICS
# ==============================================================================

def extract_radiomics_matrix(
    df: pd.DataFrame,
    config: Dict[str, Any],
    split_name: str,
) -> Tuple[np.ndarray, List[str]]:
    radiomics_cfg = config.get("radiomics", {})
    image_size = int(config.get("data", {}).get("image_size", 224))

    features = []
    names = None

    logger.info("Extrayendo radiomics para %s: %s imágenes", split_name, len(df))

    for image_path in tqdm(df["image_path"].tolist(), desc=f"radiomics_{split_name}"):
        vector, feature_names = extract_radiomics_vector(
            image=image_path,
            image_size=image_size,
            radiomics_cfg=radiomics_cfg,
        )

        features.append(vector)

        if names is None:
            names = feature_names

    x = np.vstack(features).astype(np.float32)
    names = names or [f"radiomics_{i}" for i in range(x.shape[1])]

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x, names


# ==============================================================================
# META FEATURES
# ==============================================================================

def build_group_features(
    model_probs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    groups: Dict[str, List[np.ndarray]] = {}

    for model_name, probs in model_probs.items():
        group = get_model_group(model_name)
        p_exudate = probs[:, 1]
        groups.setdefault(group, []).append(p_exudate)

    features = []
    names = []

    for group_name, values in sorted(groups.items()):
        arr = np.vstack(values).T

        group_mean = arr.mean(axis=1)
        group_std = arr.std(axis=1)
        group_max = arr.max(axis=1)
        group_min = arr.min(axis=1)

        features.extend([group_mean, group_std, group_max, group_min])
        names.extend(
            [
                f"group_{group_name}_mean",
                f"group_{group_name}_std",
                f"group_{group_name}_max",
                f"group_{group_name}_min",
            ]
        )

    if not features:
        return np.empty((0, 0), dtype=np.float32), []

    x = np.vstack(features).T.astype(np.float32)
    return x, names


def build_meta_features(
    model_outputs: Dict[str, Dict[str, Any]],
    split: str,
    radiomics_x: Optional[np.ndarray] = None,
    radiomics_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    meta_cfg = (config or {}).get("meta_model", {})

    model_probs: Dict[str, np.ndarray] = {}
    y_ref = None

    for model_name, output in model_outputs.items():
        probs = output[f"{split}_probs"]
        y = output[f"{split}_y"]

        model_probs[model_name] = probs

        if y_ref is None:
            y_ref = y
        else:
            if not np.array_equal(y_ref, y):
                raise RuntimeError(
                    f"Etiquetas desalineadas en split={split} para modelo {model_name}"
                )

    features = []
    names = []

    use_probs = bool(meta_cfg.get("use_base_probabilities", True))
    use_conf = bool(meta_cfg.get("use_model_confidence", True))

    for model_name, probs in model_probs.items():
        if use_probs:
            features.append(probs[:, 0])
            names.append(f"{model_name}_prob_healthy")

            features.append(probs[:, 1])
            names.append(f"{model_name}_prob_exudates")

        if use_conf:
            confidence = probs.max(axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)

            features.append(confidence)
            names.append(f"{model_name}_confidence")

            features.append(entropy)
            names.append(f"{model_name}_entropy")

    group_x, group_names = build_group_features(model_probs)

    if group_x.size > 0:
        for idx, name in enumerate(group_names):
            features.append(group_x[:, idx])
            names.append(name)

    if radiomics_x is not None and bool(meta_cfg.get("use_radiomics_features", True)):
        radiomics_names = radiomics_names or [
            f"radiomics_{i}" for i in range(radiomics_x.shape[1])
        ]

        for idx, name in enumerate(radiomics_names):
            features.append(radiomics_x[:, idx])
            names.append(name)

    x = np.vstack(features).T.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x, y_ref.astype(np.int64), names


# ==============================================================================
# MÉTRICAS
# ==============================================================================

def wilson_ci(
    successes: int,
    n: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0

    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower = (centre - margin) / denom
    upper = (centre + margin) / denom

    return max(0.0, lower), min(1.0, upper)


def find_optimal_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    specificity = 1.0 - fpr
    youden = tpr + specificity - 1.0

    best_idx = int(np.argmax(youden))

    threshold = float(thresholds[best_idx])
    sensitivity = float(tpr[best_idx])
    spec = float(specificity[best_idx])

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": spec,
        "youden": float(youden[best_idx]),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.0

    vpp = tp / max(tp + fp, 1)
    vpn = tn / max(tn + fn, 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(sensitivity),
        "sensibilidad": float(sensitivity),
        "especificidad": float(specificity),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "vpp": float(vpp),
        "vpn": float(vpn),
        "youden_index": float(sensitivity + specificity - 1.0),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_samples": int(len(y_true)),
        "n_positivos": int(np.sum(y_true == 1)),
        "n_negativos": int(np.sum(y_true == 0)),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)

    values = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        values.append(roc_auc_score(y_true[idx], y_prob[idx]))

    if not values:
        return 0.0, 0.0

    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))

    return lower, upper


def get_meta_classifier(config: Dict[str, Any]):
    meta_cfg = config.get("meta_model", {})
    model_type = str(meta_cfg.get("type", "xgboost")).lower()

    if model_type == "xgboost":
        xgb_cfg = meta_cfg.get("xgboost", {})
        return XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 200)),
            max_depth=int(xgb_cfg.get("max_depth", 3)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
            subsample=float(xgb_cfg.get("subsample", 0.85)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.85)),
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
            n_jobs=int(xgb_cfg.get("n_jobs", -1)),
            random_state=int(xgb_cfg.get("random_state", 42)),
        )

    if model_type == "random_forest":
        rf_cfg = meta_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth", 6),
            min_samples_split=int(rf_cfg.get("min_samples_split", 4)),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            random_state=int(rf_cfg.get("random_state", 42)),
        )

    if model_type == "logistic_regression":
        lr_cfg = meta_cfg.get("logistic_regression", {})
        return LogisticRegression(
            C=float(lr_cfg.get("C", 1.0)),
            max_iter=int(lr_cfg.get("max_iter", 2000)),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=int(lr_cfg.get("random_state", 42)),
        )

    raise ValueError(f"Meta-modelo no soportado: {model_type}")


def oof_meta_predictions(
    x: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    meta_cfg = config.get("meta_model", {})
    splits = int(meta_cfg.get("oof_splits", 5))
    seed = int(config.get("project", {}).get("seed", 42))

    skf = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=seed,
    )

    oof = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        clf = get_meta_classifier(config)
        clf.fit(x[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(x[val_idx])[:, 1]

        logger.info("OOF fold %s/%s completado.", fold, splits)

    return oof


# ==============================================================================
# REPORTES Y GRÁFICOS
# ==============================================================================

def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    figures_dir: Path,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión - Ensemble")
    plt.colorbar()

    classes = ["Healthy", "Exudates"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_ensemble.png", dpi=160)
    plt.close()


def save_roc_curve_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figures_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Falsos positivos")
    plt.ylabel("Verdaderos positivos")
    plt.title("Curva ROC - Ensemble")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve_ensemble.png", dpi=160)
    plt.close()


def save_precision_recall_curve_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figures_dir: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall / Sensibilidad")
    plt.ylabel("Precision / VPP")
    plt.title("Curva Precision-Recall - Ensemble")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve_ensemble.png", dpi=160)
    plt.close()


def save_probability_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.65, label="Healthy")
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.65, label="Exudates")
    plt.xlabel("Probabilidad predicha de exudados")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de probabilidades - Ensemble")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "probability_histogram_ensemble.png", dpi=160)
    plt.close()


def save_classification_report_outputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    reports_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["healthy", "exudates"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(reports_dir / "classification_report_ensemble.csv", index=True)

    table_df = report_df.copy()
    table_df = table_df[["precision", "recall", "f1-score", "support"]]
    table_df = table_df.round(4)

    plt.figure(figsize=(8, 3.8))
    plt.axis("off")
    plt.title("Classification report - Ensemble", fontsize=12, pad=12)

    table = plt.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(figures_dir / "classification_report_ensemble.png", dpi=180)
    plt.close()

    return report_df


def save_model_comparison_table(
    base_metrics: List[Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    reports_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    rows = []

    for item in base_metrics:
        acc = float(item["accuracy"])
        n = int(item["n_samples"])
        successes = int(round(acc * n))
        ci_low, ci_high = wilson_ci(successes, n)

        rows.append(
            {
                "model": item["model_name"],
                "accuracy": acc,
                "macro_f1": float(item["macro_f1"]),
                "weighted_f1": float(item["weighted_f1"]),
                "accuracy_ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "auc_roc": float(item["auc_roc"]),
            }
        )

    acc = float(ensemble_metrics["accuracy"])
    n = int(ensemble_metrics["n_samples"])
    successes = int(round(acc * n))
    ci_low, ci_high = wilson_ci(successes, n)

    rows.append(
        {
            "model": "ensemble_stacking",
            "accuracy": acc,
            "macro_f1": float(ensemble_metrics.get("macro_f1", ensemble_metrics["f1_score"])),
            "weighted_f1": float(ensemble_metrics.get("weighted_f1", ensemble_metrics["f1_score"])),
            "accuracy_ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
            "auc_roc": float(ensemble_metrics["auc_roc"]),
        }
    )

    df = pd.DataFrame(rows)
    df = df.sort_values("auc_roc", ascending=False)
    df.to_csv(reports_dir / "model_comparison_wilson_ci.csv", index=False)

    display_df = df.copy()
    numeric_cols = ["accuracy", "macro_f1", "weighted_f1", "auc_roc"]
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    plt.figure(figsize=(11, max(3.5, 0.45 * len(display_df) + 1.5)))
    plt.axis("off")
    plt.title("Comparación consolidada de modelos", fontsize=12, pad=12)

    table = plt.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_wilson_ci.png", dpi=180)
    plt.close()

    return df


def save_feature_correlation_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    figures_dir: Path,
    reports_dir: Path,
    top_k: int = 25,
) -> None:
    df = pd.DataFrame(x, columns=feature_names)
    df["label"] = y

    corr_with_label = df.corr(numeric_only=True)["label"].drop("label")
    corr_ranked = corr_with_label.abs().sort_values(ascending=False).head(top_k)

    selected = corr_ranked.index.tolist() + ["label"]

    corr_matrix = df[selected].corr(numeric_only=True)
    corr_matrix.to_csv(reports_dir / "feature_correlation_top.csv")

    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix.values, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Correlación")
    plt.xticks(
        np.arange(len(selected)),
        selected,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        np.arange(len(selected)),
        selected,
        fontsize=7,
    )
    plt.title("Mapa de calor de correlación - características principales")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_correlation_heatmap.png", dpi=180)
    plt.close()


def save_feature_importance(
    meta_model,
    feature_names: List[str],
    figures_dir: Path,
    reports_dir: Path,
    top_k: int = 25,
) -> None:
    if not hasattr(meta_model, "feature_importances_"):
        return

    importances = np.asarray(meta_model.feature_importances_, dtype=np.float32)

    if importances.size != len(feature_names):
        return

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    df.to_csv(reports_dir / "meta_feature_importance.csv", index=False)

    top_df = df.head(top_k).iloc[::-1]

    plt.figure(figsize=(9, 7))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importancia")
    plt.title("Top características del meta-modelo")
    plt.tight_layout()
    plt.savefig(figures_dir / "meta_feature_importance_top.png", dpi=180)
    plt.close()


# ==============================================================================
# COMPARACIÓN BASE
# ==============================================================================

def compute_base_model_metrics(
    model_outputs: Dict[str, Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    rows = []

    for model_name, output in model_outputs.items():
        y_true = output["test_y"]
        probs = output["test_probs"]
        y_prob = probs[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = compute_metrics(y_true, y_prob, threshold=threshold)

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        metrics["model_name"] = model_name
        metrics["macro_f1"] = float(report["macro avg"]["f1-score"])
        metrics["weighted_f1"] = float(report["weighted avg"]["f1-score"])

        rows.append(metrics)

    return rows


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError("No se encontró config.yaml")

    config = load_yaml_config(config_path)

    dirs = ensure_dirs(config)
    setup_logging(dirs["reports_dir"])

    logger.info("=" * 70)
    logger.info("RETINAI MVP - ENSEMBLE STACKING MULTIMODAL")
    logger.info("=" * 70)

    seed = int(config.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    device = get_device(config)

    mlflow_enabled = bool(config.get("mlflow", {}).get("enabled", False))
    logger.info("MLflow activado: %s", mlflow_enabled)

    model_cfgs = get_enabled_model_configs(config)

    # Saltar modelos fundacionales hasta integrar pesos/arquitectura.
    active_model_cfgs = []
    skipped_models = []

    for model_cfg in model_cfgs:
        name = clean_name(model_cfg.get("name", ""))

        if name in {"retfound", "retclip"}:
            skipped_models.append(name)
            continue

        active_model_cfgs.append(model_cfg)

    if not active_model_cfgs:
        raise RuntimeError("No hay modelos base activos para entrenar.")

    logger.info("Modelos activos:")
    for model_cfg in active_model_cfgs:
        logger.info(
            " - %s | input=%s | size=%s | group=%s",
            model_cfg["name"],
            model_cfg.get("input_mode", "rgb"),
            model_cfg.get("image_size", 224),
            get_model_group(model_cfg["name"]),
        )

    if skipped_models:
        logger.info("Modelos preparados pero no activos todavía: %s", skipped_models)

    # --------------------------------------------------------------------------
    # 1. Entrenar modelos base secuencialmente
    # --------------------------------------------------------------------------

    base_training_results = []

    for model_cfg in active_model_cfgs:
        result = train_base_model(
            config=config,
            model_cfg=model_cfg,
            device=device,
            dirs=dirs,
        )
        base_training_results.append(result)

    save_json(
        {
            "base_training_results": base_training_results,
            "active_models": [clean_name(m["name"]) for m in active_model_cfgs],
            "skipped_models": skipped_models,
        },
        dirs["reports_dir"] / "base_training_results.json",
    )

    # --------------------------------------------------------------------------
    # 2. Extraer probabilidades base
    # --------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRAYENDO PREDICCIONES DE MODELOS BASE")
    logger.info("=" * 70)

    model_outputs: Dict[str, Dict[str, Any]] = {}

    for model_cfg in active_model_cfgs:
        model_name = clean_name(model_cfg["name"])
        weights_path = dirs["weights_dir"] / f"{model_name}_best.pth"

        if not weights_path.exists():
            raise FileNotFoundError(f"No se encontró peso entrenado: {weights_path}")

        output = extract_probabilities(
            config=config,
            model_cfg=model_cfg,
            weights_path=weights_path,
            device=device,
        )

        model_outputs[model_name] = output

    first_output = next(iter(model_outputs.values()))
    train_df = first_output["train_df"]
    val_df = first_output["val_df"]
    test_df = first_output["test_df"]

    # --------------------------------------------------------------------------
    # 3. Radiomics
    # --------------------------------------------------------------------------

    radiomics_enabled = bool(config.get("radiomics", {}).get("enabled", True))

    if radiomics_enabled:
        x_train_rad, rad_names = extract_radiomics_matrix(train_df, config, "train")
        x_val_rad, _ = extract_radiomics_matrix(val_df, config, "val")
        x_test_rad, _ = extract_radiomics_matrix(test_df, config, "test")

        scaler = StandardScaler()
        x_val_rad_scaled = scaler.fit_transform(x_val_rad)
        x_test_rad_scaled = scaler.transform(x_test_rad)

        joblib.dump(scaler, dirs["meta_dir"] / "radiomics_scaler.pkl")

        pd.DataFrame(x_train_rad, columns=rad_names).to_csv(
            dirs["reports_dir"] / "radiomics_train.csv",
            index=False,
        )
        pd.DataFrame(x_val_rad, columns=rad_names).to_csv(
            dirs["reports_dir"] / "radiomics_val.csv",
            index=False,
        )
        pd.DataFrame(x_test_rad, columns=rad_names).to_csv(
            dirs["reports_dir"] / "radiomics_test.csv",
            index=False,
        )
    else:
        x_val_rad_scaled = None
        x_test_rad_scaled = None
        rad_names = None

    # --------------------------------------------------------------------------
    # 4. Construir meta-dataset
    # --------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONSTRUYENDO META-DATASET")
    logger.info("=" * 70)

    x_val_meta, y_val, feature_names = build_meta_features(
        model_outputs=model_outputs,
        split="val",
        radiomics_x=x_val_rad_scaled,
        radiomics_names=rad_names,
        config=config,
    )

    x_test_meta, y_test, _ = build_meta_features(
        model_outputs=model_outputs,
        split="test",
        radiomics_x=x_test_rad_scaled,
        radiomics_names=rad_names,
        config=config,
    )

    logger.info("Meta-dataset VAL:  X=%s | y=%s", x_val_meta.shape, y_val.shape)
    logger.info("Meta-dataset TEST: X=%s | y=%s", x_test_meta.shape, y_test.shape)

    pd.DataFrame(x_val_meta, columns=feature_names).assign(label=y_val).to_csv(
        dirs["reports_dir"] / "meta_features_val.csv",
        index=False,
    )

    pd.DataFrame(x_test_meta, columns=feature_names).assign(label=y_test).to_csv(
        dirs["reports_dir"] / "meta_features_test.csv",
        index=False,
    )

    save_json(
        {"feature_names": feature_names},
        dirs["meta_dir"] / "meta_feature_names.json",
    )

    save_json(
        {"base_model_names": list(model_outputs.keys())},
        dirs["meta_dir"] / "base_model_names.json",
    )

    # --------------------------------------------------------------------------
    # 5. OOF para umbral
    # --------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("OOF META-MODEL Y SELECCIÓN DE UMBRAL")
    logger.info("=" * 70)

    oof_probs = oof_meta_predictions(
        x=x_val_meta,
        y=y_val,
        config=config,
    )

    threshold_info = find_optimal_threshold_youden(y_val, oof_probs)
    optimal_threshold = float(threshold_info["threshold"])

    save_json(
        threshold_info,
        dirs["meta_dir"] / "optimal_threshold.json",
    )

    logger.info(
        "Umbral óptimo: %.4f | Sens=%.4f | Spec=%.4f | Youden=%.4f",
        threshold_info["threshold"],
        threshold_info["sensitivity"],
        threshold_info["specificity"],
        threshold_info["youden"],
    )

    # --------------------------------------------------------------------------
    # 6. Entrenar meta-modelo final
    # --------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("ENTRENANDO META-MODELO FINAL")
    logger.info("=" * 70)

    meta_model = get_meta_classifier(config)
    meta_model.fit(x_val_meta, y_val)

    joblib.dump(meta_model, dirs["meta_dir"] / "meta_model.pkl")
    logger.info("Meta-modelo guardado en: %s", dirs["meta_dir"] / "meta_model.pkl")

    # --------------------------------------------------------------------------
    # 7. Evaluación final TEST
    # --------------------------------------------------------------------------

    y_test_prob = meta_model.predict_proba(x_test_meta)[:, 1]
    y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

    ensemble_metrics = compute_metrics(
        y_true=y_test,
        y_prob=y_test_prob,
        threshold=optimal_threshold,
    )

    report = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
        zero_division=0,
    )

    ensemble_metrics["macro_f1"] = float(report["macro avg"]["f1-score"])
    ensemble_metrics["weighted_f1"] = float(report["weighted avg"]["f1-score"])

    metric_cfg = config.get("metrics", {})
    if metric_cfg.get("bootstrap_auc", True):
        ci_low, ci_high = bootstrap_auc_ci(
            y_test,
            y_test_prob,
            n_bootstraps=int(metric_cfg.get("bootstrap_iterations", 1000)),
            seed=seed,
        )
        ensemble_metrics["auc_roc_ci_lower"] = float(ci_low)
        ensemble_metrics["auc_roc_ci_upper"] = float(ci_high)

    logger.info("")
    logger.info("=" * 70)
    logger.info("MÉTRICAS CLÍNICAS - ENSEMBLE FINAL")
    logger.info("=" * 70)
    logger.info(
        "Sensibilidad: %.4f (%s TP / %s positivos)",
        ensemble_metrics["sensibilidad"],
        ensemble_metrics["tp"],
        ensemble_metrics["n_positivos"],
    )
    logger.info(
        "Especificidad: %.4f (%s TN / %s negativos)",
        ensemble_metrics["especificidad"],
        ensemble_metrics["tn"],
        ensemble_metrics["n_negativos"],
    )
    logger.info("AUC-ROC: %.4f", ensemble_metrics["auc_roc"])
    logger.info("F1-Score: %.4f", ensemble_metrics["f1_score"])
    logger.info("Precisión: %.4f", ensemble_metrics["precision"])
    logger.info("Accuracy: %.4f", ensemble_metrics["accuracy"])
    logger.info("Índice de Youden: %.4f", ensemble_metrics["youden_index"])
    logger.info("VPP: %.4f", ensemble_metrics["vpp"])
    logger.info("VPN: %.4f", ensemble_metrics["vpn"])

    if "auc_roc_ci_lower" in ensemble_metrics:
        logger.info(
            "AUC-ROC IC95%%: [%.4f, %.4f]",
            ensemble_metrics["auc_roc_ci_lower"],
            ensemble_metrics["auc_roc_ci_upper"],
        )

    # --------------------------------------------------------------------------
    # 8. Reportes
    # --------------------------------------------------------------------------

    base_metrics = compute_base_model_metrics(
        model_outputs=model_outputs,
        threshold=0.5,
    )

    pd.DataFrame(base_metrics).to_csv(
        dirs["reports_dir"] / "base_model_metrics.csv",
        index=False,
    )

    save_json(
        ensemble_metrics,
        dirs["reports_dir"] / "metrics_ensemble_stacking.json",
    )

    pd.DataFrame([ensemble_metrics]).to_csv(
        dirs["reports_dir"] / "metrics_ensemble_stacking.csv",
        index=False,
    )

    save_confusion_matrix_plot(
        y_true=y_test,
        y_prob=y_test_prob,
        threshold=optimal_threshold,
        figures_dir=dirs["figures_dir"],
    )

    save_roc_curve_plot(
        y_true=y_test,
        y_prob=y_test_prob,
        figures_dir=dirs["figures_dir"],
    )

    save_precision_recall_curve_plot(
        y_true=y_test,
        y_prob=y_test_prob,
        figures_dir=dirs["figures_dir"],
    )

    save_probability_histogram(
        y_true=y_test,
        y_prob=y_test_prob,
        figures_dir=dirs["figures_dir"],
    )

    save_classification_report_outputs(
        y_true=y_test,
        y_prob=y_test_prob,
        threshold=optimal_threshold,
        reports_dir=dirs["reports_dir"],
        figures_dir=dirs["figures_dir"],
    )

    comparison_df = save_model_comparison_table(
        base_metrics=base_metrics,
        ensemble_metrics=ensemble_metrics,
        reports_dir=dirs["reports_dir"],
        figures_dir=dirs["figures_dir"],
    )

    save_feature_correlation_heatmap(
        x=x_val_meta,
        y=y_val,
        feature_names=feature_names,
        figures_dir=dirs["figures_dir"],
        reports_dir=dirs["reports_dir"],
        top_k=25,
    )

    save_feature_importance(
        meta_model=meta_model,
        feature_names=feature_names,
        figures_dir=dirs["figures_dir"],
        reports_dir=dirs["reports_dir"],
        top_k=25,
    )

    # --------------------------------------------------------------------------
    # 9. Summary final
    # --------------------------------------------------------------------------

    training_summary = {
        "project_version": config.get("project", {}).get("version", "2.0.0"),
        "seed": seed,
        "device": str(device),
        "active_base_models": list(model_outputs.keys()),
        "skipped_models": skipped_models,
        "base_model_count": len(model_outputs),
        "meta_model": config.get("meta_model", {}).get("type", "xgboost"),
        "threshold": optimal_threshold,
        "oof_threshold_info": threshold_info,
        "val_meta_samples": int(x_val_meta.shape[0]),
        "test_meta_samples": int(x_test_meta.shape[0]),
        "meta_feature_count": int(x_val_meta.shape[1]),
        "radiomics_enabled": radiomics_enabled,
        "radiomics_feature_count": int(len(rad_names) if rad_names is not None else 0),
        "test_metrics": ensemble_metrics,
        "base_training_results": base_training_results,
    }

    save_json(
        training_summary,
        dirs["reports_dir"] / "training_summary.json",
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("Umbral óptimo: %.4f", optimal_threshold)
    logger.info(
        "Sensibilidad: %.4f | Especificidad: %.4f",
        ensemble_metrics["sensibilidad"],
        ensemble_metrics["especificidad"],
    )
    logger.info(
        "AUC-ROC: %.4f",
        ensemble_metrics["auc_roc"],
    )
    logger.info("Reportes guardados en: %s", dirs["reports_dir"])
    logger.info("Figuras guardadas en: %s", dirs["figures_dir"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()