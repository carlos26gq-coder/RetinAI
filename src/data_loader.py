# ==============================================================================
# src/data_loader.py
# RetinAI_MVP
# Carga de datos multimodal para Ensemble Stacking de exudados retinales
# ==============================================================================

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


try:
    import albumentations as A
except Exception:
    A = None


try:
    import yaml
except Exception:
    yaml = None


try:
    from src.preprocessing_branches import (
        ensure_rgb_array,
        preprocess_branch_numpy,
        to_tensor_chw,
    )
except Exception:
    from preprocessing_branches import (
        ensure_rgb_array,
        preprocess_branch_numpy,
        to_tensor_chw,
    )


# ==============================================================================
# LOGGING
# ==============================================================================

logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES
# ==============================================================================

DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ==============================================================================
# UTILIDADES GENERALES
# ==============================================================================

def setup_basic_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_yaml_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML no está instalado. Ejecuta: pip install PyYAML")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró config.yaml en: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_allowed_extensions(data_cfg: Dict[str, Any]) -> set[str]:
    extensions = data_cfg.get("allowed_extensions", None)

    if not extensions:
        return DEFAULT_IMAGE_EXTENSIONS

    return {str(ext).lower() for ext in extensions}


def is_valid_image(path: Union[str, Path]) -> bool:
    path = Path(path)

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def extract_patient_id(image_path: Union[str, Path]) -> str:
    """
    Extrae un identificador de paciente/grupo a partir del nombre de archivo.

    Ejemplos:
    - 24892_left-600.jpg  -> 24892
    - 24892_right-600.jpg -> 24892
    - C0002422.jpg        -> C0002422

    Si no detecta lateralidad, usa el stem completo.
    """
    stem = Path(image_path).stem.lower()

    stem = re.sub(r"\s+", "_", stem)

    lateral_patterns = [
        r"(.+?)[_-](left|right)(?:[_-].*)?$",
        r"(.+?)[_-](od|os)(?:[_-].*)?$",
        r"(.+?)[_-](l|r)(?:[_-].*)?$",
    ]

    for pattern in lateral_patterns:
        match = re.match(pattern, stem)
        if match:
            candidate = match.group(1)
            if candidate:
                return candidate

    return stem


# ==============================================================================
# LECTURA DE DATASET
# ==============================================================================

def read_dataset_dataframe(
    data_cfg: Dict[str, Any],
) -> pd.DataFrame:
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    classes = data_cfg.get("classes", ["healthy", "exudates"])
    allowed_extensions = get_allowed_extensions(data_cfg)

    logger.info("Leyendo dataset desde: %s", raw_dir.resolve())

    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {raw_dir}")

    records: List[Dict[str, Any]] = []

    for label, class_name in enumerate(classes):
        class_dir = raw_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")

        image_paths = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed_extensions
        ]

        valid_count = 0

        for image_path in sorted(image_paths):
            if not is_valid_image(image_path):
                logger.warning("Imagen inválida omitida: %s", image_path)
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "filename": image_path.name,
                    "class_name": class_name,
                    "label": int(label),
                    "patient_id": extract_patient_id(image_path),
                }
            )
            valid_count += 1

        logger.info("[%s] → %s imágenes válidas encontradas.", class_name, valid_count)

    if not records:
        raise RuntimeError("No se encontraron imágenes válidas en data/raw.")

    df = pd.DataFrame(records)

    logger.info("Dataset total: %s imágenes", len(df))
    logger.info("Distribución por clase:\n%s", df["class_name"].value_counts())
    logger.info("Pacientes/grupos detectados: %s", df["patient_id"].nunique())

    return df


# ==============================================================================
# SPLIT TRAIN / VAL / TEST
# ==============================================================================

def _check_mixed_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta patient_id con más de una etiqueta.
    Si existen, se reportan. No se elimina nada, pero se evita fuga por grupo.
    """
    group_labels = df.groupby("patient_id")["label"].nunique()
    mixed_groups = group_labels[group_labels > 1]

    if len(mixed_groups) > 0:
        logger.warning(
            "Se detectaron %s grupos/pacientes con etiquetas mezcladas. "
            "Revisa duplicados o nombres de archivo.",
            len(mixed_groups),
        )

    return df


def split_dataframe(
    df: pd.DataFrame,
    data_cfg: Dict[str, Any],
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_split = float(data_cfg.get("val_split", 0.15))
    test_split = float(data_cfg.get("test_split", 0.15))
    group_by_patient = bool(data_cfg.get("group_by_patient", True))

    if val_split <= 0 or test_split <= 0:
        raise ValueError("val_split y test_split deben ser mayores que 0.")

    if val_split + test_split >= 0.8:
        raise ValueError("val_split + test_split es demasiado alto.")

    if group_by_patient:
        logger.info("Split configurado: POR PACIENTE/GRUPO.")
        df = _check_mixed_groups(df)

        group_df = (
            df.groupby("patient_id")
            .agg(label=("label", "first"), count=("label", "size"))
            .reset_index()
        )

        groups = group_df["patient_id"].values
        labels = group_df["label"].values

        train_val_groups, test_groups = train_test_split(
            groups,
            test_size=test_split,
            random_state=seed,
            stratify=labels,
        )

        train_val_group_df = group_df[group_df["patient_id"].isin(train_val_groups)]
        train_val_labels = train_val_group_df["label"].values

        val_relative = val_split / (1.0 - test_split)

        train_groups, val_groups = train_test_split(
            train_val_group_df["patient_id"].values,
            test_size=val_relative,
            random_state=seed,
            stratify=train_val_labels,
        )

        train_df = df[df["patient_id"].isin(train_groups)].copy()
        val_df = df[df["patient_id"].isin(val_groups)].copy()
        test_df = df[df["patient_id"].isin(test_groups)].copy()

        logger.info(
            "Split por paciente OK → train grupos=%s | val grupos=%s | test grupos=%s",
            train_df["patient_id"].nunique(),
            val_df["patient_id"].nunique(),
            test_df["patient_id"].nunique(),
        )

    else:
        logger.info("Split configurado: POR IMAGEN.")

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_split,
            random_state=seed,
            stratify=df["label"],
        )

        val_relative = val_split / (1.0 - test_split)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative,
            random_state=seed,
            stratify=train_val_df["label"],
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split → Train: %s | Val: %s | Test: %s",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    logger.info("Train distribución:\n%s", train_df["label"].value_counts().sort_index())
    logger.info("Val distribución:\n%s", val_df["label"].value_counts().sort_index())
    logger.info("Test distribución:\n%s", test_df["label"].value_counts().sort_index())

    return train_df, val_df, test_df


def verify_no_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    train_groups = set(train_df["patient_id"].astype(str))
    val_groups = set(val_df["patient_id"].astype(str))
    test_groups = set(test_df["patient_id"].astype(str))

    leakage_train_val = train_groups.intersection(val_groups)
    leakage_train_test = train_groups.intersection(test_groups)
    leakage_val_test = val_groups.intersection(test_groups)

    has_leakage = any(
        [
            leakage_train_val,
            leakage_train_test,
            leakage_val_test,
        ]
    )

    if has_leakage:
        logger.error("❌ Fuga train/val: %s", list(leakage_train_val)[:10])
        logger.error("❌ Fuga train/test: %s", list(leakage_train_test)[:10])
        logger.error("❌ Fuga val/test: %s", list(leakage_val_test)[:10])
        return False

    logger.info("✅ Sin fuga de patient_id entre train/val/test.")
    return True


# ==============================================================================
# AUGMENTATION
# ==============================================================================

def _safe_rotate_transform(rotation_limit: int):
    if A is None:
        return None

    try:
        return A.Rotate(
            limit=int(rotation_limit),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            p=0.5,
        )
    except TypeError:
        return A.Rotate(
            limit=int(rotation_limit),
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
            p=0.5,
        )


def build_augmentation_pipeline(
    aug_cfg: Dict[str, Any],
    train: bool = True,
):
    if A is None or not train:
        return None

    transforms = []

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.2))

    rotation_limit = int(aug_cfg.get("rotation_limit", 0) or 0)
    if rotation_limit > 0:
        rotate = _safe_rotate_transform(rotation_limit)
        if rotate is not None:
            transforms.append(rotate)

    if aug_cfg.get("brightness_contrast", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.35,
            )
        )

    if aug_cfg.get("gaussian_noise", True):
        transforms.append(A.GaussNoise(p=0.20))

    if aug_cfg.get("elastic_transform", False):
        try:
            transforms.append(
                A.ElasticTransform(
                    alpha=1,
                    sigma=30,
                    p=0.15,
                )
            )
        except Exception:
            pass

    if not transforms:
        return None

    return A.Compose(transforms)


# ==============================================================================
# DATASET
# ==============================================================================

class RetinAIDataset(Dataset):
    """
    Dataset para una rama específica del ensemble.

    Retorna:
    - image_tensor: Tensor [3, H, W]
    - label: Tensor long
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_mode: str = "rgb",
        image_size: int = 224,
        preprocessing_cfg: Optional[Dict[str, Any]] = None,
        augmentation_cfg: Optional[Dict[str, Any]] = None,
        train: bool = False,
        normalize: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.input_mode = input_mode
        self.image_size = int(image_size)
        self.preprocessing_cfg = preprocessing_cfg or {}
        self.train = bool(train)
        self.normalize = bool(normalize)

        self.augment = build_augmentation_pipeline(
            augmentation_cfg or {},
            train=self.train,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        image = ensure_rgb_array(image_path)

        image = preprocess_branch_numpy(
            image=image,
            input_mode=self.input_mode,
            image_size=self.image_size,
            preprocessing_cfg=self.preprocessing_cfg,
        )

        if self.augment is not None:
            augmented = self.augment(image=image)
            image = augmented["image"]

        tensor = to_tensor_chw(image, normalize=self.normalize)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return tensor, label_tensor


# ==============================================================================
# CLASS WEIGHTS
# ==============================================================================

def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()

    weights = np.zeros(num_classes, dtype=np.float32)

    for idx in range(num_classes):
        if counts[idx] > 0:
            weights[idx] = total / (num_classes * counts[idx])
        else:
            weights[idx] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# DATALOADER BUNDLE
# ==============================================================================

@dataclass
class DataLoadersBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: torch.Tensor
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_names: List[str]
    input_mode: str
    image_size: int

    def __iter__(self) -> Iterator[Any]:
        """
        Compatibilidad con:
        train_loader, val_loader, test_loader, class_weights = build_dataloaders(...)
        """
        yield self.train_loader
        yield self.val_loader
        yield self.test_loader
        yield self.class_weights

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "train": "train_loader",
            "val": "val_loader",
            "valid": "val_loader",
            "validation": "val_loader",
            "test": "test_loader",
            "weights": "class_weights",
            "class_weights": "class_weights",
            "train_df": "train_df",
            "val_df": "val_df",
            "test_df": "test_df",
            "class_names": "class_names",
            "input_mode": "input_mode",
            "image_size": "image_size",
        }

        attr = aliases.get(key, key)

        if not hasattr(self, attr):
            raise KeyError(key)

        return getattr(self, attr)

    def keys(self) -> List[str]:
        return [
            "train_loader",
            "val_loader",
            "test_loader",
            "class_weights",
            "train_df",
            "val_df",
            "test_df",
            "class_names",
            "input_mode",
            "image_size",
        ]


# ==============================================================================
# CONFIG DE MODELO/RAMA
# ==============================================================================

def _resolve_model_cfg(
    config: Dict[str, Any],
    model_cfg: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if isinstance(model_cfg, dict):
        return model_cfg

    if isinstance(model_cfg, str):
        for item in config.get("base_models", []):
            if item.get("name") == model_cfg:
                return item
        return {"name": model_cfg}

    enabled_models = [
        item for item in config.get("base_models", [])
        if item.get("enabled", True)
    ]

    if enabled_models:
        return enabled_models[0]

    return {
        "name": "default_rgb",
        "input_mode": "rgb",
        "image_size": config.get("data", {}).get("image_size", 224),
    }


# ==============================================================================
# CREACIÓN DE DATALOADERS
# ==============================================================================

def build_dataloaders(
    config: Union[Dict[str, Any], str, Path],
    model_cfg: Optional[Union[str, Dict[str, Any]]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle_train: bool = True,
) -> DataLoadersBundle:
    """
    Construye dataloaders para una rama específica.

    Compatible con:
    - build_dataloaders(config)
    - build_dataloaders(config, model_cfg)
    - train_loader, val_loader, test_loader, class_weights = build_dataloaders(...)
    """
    if not isinstance(config, dict):
        config = load_yaml_config(config)

    seed = int(config.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    preprocessing_cfg = config.get("branch_preprocessing", {})

    selected_model_cfg = _resolve_model_cfg(config, model_cfg)

    input_mode = selected_model_cfg.get("input_mode", "rgb")
    image_size = int(selected_model_cfg.get("image_size", data_cfg.get("image_size", 224)))

    if batch_size is None:
        batch_size = int(training_cfg.get("batch_size", 16))

    if num_workers is None:
        num_workers = int(training_cfg.get("num_workers", 0))

    class_names = list(data_cfg.get("classes", ["healthy", "exudates"]))
    num_classes = len(class_names)

    df = read_dataset_dataframe(data_cfg)
    train_df, val_df, test_df = split_dataframe(df, data_cfg, seed=seed)

    if data_cfg.get("group_by_patient", True):
        verify_no_patient_leakage(train_df, val_df, test_df)

    train_dataset = RetinAIDataset(
        dataframe=train_df,
        input_mode=input_mode,
        image_size=image_size,
        preprocessing_cfg=preprocessing_cfg,
        augmentation_cfg=aug_cfg,
        train=True,
        normalize=bool(aug_cfg.get("normalize_imagenet", True)),
    )

    val_dataset = RetinAIDataset(
        dataframe=val_df,
        input_mode=input_mode,
        image_size=image_size,
        preprocessing_cfg=preprocessing_cfg,
        augmentation_cfg=aug_cfg,
        train=False,
        normalize=bool(aug_cfg.get("normalize_imagenet", True)),
    )

    test_dataset = RetinAIDataset(
        dataframe=test_df,
        input_mode=input_mode,
        image_size=image_size,
        preprocessing_cfg=preprocessing_cfg,
        augmentation_cfg=aug_cfg,
        train=False,
        normalize=bool(aug_cfg.get("normalize_imagenet", True)),
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    class_weights = compute_class_weights(
        labels=train_df["label"].values,
        num_classes=num_classes,
    )

    logger.info("DataLoaders construidos correctamente.")
    logger.info("Rama input_mode: %s | image_size: %s", input_mode, image_size)
    logger.info("Train batches: %s", len(train_loader))
    logger.info("Val batches:   %s", len(val_loader))
    logger.info("Test batches:  %s", len(test_loader))
    logger.info("Class weights: %s", class_weights)

    return DataLoadersBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_names=class_names,
        input_mode=input_mode,
        image_size=image_size,
    )


def create_dataloaders(
    config: Union[Dict[str, Any], str, Path],
    model_cfg: Optional[Union[str, Dict[str, Any]]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoadersBundle:
    return build_dataloaders(
        config=config,
        model_cfg=model_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_dataloaders(
    config: Union[Dict[str, Any], str, Path],
    model_cfg: Optional[Union[str, Dict[str, Any]]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoadersBundle:
    return build_dataloaders(
        config=config,
        model_cfg=model_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
    )


# ==============================================================================
# INFERENCIA
# ==============================================================================

def preprocess_single_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    image_size: int = 224,
    config: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    input_mode: Optional[str] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocesa una imagen individual para inferencia.

    Mantiene compatibilidad con predictor.py antiguo:
    - preprocess_single_image(image, image_size=224)
    """
    preprocessing_cfg = {}

    if config is not None:
        preprocessing_cfg = config.get("branch_preprocessing", {})

    if model_cfg is not None:
        input_mode = model_cfg.get("input_mode", input_mode or "rgb")
        image_size = int(model_cfg.get("image_size", image_size))

    if input_mode is None:
        input_mode = "rgb"

    image_np = preprocess_branch_numpy(
        image=image,
        input_mode=input_mode,
        image_size=image_size,
        preprocessing_cfg=preprocessing_cfg,
    )

    tensor = to_tensor_chw(image_np, normalize=normalize)
    return tensor


# ==============================================================================
# MAIN DE VERIFICACIÓN
# ==============================================================================

def _print_header(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def main() -> None:
    setup_basic_logging()

    _print_header("Verificando pipeline de datos multimodal...")

    config = load_yaml_config("config.yaml")
    model_cfgs = [
        m for m in config.get("base_models", [])
        if m.get("enabled", True)
    ]

    if not model_cfgs:
        raise RuntimeError("No hay modelos habilitados en config.yaml")

    # Verificar primera rama habilitada para no cargar todo innecesariamente.
    first_model_cfg = model_cfgs[0]

    logger.info(
        "Verificando con rama: %s | input_mode=%s | image_size=%s",
        first_model_cfg.get("name"),
        first_model_cfg.get("input_mode", "rgb"),
        first_model_cfg.get("image_size", config.get("data", {}).get("image_size", 224)),
    )

    bundle = build_dataloaders(
        config=config,
        model_cfg=first_model_cfg,
    )

    logger.info("Batches en train: %s", len(bundle.train_loader))
    logger.info("Batches en val:   %s", len(bundle.val_loader))
    logger.info("Batches en test:  %s", len(bundle.test_loader))
    logger.info("Pesos de clase:   %s", bundle.class_weights)

    images, labels = next(iter(bundle.train_loader))

    logger.info("Shape de batch: %s", tuple(images.shape))
    logger.info("Etiquetas batch: %s", labels.tolist())
    logger.info("Train muestras: %s", len(bundle.train_df))
    logger.info("Val muestras:   %s", len(bundle.val_df))
    logger.info("Test muestras:  %s", len(bundle.test_df))

    assert images.ndim == 4, "El batch debe tener forma [B, C, H, W]"
    assert images.shape[1] == 3, "Los modelos esperan 3 canales"
    assert labels.ndim == 1, "Las etiquetas deben ser vector [B]"

    _print_header("Verificando modos de preprocesamiento habilitados...")

    sample_path = bundle.train_df.iloc[0]["image_path"]

    for model_cfg in model_cfgs:
        if model_cfg.get("name") in {"retfound", "retclip"}:
            continue

        name = model_cfg.get("name")
        input_mode = model_cfg.get("input_mode", "rgb")
        image_size = int(model_cfg.get("image_size", config.get("data", {}).get("image_size", 224)))

        tensor = preprocess_single_image(
            image=sample_path,
            image_size=image_size,
            config=config,
            model_cfg=model_cfg,
        )

        logger.info(
            "Modelo=%s | mode=%s | tensor=%s",
            name,
            input_mode,
            tuple(tensor.shape),
        )

    logger.info("✅ Pipeline de datos multimodal verificado correctamente.")


if __name__ == "__main__":
    main()