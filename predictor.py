# ==============================================================================
# predictor.py
# RetinAI_MVP
# Motor de inferencia para Ensemble Stacking Multimodal
# ==============================================================================

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from PIL import Image

from src.base_models import build_model_from_config
from src.data_loader import load_yaml_config
from src.preprocessing_branches import (
    extract_radiomics_features,
    extract_radiomics_vector,
    preprocess_branch_tensor,
)


logger = logging.getLogger(__name__)


ImageInput = Union[str, Path, Image.Image, np.ndarray]


# ==============================================================================
# UTILIDADES
# ==============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _clean_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def _torch_load(path: Union[str, Path], device: torch.device) -> Any:
    """
    Carga checkpoints de forma compatible con distintas versiones de PyTorch.
    """
    path = Path(path)

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_json(path: Union[str, Path], default: Optional[Any] = None) -> Any:
    path = Path(path)

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_device(config: Dict[str, Any]) -> torch.device:
    requested = str(config.get("hardware", {}).get("device", "auto")).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(requested)


def _get_model_group(model_name: str) -> str:
    """
    Debe coincidir con la lógica usada en train_ensemble.py.
    """
    name = _clean_name(model_name)

    if name in {"retfound", "retclip", "efficientnet_b0", "efficientnet_b3", "inception_v3"}:
        return "global_multiscale"

    if name in {"densenet121", "unet_encoder", "mobilenetv3_exudate_map"}:
        return "lesion_focused"

    if name in {"resnet50_cbam", "resnet50"}:
        return "attention_structure"

    return "other"


def _get_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        return output[0]

    if hasattr(output, "logits"):
        return output.logits

    raise TypeError(f"Salida de modelo no soportada: {type(output)}")


def _enabled_model_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    models_cfg = config.get("base_models", [])

    return [
        cfg for cfg in models_cfg
        if cfg.get("enabled", True)
    ]


def _find_model_cfg(
    config: Dict[str, Any],
    model_name: str,
) -> Optional[Dict[str, Any]]:
    target = _clean_name(model_name)

    for model_cfg in config.get("base_models", []):
        if _clean_name(model_cfg.get("name", "")) == target:
            return model_cfg

    return None


def _risk_level_from_probability(
    probability: float,
    config: Dict[str, Any],
) -> str:
    levels = config.get("inference", {}).get("risk_levels", None)

    if not levels:
        if probability < 0.40:
            return "Bajo"
        if probability < 0.70:
            return "Moderado"
        return "Alto"

    for _, item in levels.items():
        min_p = float(item.get("min_probability", 0.0))
        max_p = float(item.get("max_probability", 1.0))

        if min_p <= probability <= max_p:
            return str(item.get("label", "No definido"))

    return "No definido"


# ==============================================================================
# PREDICTOR PRINCIPAL
# ==============================================================================

class RetinAIPredictor:
    """
    Predictor multimodal:
    - carga modelos base entrenados
    - aplica preprocesamiento por rama
    - extrae radiomics
    - arma vector meta
    - aplica meta-modelo stacking
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "config.yaml",
        auto_load: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = load_yaml_config(self.config_path)

        self.device = _get_device(self.config)

        paths_cfg = self.config.get("paths", {})
        self.weights_dir = Path(paths_cfg.get("weights_dir", "models/weights"))
        self.meta_dir = Path(paths_cfg.get("meta_dir", "models/meta"))

        self.models: Dict[str, torch.nn.Module] = {}
        self.model_cfgs: Dict[str, Dict[str, Any]] = {}

        self.meta_model = None
        self.radiomics_scaler = None

        self.base_model_names: List[str] = []
        self.meta_feature_names: List[str] = []

        self.threshold: float = float(
            self.config.get("inference", {}).get("default_threshold", 0.5)
        )

        self.class_names: List[str] = list(
            self.config.get("data", {}).get("classes", ["healthy", "exudates"])
        )

        self.is_loaded: bool = False
        self.load_errors: List[str] = []

        if auto_load:
            self.load()

    # --------------------------------------------------------------------------
    # CARGA DE ARTEFACTOS
    # --------------------------------------------------------------------------

    def load(self) -> None:
        logger.info("Cargando RetinAI Predictor multimodal...")

        self.load_errors.clear()

        self._load_meta_artifacts()
        self._load_base_models()

        self.is_loaded = (
            len(self.models) > 0
            and self.meta_model is not None
            and len(self.meta_feature_names) > 0
        )

        if self.is_loaded:
            logger.info("✅ Predictor cargado correctamente.")
            logger.info("Modelos cargados: %s", list(self.models.keys()))
            logger.info("Umbral: %.4f", self.threshold)
            logger.info("Meta-features: %s", len(self.meta_feature_names))
        else:
            logger.warning("⚠️ Predictor no quedó completamente cargado.")
            logger.warning("Errores: %s", self.load_errors)

    def _load_meta_artifacts(self) -> None:
        meta_model_path = self.meta_dir / "meta_model.pkl"
        threshold_path = self.meta_dir / "optimal_threshold.json"
        base_names_path = self.meta_dir / "base_model_names.json"
        feature_names_path = self.meta_dir / "meta_feature_names.json"
        scaler_path = self.meta_dir / "radiomics_scaler.pkl"

        if not meta_model_path.exists():
            self.load_errors.append(f"No existe meta_model.pkl: {meta_model_path}")
        else:
            self.meta_model = joblib.load(meta_model_path)

        threshold_data = _load_json(threshold_path, default=None)

        if isinstance(threshold_data, dict):
            self.threshold = float(
                threshold_data.get(
                    "threshold",
                    self.config.get("inference", {}).get("default_threshold", 0.5),
                )
            )

        base_names_data = _load_json(base_names_path, default=None)

        if isinstance(base_names_data, dict):
            self.base_model_names = [
                _clean_name(x)
                for x in base_names_data.get("base_model_names", [])
            ]
        elif isinstance(base_names_data, list):
            self.base_model_names = [_clean_name(x) for x in base_names_data]

        feature_names_data = _load_json(feature_names_path, default=None)

        if isinstance(feature_names_data, dict):
            self.meta_feature_names = list(feature_names_data.get("feature_names", []))
        elif isinstance(feature_names_data, list):
            self.meta_feature_names = list(feature_names_data)

        if scaler_path.exists():
            self.radiomics_scaler = joblib.load(scaler_path)

    def _load_base_models(self) -> None:
        if not self.base_model_names:
            enabled = _enabled_model_configs(self.config)
            self.base_model_names = [
                _clean_name(cfg.get("name", ""))
                for cfg in enabled
                if _clean_name(cfg.get("name", "")) not in {"retfound", "retclip"}
            ]

        for model_name in self.base_model_names:
            if model_name in {"retfound", "retclip"}:
                continue

            weights_path = self.weights_dir / f"{model_name}_best.pth"

            if not weights_path.exists():
                self.load_errors.append(f"No existe peso para {model_name}: {weights_path}")
                continue

            try:
                checkpoint = _torch_load(weights_path, self.device)

                checkpoint_cfg = None
                state_dict = None

                if isinstance(checkpoint, dict):
                    checkpoint_cfg = checkpoint.get("model_cfg", None)
                    state_dict = checkpoint.get("state_dict", checkpoint)
                else:
                    state_dict = checkpoint

                config_model_cfg = _find_model_cfg(self.config, model_name)

                model_cfg = checkpoint_cfg or config_model_cfg

                if model_cfg is None:
                    self.load_errors.append(f"No se encontró configuración para {model_name}")
                    continue

                model_cfg = dict(model_cfg)
                model_cfg["name"] = model_name

                # En inferencia NO necesitamos descargar pesos preentrenados.
                # Primero se crea la arquitectura sin pesos externos y luego se cargan
                # los pesos entrenados guardados en models/weights/*.pth.
                model_cfg["pretrained"] = False
                if "encoder_weights" in model_cfg:
                    model_cfg["encoder_weights"] = None

                model = build_model_from_config(
                    model_cfg=model_cfg,
                    num_classes=len(self.class_names),
                )

                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                model.eval()

                self.models[model_name] = model
                self.model_cfgs[model_name] = model_cfg

            except Exception as exc:
                self.load_errors.append(f"Error cargando {model_name}: {exc}")

    # --------------------------------------------------------------------------
    # PREDICCIÓN BASE
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def _predict_base_models(
        self,
        image: ImageInput,
    ) -> Dict[str, Dict[str, float]]:
        if not self.models:
            raise RuntimeError("No hay modelos base cargados.")

        preprocessing_cfg = self.config.get("branch_preprocessing", {})

        outputs: Dict[str, Dict[str, float]] = {}

        for model_name, model in self.models.items():
            model_cfg = self.model_cfgs[model_name]

            input_mode = model_cfg.get("input_mode", "rgb")
            image_size = int(model_cfg.get("image_size", 224))

            tensor = preprocess_branch_tensor(
                image=image,
                input_mode=input_mode,
                image_size=image_size,
                preprocessing_cfg=preprocessing_cfg,
                normalize=True,
            )

            tensor = tensor.unsqueeze(0).to(self.device)

            logits = _get_logits(model(tensor))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

            prob_healthy = _safe_float(probs[0])
            prob_exudates = _safe_float(probs[1])

            confidence = max(prob_healthy, prob_exudates)
            entropy = -float(
                prob_healthy * np.log(prob_healthy + 1e-8)
                + prob_exudates * np.log(prob_exudates + 1e-8)
            )

            outputs[model_name] = {
                "prob_healthy": prob_healthy,
                "prob_exudates": prob_exudates,
                "confidence": _safe_float(confidence),
                "entropy": _safe_float(entropy),
                "input_mode": str(input_mode),
                "image_size": int(image_size),
                "group": _get_model_group(model_name),
            }

        return outputs

    # --------------------------------------------------------------------------
    # RADIOMICS
    # --------------------------------------------------------------------------

    def _extract_radiomics(
        self,
        image: ImageInput,
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        radiomics_cfg = self.config.get("radiomics", {})
        image_size = int(self.config.get("data", {}).get("image_size", 224))

        vector, names = extract_radiomics_vector(
            image=image,
            image_size=image_size,
            radiomics_cfg=radiomics_cfg,
        )

        feature_dict = extract_radiomics_features(
            image=image,
            image_size=image_size,
            radiomics_cfg=radiomics_cfg,
        )

        vector = vector.reshape(1, -1)

        if self.radiomics_scaler is not None:
            vector_scaled = self.radiomics_scaler.transform(vector)
        else:
            vector_scaled = vector

        radiomics_summary = {
            "yellow_pixel_ratio": _safe_float(feature_dict.get("color_yellow_pixel_ratio", 0.0)),
            "bright_pixel_ratio": _safe_float(feature_dict.get("color_bright_pixel_ratio", 0.0)),
            "candidate_regions": _safe_float(feature_dict.get("morph_num_candidate_regions", 0.0)),
            "candidate_area_ratio": _safe_float(feature_dict.get("morph_candidate_area_ratio", 0.0)),
            "central_region_ratio": _safe_float(feature_dict.get("spatial_central_region_ratio", 0.0)),
            "peripheral_region_ratio": _safe_float(feature_dict.get("spatial_peripheral_region_ratio", 0.0)),
        }

        return vector_scaled.astype(np.float32), names, radiomics_summary

    # --------------------------------------------------------------------------
    # META FEATURES
    # --------------------------------------------------------------------------

    def _build_meta_feature_dict(
        self,
        base_outputs: Dict[str, Dict[str, float]],
        radiomics_vector: Optional[np.ndarray],
        radiomics_names: Optional[List[str]],
    ) -> Dict[str, float]:
        meta_cfg = self.config.get("meta_model", {})

        feature_dict: Dict[str, float] = {}

        use_probs = bool(meta_cfg.get("use_base_probabilities", True))
        use_conf = bool(meta_cfg.get("use_model_confidence", True))
        use_radiomics = bool(meta_cfg.get("use_radiomics_features", True))

        model_probs_by_group: Dict[str, List[float]] = {}

        for model_name in self.base_model_names:
            if model_name not in base_outputs:
                continue

            item = base_outputs[model_name]

            prob_healthy = _safe_float(item["prob_healthy"])
            prob_exudates = _safe_float(item["prob_exudates"])
            confidence = _safe_float(item["confidence"])
            entropy = _safe_float(item["entropy"])

            if use_probs:
                feature_dict[f"{model_name}_prob_healthy"] = prob_healthy
                feature_dict[f"{model_name}_prob_exudates"] = prob_exudates

            if use_conf:
                feature_dict[f"{model_name}_confidence"] = confidence
                feature_dict[f"{model_name}_entropy"] = entropy

            group = _get_model_group(model_name)
            model_probs_by_group.setdefault(group, []).append(prob_exudates)

        for group_name, values in sorted(model_probs_by_group.items()):
            arr = np.asarray(values, dtype=np.float32)

            feature_dict[f"group_{group_name}_mean"] = _safe_float(arr.mean())
            feature_dict[f"group_{group_name}_std"] = _safe_float(arr.std())
            feature_dict[f"group_{group_name}_max"] = _safe_float(arr.max())
            feature_dict[f"group_{group_name}_min"] = _safe_float(arr.min())

        if (
            use_radiomics
            and radiomics_vector is not None
            and radiomics_names is not None
            and len(radiomics_names) == radiomics_vector.shape[1]
        ):
            flat = radiomics_vector.reshape(-1)

            for idx, name in enumerate(radiomics_names):
                feature_dict[name] = _safe_float(flat[idx])

        return feature_dict

    def _build_meta_vector(
        self,
        feature_dict: Dict[str, float],
    ) -> np.ndarray:
        if not self.meta_feature_names:
            raise RuntimeError("No se cargaron meta_feature_names.json.")

        values = [
            _safe_float(feature_dict.get(name, 0.0))
            for name in self.meta_feature_names
        ]

        x = np.asarray(values, dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x

    # --------------------------------------------------------------------------
    # PREDICCIÓN FINAL
    # --------------------------------------------------------------------------

    def predict(
        self,
        image: ImageInput,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        if not self.is_loaded:
            raise RuntimeError(
                "El predictor no está cargado correctamente. "
                f"Errores: {self.load_errors}"
            )

        base_outputs = self._predict_base_models(image)

        radiomics_vector, radiomics_names, radiomics_summary = self._extract_radiomics(image)

        feature_dict = self._build_meta_feature_dict(
            base_outputs=base_outputs,
            radiomics_vector=radiomics_vector,
            radiomics_names=radiomics_names,
        )

        x_meta = self._build_meta_vector(feature_dict)

        probability = float(self.meta_model.predict_proba(x_meta)[0, 1])
        predicted_index = int(probability >= self.threshold)

        predicted_label = self.class_names[predicted_index]
        risk_level = _risk_level_from_probability(probability, self.config)

        result: Dict[str, Any] = {
            "probability": probability,
            "probability_percent": round(probability * 100.0, 2),
            "predicted_label": predicted_label,
            "predicted_class": predicted_label,
            "predicted_index": predicted_index,
            "threshold": float(self.threshold),
            "risk_level": risk_level,
            "models_loaded": True,
            "base_model_count": len(self.models),
            "meta_feature_count": len(self.meta_feature_names),
        }

        # Compatibilidad con app/api anteriores
        result["base_model_probs"] = {
            name: round(item["prob_exudates"], 6)
            for name, item in base_outputs.items()
        }

        if include_details:
            result["model_probabilities"] = {
                name: {
                    "prob_healthy": round(item["prob_healthy"], 6),
                    "prob_exudates": round(item["prob_exudates"], 6),
                    "confidence": round(item["confidence"], 6),
                    "entropy": round(item["entropy"], 6),
                    "input_mode": item["input_mode"],
                    "group": item["group"],
                }
                for name, item in base_outputs.items()
            }

            result["group_probabilities"] = self._summarize_groups(base_outputs)
            result["radiomics_summary"] = radiomics_summary

        return result

    def _summarize_groups(
        self,
        base_outputs: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[float]] = {}

        for model_name, item in base_outputs.items():
            group = _get_model_group(model_name)
            groups.setdefault(group, []).append(float(item["prob_exudates"]))

        summary: Dict[str, Dict[str, float]] = {}

        for group_name, values in sorted(groups.items()):
            arr = np.asarray(values, dtype=np.float32)

            summary[group_name] = {
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std()), 6),
                "min": round(float(arr.min()), 6),
                "max": round(float(arr.max()), 6),
                "models": int(len(values)),
            }

        return summary

    # --------------------------------------------------------------------------
    # INFO / HEALTH
    # --------------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return {
            "models_loaded": bool(self.is_loaded),
            "device": str(self.device),
            "base_models": list(self.models.keys()),
            "base_model_count": len(self.models),
            "threshold": float(self.threshold),
            "meta_model_loaded": self.meta_model is not None,
            "radiomics_scaler_loaded": self.radiomics_scaler is not None,
            "meta_feature_count": len(self.meta_feature_names),
            "errors": self.load_errors,
        }

    def info(self) -> Dict[str, Any]:
        return {
            "project": self.config.get("project", {}),
            "classes": self.class_names,
            "device": str(self.device),
            "base_models": {
                name: {
                    "input_mode": cfg.get("input_mode", "rgb"),
                    "image_size": cfg.get("image_size", 224),
                    "type": cfg.get("type", "cnn"),
                    "group": _get_model_group(name),
                }
                for name, cfg in self.model_cfgs.items()
            },
            "threshold": float(self.threshold),
            "risk_levels": self.config.get("inference", {}).get("risk_levels", {}),
        }


# ==============================================================================
# SINGLETON
# ==============================================================================

_GLOBAL_PREDICTOR: Optional[RetinAIPredictor] = None
_GLOBAL_LOCK = threading.Lock()


def get_predictor(
    config_path: Union[str, Path] = "config.yaml",
) -> RetinAIPredictor:
    global _GLOBAL_PREDICTOR

    with _GLOBAL_LOCK:
        if _GLOBAL_PREDICTOR is None:
            _GLOBAL_PREDICTOR = RetinAIPredictor(
                config_path=config_path,
                auto_load=True,
            )

    return _GLOBAL_PREDICTOR


def predict(
    image: ImageInput,
    config_path: Union[str, Path] = "config.yaml",
    include_details: bool = True,
) -> Dict[str, Any]:
    predictor = get_predictor(config_path=config_path)
    return predictor.predict(image=image, include_details=include_details)


# Alias de compatibilidad
Predictor = RetinAIPredictor


# ==============================================================================
# PRUEBA RÁPIDA
# ==============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Prueba rápida de predictor.py")
    parser.add_argument("--image", type=str, required=False, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()

    predictor = RetinAIPredictor(
        config_path=args.config,
        auto_load=True,
    )

    print("HEALTH:")
    print(json.dumps(predictor.health(), indent=2, ensure_ascii=False))

    if args.image:
        output = predictor.predict(args.image)
        print("PREDICCIÓN:")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("No se indicó imagen. Solo se verificó la carga del predictor.")