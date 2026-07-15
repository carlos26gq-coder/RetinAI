# ==============================================================================
# generate_gradcam_examples.py
# RetinAI_MVP
# Grad-CAM para evidencia visual de regiones relevantes en fondo de ojo
# ==============================================================================

from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.base_models import build_model_from_config
from src.data_loader import build_dataloaders, load_yaml_config, set_global_seed
from src.preprocessing_branches import preprocess_branch_tensor


# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("reports/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["healthy", "exudates"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        return output[0]

    if hasattr(output, "logits"):
        return output.logits

    raise TypeError(f"Salida no soportada: {type(output)}")


def find_model_cfg(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    target = clean_name(model_name)

    for model_cfg in config.get("base_models", []):
        if clean_name(model_cfg.get("name", "")) == target:
            return model_cfg

    raise ValueError(f"No se encontró modelo en config.yaml: {model_name}")


def find_last_conv_layer(model: nn.Module, model_name: str = "") -> nn.Module:
    """
    Busca una capa convolucional útil para Grad-CAM.
    Evita capas de clasificación, atención final o CBAM spatial,
    porque suelen generar mapas demasiado gruesos o poco interpretables.
    """
    excluded_keywords = [
        "classifier",
        "fc",
        "head",
        "cbam",
        "attention",
        "spatial",
        "channel",
    ]

    candidates = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            lname = name.lower()

            if any(k in lname for k in excluded_keywords):
                continue

            candidates.append((name, module))

    if candidates:
        selected_name, selected_module = candidates[-1]
        print("Target layer seleccionado:", selected_name, "->", selected_module)
        return selected_module

    # Fallback: última Conv2d aunque sea de atención.
    last_conv = None
    last_name = ""

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_conv = module

    if last_conv is None:
        raise RuntimeError("No se encontró ninguna capa Conv2d para Grad-CAM.")

    print("Target layer fallback:", last_name, "->", last_conv)
    return last_conv


def pil_to_rgb_array(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((size, size))
    return np.asarray(image).astype(np.uint8)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)

    cam_min = float(cam.min())
    cam_max = float(cam.max())

    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)

    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.astype(np.float32)


def overlay_cam_on_image(
    rgb_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.42,
) -> np.ndarray:
    """
    Crea overlay Grad-CAM sobre imagen RGB.
    """
    h, w = rgb_image.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8((1 - alpha) * rgb_image + alpha * heatmap)

    return overlay


def save_side_by_side(
    original: np.ndarray,
    cam_only: np.ndarray,
    overlay: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Guarda una imagen compuesta: original | heatmap | overlay.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(cam_only, cmap="jet")
    axes[1].set_title("Mapa Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Superposición")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def build_model_for_gradcam(
    config: Dict[str, Any],
    model_name: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model_name = clean_name(model_name)
    model_cfg = find_model_cfg(config, model_name)

    weights_path = Path(config.get("paths", {}).get("weights_dir", "models/weights")) / f"{model_name}_best.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"No existe peso entrenado: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    checkpoint_cfg = checkpoint.get("model_cfg", model_cfg)
    state_dict = checkpoint.get("state_dict", checkpoint)

    final_cfg = copy.deepcopy(checkpoint_cfg)
    final_cfg["name"] = model_name

    # Importante: no descargar pesos externos para Grad-CAM.
    final_cfg["pretrained"] = False
    if "encoder_weights" in final_cfg:
        final_cfg["encoder_weights"] = None

    num_classes = len(config.get("data", {}).get("classes", CLASS_NAMES))

    model = build_model_from_config(
        model_cfg=final_cfg,
        num_classes=num_classes,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Para Grad-CAM se necesitan gradientes dentro del backbone.
    # Esto NO entrena ni modifica pesos; solo permite calcular activaciones/gradientes.
    for param in model.parameters():
        param.requires_grad_(True)

    return model, final_cfg


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Asegura que se construya el grafo de gradientes para Grad-CAM,
        # incluso si el modelo fue entrenado con backbone congelado.
        input_tensor = input_tensor.detach().clone().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        logits = get_logits(output)

        probs = torch.softmax(logits, dim=1)
        predicted_class = int(torch.argmax(probs, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No se capturaron activaciones o gradientes.")

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze(0)

        cam_np = cam.detach().cpu().numpy()
        cam_np = normalize_cam(cam_np)

        probs_np = probs.detach().cpu().numpy()[0]

        return cam_np, probs_np


# ==============================================================================
# SELECCIÓN DE IMÁGENES
# ==============================================================================

def get_test_images_from_split(
    config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    n_per_class: int = 3,
) -> List[Dict[str, Any]]:
    """
    Reconstruye el split del data_loader y toma ejemplos del test set.
    """
    bundle = build_dataloaders(
        config=config,
        model_cfg=model_cfg,
        batch_size=8,
        num_workers=0,
        shuffle_train=False,
    )

    test_df = bundle.test_df.copy()

    selected_rows = []

    for label in sorted(test_df["label"].unique()):
        class_df = test_df[test_df["label"] == label].copy()

        if len(class_df) == 0:
            continue

        sample_n = min(n_per_class, len(class_df))
        class_df = class_df.sample(n=sample_n, random_state=42)

        selected_rows.append(class_df)

    if not selected_rows:
        raise RuntimeError("No se encontraron imágenes de test.")

    selected = []
    final_df = __import__("pandas").concat(selected_rows, ignore_index=True)

    for _, row in final_df.iterrows():
        selected.append(
            {
                "image_path": Path(row["image_path"]),
                "label": int(row["label"]),
                "class_name": str(row.get("class_name", CLASS_NAMES[int(row["label"])])),
            }
        )

    return selected


# ==============================================================================
# MAIN GRADCAM
# ==============================================================================

def run_gradcam_for_model(
    model_name: str = "resnet50_cbam",
    n_per_class: int = 3,
    target_class: int = 1,
) -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("No se encontró config.yaml")

    config = load_yaml_config(CONFIG_PATH)
    set_global_seed(int(config.get("project", {}).get("seed", 42)))

    device = get_device()

    print("=" * 80)
    print("GENERANDO GRAD-CAM")
    print("=" * 80)
    print("Modelo:", model_name)
    print("Device:", device)

    model, model_cfg = build_model_for_gradcam(config, model_name, device)
    target_layer = find_last_conv_layer(model, model_name)

    print("Input mode:", model_cfg.get("input_mode", "rgb"))
    print("Image size:", model_cfg.get("image_size", 224))
    print("Target layer:", target_layer)

    gradcam = GradCAM(model, target_layer)

    selected_images = get_test_images_from_split(
        config=config,
        model_cfg=model_cfg,
        n_per_class=n_per_class,
    )

    preprocessing_cfg = config.get("branch_preprocessing", {})
    input_mode = model_cfg.get("input_mode", "rgb")
    image_size = int(model_cfg.get("image_size", 224))

    rows = []

    for idx, item in enumerate(selected_images, start=1):
        image_path = Path(item["image_path"])
        label = int(item["label"])
        class_name = item["class_name"]

        if not image_path.exists():
            print("No existe imagen:", image_path)
            continue

        image = Image.open(image_path).convert("RGB")

        tensor = preprocess_branch_tensor(
            image=image,
            input_mode=input_mode,
            image_size=image_size,
            preprocessing_cfg=preprocessing_cfg,
            normalize=True,
        )

        input_tensor = tensor.unsqueeze(0).to(device)

        cam, probs = gradcam.generate(
            input_tensor=input_tensor,
            target_class=target_class,
        )

        prob_healthy = float(probs[0])
        prob_exudates = float(probs[1])
        predicted_label = int(np.argmax(probs))
        predicted_name = CLASS_NAMES[predicted_label]

        original = pil_to_rgb_array(image, image_size)
        overlay = overlay_cam_on_image(original, cam)

        safe_stem = image_path.stem.replace(" ", "_").replace("/", "_")
        out_name = f"{model_name}_{class_name}_{safe_stem}_gradcam.png"
        out_path = OUTPUT_DIR / out_name

        title = (
            f"{pretty_model_title(model_name)} | Real: {class_name} | "
            f"Pred: {predicted_name} | P(exudados): {prob_exudates*100:.1f}%"
        )

        save_side_by_side(
            original=original,
            cam_only=cam,
            overlay=overlay,
            out_path=out_path,
            title=title,
        )

        rows.append(
            {
                "model": model_name,
                "image_path": str(image_path),
                "output_gradcam": str(out_path),
                "true_label": label,
                "true_class": class_name,
                "predicted_label": predicted_label,
                "predicted_class": predicted_name,
                "prob_healthy": prob_healthy,
                "prob_exudates": prob_exudates,
                "correct": int(predicted_label == label),
            }
        )

        print(
            f"[{idx}/{len(selected_images)}] {image_path.name} | "
            f"real={class_name} | pred={predicted_name} | "
            f"P(exud)={prob_exudates*100:.1f}% | guardado={out_path}"
        )

    gradcam.remove_hooks()

    csv_path = OUTPUT_DIR / f"{model_name}_gradcam_summary.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "image_path",
                "output_gradcam",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "prob_healthy",
                "prob_exudates",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nResumen guardado en:", csv_path)
    print("Grad-CAMs guardados en:", OUTPUT_DIR)
    print("✅ Grad-CAM completado.")


def pretty_model_title(name: str) -> str:
    mapping = {
        "resnet50_cbam": "ResNet50-CBAM",
        "densenet121": "DenseNet121",
        "efficientnet_b0": "EfficientNetB0",
        "efficientnet_b3": "EfficientNetB3",
        "mobilenetv3_exudate_map": "MobileNetV3 exudate map",
    }

    return mapping.get(clean_name(name), name)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generar Grad-CAM para RetinAI.")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50_cbam",
        help="Modelo para Grad-CAM. Recomendado: resnet50_cbam, densenet121, efficientnet_b0.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=3,
        help="Cantidad de imágenes por clase del test set.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="Clase objetivo para Grad-CAM. 1 = exudates.",
    )

    args = parser.parse_args()

    run_gradcam_for_model(
        model_name=args.model,
        n_per_class=args.n_per_class,
        target_class=args.target_class,
    )