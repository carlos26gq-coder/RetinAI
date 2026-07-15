# ==============================================================================
# verify_training_connection.py
# RetinAI_MVP
# Verificación previa al entrenamiento en Kaggle / Laptop
# ==============================================================================

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from src.base_models import (
    build_model_from_config,
    count_total_parameters,
    count_trainable_parameters,
    get_enabled_model_configs,
)
from src.data_loader import build_dataloaders, load_yaml_config, set_global_seed
from src.preprocessing_branches import (
    extract_radiomics_vector,
    preprocess_branch_tensor,
)


# ==============================================================================
# UTILIDADES
# ==============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def clean_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        return output[0]

    if hasattr(output, "logits"):
        return output.logits

    raise TypeError(f"Salida de modelo no soportada: {type(output)}")


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_dataset_images(config: Dict[str, Any]) -> Dict[str, int]:
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    classes = data_cfg.get("classes", ["healthy", "exudates"])

    allowed_extensions = set(
        ext.lower()
        for ext in data_cfg.get(
            "allowed_extensions",
            list(IMAGE_EXTENSIONS),
        )
    )

    counts = {}

    for class_name in classes:
        class_dir = raw_dir / class_name

        if not class_dir.exists():
            counts[class_name] = 0
            continue

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed_extensions
        ]

        counts[class_name] = len(files)

    return counts


def get_first_available_image(config: Dict[str, Any]) -> Optional[Path]:
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    classes = data_cfg.get("classes", ["healthy", "exudates"])

    allowed_extensions = set(
        ext.lower()
        for ext in data_cfg.get(
            "allowed_extensions",
            list(IMAGE_EXTENSIONS),
        )
    )

    for class_name in classes:
        class_dir = raw_dir / class_name

        if not class_dir.exists():
            continue

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                return path

    return None


def make_dummy_image() -> Image.Image:
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[:, :] = (55, 38, 30)

    yy, xx = np.indices((512, 512))
    center = (256, 256)
    radius = 210
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2

    image[mask] = (95, 60, 45)

    # Simular zonas brillantes compatibles con posibles exudados
    image[230:250, 285:305] = (235, 220, 120)
    image[300:315, 250:265] = (230, 210, 110)

    return Image.fromarray(image)


def make_verify_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configuración ligera para verificar sin cargar demasiado la RAM.
    No modifica tu config.yaml real.
    """
    cfg = copy.deepcopy(config)

    cfg.setdefault("training", {})
    cfg["training"]["batch_size"] = 2
    cfg["training"]["num_workers"] = 0

    return cfg


def can_use_real_dataset(counts: Dict[str, int]) -> bool:
    """
    Para split estratificado mínimo necesitamos más de unas pocas imágenes por clase.
    En Kaggle con 4000+ imágenes será true.
    En laptop, si aún no cargaste imágenes, será false.
    """
    if not counts:
        return False

    return all(v >= 10 for v in counts.values())


def get_batch_for_model(
    config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: torch.device,
    use_real_data: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    """
    Devuelve un batch real si hay dataset suficiente.
    Si no, devuelve batch dummy para verificar forward/backward.
    """
    num_classes = len(config.get("data", {}).get("classes", ["healthy", "exudates"]))

    if use_real_data:
        try:
            bundle = build_dataloaders(
                config=config,
                model_cfg=model_cfg,
                batch_size=2,
                num_workers=0,
                shuffle_train=True,
            )

            images, labels = next(iter(bundle.train_loader))

            class_weights = bundle.class_weights

            return (
                images.to(device),
                labels.to(device),
                class_weights.to(device),
                len(bundle.class_names),
                "real",
            )

        except Exception as exc:
            print(f"⚠️ No se pudo usar batch real para {model_cfg['name']}: {exc}")
            print("   Se usará batch dummy solo para prueba técnica.")

    image_size = int(model_cfg.get("image_size", config.get("data", {}).get("image_size", 224)))
    input_mode = model_cfg.get("input_mode", "rgb")
    preprocessing_cfg = config.get("branch_preprocessing", {})

    dummy_image = make_dummy_image()

    tensor = preprocess_branch_tensor(
        image=dummy_image,
        input_mode=input_mode,
        image_size=image_size,
        preprocessing_cfg=preprocessing_cfg,
        normalize=True,
    )

    images = torch.stack([tensor, tensor], dim=0).to(device)
    labels = torch.tensor([0, 1], dtype=torch.long).to(device)
    class_weights = torch.ones(num_classes, dtype=torch.float32).to(device)

    return images, labels, class_weights, num_classes, "dummy"


# ==============================================================================
# VERIFICACIONES
# ==============================================================================

def verify_environment() -> Dict[str, Any]:
    print_header("1. VERIFICACIÓN DEL ENTORNO")

    device = get_device()

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    print("Python:", info["python"].split("\n")[0])
    print("Torch:", info["torch"])
    print("CUDA disponible:", info["cuda_available"])
    print("Device:", info["device"])
    print("GPU:", info["cuda_device"])
    print("Cantidad GPUs:", info["cuda_count"])

    return info


def verify_config(config: Dict[str, Any]) -> Dict[str, Any]:
    print_header("2. VERIFICACIÓN DE CONFIG.YAML")

    project = config.get("project", {})
    data = config.get("data", {})
    training = config.get("training", {})

    active_models = [
        m for m in get_enabled_model_configs(config)
        if clean_name(m.get("name", "")) not in {"retfound", "retclip"}
    ]

    disabled_foundation = [
        m.get("name")
        for m in config.get("base_models", [])
        if clean_name(m.get("name", "")) in {"retfound", "retclip"}
    ]

    print("Proyecto:", project.get("name"))
    print("Versión:", project.get("version"))
    print("Clases:", data.get("classes"))
    print("Batch entrenamiento configurado:", training.get("batch_size"))
    print("Épocas configuradas:", training.get("epochs_base"))

    print("\nModelos activos para entrenamiento:")
    for model_cfg in active_models:
        print(
            f" - {model_cfg['name']} | "
            f"input_mode={model_cfg.get('input_mode', 'rgb')} | "
            f"size={model_cfg.get('image_size', data.get('image_size', 224))}"
        )

    if disabled_foundation:
        print("\nModelos preparados para fase posterior:")
        for name in disabled_foundation:
            print(f" - {name} desactivado por ahora")

    return {
        "project": project,
        "active_model_count": len(active_models),
        "active_models": [m["name"] for m in active_models],
        "foundation_prepared": disabled_foundation,
    }


def verify_dataset(config: Dict[str, Any]) -> Tuple[Dict[str, int], bool]:
    print_header("3. VERIFICACIÓN DE DATASET")

    raw_dir = Path(config.get("data", {}).get("raw_dir", "data/raw"))
    counts = count_dataset_images(config)
    use_real_data = can_use_real_dataset(counts)

    print("Carpeta raw:", raw_dir.resolve())

    for class_name, count in counts.items():
        print(f"{class_name}: {count} imágenes")

    total = sum(counts.values())
    print("Total:", total)

    if use_real_data:
        print("✅ Dataset suficiente para prueba real de dataloaders.")
    else:
        print("⚠️ Dataset insuficiente o aún no cargado.")
        print("   En laptop esto es aceptable si todavía no colocaste las imágenes.")
        print("   En Kaggle debe mostrar las 4000+ imágenes antes de entrenar.")

    return counts, use_real_data


def verify_preprocessing_and_radiomics(config: Dict[str, Any]) -> Dict[str, Any]:
    print_header("4. VERIFICACIÓN DE PREPROCESAMIENTO Y RADIOMICS")

    sample_path = get_first_available_image(config)

    if sample_path is not None:
        image_source: Any = sample_path
        print("Imagen de prueba:", sample_path)
    else:
        image_source = make_dummy_image()
        print("Imagen de prueba: dummy")

    preprocessing_cfg = config.get("branch_preprocessing", {})
    active_models = [
        m for m in get_enabled_model_configs(config)
        if clean_name(m.get("name", "")) not in {"retfound", "retclip"}
    ]

    preprocessing_results = {}

    for model_cfg in active_models:
        model_name = clean_name(model_cfg["name"])
        input_mode = model_cfg.get("input_mode", "rgb")
        image_size = int(model_cfg.get("image_size", config.get("data", {}).get("image_size", 224)))

        tensor = preprocess_branch_tensor(
            image=image_source,
            input_mode=input_mode,
            image_size=image_size,
            preprocessing_cfg=preprocessing_cfg,
            normalize=True,
        )

        preprocessing_results[model_name] = {
            "input_mode": input_mode,
            "shape": list(tensor.shape),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
        }

        print(
            f"{model_name:28s} | mode={input_mode:16s} | "
            f"shape={tuple(tensor.shape)} | "
            f"mean={float(tensor.mean()):.4f} | std={float(tensor.std()):.4f}"
        )

    vector, feature_names = extract_radiomics_vector(
        image=image_source,
        image_size=int(config.get("data", {}).get("image_size", 224)),
        radiomics_cfg=config.get("radiomics", {}),
    )

    print("\nRadiomics shape:", vector.shape)
    print("Primeras características:", feature_names[:10])
    print("✅ Preprocesamiento y radiomics verificados.")

    return {
        "preprocessing": preprocessing_results,
        "radiomics_shape": list(vector.shape),
        "radiomics_feature_count": int(len(feature_names)),
        "radiomics_first_features": feature_names[:10],
    }


def verify_model_forward_backward(
    config: Dict[str, Any],
    device: torch.device,
    use_real_data: bool,
    use_pretrained: bool = False,
) -> Dict[str, Any]:
    print_header("5. VERIFICACIÓN MODELO → LOSS → GRADIENTES → ACTUALIZACIÓN")

    active_models = [
        m for m in get_enabled_model_configs(config)
        if clean_name(m.get("name", "")) not in {"retfound", "retclip"}
    ]

    results = {}

    for original_model_cfg in active_models:
        model_cfg = copy.deepcopy(original_model_cfg)
        model_name = clean_name(model_cfg["name"])

        # Para verificación rápida evitamos descargar pesos.
        # En entrenamiento real se usará el valor de config.yaml.
        model_cfg["pretrained"] = bool(use_pretrained)

        print("\n" + "-" * 80)
        print(f"Probando modelo: {model_name}")
        print("-" * 80)

        start = time.time()

        try:
            images, labels, class_weights, num_classes, batch_source = get_batch_for_model(
                config=config,
                model_cfg=model_cfg,
                device=device,
                use_real_data=use_real_data,
            )

            print("Fuente batch:", batch_source)
            print("Shape imágenes:", tuple(images.shape))
            print("Shape etiquetas:", tuple(labels.shape))
            print("Etiquetas:", labels.detach().cpu().tolist())
            print("Min tensor:", float(images.min()))
            print("Max tensor:", float(images.max()))
            print("Media tensor:", float(images.mean()))
            print("Std tensor:", float(images.std()))

            model = build_model_from_config(
                model_cfg=model_cfg,
                num_classes=num_classes,
            ).to(device)

            total_params = count_total_parameters(model)
            trainable_params = count_trainable_parameters(model)

            print("Parámetros totales:", total_params)
            print("Parámetros entrenables:", trainable_params)

            if trainable_params <= 0:
                raise RuntimeError("El modelo no tiene parámetros entrenables.")

            criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(config.get("training", {}).get("learning_rate", 0.0002)),
                weight_decay=float(config.get("training", {}).get("weight_decay", 0.0001)),
            )

            trainable_before = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

            model.train()
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            logits = get_logits(outputs)

            loss = criterion(logits, labels)
            loss.backward()

            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += float(param.grad.detach().abs().sum().cpu())

            optimizer.step()

            total_change = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in trainable_before:
                    total_change += float(
                        (param.detach().cpu() - trainable_before[name].cpu()).abs().sum()
                    )

            elapsed = time.time() - start

            print("Output shape:", tuple(logits.shape))
            print("Loss:", float(loss.detach().cpu()))
            print("Grad norm:", grad_norm)
            print("Cambio pesos entrenables:", total_change)
            print("Tiempo prueba:", round(elapsed, 2), "s")

            forward_ok = logits.shape[0] == images.shape[0] and logits.shape[1] == num_classes
            grad_ok = grad_norm > 0
            update_ok = total_change > 0

            if forward_ok:
                print("✅ Forward correcto")
            else:
                print("❌ Forward incorrecto")

            if grad_ok:
                print("✅ Gradientes correctos")
            else:
                print("❌ Sin gradientes")

            if update_ok:
                print("✅ Pesos entrenables actualizados")
            else:
                print("❌ Pesos no cambiaron")

            results[model_name] = {
                "ok": bool(forward_ok and grad_ok and update_ok),
                "batch_source": batch_source,
                "output_shape": list(logits.shape),
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm),
                "trainable_weight_change": float(total_change),
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "elapsed_seconds": float(elapsed),
            }

            del model, images, labels, outputs, logits, loss
            clear_memory()

        except Exception as exc:
            print(f"❌ Error en {model_name}: {exc}")

            results[model_name] = {
                "ok": False,
                "error": str(exc),
            }

            clear_memory()

    return results


def verify_expected_artifact_outputs(config: Dict[str, Any]) -> Dict[str, Any]:
    print_header("6. ARCHIVOS QUE SE GENERARÁN DESPUÉS DEL ENTRENAMIENTO")

    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    figures_dir = Path(config.get("paths", {}).get("figures_dir", "reports/figures"))
    weights_dir = Path(config.get("paths", {}).get("weights_dir", "models/weights"))
    meta_dir = Path(config.get("paths", {}).get("meta_dir", "models/meta"))

    expected = {
        "weights": [
            "efficientnet_b0_best.pth",
            "efficientnet_b3_best.pth",
            "densenet121_best.pth",
            "inception_v3_best.pth",
            "resnet50_cbam_best.pth",
            "unet_encoder_best.pth",
            "mobilenetv3_exudate_map_best.pth",
        ],
        "meta": [
            "meta_model.pkl",
            "optimal_threshold.json",
            "base_model_names.json",
            "meta_feature_names.json",
            "radiomics_scaler.pkl",
        ],
        "reports": [
            "training_summary.json",
            "base_training_results.json",
            "base_model_metrics.csv",
            "classification_report_ensemble.csv",
            "model_comparison_wilson_ci.csv",
            "feature_correlation_top.csv",
            "meta_feature_importance.csv",
            "metrics_ensemble_stacking.csv",
            "metrics_ensemble_stacking.json",
        ],
        "figures": [
            "confusion_matrix_ensemble.png",
            "roc_curve_ensemble.png",
            "precision_recall_curve_ensemble.png",
            "probability_histogram_ensemble.png",
            "classification_report_ensemble.png",
            "model_comparison_wilson_ci.png",
            "feature_correlation_heatmap.png",
            "meta_feature_importance_top.png",
        ],
    }

    print("Pesos:", weights_dir)
    for item in expected["weights"]:
        print(" -", item)

    print("\nMeta-modelo:", meta_dir)
    for item in expected["meta"]:
        print(" -", item)

    print("\nReportes:", reports_dir)
    for item in expected["reports"]:
        print(" -", item)

    print("\nFiguras:", figures_dir)
    for item in expected["figures"]:
        print(" -", item)

    return expected


def save_verify_report(report: Dict[str, Any]) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / "verify_training_connection.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nReporte guardado en:", out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifica conexión dataset → preprocesamiento → modelos → loss → gradientes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Ruta del config.yaml",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Probar carga de pesos preentrenados. Puede requerir internet y demorar más.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Falla si no hay dataset suficiente.",
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    print_header("RETINAI - VERIFICACIÓN PREVIA AL ENTRENAMIENTO")
    print("Config:", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"No existe config.yaml: {config_path}")

    config = load_yaml_config(config_path)
    config = make_verify_config(config)

    seed = int(config.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    env_info = verify_environment()
    config_info = verify_config(config)
    dataset_counts, use_real_data = verify_dataset(config)

    if args.strict_data and not use_real_data:
        raise RuntimeError(
            "Dataset insuficiente para verificación estricta. "
            "Coloca imágenes en data/raw/healthy y data/raw/exudates."
        )

    preprocessing_info = verify_preprocessing_and_radiomics(config)

    device = get_device()

    model_results = verify_model_forward_backward(
        config=config,
        device=device,
        use_real_data=use_real_data,
        use_pretrained=bool(args.pretrained),
    )

    expected_outputs = verify_expected_artifact_outputs(config)

    all_models_ok = all(item.get("ok", False) for item in model_results.values())

    print_header("RESUMEN FINAL")

    print("Dataset real usado:", use_real_data)
    print("Modelos activos:", config_info["active_model_count"])
    print("Modelos verificados OK:", sum(1 for item in model_results.values() if item.get("ok", False)))
    print("Total modelos probados:", len(model_results))

    if all_models_ok:
        print("\n✅ Verificación completa correcta.")
        print("   El proyecto está listo para pasar a Kaggle y entrenar.")
    else:
        print("\n⚠️ Algunos modelos no pasaron la verificación.")
        print("   Revisa los errores antes de entrenar.")

    report = {
        "environment": env_info,
        "config": config_info,
        "dataset_counts": dataset_counts,
        "use_real_data": use_real_data,
        "preprocessing": preprocessing_info,
        "model_results": model_results,
        "expected_outputs": expected_outputs,
        "all_models_ok": all_models_ok,
    }

    save_verify_report(report)


if __name__ == "__main__":
    main()