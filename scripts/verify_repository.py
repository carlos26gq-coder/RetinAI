from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED = {
    "efficientnet_b0",
    "efficientnet_b3",
    "densenet121",
    "inception_v3",
    "unet_encoder",
    "mobilenetv3_exudate_map",
    "resnet50_cbam",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "data",
    "models",
}

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    errors = []

    required = [
        "api.py",
        "app.py",
        "predictor.py",
        "train_ensemble.py",
        "compare_stacking_meta_models.py",
        "config.yaml",
        "src/base_models.py",
        "src/data_loader.py",
        "src/metrics.py",
        "src/preprocessing_branches.py",
    ]

    for relative in required:
        if not (root / relative).exists():
            errors.append(f"Falta: {relative}")

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        rel_text = relative.as_posix()

        if any(part in SKIP_DIRS for part in relative.parts):
            continue

        if "backup_" in rel_text or rel_text.startswith("src/src"):
            errors.append(f"Ruta no permitida: {rel_text}")

        if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"Archivo público >100 MiB: {rel_text}")

    if args.require_models:
        meta = root / "models/meta"
        weights = root / "models/weights"

        for relative in [
            "base_model_names.json",
            "meta_feature_names.json",
            "meta_model.pkl",
            "optimal_threshold.json",
            "radiomics_scaler.pkl",
        ]:
            if not (meta / relative).exists():
                errors.append(f"Falta artefacto: models/meta/{relative}")

        base_names_path = meta / "base_model_names.json"
        if base_names_path.exists():
            data = read_json(base_names_path)
            names = data.get("base_model_names", data) if isinstance(data, dict) else data
            normalized = {str(name).lower() for name in names}
            if len(names) != 7 or normalized != EXPECTED:
                errors.append(
                    "base_model_names.json no contiene exactamente los 7 "
                    f"modelos finales: {names}"
                )

        feature_names_path = meta / "meta_feature_names.json"
        if feature_names_path.exists():
            data = read_json(feature_names_path)
            names = data.get("feature_names", data) if isinstance(data, dict) else data
            if len(names) != 106:
                errors.append(
                    f"Meta-features: {len(names)}; se esperaban 106"
                )

        threshold_path = meta / "optimal_threshold.json"
        if threshold_path.exists():
            data = read_json(threshold_path)
            value = data.get("threshold", data) if isinstance(data, dict) else data
            if abs(float(value) - 0.3798) > 0.005:
                errors.append(
                    f"Umbral inesperado: {value}; se esperaba aproximadamente 0.3798"
                )

        for name in EXPECTED:
            if not (weights / f"{name}_best.pth").exists():
                errors.append(f"Falta peso: {name}_best.pth")

    if errors:
        print("\n".join(f"[ERROR] {item}" for item in errors))
        sys.exit(1)

    print("Verificación OK")

if __name__ == "__main__":
    main()
