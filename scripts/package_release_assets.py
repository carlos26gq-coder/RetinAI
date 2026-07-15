from __future__ import annotations
import argparse, hashlib, zipfile
from pathlib import Path

MODEL_FILES = [
    "models/weights/efficientnet_b0_best.pth",
    "models/weights/efficientnet_b3_best.pth",
    "models/weights/densenet121_best.pth",
    "models/weights/inception_v3_best.pth",
    "models/weights/unet_encoder_best.pth",
    "models/weights/mobilenetv3_exudate_map_best.pth",
    "models/weights/resnet50_cbam_best.pth",
    "models/meta/base_model_names.json",
    "models/meta/meta_feature_names.json",
    "models/meta/meta_model.pkl",
    "models/meta/optimal_threshold.json",
    "models/meta/radiomics_scaler.pkl",
]
REPRO_FILES = [
    "reports/meta_features_val.csv",
    "reports/meta_features_test.csv",
    "reports/stacking_meta_model_oof_predictions.csv",
    "reports/stacking_meta_model_test_predictions.csv",
    "reports/stacking_meta_model_comparison.csv",
    "reports/base_training_results.json",
    "reports/environment_versions.txt",
    "reports/table_v_oof_final.csv",
    "reports/table_test_final.csv",
    "reports/xgboost_threshold_comparison.csv",
    "reports/xgboost_threshold_summary.json",
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def make_zip(root, output, files):
    missing = [r for r in files if not (root / r).exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(root / rel, rel)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--include-reproducibility", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    dist = root / "dist"
    outputs = []
    model_zip = dist / f"retinai_models_v{a.version}.zip"
    make_zip(root, model_zip, MODEL_FILES)
    outputs.append(model_zip)
    if a.include_reproducibility:
        repro_zip = dist / f"retinai_reproducibility_v{a.version}.zip"
        make_zip(root, repro_zip, REPRO_FILES)
        outputs.append(repro_zip)
    sums = "".join(f"{sha256(x)}  {x.name}\n" for x in outputs)
    (dist / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    for x in outputs:
        print(x, sha256(x))

if __name__ == "__main__":
    main()
