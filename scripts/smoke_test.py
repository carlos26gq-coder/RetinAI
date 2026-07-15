from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from predictor import RetinAIPredictor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    a = p.parse_args()

    predictor = RetinAIPredictor(a.config, auto_load=True)
    if not predictor.is_loaded:
        raise RuntimeError("\n".join(predictor.load_errors))

    with Image.open(a.image) as image:
        result = predictor.predict(image.convert("RGB"), include_details=False)

    print(json.dumps({
        "predicted_label": result.get("predicted_label"),
        "probability": result.get("probability"),
        "threshold": result.get("threshold"),
        "risk_level": result.get("risk_level"),
        "base_model_count": result.get("base_model_count"),
        "meta_feature_count": result.get("meta_feature_count"),
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
