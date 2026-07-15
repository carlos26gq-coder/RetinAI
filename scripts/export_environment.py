import importlib.metadata, platform
from pathlib import Path

packages = [
    "torch", "torchvision", "numpy", "pandas", "scikit-learn",
    "scipy", "joblib", "xgboost", "Pillow",
    "opencv-python-headless", "albumentations", "scikit-image",
    "timm", "segmentation-models-pytorch", "matplotlib",
    "seaborn", "plotly", "fastapi", "uvicorn",
    "python-multipart", "streamlit", "requests", "PyYAML",
    "tqdm", "pytest", "httpx",
]
lines = [f"# Python {platform.python_version()}"]
for package in packages:
    try:
        lines.append(f"{package}=={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        pass
Path("requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("requirements-lock.txt creado")
