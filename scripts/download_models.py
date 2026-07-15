from __future__ import annotations
import argparse, hashlib, tempfile, urllib.request, zipfile
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_extract(z, destination):
    destination = destination.resolve()
    for member in z.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"Ruta insegura: {member.filename}")
    z.extractall(destination)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--sha256", default="")
    p.add_argument("--destination", type=Path, default=Path.cwd())
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "models.zip"
        urllib.request.urlretrieve(a.url, path)
        actual = sha256(path)
        print("SHA256:", actual)
        if a.sha256 and actual.lower() != a.sha256.lower():
            raise RuntimeError("SHA-256 no coincide")
        with zipfile.ZipFile(path) as z:
            safe_extract(z, a.destination.resolve())
    print("Modelos instalados.")

if __name__ == "__main__":
    main()
