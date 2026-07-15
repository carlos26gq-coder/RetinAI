"""
tests/test_api.py
=================
Tests de integración para la API FastAPI.
Ejecutar: pytest tests/ -v

Autor: MVP Tesis — Ensemble Stacking Retinopatía Diabética
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api import app

client = TestClient(app)

health_data = client.get("/health").json()
MODELS_AVAILABLE = bool(health_data.get("models_loaded", False))


def create_dummy_image(
    width: int = 224,
    height: int = 224,
    color: str = "RGB",
) -> bytes:
    """Crea una imagen de prueba en memoria."""
    array = np.random.randint(
        0,
        255,
        (height, width, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(array, color)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    return image_bytes.read()


class TestSystemEndpoints:

    def test_root_endpoint(self) -> None:
        """El endpoint raíz debe responder correctamente."""
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self) -> None:
        """El health check debe retornar el estado del sistema."""
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], bool)

    def test_info_endpoint(self) -> None:
        """El endpoint de información debe responder correctamente."""
        response = client.get("/info")

        assert response.status_code == 200


class TestImageValidation:

    def test_upload_valid_jpeg(self) -> None:
        """Una imagen JPEG válida debe ser aceptada."""
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    create_dummy_image(),
                    "image/jpeg",
                )
            },
        )

        assert response.status_code in {200, 503}

    @pytest.mark.skipif(
        not MODELS_AVAILABLE,
        reason="La API valida el archivo después de cargar los modelos.",
    )
    def test_upload_invalid_extension(self) -> None:
        """Una extensión no soportada debe retornar 415."""
        response = client.post(
            "/predict",
            files={
                "file": (
                    "documento.pdf",
                    b"fake content",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 415

    @pytest.mark.skipif(
        not MODELS_AVAILABLE,
        reason="La API valida el archivo después de cargar los modelos.",
    )
    def test_upload_corrupted_file(self) -> None:
        """Un archivo corrupto debe retornar 400."""
        response = client.post(
            "/predict",
            files={
                "file": (
                    "imagen.jpg",
                    b"esto no es una imagen",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400

    @pytest.mark.skipif(
        not MODELS_AVAILABLE,
        reason="La API valida el archivo después de cargar los modelos.",
    )
    def test_upload_too_small_image(self) -> None:
        """Una imagen demasiado pequeña debe retornar 400."""
        array = np.zeros((5, 5, 3), dtype=np.uint8)
        image = Image.fromarray(array)

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "tiny.jpg",
                    image_bytes.read(),
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400

    def test_upload_no_file(self) -> None:
        """Una solicitud sin archivo debe retornar 422."""
        response = client.post("/predict")

        assert response.status_code == 422

    def test_upload_png_image(self) -> None:
        """Una imagen PNG válida debe ser aceptada."""
        array = np.random.randint(
            0,
            255,
            (224, 224, 3),
            dtype=np.uint8,
        )
        image = Image.fromarray(array)

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "retina.png",
                    image_bytes.read(),
                    "image/png",
                )
            },
        )

        assert response.status_code in {200, 503}


class TestResponseStructure:

    def test_health_response_has_required_fields(self) -> None:
        """La respuesta de health debe contener los campos requeridos."""
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data
        assert "model_names" in data or "base_models" in data

    def test_root_response_has_docs_link(self) -> None:
        """La respuesta raíz debe incluir el enlace a la documentación."""
        response = client.get("/")

        assert response.status_code == 200
        assert "docs" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])