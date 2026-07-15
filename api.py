# ==============================================================================
# api.py
# RetinAI_MVP
# Backend REST API con FastAPI para Ensemble Stacking Multimodal
# ==============================================================================

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from predictor import RetinAIPredictor


# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"No se encontró config.yaml en: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

APP_VERSION = str(cfg.get("project", {}).get("version", "2.0.0"))

api_cfg = cfg.get("api", {})
MAX_FILE_SIZE_MB = int(api_cfg.get("max_file_size_mb", 10))
ALLOWED_EXTENSIONS = set(
    api_cfg.get(
        "allowed_extensions",
        [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    )
)

API_HOST = str(api_cfg.get("host", "127.0.0.1"))
API_PORT = int(api_cfg.get("port", 8000))


# ==============================================================================
# ESTADO GLOBAL
# ==============================================================================

predictor: Optional[RetinAIPredictor] = None


# ==============================================================================
# UTILIDADES DE MODELOS
# ==============================================================================

def check_required_artifacts(config_path: Path = CONFIG_PATH) -> tuple[bool, str]:
    """
    Verifica si existen los artefactos mínimos para inferencia.
    La API puede iniciar aunque falten modelos, pero /predict devolverá 503.
    """
    if not config_path.exists():
        return False, f"No existe config.yaml: {config_path}"

    with config_path.open("r", encoding="utf-8") as f:
        local_cfg = yaml.safe_load(f)

    paths_cfg = local_cfg.get("paths", {})
    weights_dir = BASE_DIR / paths_cfg.get("weights_dir", "models/weights")
    meta_dir = BASE_DIR / paths_cfg.get("meta_dir", "models/meta")

    meta_model_path = meta_dir / "meta_model.pkl"
    threshold_path = meta_dir / "optimal_threshold.json"
    base_names_path = meta_dir / "base_model_names.json"
    feature_names_path = meta_dir / "meta_feature_names.json"

    missing = []

    for path in [
        meta_model_path,
        threshold_path,
        base_names_path,
        feature_names_path,
    ]:
        if not path.exists():
            missing.append(str(path))

    base_model_names = []

    if base_names_path.exists():
        try:
            import json

            with base_names_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                base_model_names = data.get("base_model_names", [])
            elif isinstance(data, list):
                base_model_names = data
        except Exception as exc:
            return False, f"No se pudo leer base_model_names.json: {exc}"

    if base_model_names:
        for name in base_model_names:
            weight_path = weights_dir / f"{name}_best.pth"
            if not weight_path.exists():
                missing.append(str(weight_path))

    if missing:
        return False, "Faltan artefactos: " + " | ".join(missing[:8])

    return True, "Artefactos mínimos disponibles."


def get_current_predictor() -> Optional[RetinAIPredictor]:
    global predictor

    if predictor is None:
        return None

    if not getattr(predictor, "is_loaded", False):
        return None

    return predictor


# ==============================================================================
# LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    logger.info("🚀 Iniciando RetinAI API multimodal...")

    artifacts_ok, message = check_required_artifacts(CONFIG_PATH)

    if not artifacts_ok:
        predictor = None
        logger.warning("⚠️ Modelos no disponibles todavía.")
        logger.warning(message)
        logger.warning(
            "La API iniciará en modo degradado. "
            "Después de entrenar en Kaggle y copiar models/ y reports/, /predict funcionará."
        )
    else:
        try:
            predictor = RetinAIPredictor(
                config_path=CONFIG_PATH,
                auto_load=True,
            )

            if predictor.is_loaded:
                logger.info("✅ Predictor multimodal cargado correctamente.")
            else:
                logger.warning("⚠️ Predictor creado, pero no cargó completo.")
                logger.warning("Errores: %s", predictor.load_errors)

        except Exception as exc:
            predictor = None
            logger.exception("❌ Error cargando predictor: %s", exc)

    yield

    logger.info("🛑 Apagando RetinAI API...")
    predictor = None


# ==============================================================================
# FASTAPI
# ==============================================================================

app = FastAPI(
    title="RetinAI API — Detección de Exudados Retinales",
    description=(
        "API para detección temprana de exudados retinales en pacientes con diabetes "
        "mediante un ensemble stacking multimodal con CNNs, atención, mapas candidatos "
        "y características radiomics."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==============================================================================
# MODELOS PYDANTIC
# ==============================================================================

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    base_models: list[str]
    base_model_count: int
    threshold: Optional[float] = None
    meta_model_loaded: bool = False
    radiomics_scaler_loaded: bool = False
    meta_feature_count: int = 0
    device: str = "unknown"
    version: str = APP_VERSION
    errors: list[str] = Field(default_factory=list)


class BaseModelOutput(BaseModel):
    prob_healthy: float
    prob_exudates: float
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    input_mode: Optional[str] = None
    group: Optional[str] = None


class GroupOutput(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    models: int


class RadiomicsSummary(BaseModel):
    yellow_pixel_ratio: float = 0.0
    bright_pixel_ratio: float = 0.0
    candidate_regions: float = 0.0
    candidate_area_ratio: float = 0.0
    central_region_ratio: float = 0.0
    peripheral_region_ratio: float = 0.0


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Clase predicha: healthy o exudates")
    predicted_label: str = Field(..., description="Clase predicha en texto")
    predicted_index: int = Field(..., description="Índice de clase predicha")
    probability_exudates: float = Field(..., description="Probabilidad final de exudados")
    probability_healthy: float = Field(..., description="Probabilidad final de imagen sana")
    probability_percent: float = Field(..., description="Probabilidad de exudados en porcentaje")
    risk_level: str = Field(..., description="Nivel de riesgo")
    threshold_used: float = Field(..., description="Umbral usado")
    base_model_count: int = Field(..., description="Cantidad de modelos base cargados")
    meta_feature_count: int = Field(..., description="Cantidad de características del meta-modelo")
    inference_time_ms: float = Field(..., description="Tiempo de inferencia")
    model_version: str = Field(default=APP_VERSION)

    base_model_probs: Dict[str, float] = Field(default_factory=dict)
    model_probabilities: Optional[Dict[str, BaseModelOutput]] = None
    group_probabilities: Optional[Dict[str, GroupOutput]] = None
    radiomics_summary: Optional[RadiomicsSummary] = None


class InfoResponse(BaseModel):
    project: Dict[str, Any]
    ensemble_type: str
    classes: list[str]
    base_models: Dict[str, Any]
    threshold: Optional[float]
    risk_levels: Dict[str, Any]
    framework_cnn: str
    framework_meta: str
    version: str


# ==============================================================================
# VALIDACIÓN DE IMAGEN
# ==============================================================================

def validate_image_file(file: UploadFile) -> None:
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Formato no soportado: '{file_ext}'. "
                f"Formatos aceptados: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


async def read_image_from_upload(file: UploadFile) -> Image.Image:
    contents = await file.read()

    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Archivo demasiado grande: {size_mb:.1f} MB. "
                f"Máximo permitido: {MAX_FILE_SIZE_MB} MB."
            ),
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo no es una imagen válida o está corrupto.",
        )

    min_size = 50

    if image.width < min_size or image.height < min_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Imagen demasiado pequeña: {image.width}x{image.height}. "
                f"Mínimo permitido: {min_size}x{min_size}px."
            ),
        )

    return image


def normalize_prediction_response(
    result: Dict[str, Any],
    inference_time_ms: float,
) -> Dict[str, Any]:
    """
    Convierte la salida interna del predictor al formato estable de la API.
    """
    probability_exudates = float(result.get("probability", 0.0))
    probability_healthy = 1.0 - probability_exudates

    predicted_label = str(result.get("predicted_label", "unknown"))
    predicted_index = int(result.get("predicted_index", 0))

    return {
        "predicted_class": predicted_label,
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "probability_exudates": round(probability_exudates, 6),
        "probability_healthy": round(probability_healthy, 6),
        "probability_percent": round(probability_exudates * 100.0, 2),
        "risk_level": str(result.get("risk_level", "No definido")),
        "threshold_used": round(float(result.get("threshold", 0.5)), 6),
        "base_model_count": int(result.get("base_model_count", 0)),
        "meta_feature_count": int(result.get("meta_feature_count", 0)),
        "inference_time_ms": round(inference_time_ms, 2),
        "model_version": APP_VERSION,
        "base_model_probs": result.get("base_model_probs", {}),
        "model_probabilities": result.get("model_probabilities", None),
        "group_probabilities": result.get("group_probabilities", None),
        "radiomics_summary": result.get("radiomics_summary", None),
    }


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", tags=["Sistema"])
async def root():
    return {
        "message": "RetinAI API — Detección temprana de exudados retinales",
        "description": "Ensemble stacking multimodal para imágenes de fondo de ojo.",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "version": APP_VERSION,
    }


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    p = get_current_predictor()

    if p is None:
        artifacts_ok, message = check_required_artifacts(CONFIG_PATH)

        return HealthResponse(
            status="degraded",
            models_loaded=False,
            base_models=[],
            base_model_count=0,
            threshold=None,
            meta_model_loaded=False,
            radiomics_scaler_loaded=False,
            meta_feature_count=0,
            device="unknown",
            version=APP_VERSION,
            errors=[] if artifacts_ok else [message],
        )

    health = p.health()

    return HealthResponse(
        status="healthy" if health.get("models_loaded", False) else "degraded",
        models_loaded=bool(health.get("models_loaded", False)),
        base_models=list(health.get("base_models", [])),
        base_model_count=int(health.get("base_model_count", 0)),
        threshold=health.get("threshold", None),
        meta_model_loaded=bool(health.get("meta_model_loaded", False)),
        radiomics_scaler_loaded=bool(health.get("radiomics_scaler_loaded", False)),
        meta_feature_count=int(health.get("meta_feature_count", 0)),
        device=str(health.get("device", "unknown")),
        version=APP_VERSION,
        errors=list(health.get("errors", [])),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predicción"],
    summary="Predice presencia de exudados retinales en una imagen de fondo de ojo",
    responses={
        200: {"description": "Predicción exitosa"},
        400: {"description": "Imagen inválida o corrupta"},
        413: {"description": "Archivo demasiado grande"},
        415: {"description": "Formato de imagen no soportado"},
        503: {"description": "Modelos no disponibles"},
    },
)
async def predict_image(
    file: UploadFile = File(
        ...,
        description="Imagen de fondo de ojo: JPG, PNG, BMP o TIFF.",
    ),
    include_details: bool = Query(
        True,
        description="Incluye probabilidades por modelo, grupos y resumen radiomics.",
    ),
):
    p = get_current_predictor()

    if p is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Los modelos todavía no están disponibles. "
                "Entrena en Kaggle, descarga retinai_resultados_kaggle.zip y copia "
                "las carpetas models/ y reports/ al proyecto local."
            ),
        )

    validate_image_file(file)
    image = await read_image_from_upload(file)

    start_time = time.perf_counter()

    try:
        result = await run_in_threadpool(
            p.predict,
            image,
            include_details,
        )
    except Exception as exc:
        logger.exception("Error durante la inferencia: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia: {str(exc)}",
        )

    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    response = normalize_prediction_response(
        result=dict(result),
        inference_time_ms=inference_time_ms,
    )

    logger.info(
        "Predicción | archivo=%s | clase=%s | P(exudates)=%.4f | riesgo=%s | %.1f ms",
        file.filename,
        response["predicted_class"],
        response["probability_exudates"],
        response["risk_level"],
        inference_time_ms,
    )

    return PredictionResponse(**response)


@app.get("/info", response_model=InfoResponse, tags=["Sistema"])
async def model_info():
    p = get_current_predictor()

    if p is None:
        return InfoResponse(
            project=cfg.get("project", {}),
            ensemble_type="Stacking multimodal",
            classes=cfg.get("data", {}).get("classes", ["healthy", "exudates"]),
            base_models={},
            threshold=None,
            risk_levels=cfg.get("inference", {}).get("risk_levels", {}),
            framework_cnn="PyTorch",
            framework_meta="Scikit-Learn / XGBoost",
            version=APP_VERSION,
        )

    info = p.info()

    return InfoResponse(
        project=info.get("project", cfg.get("project", {})),
        ensemble_type="Stacking multimodal",
        classes=info.get("classes", cfg.get("data", {}).get("classes", [])),
        base_models=info.get("base_models", {}),
        threshold=info.get("threshold", None),
        risk_levels=info.get("risk_levels", {}),
        framework_cnn="PyTorch",
        framework_meta="Scikit-Learn / XGBoost",
        version=APP_VERSION,
    )


@app.get("/artifacts", tags=["Sistema"])
async def artifacts_status():
    artifacts_ok, message = check_required_artifacts(CONFIG_PATH)

    paths_cfg = cfg.get("paths", {})
    weights_dir = BASE_DIR / paths_cfg.get("weights_dir", "models/weights")
    meta_dir = BASE_DIR / paths_cfg.get("meta_dir", "models/meta")
    reports_dir = BASE_DIR / paths_cfg.get("reports_dir", "reports")

    weights = sorted([p.name for p in weights_dir.glob("*.pth")]) if weights_dir.exists() else []
    meta_files = sorted([p.name for p in meta_dir.glob("*")]) if meta_dir.exists() else []
    reports = sorted([p.name for p in reports_dir.glob("*") if p.is_file()]) if reports_dir.exists() else []

    return {
        "artifacts_ok": artifacts_ok,
        "message": message,
        "weights_dir": str(weights_dir),
        "meta_dir": str(meta_dir),
        "reports_dir": str(reports_dir),
        "weights": weights,
        "meta_files": meta_files,
        "reports": reports,
    }


# ==============================================================================
# EJECUCIÓN LOCAL
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )