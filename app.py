# ==============================================================================
# app.py
# RetinAI_MVP
# Interfaz Streamlit para detección temprana de exudados retinales
# ==============================================================================

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="RetinAI — Detección de Exudados Retinales",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==============================================================================
# RUTAS / CONFIG
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

DEFAULT_API_URL = os.getenv("RETINAI_API_URL", "http://127.0.0.1:8000")

RISK_COLORS = {
    "Bajo": "#2ECC71",
    "Moderado": "#F39C12",
    "Alto": "#E74C3C",
}

RISK_ICONS = {
    "Bajo": "✅",
    "Moderado": "⚠️",
    "Alto": "🚨",
}


# ==============================================================================
# CONFIG HELPERS
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_config() -> Dict[str, Any]:
    import yaml

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No se encontró config.yaml en: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


APP_CFG = load_config()
APP_VERSION = str(APP_CFG.get("project", {}).get("version", "2.0.0"))

SUPPORTED_EXTENSIONS = [
    ext.lower().replace(".", "")
    for ext in APP_CFG.get("api", {}).get(
        "allowed_extensions",
        [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    )
]

MAX_FILE_SIZE_MB = int(APP_CFG.get("api", {}).get("max_file_size_mb", 10))


def normalize_risk_label(risk_level: str) -> str:
    value = str(risk_level or "").strip()

    if "Alto" in value:
        return "Alto"
    if "Moderado" in value:
        return "Moderado"
    if "Bajo" in value:
        return "Bajo"

    return "Bajo"


def get_api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL)


def set_api_url(value: str) -> None:
    st.session_state["api_url"] = (value or "").strip() or DEFAULT_API_URL


def validate_local_file_size(uploaded_file) -> bool:
    return (uploaded_file.size / (1024 * 1024)) <= MAX_FILE_SIZE_MB


def display_image_compat(image: Image.Image, caption: str) -> None:
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption)


def plotly_chart_compat(fig: go.Figure, key: Optional[str] = None) -> None:
    try:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key=key,
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def image_to_api_file(image: Image.Image) -> Tuple[io.BytesIO, str]:
    img_bytes = io.BytesIO()
    rgb_image = image.convert("RGB")
    rgb_image.save(img_bytes, format="JPEG", quality=95, optimize=True)
    img_bytes.seek(0)
    return img_bytes, "imagen_fondo_ojo.jpg"


# ==============================================================================
# CSS
# ==============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #0f3460 100%);
            padding: 2rem 2rem 1.6rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
            text-align: center;
            color: white;
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
        }

        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
        }

        .main-header p {
            font-size: 1.05rem;
            opacity: 0.88;
            margin: 0.55rem 0 0;
            font-weight: 400;
        }

        .result-card {
            border-radius: 16px;
            padding: 1.5rem;
            text-align: left;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            margin: 0.8rem 0 1rem;
            border: 1px solid #eeeeee;
        }

        .result-high {
            background: linear-gradient(135deg, #fff5f5, #ffe5e5);
            border-left: 6px solid #E74C3C;
        }

        .result-moderate {
            background: linear-gradient(135deg, #fffdf0, #fff3cd);
            border-left: 6px solid #F39C12;
        }

        .result-low {
            background: linear-gradient(135deg, #f0fff4, #d4edda);
            border-left: 6px solid #2ECC71;
        }

        .metric-card {
            background: white;
            border-radius: 14px;
            padding: 1.05rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            border: 1px solid #eeeeee;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f3460;
        }

        .metric-label {
            font-size: 0.78rem;
            color: #666;
            font-weight: 600;
            text-transform: uppercase;
            margin-top: 0.25rem;
        }

        .clinical-note {
            background: #f8f9fa;
            border-left: 4px solid #0f3460;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: #333;
            font-size: 0.92rem;
        }

        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 9px;
            padding: 0.85rem 1rem;
            font-size: 0.85rem;
            color: #856404;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# NORMALIZACIÓN DE RESULTADOS
# ==============================================================================

def normalize_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unifica salida de API y modo standalone.
    """
    probability_exudates = result.get("probability_exudates", None)

    if probability_exudates is None:
        probability_exudates = result.get("probability", 0.0)

    probability_exudates = float(probability_exudates)
    probability_exudates = max(0.0, min(1.0, probability_exudates))

    probability_healthy = result.get("probability_healthy", None)

    if probability_healthy is None:
        probability_healthy = 1.0 - probability_exudates

    probability_healthy = float(probability_healthy)
    probability_healthy = max(0.0, min(1.0, probability_healthy))

    threshold = result.get("threshold_used", result.get("threshold", 0.5))
    threshold = float(threshold)

    predicted_index = result.get("predicted_index", None)

    if predicted_index is None:
        predicted_index = int(probability_exudates >= threshold)

    predicted_index = int(predicted_index)

    predicted_class = result.get("predicted_class", result.get("predicted_label", ""))

    if isinstance(predicted_class, int):
        predicted_class = "exudates" if predicted_class == 1 else "healthy"

    predicted_class = str(predicted_class or "").lower()

    if predicted_class not in {"healthy", "exudates"}:
        predicted_class = "exudates" if predicted_index == 1 else "healthy"

    normalized = dict(result)
    normalized["probability_exudates"] = probability_exudates
    normalized["probability_healthy"] = probability_healthy
    normalized["probability_percent"] = round(probability_exudates * 100.0, 2)
    normalized["threshold_used"] = threshold
    normalized["predicted_index"] = predicted_index
    normalized["predicted_class"] = predicted_class
    normalized["predicted_label"] = predicted_class
    normalized["risk_level"] = normalize_risk_label(result.get("risk_level", "Bajo"))

    return normalized


# ==============================================================================
# API MODE
# ==============================================================================

def check_api_health(api_url: str) -> Tuple[bool, str]:
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)

        if resp.status_code != 200:
            return False, f"API respondió HTTP {resp.status_code}"

        payload = resp.json()

        if payload.get("models_loaded", False):
            count = int(payload.get("base_model_count", 0))
            return True, f"✅ API online — {count} modelos cargados"

        return False, "⚠️ API online — modelos aún no cargados"

    except requests.exceptions.ConnectionError:
        return False, "❌ API offline"
    except requests.exceptions.Timeout:
        return False, "❌ API sin respuesta"
    except Exception as exc:
        return False, f"❌ Error: {exc}"


def predict_via_api(image: Image.Image) -> Optional[Dict[str, Any]]:
    img_bytes, filename = image_to_api_file(image)

    try:
        response = requests.post(
            f"{get_api_url()}/predict",
            params={"include_details": "true"},
            files={"file": (filename, img_bytes, "image/jpeg")},
            timeout=120,
        )

        if response.status_code == 200:
            return normalize_result_payload(response.json())

        try:
            error_detail = response.json().get("detail", "Error desconocido")
        except Exception:
            error_detail = response.text or "Error desconocido"

        st.error(f"❌ Error de la API ({response.status_code}): {error_detail}")
        return None

    except requests.exceptions.ConnectionError:
        st.error(f"❌ No se puede conectar con la API en {get_api_url()}.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ La API tardó demasiado en responder.")
        return None
    except Exception as exc:
        st.error(f"❌ Error inesperado al contactar la API: {exc}")
        return None


# ==============================================================================
# STANDALONE MODE
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_standalone_predictor(config_path_str: str):
    from predictor import RetinAIPredictor

    return RetinAIPredictor(
        config_path=config_path_str,
        auto_load=True,
    )


def predict_standalone(image: Image.Image) -> Optional[Dict[str, Any]]:
    try:
        predictor = load_standalone_predictor(str(CONFIG_PATH))

        if not predictor.is_loaded:
            st.error(
                "❌ Los modelos aún no están cargados. "
                "Primero entrena en Kaggle y copia la carpeta models/ al proyecto local."
            )
            return None

        start = time.perf_counter()
        result = predictor.predict(image, include_details=True)
        result["inference_time_ms"] = round((time.perf_counter() - start) * 1000, 2)

        return normalize_result_payload(result)

    except Exception as exc:
        st.error(f"❌ Error en modo standalone: {exc}")
        return None


# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

def render_risk_gauge(result: Dict[str, Any]) -> go.Figure:
    """
    Indicador angular dinámico.
    El valor se mueve según la probabilidad real de exudados de cada imagen.
    No muestra umbral para evitar redundancia visual.
    """
    probability = float(result.get("probability_exudates", 0.0))
    probability = max(0.0, min(1.0, probability))
    probability_percent = probability * 100.0

    risk = normalize_risk_label(result.get("risk_level", "Bajo"))
    color = RISK_COLORS.get(risk, "#0f3460")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                    "color": "#111827",
                },
            },
            title={
                "text": "Compatibilidad con exudados retinales",
                "font": {
                    "size": 15,
                    "color": "#374151",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#6b7280",
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticksuffix": "%",
                },
                "bar": {
                    "color": color,
                    "thickness": 0.28,
                },
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e5e7eb",
                "steps": [
                    {"range": [0, 40], "color": "#d1fae5"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
                # Esta línea se mueve con el resultado, ya no representa el umbral.
                "threshold": {
                    "line": {
                        "color": color,
                        "width": 5,
                    },
                    "thickness": 0.85,
                    "value": probability_percent,
                },
            },
        )
    )

    fig.update_layout(
        height=310,
        margin={"t": 45, "b": 10, "l": 25, "r": 25},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )

    return fig


def extract_model_probabilities(result: Dict[str, Any]) -> Dict[str, float]:
    model_probs = {}

    detailed = result.get("model_probabilities", None)

    if isinstance(detailed, dict) and detailed:
        for model_name, item in detailed.items():
            if isinstance(item, dict):
                model_probs[model_name] = float(item.get("prob_exudates", 0.0))
        return model_probs

    base = result.get("base_model_probs", {})

    if isinstance(base, dict):
        for model_name, value in base.items():
            if isinstance(value, dict):
                model_probs[model_name] = float(
                    value.get("prob_exudates", value.get("p_exudates", 0.0))
                )
            else:
                model_probs[model_name] = float(value)

    return model_probs


def pretty_model_name(name: str) -> str:
    mapping = {
        "efficientnet_b0": "EfficientNetB0",
        "efficientnet_b3": "EfficientNetB3",
        "densenet121": "DenseNet121",
        "inception_v3": "Inception-v3",
        "resnet50_cbam": "ResNet50 + CBAM",
        "unet_encoder": "U-Net Encoder",
        "mobilenetv3_exudate_map": "MobileNetV3 Map",
        "retfound": "RETFound",
        "retclip": "RET-CLIP",
    }

    return mapping.get(name, name)


def render_base_model_chart(result: Dict[str, Any]) -> go.Figure:
    model_probs = extract_model_probabilities(result)

    if not model_probs:
        return go.Figure()

    models = list(model_probs.keys())
    values = [model_probs[m] * 100.0 for m in models]
    labels = [pretty_model_name(m) for m in models]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Probabilidad de exudados por modelo base",
        yaxis={"title": "Probabilidad (%)", "range": [0, 110], "ticksuffix": "%"},
        xaxis={"title": "Modelo"},
        height=360,
        margin={"t": 55, "b": 80, "l": 45, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,1)",
        font={"family": "Inter"},
        showlegend=False,
    )

    return fig


def render_group_chart(result: Dict[str, Any]) -> go.Figure:
    groups = result.get("group_probabilities", {})

    if not isinstance(groups, dict) or not groups:
        return go.Figure()

    label_map = {
        "global_multiscale": "Global / multiescala",
        "lesion_focused": "Focalizado en lesión",
        "attention_structure": "Atención / estructura",
        "other": "Otros",
    }

    names = []
    values = []
    counts = []

    for group_name, item in groups.items():
        if isinstance(item, dict):
            names.append(label_map.get(group_name, group_name))
            values.append(float(item.get("mean", 0.0)) * 100.0)
            counts.append(int(item.get("models", 0)))

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            text=[f"{v:.1f}%<br>{c} modelos" for v, c in zip(values, counts)],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Resultado agrupado por enfoque de análisis",
        yaxis={"title": "Promedio de probabilidad (%)", "range": [0, 110], "ticksuffix": "%"},
        xaxis={"title": "Grupo de modelos"},
        height=330,
        margin={"t": 55, "b": 70, "l": 45, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,1)",
        font={"family": "Inter"},
        showlegend=False,
    )

    return fig


def render_radiomics_chart(result: Dict[str, Any]) -> go.Figure:
    summary = result.get("radiomics_summary", {})

    if not isinstance(summary, dict) or not summary:
        return go.Figure()

    items = {
        "Píxeles amarillentos": float(summary.get("yellow_pixel_ratio", 0.0)) * 100.0,
        "Píxeles brillantes": float(summary.get("bright_pixel_ratio", 0.0)) * 100.0,
        "Área candidata": float(summary.get("candidate_area_ratio", 0.0)) * 100.0,
        "Región central": float(summary.get("central_region_ratio", 0.0)) * 100.0,
        "Región periférica": float(summary.get("peripheral_region_ratio", 0.0)) * 100.0,
    }

    fig = go.Figure(
        go.Bar(
            x=list(items.keys()),
            y=list(items.values()),
            text=[f"{v:.2f}%" for v in items.values()],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Características visuales extraídas de la imagen",
        yaxis={"title": "Proporción (%)", "range": [0, max(5, max(items.values()) * 1.25)]},
        xaxis={"title": "Característica"},
        height=330,
        margin={"t": 55, "b": 90, "l": 45, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,1)",
        font={"family": "Inter"},
        showlegend=False,
    )

    return fig


def render_multi_image_comparison(records: List[Dict[str, Any]]) -> go.Figure:
    names = [item["filename"] for item in records]
    probs = [float(item["result"].get("probability_exudates", 0.0)) * 100.0 for item in records]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=probs,
            text=[f"{v:.1f}%" for v in probs],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Comparación entre imágenes analizadas",
        yaxis={"title": "Probabilidad de exudados (%)", "range": [0, 110], "ticksuffix": "%"},
        xaxis={"title": "Imagen"},
        height=330,
        margin={"t": 55, "b": 90, "l": 45, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,1)",
        font={"family": "Inter"},
        showlegend=False,
    )

    return fig


# ==============================================================================
# RESULTADO CLÍNICO
# ==============================================================================

def clinical_text(result: Dict[str, Any]) -> Tuple[str, str]:
    prob = float(result.get("probability_exudates", 0.0))
    threshold = float(result.get("threshold_used", 0.5))
    predicted_index = int(result.get("predicted_index", 0))

    if predicted_index == 1:
        title = "Hallazgos compatibles con exudados retinales"
        message = (
            "La imagen presenta una probabilidad por encima del umbral configurado. "
            "Se recomienda revisión por especialista y correlación con evaluación oftalmológica."
        )
    else:
        title = "No se evidencian exudados retinales significativos"
        message = (
            "La probabilidad se encuentra por debajo del umbral configurado. "
            "El resultado no descarta otros signos de retinopatía diabética y debe interpretarse como apoyo."
        )

    if abs(prob - threshold) <= 0.08:
        message += (
            " El resultado está cerca del umbral, por lo que conviene revisar calidad de imagen "
            "y considerar una segunda evaluación."
        )

    return title, message


def render_result_card(result: Dict[str, Any], filename: str = "") -> None:
    """
    Tarjeta principal sin umbral ni texto interpretativo redundante.
    """
    risk = normalize_risk_label(result.get("risk_level", "Bajo"))
    icon = RISK_ICONS.get(risk, "ℹ️")
    color = RISK_COLORS.get(risk, "#888")

    css_class = {
        "Alto": "result-high",
        "Moderado": "result-moderate",
        "Bajo": "result-low",
    }.get(risk, "")

    predicted_index = int(result.get("predicted_index", 0))

    if predicted_index == 1:
        title = "Hallazgos compatibles con exudados retinales"
    else:
        title = "Sin hallazgos compatibles con exudados retinales"

    probability = float(result.get("probability_exudates", 0.0)) * 100.0

    file_label = (
        f"<div style='font-size:0.82rem; color:#777; margin-bottom:0.3rem'>{filename}</div>"
        if filename else ""
    )

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            {file_label}
            <div style="display:flex; gap:1rem; align-items:center;">
                <div style="font-size:2.9rem;">{icon}</div>
                <div>
                    <div style="font-size:1.35rem; font-weight:750; color:{color};">
                        {title}
                    </div>
                    <div style="font-size:0.95rem; color:#444; margin-top:0.25rem;">
                        Nivel de alerta: <strong style="color:{color};">{risk}</strong>
                    </div>
                    <div style="font-size:0.90rem; color:#555; margin-top:0.35rem;">
                        Probabilidad estimada: <strong>{probability:.1f}%</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(result: Dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{float(result.get('probability_exudates', 0.0))*100:.1f}%</div>
                <div class="metric-label">Compatibilidad con exudados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{float(result.get('probability_healthy', 0.0))*100:.1f}%</div>
                <div class="metric-label">Compatibilidad sin exudados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{normalize_risk_label(result.get('risk_level', 'Bajo'))}</div>
                <div class="metric-label">Nivel de alerta</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        if "api_url" not in st.session_state:
            st.session_state["api_url"] = DEFAULT_API_URL

        api_url = st.text_input(
            "URL de la API",
            value=st.session_state["api_url"],
            help="Por defecto: http://127.0.0.1:8000",
        )
        set_api_url(api_url)

        mode = st.radio(
            "Modo de inferencia",
            options=["API", "Standalone"],
            index=0,
            help="API usa FastAPI. Standalone carga los modelos directamente en Streamlit.",
        )

        st.markdown("---")
        st.markdown("### 📡 Estado")

        if mode == "API":
            online, message = check_api_health(get_api_url())

            if online:
                st.success(message)
            else:
                st.warning(message)
                st.code("uvicorn api:app --host 127.0.0.1 --port 8000 --reload", language="bash")
        else:
            st.info("Modo standalone activo")
            st.caption("Útil solo cuando los modelos ya fueron entrenados y copiados localmente.")

        st.markdown("---")
        st.markdown("### ℹ️ Proyecto")
        st.markdown(
            f"""
            **Versión:** {APP_VERSION}  
            **Formatos:** {", ".join(SUPPORTED_EXTENSIONS)}  
            **Máximo por imagen:** {MAX_FILE_SIZE_MB} MB  

            **Modelo:** Ensemble stacking multimodal  
            **Entrada:** Imágenes de fondo de ojo  
            **Salida:** Probabilidad de hallazgos compatibles con exudados retinales
            """
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="disclaimer">
            ⚠️ <strong>Aviso:</strong> Este sistema es una herramienta de apoyo.
            No reemplaza el diagnóstico ni la evaluación de un oftalmólogo.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return mode


# ==============================================================================
# ANÁLISIS DE IMÁGENES
# ==============================================================================

def open_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    if not validate_local_file_size(uploaded_file):
        st.error(f"❌ {uploaded_file.name} supera el límite de {MAX_FILE_SIZE_MB} MB.")
        return None

    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except UnidentifiedImageError:
        st.error(f"❌ {uploaded_file.name} no es una imagen válida o está corrupta.")
        return None
    except Exception as exc:
        st.error(f"❌ No fue posible abrir {uploaded_file.name}: {exc}")
        return None


def run_prediction(image: Image.Image, mode: str) -> Optional[Dict[str, Any]]:
    if mode == "API":
        return predict_via_api(image)

    return predict_standalone(image)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    inject_css()

    st.markdown(
        f"""
        <div class="main-header">
            <h1>👁️ RetinAI</h1>
            <p>Detección temprana de exudados retinales en pacientes con diabetes mediante modelos preentrenados</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = render_sidebar()

    col_upload, col_result = st.columns([0.95, 1.05], gap="large")

    with col_upload:
        st.markdown("### 📁 Cargar imágenes de fondo de ojo")

        uploaded_files = st.file_uploader(
            "Selecciona una, dos o tres imágenes",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Formatos permitidos: {', '.join(SUPPORTED_EXTENSIONS)}. Máximo {MAX_FILE_SIZE_MB} MB por imagen.",
        )

        if uploaded_files and len(uploaded_files) > 3:
            st.warning("Solo se analizarán las primeras 3 imágenes cargadas.")
            uploaded_files = uploaded_files[:3]

        loaded_images: List[Dict[str, Any]] = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = open_uploaded_image(uploaded_file)

                if image is None:
                    continue

                loaded_images.append(
                    {
                        "filename": uploaded_file.name,
                        "image": image,
                        "size_kb": uploaded_file.size / 1024,
                    }
                )

            if loaded_images:
                preview_cols = st.columns(len(loaded_images))

                for idx, item in enumerate(loaded_images):
                    with preview_cols[idx]:
                        display_image_compat(
                            item["image"],
                            caption=f"{item['filename']} · {item['image'].width}×{item['image'].height}px",
                        )

                if st.button("🔬 Analizar imagen/es", type="primary", use_container_width=True):
                    records = []

                    progress = st.progress(0)
                    status_box = st.empty()

                    for idx, item in enumerate(loaded_images, start=1):
                        status_box.info(f"Analizando {item['filename']} ({idx}/{len(loaded_images)})...")

                        result = run_prediction(item["image"], mode)

                        if result is not None:
                            records.append(
                                {
                                    "filename": item["filename"],
                                    "image": item["image"],
                                    "result": result,
                                }
                            )

                        progress.progress(idx / len(loaded_images))

                    status_box.empty()

                    if records:
                        st.session_state["analysis_records"] = records
                        st.success("✅ Análisis completado.")
                    else:
                        st.error("No se pudo completar el análisis.")

        else:
            st.info("Sube una imagen de fondo de ojo para iniciar el análisis.")

    with col_result:
        st.markdown("### 📊 Resultado principal")

        records = st.session_state.get("analysis_records", [])

        if not records:
            st.info("Los resultados aparecerán aquí después del análisis.")
        else:
            selected_index = 0

            if len(records) > 1:
                selected_name = st.selectbox(
                    "Selecciona imagen para revisar",
                    options=[item["filename"] for item in records],
                )

                selected_index = [item["filename"] for item in records].index(selected_name)

            selected = records[selected_index]
            result = selected["result"]

            render_result_card(result, filename=selected["filename"])
            render_summary_metrics(result)

            plotly_chart_compat(
                render_risk_gauge(result),
                key=f"risk_gauge_{selected['filename']}_{result.get('probability_percent', 0)}",
            )

    records = st.session_state.get("analysis_records", [])

    if records:
        st.markdown("---")
        st.markdown("### 📌 Análisis complementario")

        if len(records) > 1:
            summary_rows = []

            for item in records:
                result = item["result"]
                summary_rows.append(
                    {
                        "Imagen": item["filename"],
                        "Resultado": "Compatible con exudados" if int(result.get("predicted_index", 0)) == 1 else "Sin exudados significativos",
                        "Probabilidad exudados (%)": round(float(result.get("probability_exudates", 0.0)) * 100.0, 2),
                        "Riesgo": normalize_risk_label(result.get("risk_level", "Bajo")),
                    }
                )

            st.dataframe(summary_rows, use_container_width=True)

            plotly_chart_compat(
                render_multi_image_comparison(records),
                key="multi_image_comparison",
            )

        selected = records[0]

        if len(records) > 1:
            selected_name = st.selectbox(
                "Imagen para gráficos complementarios",
                options=[item["filename"] for item in records],
                key="selected_graph_image",
            )
            selected = [item for item in records if item["filename"] == selected_name][0]

        result = selected["result"]

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🔎 Enfoques del modelo",
                "🧠 Modelos base",
                "🧬 Características visuales",
                "📄 Detalles técnicos",
            ]
        )

        with tab1:
            fig = render_group_chart(result)
            if fig.data:
                plotly_chart_compat(fig, key=f"group_chart_{selected['filename']}")
                st.caption(
                    "Agrupa los modelos según el tipo de análisis: visión global/multiescala, "
                    "enfoque en lesión y atención/estructura retinal."
                )
            else:
                st.info("No se recibieron datos agrupados del modelo.")

        with tab2:
            fig = render_base_model_chart(result)
            if fig.data:
                plotly_chart_compat(fig, key=f"base_chart_{selected['filename']}")
                st.caption(
                    "Muestra la probabilidad de exudados estimada por cada modelo base antes del stacking final."
                )
            else:
                st.info("No se recibieron probabilidades por modelo base.")

        with tab3:
            fig = render_radiomics_chart(result)
            if fig.data:
                plotly_chart_compat(fig, key=f"radiomics_chart_{selected['filename']}")
                st.caption(
                    "Resume señales visuales usadas como apoyo: brillo, tonalidad amarillenta, área candidata "
                    "y distribución central/periférica."
                )
            else:
                st.info("No se recibieron características visuales complementarias.")

        with tab4:
            st.markdown(
                "Estos datos son técnicos y sirven para depuración o validación del sistema, "
                "no para lectura clínica directa."
            )
            technical_result = dict(result)
            technical_result.pop("threshold", None)
            technical_result.pop("threshold_used", None)
            st.json(technical_result)


if __name__ == "__main__":
    main()