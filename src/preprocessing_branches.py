# ==============================================================================
# src/preprocessing_branches.py
# RetinAI_MVP
# Preprocesamiento multimodal por rama + extracción de características radiomics
# ==============================================================================

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


try:
    from skimage.feature import local_binary_pattern
except Exception:
    local_binary_pattern = None

try:
    from skimage.filters import gabor
except Exception:
    gabor = None

try:
    from skimage.measure import regionprops
except Exception:
    regionprops = None


ImageInput = Union[str, Path, Image.Image, np.ndarray]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================================
# UTILIDADES BASE
# ==============================================================================

def ensure_rgb_array(image: ImageInput) -> np.ndarray:
    """
    Convierte una imagen PIL, ruta o ndarray a RGB uint8 con forma HxWx3.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
        return np.array(image, dtype=np.uint8)

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)

    if isinstance(image, np.ndarray):
        arr = image.copy()

        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    raise TypeError(f"Tipo de imagen no soportado: {type(image)}")


def resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    """
    Redimensiona una imagen RGB manteniendo salida uint8 HxWx3.
    """
    image = ensure_rgb_array(image)
    return cv2.resize(
        image,
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_AREA,
    )


def normalize_uint8_to_float(image: np.ndarray) -> np.ndarray:
    """
    Convierte uint8 0-255 a float32 0-1.
    """
    image = ensure_rgb_array(image)
    return image.astype(np.float32) / 255.0


def imagenet_normalize(image_float: np.ndarray) -> np.ndarray:
    """
    Normalización ImageNet para tensores CNN.
    Entrada esperada: float32 HxWx3 en rango 0-1.
    """
    return (image_float - IMAGENET_MEAN) / IMAGENET_STD


def to_tensor_chw(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    Convierte imagen RGB uint8 HxWx3 a tensor CHW float32.
    """
    image_float = normalize_uint8_to_float(image)

    if normalize:
        image_float = imagenet_normalize(image_float)

    tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).float()
    return tensor


def _get_mode_cfg(
    preprocessing_cfg: Optional[Dict[str, Any]],
    input_mode: str,
) -> Dict[str, Any]:
    """
    Obtiene la configuración específica de una rama desde branch_preprocessing.
    """
    if preprocessing_cfg is None:
        return {}

    if input_mode in preprocessing_cfg:
        return preprocessing_cfg.get(input_mode, {}) or {}

    return preprocessing_cfg or {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _scale_to_uint8(x: np.ndarray) -> np.ndarray:
    """
    Escala una matriz numérica cualquiera a uint8 0-255.
    """
    x = x.astype(np.float32)
    min_v = float(np.min(x))
    max_v = float(np.max(x))

    if max_v - min_v < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    x = (x - min_v) / (max_v - min_v)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


# ==============================================================================
# PREPROCESAMIENTOS VISUALES
# ==============================================================================

def apply_clahe_gray(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Aplica CLAHE a una imagen en escala de grises.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(gray)


def apply_clahe_rgb_luminance(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Aplica CLAHE sobre la luminancia L en espacio LAB.
    Mantiene la apariencia RGB, pero mejora contraste local.
    """
    image = ensure_rgb_array(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_clahe = apply_clahe_gray(
        l_channel,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe.astype(np.uint8)


def make_clahe_green_image(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Extrae canal verde, aplica CLAHE y lo replica a 3 canales.
    Útil para resaltar contraste vascular y lesiones brillantes.
    """
    image = ensure_rgb_array(image)
    green = image[:, :, 1]

    green_clahe = apply_clahe_gray(
        green,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    return cv2.merge([green_clahe, green_clahe, green_clahe]).astype(np.uint8)


def create_candidate_mask(
    image: np.ndarray,
    use_green_channel: bool = True,
    use_clahe: bool = True,
    use_tophat: bool = True,
    threshold_method: str = "otsu",
    morphology_kernel_size: int = 5,
) -> np.ndarray:
    """
    Genera una máscara candidata de zonas brillantes compatibles con exudados.
    No es una segmentación clínica definitiva; es una entrada auxiliar para ramas.
    """
    image = ensure_rgb_array(image)

    if use_green_channel:
        base = image[:, :, 1]
    else:
        base = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    base = base.astype(np.uint8)

    if use_clahe:
        base = apply_clahe_gray(base, clip_limit=2.0, tile_grid_size=8)

    if use_tophat:
        k = max(3, int(morphology_kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        enhanced = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, kernel)
    else:
        enhanced = base

    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    if threshold_method.lower() == "otsu":
        _, mask = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        p = np.percentile(enhanced, 90)
        _, mask = cv2.threshold(enhanced, p, 255, cv2.THRESH_BINARY)

    k = max(3, int(morphology_kernel_size))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask.astype(np.uint8)


def create_exudate_map(
    image: np.ndarray,
    use_hsv_yellow: bool = True,
    use_brightness: bool = True,
    use_green_blue_contrast: bool = True,
    use_morphology: bool = True,
    morphology_kernel_size: int = 5,
) -> np.ndarray:
    """
    Crea un mapa de calor candidato para exudados usando brillo, color amarillento
    y contraste verde/azul. Salida RGB uint8 HxWx3.
    """
    image = ensure_rgb_array(image).astype(np.uint8)

    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    score = np.zeros(image.shape[:2], dtype=np.float32)

    if use_hsv_yellow:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        yellow_mask = (
            (h >= 10) & (h <= 45) &
            (s >= 35) &
            (v >= 90)
        ).astype(np.float32)

        yellow_score = yellow_mask * (v / 255.0)
        score += 0.40 * yellow_score

    if use_brightness:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bright_score = gray / 255.0
        score += 0.30 * bright_score

    if use_green_blue_contrast:
        yellow_like = ((r + g) / 2.0) - b
        gb_contrast = g - b

        yellow_like = _scale_to_uint8(yellow_like).astype(np.float32) / 255.0
        gb_contrast = _scale_to_uint8(gb_contrast).astype(np.float32) / 255.0

        score += 0.20 * yellow_like
        score += 0.10 * gb_contrast

    score = np.clip(score, 0.0, 1.0)
    score_u8 = np.clip(score * 255.0, 0, 255).astype(np.uint8)

    if use_morphology:
        k = max(3, int(morphology_kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        score_u8 = cv2.morphologyEx(score_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        score_u8 = cv2.morphologyEx(score_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    score_u8 = cv2.GaussianBlur(score_u8, (3, 3), 0)

    return cv2.merge([score_u8, score_u8, score_u8]).astype(np.uint8)


# ==============================================================================
# PREPROCESAMIENTO POR RAMA
# ==============================================================================

def preprocess_branch_numpy(
    image: ImageInput,
    input_mode: str = "rgb",
    image_size: int = 224,
    preprocessing_cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Genera la imagen preprocesada para una rama específica.
    Salida: RGB uint8 HxWx3.
    """
    image_rgb = ensure_rgb_array(image)
    input_mode = (input_mode or "rgb").lower()

    cfg = _get_mode_cfg(preprocessing_cfg, input_mode)

    if input_mode in {"rgb", "rgb_highres", "inception_rgb", "retfound_rgb", "retclip_rgb"}:
        processed = image_rgb

    elif input_mode == "clahe_green":
        processed = make_clahe_green_image(
            image_rgb,
            clip_limit=cfg.get("clahe_clip_limit", 2.0),
            tile_grid_size=cfg.get("clahe_tile_grid_size", 8),
        )

    elif input_mode == "rgb_clahe":
        processed = apply_clahe_rgb_luminance(
            image_rgb,
            clip_limit=cfg.get("clahe_clip_limit", 2.0),
            tile_grid_size=cfg.get("clahe_tile_grid_size", 8),
        )

    elif input_mode == "candidate_mask":
        mask = create_candidate_mask(
            image_rgb,
            use_green_channel=cfg.get("use_green_channel", True),
            use_clahe=cfg.get("use_clahe", True),
            use_tophat=cfg.get("use_tophat", True),
            threshold_method=cfg.get("threshold_method", "otsu"),
            morphology_kernel_size=cfg.get("morphology_kernel_size", 5),
        )
        processed = cv2.merge([mask, mask, mask]).astype(np.uint8)

    elif input_mode == "exudate_map":
        processed = create_exudate_map(
            image_rgb,
            use_hsv_yellow=cfg.get("use_hsv_yellow", True),
            use_brightness=cfg.get("use_brightness", True),
            use_green_blue_contrast=cfg.get("use_green_blue_contrast", True),
            use_morphology=cfg.get("use_morphology", True),
            morphology_kernel_size=cfg.get("morphology_kernel_size", 5),
        )

    else:
        processed = image_rgb

    processed = resize_rgb(processed, int(image_size))
    return processed.astype(np.uint8)


def preprocess_branch_tensor(
    image: ImageInput,
    input_mode: str = "rgb",
    image_size: int = 224,
    preprocessing_cfg: Optional[Dict[str, Any]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Genera tensor CHW float32 para una rama específica.
    """
    processed = preprocess_branch_numpy(
        image=image,
        input_mode=input_mode,
        image_size=image_size,
        preprocessing_cfg=preprocessing_cfg,
    )
    return to_tensor_chw(processed, normalize=normalize)


def preprocess_multiple_branches(
    image: ImageInput,
    branches: Iterable[Dict[str, Any]],
    preprocessing_cfg: Optional[Dict[str, Any]] = None,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Preprocesa una imagen para varias ramas definidas en config.yaml.
    Devuelve: {nombre_modelo: tensor}
    """
    output: Dict[str, torch.Tensor] = {}

    for branch in branches:
        if not branch.get("enabled", True):
            continue

        name = branch["name"]
        input_mode = branch.get("input_mode", "rgb")
        image_size = int(branch.get("image_size", 224))

        output[name] = preprocess_branch_tensor(
            image=image,
            input_mode=input_mode,
            image_size=image_size,
            preprocessing_cfg=preprocessing_cfg,
            normalize=normalize,
        )

    return output


# ==============================================================================
# RADIOMICS / CARACTERÍSTICAS BIOMÉDICAS
# ==============================================================================

def _binary_candidate_mask_for_features(image: np.ndarray) -> np.ndarray:
    mask = create_candidate_mask(
        image,
        use_green_channel=True,
        use_clahe=True,
        use_tophat=True,
        threshold_method="otsu",
        morphology_kernel_size=5,
    )
    return (mask > 0).astype(np.uint8)


def _color_features(image: np.ndarray) -> OrderedDict:
    features = OrderedDict()

    image = ensure_rgb_array(image)
    image_f = image.astype(np.float32)

    r = image_f[:, :, 0]
    g = image_f[:, :, 1]
    b = image_f[:, :, 2]

    for name, channel in [("r", r), ("g", g), ("b", b)]:
        features[f"color_rgb_mean_{name}"] = _safe_float(np.mean(channel))
        features[f"color_rgb_std_{name}"] = _safe_float(np.std(channel))
        features[f"color_rgb_p90_{name}"] = _safe_float(np.percentile(channel, 90))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    for name, channel in [("h", h), ("s", s), ("v", v)]:
        features[f"color_hsv_mean_{name}"] = _safe_float(np.mean(channel))
        features[f"color_hsv_std_{name}"] = _safe_float(np.std(channel))
        features[f"color_hsv_p90_{name}"] = _safe_float(np.percentile(channel, 90))

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, lab_b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    for name, channel in [("l", l), ("a", a), ("b", lab_b)]:
        features[f"color_lab_mean_{name}"] = _safe_float(np.mean(channel))
        features[f"color_lab_std_{name}"] = _safe_float(np.std(channel))

    yellow_mask = (
        (h >= 10) & (h <= 45) &
        (s >= 35) &
        (v >= 90)
    )

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright_threshold = max(180.0, float(np.percentile(gray, 90)))
    bright_mask = gray >= bright_threshold

    green_blue = g - b
    yellow_like = ((r + g) / 2.0) - b

    features["color_yellow_pixel_ratio"] = _safe_float(np.mean(yellow_mask))
    features["color_bright_pixel_ratio"] = _safe_float(np.mean(bright_mask))
    features["color_green_blue_contrast_mean"] = _safe_float(np.mean(green_blue))
    features["color_green_blue_contrast_std"] = _safe_float(np.std(green_blue))
    features["color_yellow_like_mean"] = _safe_float(np.mean(yellow_like))
    features["color_yellow_like_std"] = _safe_float(np.std(yellow_like))

    return features


def _texture_features(image: np.ndarray) -> OrderedDict:
    features = OrderedDict()

    image = ensure_rgb_array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    gray_f = gray.astype(np.float32)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(prob * np.log2(prob + 1e-8))

    features["texture_gray_mean"] = _safe_float(np.mean(gray_f))
    features["texture_gray_std"] = _safe_float(np.std(gray_f))
    features["texture_entropy"] = _safe_float(entropy)
    features["texture_laplacian_var"] = _safe_float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if local_binary_pattern is not None:
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)

        for i, value in enumerate(lbp_hist):
            features[f"texture_lbp_bin_{i}"] = _safe_float(value)
    else:
        for i in range(10):
            features[f"texture_lbp_bin_{i}"] = 0.0

    if gabor is not None:
        for angle_name, theta in [
            ("0", 0.0),
            ("45", np.pi / 4),
            ("90", np.pi / 2),
            ("135", 3 * np.pi / 4),
        ]:
            try:
                real, imag = gabor(gray_f / 255.0, frequency=0.2, theta=theta)
                energy = np.sqrt(real ** 2 + imag ** 2).mean()
                features[f"texture_gabor_energy_{angle_name}"] = _safe_float(energy)
            except Exception:
                features[f"texture_gabor_energy_{angle_name}"] = 0.0
    else:
        for angle_name in ["0", "45", "90", "135"]:
            features[f"texture_gabor_energy_{angle_name}"] = 0.0

    return features


def _morphology_features(image: np.ndarray, mask: np.ndarray) -> OrderedDict:
    features = OrderedDict()

    image = ensure_rgb_array(image)
    h, w = mask.shape[:2]
    total_area = float(h * w)

    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    areas = []
    valid_labels = []

    min_area = max(3, int(total_area * 0.00002))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            areas.append(area)
            valid_labels.append(label_id)

    areas_np = np.array(areas, dtype=np.float32) if areas else np.array([], dtype=np.float32)

    features["morph_num_candidate_regions"] = _safe_float(len(areas))
    features["morph_total_candidate_area"] = _safe_float(areas_np.sum() if areas_np.size else 0.0)
    features["morph_candidate_area_ratio"] = _safe_float((areas_np.sum() / total_area) if areas_np.size else 0.0)
    features["morph_mean_candidate_area"] = _safe_float(areas_np.mean() if areas_np.size else 0.0)
    features["morph_max_candidate_area"] = _safe_float(areas_np.max() if areas_np.size else 0.0)

    candidate_pixels = mask_u8 > 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if np.any(candidate_pixels):
        features["morph_mean_candidate_intensity"] = _safe_float(gray[candidate_pixels].mean())
        features["morph_max_candidate_intensity"] = _safe_float(gray[candidate_pixels].max())
    else:
        features["morph_mean_candidate_intensity"] = 0.0
        features["morph_max_candidate_intensity"] = 0.0

    contours, _ = cv2.findContours(
        (mask_u8 * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    compactness_values = []
    solidity_values = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area <= 0 or perimeter <= 0:
            continue

        compactness = (4.0 * np.pi * area) / (perimeter ** 2 + 1e-8)
        compactness_values.append(compactness)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity_values.append(area / hull_area)

    features["morph_mean_compactness"] = _safe_float(np.mean(compactness_values) if compactness_values else 0.0)
    features["morph_mean_solidity"] = _safe_float(np.mean(solidity_values) if solidity_values else 0.0)

    if regionprops is not None:
        try:
            props = regionprops(labels)
            ecc_values = [
                p.eccentricity
                for p in props
                if p.area >= min_area
            ]
            features["morph_mean_eccentricity"] = _safe_float(np.mean(ecc_values) if ecc_values else 0.0)
        except Exception:
            features["morph_mean_eccentricity"] = 0.0
    else:
        features["morph_mean_eccentricity"] = 0.0

    return features


def _spatial_features(mask: np.ndarray) -> OrderedDict:
    features = OrderedDict()

    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape[:2]
    total_pixels = float(h * w)

    yy, xx = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) + 1e-8

    central_region = distance <= (0.30 * max(h, w))
    peripheral_region = distance >= (0.55 * max(h, w))

    lesion_pixels = mask_u8 > 0
    lesion_count = float(np.sum(lesion_pixels))

    if lesion_count <= 0:
        features["spatial_central_region_ratio"] = 0.0
        features["spatial_peripheral_region_ratio"] = 0.0
        features["spatial_superior_ratio"] = 0.0
        features["spatial_inferior_ratio"] = 0.0
        features["spatial_left_ratio"] = 0.0
        features["spatial_right_ratio"] = 0.0
        features["spatial_mean_distance_to_center"] = 0.0
        features["spatial_lesion_density"] = 0.0
        return features

    features["spatial_central_region_ratio"] = _safe_float(np.sum(lesion_pixels & central_region) / lesion_count)
    features["spatial_peripheral_region_ratio"] = _safe_float(np.sum(lesion_pixels & peripheral_region) / lesion_count)
    features["spatial_superior_ratio"] = _safe_float(np.sum(lesion_pixels & (yy < cy)) / lesion_count)
    features["spatial_inferior_ratio"] = _safe_float(np.sum(lesion_pixels & (yy >= cy)) / lesion_count)
    features["spatial_left_ratio"] = _safe_float(np.sum(lesion_pixels & (xx < cx)) / lesion_count)
    features["spatial_right_ratio"] = _safe_float(np.sum(lesion_pixels & (xx >= cx)) / lesion_count)
    features["spatial_mean_distance_to_center"] = _safe_float(np.mean(distance[lesion_pixels] / max_dist))
    features["spatial_lesion_density"] = _safe_float(lesion_count / total_pixels)

    return features


def extract_radiomics_features(
    image: ImageInput,
    image_size: int = 224,
    radiomics_cfg: Optional[Dict[str, Any]] = None,
) -> OrderedDict:
    """
    Extrae características cromáticas, texturales, morfológicas y espaciales.
    Devuelve un OrderedDict con longitud fija.
    """
    cfg = radiomics_cfg or {}

    image_rgb = ensure_rgb_array(image)
    image_rgb = resize_rgb(image_rgb, int(image_size))

    features = OrderedDict()

    if cfg.get("color_features", True):
        features.update(_color_features(image_rgb))

    if cfg.get("texture_features", True):
        features.update(_texture_features(image_rgb))

    mask = _binary_candidate_mask_for_features(image_rgb)

    if cfg.get("morphology_features", True):
        features.update(_morphology_features(image_rgb, mask))

    if cfg.get("spatial_features", True):
        features.update(_spatial_features(mask))

    return features


def extract_radiomics_vector(
    image: ImageInput,
    image_size: int = 224,
    radiomics_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, list[str]]:
    """
    Devuelve vector numpy y nombres de características.
    """
    features = extract_radiomics_features(
        image=image,
        image_size=image_size,
        radiomics_cfg=radiomics_cfg,
    )

    names = list(features.keys())
    values = np.array([_safe_float(features[name]) for name in names], dtype=np.float32)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return values, names


# ==============================================================================
# PRUEBA RÁPIDA
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prueba de preprocessing_branches.py")
    parser.add_argument("--image", type=str, required=False, default=None)
    parser.add_argument("--mode", type=str, default="exudate_map")
    parser.add_argument("--size", type=int, default=224)

    args = parser.parse_args()

    if args.image is None:
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(dummy, (256, 256), 180, (80, 55, 40), -1)
        cv2.circle(dummy, (300, 250), 15, (235, 220, 120), -1)
        image = dummy
    else:
        image = args.image

    processed = preprocess_branch_numpy(
        image=image,
        input_mode=args.mode,
        image_size=args.size,
    )

    tensor = preprocess_branch_tensor(
        image=image,
        input_mode=args.mode,
        image_size=args.size,
    )

    vector, names = extract_radiomics_vector(image=image, image_size=args.size)

    print("Modo:", args.mode)
    print("Imagen procesada:", processed.shape, processed.dtype, processed.min(), processed.max())
    print("Tensor:", tuple(tensor.shape), tensor.dtype)
    print("Radiomics:", vector.shape)
    print("Primeras características:", names[:10])