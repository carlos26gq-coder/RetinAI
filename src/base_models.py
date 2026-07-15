# ==============================================================================
# src/base_models.py
# RetinAI_MVP
# Modelos base para Ensemble Stacking Multimodal
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


try:
    import timm
except Exception:
    timm = None


try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None


# ==============================================================================
# UTILIDADES
# ==============================================================================

SUPPORTED_MODELS = [
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet50",
    "efficientnet_b3",
    "densenet121",
    "inception_v3",
    "resnet50_cbam",
    "unet_encoder",
    "mobilenetv3_exudate_map",
    "mobilenet_v3_large",
    "retfound",
    "retclip",
]


def list_supported_models() -> List[str]:
    return SUPPORTED_MODELS.copy()


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _freeze_all_except_classifier(model: nn.Module) -> None:
    """
    Congela todo excepto capas típicas de clasificación.
    Útil para modelos timm.
    """
    for name, param in model.named_parameters():
        trainable = any(
            key in name.lower()
            for key in ["classifier", "fc", "head", "last_linear"]
        )
        param.requires_grad = trainable


def _get_cfg_value(model_cfg: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if model_cfg is None:
        return default
    return model_cfg.get(key, default)


def _load_torchvision_weights(model_name: str, pretrained: bool):
    """
    Carga pesos torchvision de forma compatible con versiones nuevas.
    """
    if not pretrained:
        return None

    name = model_name.lower()

    try:
        if name == "mobilenet_v2":
            return models.MobileNet_V2_Weights.DEFAULT

        if name == "mobilenet_v3_large":
            return models.MobileNet_V3_Large_Weights.DEFAULT

        if name == "efficientnet_b0":
            return models.EfficientNet_B0_Weights.DEFAULT

        if name == "densenet121":
            return models.DenseNet121_Weights.DEFAULT

        if name == "resnet50":
            return models.ResNet50_Weights.DEFAULT

        if name == "inception_v3":
            return models.Inception_V3_Weights.DEFAULT

    except Exception:
        return None

    return None


# ==============================================================================
# WRAPPERS BASE
# ==============================================================================

class FeatureClassifier(nn.Module):
    """
    Wrapper genérico:
    feature_extractor -> pooling -> classifier
    Retorna logits de tamaño [B, num_classes].
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        in_features: int,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]

        if x.ndim == 4:
            x = self.pool(x)
            x = torch.flatten(x, 1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


# ==============================================================================
# CBAM: CONVOLUTIONAL BLOCK ATTENTION MODULE
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention))

        return x * attention


class CBAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResNet50CBAMClassifier(nn.Module):
    """
    ResNet50 + CBAM.
    Se usa como rama de atención espacial/canal.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = _load_torchvision_weights("resnet50", pretrained)
        backbone = models.resnet50(weights=weights)

        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.cbam = CBAM(channels=2048, reduction=16, spatial_kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

        if freeze_backbone:
            _set_requires_grad(self.feature_extractor, False)
            _set_requires_grad(self.cbam, True)
            _set_requires_grad(self.classifier, True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


# ==============================================================================
# U-NET ENCODER CLASSIFIER
# ==============================================================================

class UNetEncoderClassifier(nn.Module):
    """
    Usa solo el encoder de una arquitectura U-Net.
    Entrada esperada: 3 canales.
    Salida: clasificación binaria.
    """

    def __init__(
        self,
        num_classes: int = 2,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if smp is None:
            raise ImportError(
                "No se encontró segmentation_models_pytorch. "
                "Instala con: pip install segmentation-models-pytorch"
            )

        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        out_channels = self.encoder.out_channels[-1]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_channels, num_classes),
        )

        if freeze_backbone:
            _set_requires_grad(self.encoder, False)
            _set_requires_grad(self.classifier, True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = features[-1]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


# ==============================================================================
# MODELOS TORCHVISION
# ==============================================================================

def build_mobilenet_v2(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = _load_torchvision_weights("mobilenet_v2", pretrained)
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        _set_requires_grad(model.features, False)
        _set_requires_grad(model.classifier, True)

    return model


def build_mobilenet_v3_large(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = _load_torchvision_weights("mobilenet_v3_large", pretrained)
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        _set_requires_grad(model.features, False)
        _set_requires_grad(model.classifier, True)

    return model


def build_efficientnet_b0(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = _load_torchvision_weights("efficientnet_b0", pretrained)
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        _set_requires_grad(model.features, False)
        _set_requires_grad(model.classifier, True)

    return model


def build_densenet121(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = _load_torchvision_weights("densenet121", pretrained)
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        _set_requires_grad(model.features, False)
        _set_requires_grad(model.classifier, True)

    return model


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = _load_torchvision_weights("resnet50", pretrained)
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model


def build_inception_v3(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = _load_torchvision_weights("inception_v3", pretrained)

    model = models.inception_v3(
        weights=weights,
        aux_logits=True if weights is not None else False,
    )

    # Evita que forward retorne InceptionOutputs durante entrenamiento.
    model.aux_logits = False
    model.AuxLogits = None

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model


# ==============================================================================
# MODELOS TIMM
# ==============================================================================

def build_timm_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if timm is None:
        raise ImportError(
            "No se encontró timm. Instala con: pip install timm"
        )

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        _freeze_all_except_classifier(model)

    return model


# ==============================================================================
# FOUNDATION MODELS PLACEHOLDER
# ==============================================================================

class FoundationModelNotIntegrated(nn.Module):
    """
    Placeholder controlado para evitar errores silenciosos.
    RETFound/RET-CLIP se activarán cuando integremos pesos y arquitectura.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.model_name} todavía no está integrado en base_models.py. "
            "Debe permanecer con enabled: false en config.yaml hasta implementar "
            "la carga oficial de pesos y arquitectura."
        )


# ==============================================================================
# FACTORY PRINCIPAL
# ==============================================================================

def build_model_from_config(
    model_cfg: Dict[str, Any],
    num_classes: int = 2,
) -> nn.Module:
    """
    Construye un modelo usando el bloque correspondiente de config.yaml.
    """
    name = str(model_cfg.get("name", "")).lower().strip()
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))

    if name == "mobilenet_v2":
        return build_mobilenet_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name in {"mobilenet_v3_large", "mobilenetv3_exudate_map"}:
        return build_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "efficientnet_b3":
        return build_timm_model(
            model_name="efficientnet_b3",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "densenet121":
        return build_densenet121(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "inception_v3":
        return build_inception_v3(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "resnet50_cbam":
        return ResNet50CBAMClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "unet_encoder":
        encoder_name = model_cfg.get("encoder_name", "resnet34")
        encoder_weights = model_cfg.get("encoder_weights", "imagenet")

        return UNetEncoderClassifier(
            num_classes=num_classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            freeze_backbone=freeze_backbone,
        )

    if name in {"retfound", "retclip"}:
        return FoundationModelNotIntegrated(model_name=name)

    raise ValueError(
        f"Modelo no soportado: {name}. "
        f"Modelos disponibles: {SUPPORTED_MODELS}"
    )


def build_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Constructor directo por nombre.
    Mantiene compatibilidad con scripts anteriores.
    """
    model_cfg = {
        "name": name,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
    }
    model_cfg.update(kwargs)

    return build_model_from_config(model_cfg, num_classes=num_classes)


def get_base_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Alias de compatibilidad.
    """
    return build_model(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


def create_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Alias de compatibilidad.
    """
    return build_model(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


def get_enabled_model_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Devuelve solo modelos habilitados desde config.yaml.
    """
    models_cfg = config.get("base_models", [])
    return [
        model_cfg
        for model_cfg in models_cfg
        if model_cfg.get("enabled", True)
    ]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def summarize_model(
    model: nn.Module,
    model_name: str = "model",
) -> Dict[str, Any]:
    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)

    return {
        "model_name": model_name,
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


# ==============================================================================
# PRUEBA RÁPIDA
# ==============================================================================

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config_path = Path("config.yaml")

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_cfgs = get_enabled_model_configs(cfg)

        print("Modelos habilitados:")
        for model_cfg in model_cfgs:
            print(" -", model_cfg["name"])
    else:
        model_cfgs = [
            {"name": "efficientnet_b0", "pretrained": False, "freeze_backbone": True},
            {"name": "densenet121", "pretrained": False, "freeze_backbone": True},
            {"name": "resnet50_cbam", "pretrained": False, "freeze_backbone": True},
            {"name": "unet_encoder", "pretrained": False, "freeze_backbone": True},
            {"name": "mobilenetv3_exudate_map", "pretrained": False, "freeze_backbone": True},
        ]

    for model_cfg in model_cfgs:
        if not model_cfg.get("enabled", True):
            continue

        name = model_cfg["name"]

        if name in {"retfound", "retclip"}:
            print(f"[SKIP] {name}: todavía no integrado.")
            continue

        print(f"\nConstruyendo: {name}")

        try:
            test_cfg = dict(model_cfg)
            test_cfg["pretrained"] = False

            model = build_model_from_config(test_cfg, num_classes=2)
            model.eval()

            image_size = int(test_cfg.get("image_size", 224))
            x = torch.randn(2, 3, image_size, image_size)

            with torch.no_grad():
                y = model(x)

            summary = summarize_model(model, name)

            print("Output:", tuple(y.shape))
            print("Total params:", summary["total_parameters"])
            print("Trainable params:", summary["trainable_parameters"])

        except Exception as exc:
            print(f"ERROR en {name}: {exc}")