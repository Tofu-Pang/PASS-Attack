import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


STANDARD_MODELS = (
    "resnet50",
    "densenet121",
    "vgg16",
    "inception_v3",
)
DIVERSE_MODELS = (
    "inception_v4",
    "inception_resnet_v2",
    "xception",
    "vit_base_patch16_224",
)
MODEL_NAMES = STANDARD_MODELS + DIVERSE_MODELS
TIMM_NAMES = {
    "inception_v4": "inception_v4",
    "inception_resnet_v2": "inception_resnet_v2",
    "xception": "legacy_xception",
    "vit_base_patch16_224": "vit_base_patch16_224",
}
INPUT_SIZES = {
    "inception_v4": 299,
    "inception_resnet_v2": 299,
    "xception": 299,
    "vit_base_patch16_224": 224,
}


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, inputs):
        return (inputs - self.mean) / self.std


class ImageNetModel(nn.Module):
    def __init__(self, backbone, mean, std, input_size=None, interpolation="bilinear"):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.backbone = backbone
        self.input_size = input_size
        self.interpolation = interpolation

    def forward(self, inputs):
        if self.input_size is not None and inputs.shape[-2:] != (self.input_size, self.input_size):
            options = {
                "size": (self.input_size, self.input_size),
                "mode": self.interpolation,
            }
            if self.interpolation in ("bilinear", "bicubic"):
                options.update(align_corners=False, antialias=True)
            inputs = F.interpolate(inputs, **options)
        return self.backbone(self.normalize(inputs))


def _build_torchvision_backbone(name):
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if name == "densenet121":
        return models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if name == "vgg16":
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if name == "inception_v3":
        backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        backbone.transform_input = False
        return backbone
    raise ValueError(f"Unsupported model: {name}")


def _build_timm_backbone(name):
    try:
        import timm
    except ImportError as error:
        raise RuntimeError("timm is required for diverse architectures") from error
    backbone = timm.create_model(TIMM_NAMES[name], pretrained=True)
    data_config = timm.data.resolve_model_data_config(backbone)
    return backbone, data_config


def build_model(name, device):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {name}")
    try:
        if name in STANDARD_MODELS:
            backbone = _build_torchvision_backbone(name)
            input_size = None
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            interpolation = "bilinear"
        else:
            backbone, data_config = _build_timm_backbone(name)
            input_size = int(data_config["input_size"][-1])
            if input_size != INPUT_SIZES[name]:
                raise RuntimeError(f"Unexpected input size for {name}")
            mean = data_config["mean"]
            std = data_config["std"]
            interpolation = data_config["interpolation"]
    except Exception as error:
        raise RuntimeError(f"Unable to load pretrained weights for {name}") from error
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model = ImageNetModel(backbone, mean, std, input_size, interpolation).to(device)
    model.eval()
    return model


def resolve_feature_layer(model, name, depth):
    backbone = model.backbone
    depth = int(depth)
    if depth not in (1, 2, 3, 4):
        raise ValueError("feature_layer must be 1, 2, 3, or 4")
    if name == "resnet50":
        return getattr(backbone, f"layer{depth}")[-1]
    if name == "densenet121":
        return getattr(backbone.features, f"denseblock{depth}")
    if name == "vgg16":
        return backbone.features[{1: 8, 2: 15, 3: 22, 4: 29}[depth]]
    if name == "inception_v3":
        return getattr(
            backbone,
            {1: "Conv2d_4a_3x3", 2: "Mixed_5d", 3: "Mixed_6e", 4: "Mixed_7c"}[depth],
        )
    if name == "xception":
        blocks = [
            backbone.block1,
            backbone.block2,
            backbone.block3,
            backbone.block4,
            backbone.block5,
            backbone.block6,
            backbone.block7,
            backbone.block8,
            backbone.block9,
            backbone.block10,
            backbone.block11,
            backbone.block12,
        ]
        return blocks[{1: 2, 2: 5, 3: 8, 4: 11}[depth]]
    if name == "inception_v4":
        return backbone.features[{1: 2, 2: 4, 3: 6, 4: 8}[depth]]
    if name == "inception_resnet_v2":
        return {
            1: backbone.mixed_5b,
            2: backbone.mixed_6a,
            3: backbone.mixed_7a,
            4: backbone.block8,
        }[depth]
    if name == "vit_base_patch16_224":
        return backbone.blocks[{1: 4, 2: 8, 3: 11, 4: 11}[depth]]
    raise ValueError(f"Unsupported model: {name}")
