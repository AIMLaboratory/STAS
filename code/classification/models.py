import torch
import torch.nn as nn
from config import Config
import monai.networks.nets as monai_nets


# =========================================================
# 1. 3D Models
# =========================================================
class ResNet18_3D(nn.Module):
    """3D ResNet18 for T-stage classification"""
    def __init__(self, num_classes=2, in_channels=1, dropout=0.2, pretrained=False, **kwargs):
        super().__init__()

        self.model = monai_nets.ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type="B",
            widen_factor=1.0,
            num_classes=num_classes,
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ResNet34_3D(nn.Module):
    """3D ResNet34 for T-stage classification"""
    def __init__(self, num_classes=2, in_channels=1, dropout=0.2, pretrained=False, **kwargs):
        super().__init__()

        self.model = monai_nets.ResNet(
            block="basic",
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type="B",
            widen_factor=1.0,
            num_classes=num_classes,
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class DenseNet121_3D(nn.Module):
    """
    3D DenseNet121 adapted for shallow-depth volumes like (24, 400, 400).
    Key idea:
    - downsample in XY
    - avoid aggressive downsampling in Z
    """
    def __init__(self, num_classes=2, in_channels=1, dropout=0.2, pretrained=False, **kwargs):
        super().__init__()
        self.backbone = monai_nets.DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            pretrained=False,
            dropout_prob=dropout
        )

        # stem conv: stride (1,2,2) to preserve Z resolution
        self.backbone.features.conv0 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )

        # stem pool: only downsample in XY
        self.backbone.features.pool0 = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

        # transition pools: replace AvgPool3d to XY only
        for trans_name in ["transition1", "transition2", "transition3"]:
            if hasattr(self.backbone.features, trans_name):
                trans = getattr(self.backbone.features, trans_name)
                self._replace_avgpool3d_to_xy_only(trans)

    def _replace_avgpool3d_to_xy_only(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.AvgPool3d):
                setattr(
                    module,
                    name,
                    nn.AvgPool3d(
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2)
                    )
                )
            else:
                self._replace_avgpool3d_to_xy_only(child)

    def forward(self, x):
        return self.backbone(x)


class Med3DResNet(nn.Module):
    """MONAI-based Med3D-style ResNet"""
    def __init__(self, num_classes=2, in_channels=1, dropout=0.2, pretrained=False, **kwargs):
        super().__init__()

        self.model = monai_nets.ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type="B",
            widen_factor=1.0,
            num_classes=num_classes
        )

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# 2. Model Factory
# =========================================================
def create_model(model_name=None, num_classes=None, in_channels=None, **kwargs):
    if model_name is None:
        model_name = Config.model_name
    if num_classes is None:
        num_classes = Config.get_num_classes()
    if in_channels is None:
        in_channels = Config.get_num_input_channels()

    model_configs = {
        "resnet18_3d": {
            "class": ResNet18_3D,
            "args": {"num_classes": num_classes, "in_channels": in_channels,
                     "dropout": 0.2, "pretrained": False}
        },
        "resnet34_3d": {
            "class": ResNet34_3D,
            "args": {"num_classes": num_classes, "in_channels": in_channels,
                     "dropout": 0.2, "pretrained": False}
        },
        "densenet121_3d": {
            "class": DenseNet121_3D,
            "args": {"num_classes": num_classes, "in_channels": in_channels,
                     "dropout": 0.2, "pretrained": False}
        },
        "med3d": {
            "class": Med3DResNet,
            "args": {"num_classes": num_classes, "in_channels": in_channels,
                     "dropout": 0.2, "pretrained": False}
        },
        "Med3DResNet": {
            "class": Med3DResNet,
            "args": {"num_classes": num_classes, "in_channels": in_channels,
                     "dropout": 0.2, "pretrained": False}
        },
    }

    if model_name not in model_configs:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(model_configs.keys())}"
        )

    config = model_configs[model_name]
    base_args = config["args"].copy()

    model_config = Config.get_model_config(model_name) if hasattr(Config, "get_model_config") else {}
    final_args = {**base_args, **model_config, **kwargs}

    # Remove training-specific parameters
    for param in ["batch_size", "learning_rate", "dimension"]:
        final_args.pop(param, None)

    final_args["num_classes"] = num_classes
    final_args["in_channels"] = in_channels

    model = config["class"](**final_args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("Model Created")
    print("=" * 70)
    print(f"Model name           : {model_name}")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Number of classes    : {num_classes}")
    print(f"Input channels       : {in_channels}")
    print(f"Input mode           : {Config.input_mode}")
    print(f"Label mode           : {Config.label_mode}")
    print(f"Class names          : {Config.get_class_names()}")
    print("=" * 70)

    return model


# =========================================================
# 3. Debug / Inspection
# =========================================================
def inspect_model(model_name=None):
    if model_name is None:
        model_name = Config.model_name
    model = create_model(model_name=model_name)
    model.eval()
    batch_size = 2
    channels = Config.get_num_input_channels()
    d, h, w = Config.input_size_3d
    x = torch.randn(batch_size, channels, d, h, w)
    with torch.no_grad():
        y = model(x)
    print("\n" + "=" * 70)
    print("Model Inspection")
    print("=" * 70)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")
    print("=" * 70)
    return model


if __name__ == "__main__":
    inspect_model()