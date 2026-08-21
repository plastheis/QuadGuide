"""SiamFC adapter: vendored huanglianghua/siamfc-pytorch AlexNetV1 backbone + batch-1
cross-correlation head, registered for the `siamfc_generic` manifest. The module is
self-contained so the original repo need not be installed."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_lib.registry import Adapter, register


def _bn(c: int) -> nn.BatchNorm2d:
    # huanglianghua uses eps=1e-6, momentum=0.05; eval-time eps affects numerics.
    return nn.BatchNorm2d(c, eps=1e-6, momentum=0.05)


class AlexNetV1(nn.Module):
    """Backbone matching siamfc-pytorch state_dict keys (backbone.conv1..conv5).
    Conv layers keep their default bias (the checkpoint stores conv biases)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2), _bn(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2), _bn(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1), _bn(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2), _bn(384), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class SiamFCHead(nn.Module):
    """Batch-1 cross-correlation: conv2d(search_feat, exemplar_feat) * out_scale.
    Numerically identical to the repo's _fast_xcorr for batch size 1."""

    def __init__(self, out_scale: float = 0.001) -> None:
        super().__init__()
        self.out_scale = out_scale

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, z) * self.out_scale


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AlexNetV1()
        self.head = SiamFCHead()

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # forward arg order (z=exemplar, x=search) MUST match manifest io.inputs order.
        return self.head(self.backbone(z), self.backbone(x))


def build(checkpoint: str) -> Net:
    """Load a checkpoint state_dict into Net (strict=True) and return it in eval mode."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:   # tolerate wrapped checkpoints
        sd = sd["state_dict"]
    net = Net()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


register(Adapter(name="siamfc_generic", build=build))
