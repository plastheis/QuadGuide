"""NanoTrack V3 adapter: vendored MobileNetV3-small-v3 backbone + AdjustLayer
neck + DepthwiseBAN anchor-free head, registered for the `nanotrack` manifest.
The module is self-contained so the original repo need not be installed.

Reference: HonglinChu/SiamTrackers/NanoTrack configv3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_lib.registry import Adapter, register

# ── backbone: mobilenetv3_small_v3 ──────────────────────────────────────────


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.hard_sigmoid = nn.Hardsigmoid(inplace=inplace)

    def forward(self, x):
        return self.hard_sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.hard_swish = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.hard_swish(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish(),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish(),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0):
        super().__init__()
        self.cfgs = cfgs
        assert mode in ["large", "small"]
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(InvertedResidual(input_channel, exp_size, output_channel,
                                           k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.features(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_small_v3(**kwargs):
    cfgs = [
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],  # s=2 -> s=1 in V3
    ]
    return MobileNetV3(cfgs, mode="small", **kwargs)


# ── neck: AdjustLayer ────────────────────────────────────────────────────────


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.downsample(x)
        # NOTE: centre-crop removed for V3 compatibility — the 127×127 exemplar
        # yields an 8×8 backbone feature which pixelwise correlation handles
        # directly (64-channel output matching CAModule(channels=64)). Removing
        # the crop changes no state_dict keys; AdjustLayer owns only
        # downsample.{0,1}.* in the checkpoint.
        return x


# ── head: DepthwiseBAN (V3) ──────────────────────────────────────────────────


def _xcorr_depthwise(x, kernel):
    """depthwise cross correlation"""
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, padding=1, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def _xcorr_pixelwise(x, kernel):
    """Pixel-wise correlation (matrix multiplication)."""
    b, c, h, w = x.size()
    kernel_mat = kernel.view((b, c, -1)).transpose(1, 2)  # (b, hz*wz, c)
    x_mat = x.view((b, c, -1))  # (b, c, hx*wx)
    return torch.matmul(kernel_mat, x_mat).view((b, -1, h, w))


class CAModule(nn.Module):
    """Channel attention module."""
    def __init__(self, channels=64, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = _xcorr_depthwise(search, kernel)
        return feature


class PixelwiseXCorr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        channels = 64
        self.CA_layer = CAModule(channels)
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = _xcorr_pixelwise(search, kernel)
        corr = self.CA_layer(feature)
        corr = self.conv(corr)
        return corr


class DepthwiseBAN(nn.Module):
    def __init__(self, in_channels=96, out_channels=96, weighted=False):
        super().__init__()
        self.corr_dw_reg = DepthwiseXCorr(in_channels, out_channels)
        self.corr_pw_reg = PixelwiseXCorr(in_channels, out_channels)
        self.corr_dw_cls = DepthwiseXCorr(in_channels, out_channels)
        self.corr_pw_cls = PixelwiseXCorr(in_channels, out_channels)

        cls_tower = []
        bbox_tower = []
        for _ in range(6):
            cls_tower.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(in_channels),
            ])
            bbox_tower.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(in_channels),
            ])
        self.add_module("cls_tower", nn.Sequential(*cls_tower))
        self.add_module("bbox_tower", nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0),
        )
        self.bbox_pred = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, padding=0),
        )
        self.down_reg = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, kernel_size=1, stride=1, padding=0),
        )
        self.down_cls = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, kernel_size=1, stride=1, padding=0),
        )

    @staticmethod
    def crop(x):
        if x.size(3) > 4:
            off = 2
            hi = off + 4
            x = x[:, :, off:hi, off:hi]
        return x

    def forward(self, z_f, x_f):
        crop_z_f = self.crop(z_f)
        x_pw_reg = self.corr_pw_reg(z_f, x_f)
        x_pw_cls = self.corr_pw_cls(z_f, x_f)
        x_dw_reg = self.corr_dw_reg(crop_z_f, x_f)
        x_dw_cls = self.corr_dw_cls(crop_z_f, x_f)
        x_reg = self.down_reg(torch.cat((x_pw_reg, x_dw_reg), 1))
        x_cls = self.down_cls(torch.cat((x_pw_cls, x_dw_cls), 1))
        cls_tower = self.cls_tower(x_cls)
        logits = self.cls_logits(cls_tower)
        bbox_tower = self.bbox_tower(x_reg)
        bbox_reg = self.bbox_pred(bbox_tower)
        bbox_reg = torch.exp(bbox_reg)
        return logits, bbox_reg


# ── Net wrapper + build + register ───────────────────────────────────────────


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = mobilenetv3_small_v3()
        self.neck = AdjustLayer(in_channels=96, out_channels=96)
        self.head = DepthwiseBAN(in_channels=96, out_channels=96)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        # forward arg order (z=exemplar, x=search) MUST match manifest io.inputs order.
        zf = self.neck(self.backbone(z))
        xf = self.neck(self.backbone(x))
        cls, loc = self.head(zf, xf)            # cls first, loc second
        return cls, loc


def build(checkpoint: str) -> Net:
    """Load a checkpoint state_dict into Net (strict=True) and return it in eval mode."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:   # tolerate wrapped checkpoints
        sd = sd["state_dict"]
    net = Net()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


register(Adapter(
    name="nanotrack",
    build=build,
    dynamic_axes={
        "cls": {2: "S", 3: "S"},
        "loc": {2: "S", 3: "S"},
    },
))
