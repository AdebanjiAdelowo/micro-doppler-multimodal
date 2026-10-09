"""Neural classifiers for (B, C, n_doppler, n_time) spectrogram inputs.

Every encoder exposes ``feat_dim`` and ``features(x) -> (B, feat_dim)`` so it
can be reused inside the fusion models at the bottom of the file.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _Encoder(nn.Module):
    feat_dim: int

    def features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x)))


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class LightCNN(_Encoder):
    """Four conv-BN-ReLU-pool stages and global average pooling (~0.1 M params)."""

    def __init__(self, n_classes: int, in_ch: int = 1, widths=(16, 32, 64, 96), dropout: float = 0.2):
        super().__init__()
        chans = (in_ch, *widths)
        self.body = nn.Sequential(*[_conv_block(a, b) for a, b in zip(chans[:-1], chans[1:], strict=True)])
        self.feat_dim = widths[-1]
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feat_dim, n_classes)

    def feature_map(self, x):
        return self.body(x)

    def features(self, x):
        return self.body(x).mean(dim=(-2, -1))


class _BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + (x if self.short is None else self.short(x)))


class ResNetSmall(_Encoder):
    """ResNet-10-style network: one residual basic block per stage (~1.2 M params)."""

    def __init__(self, n_classes: int, in_ch: int = 1, widths=(24, 48, 96, 192), dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)
        )
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(_BasicBlock(cin, w, 1 if i == 0 else 2))
            cin = w
        self.body = nn.Sequential(*blocks)
        self.feat_dim = widths[-1]
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feat_dim, n_classes)

    def feature_map(self, x):
        return self.body(self.stem(x))

    def features(self, x):
        return self.feature_map(x).mean(dim=(-2, -1))


class _TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class TemporalCNN(_Encoder):
    """Dilated 1-D convolutions over time; Doppler bins are input channels.

    Unlike the 2-D CNNs this model is not translation-equivariant along
    Doppler: each Doppler bin has its own weights.
    """

    def __init__(self, n_classes: int, in_ch: int = 1, n_doppler: int = 128, width: int = 64,
                 dilations=(1, 2, 4, 8, 16), dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(in_ch * n_doppler, width, 1, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        self.body = nn.Sequential(*[_TCNBlock(width, d, dropout) for d in dilations])
        self.feat_dim = width
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)

    def features(self, x):
        return self.body(self.proj(x.flatten(1, 2))).mean(dim=-1)


class SpecTransformer(_Encoder):
    """Transformer encoder over time frames (one token per STFT column)."""

    def __init__(self, n_classes: int, in_ch: int = 1, n_doppler: int = 128, n_time: int = 128,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_ch * n_doppler, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_time, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.feat_dim = d_model
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def features(self, x):
        tok = self.embed(x.flatten(1, 2).transpose(1, 2)) + self.pos
        return self.norm(self.encoder(tok)).mean(dim=1)


MODELS = {
    "light_cnn": LightCNN,
    "resnet_small": ResNetSmall,
    "temporal_cnn": TemporalCNN,
    "transformer": SpecTransformer,
}


def build_model(name: str, n_classes: int, in_ch: int = 1, **kw) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](n_classes=n_classes, in_ch=in_ch, **kw)


# --------------------------------------------------------------------------
# Fusion of two aligned inputs (a, b) describing the SAME recording.
#
# These classes are modality-agnostic. In this repository they are only ever
# run on two radar-derived views of one recording (Doppler-time and
# range-time maps). They have not been run on radar + vision because no
# paired dataset is used here; see docs/DATASETS.md.
# --------------------------------------------------------------------------
def late_fusion(prob_a: np.ndarray, prob_b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    """Decision-level fusion: convex combination of class probabilities."""
    return weight_a * prob_a + (1.0 - weight_a) * prob_b


class EarlyFusion(nn.Module):
    """Input-level fusion: stack the two views as channels of one network."""

    def __init__(self, backbone: str, n_classes: int, **kw):
        super().__init__()
        self.net = build_model(backbone, n_classes, in_ch=2, **kw)

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1))


class FeatureFusion(nn.Module):
    """Intermediate fusion: concatenate the two encoders' embeddings."""

    def __init__(self, backbone: str, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.enc_a = build_model(backbone, n_classes)
        self.enc_b = build_model(backbone, n_classes)
        d = self.enc_a.feat_dim + self.enc_b.feat_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, n_classes))

    def forward(self, a, b):
        return self.head(torch.cat([self.enc_a.features(a), self.enc_b.features(b)], dim=1))


class CrossAttentionFusion(nn.Module):
    """Learned cross-modal fusion: each view's feature-map tokens attend to
    the other view's tokens; pooled results are concatenated."""

    def __init__(self, n_classes: int, backbone: str = "light_cnn", n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if backbone not in ("light_cnn", "resnet_small"):
            raise ValueError("cross-attention fusion needs a 2-D CNN backbone")
        self.enc_a = build_model(backbone, n_classes)
        self.enc_b = build_model(backbone, n_classes)
        d = self.enc_a.feat_dim
        self.a_from_b = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.b_from_a = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_a, self.norm_b = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, n_classes))

    @staticmethod
    def _tokens(fmap):
        return fmap.flatten(2).transpose(1, 2)

    def forward(self, a, b):
        ta, tb = self._tokens(self.enc_a.feature_map(a)), self._tokens(self.enc_b.feature_map(b))
        fa = self.norm_a(ta + self.a_from_b(ta, tb, tb, need_weights=False)[0]).mean(dim=1)
        fb = self.norm_b(tb + self.b_from_a(tb, ta, ta, need_weights=False)[0]).mean(dim=1)
        return self.head(torch.cat([fa, fb], dim=1))


def build_fusion(kind: str, n_classes: int, backbone: str = "light_cnn") -> nn.Module:
    if kind == "early":
        return EarlyFusion(backbone, n_classes)
    if kind == "feature":
        return FeatureFusion(backbone, n_classes)
    if kind == "cross_attention":
        return CrossAttentionFusion(n_classes, backbone)
    raise KeyError(f"unknown fusion {kind!r}")
