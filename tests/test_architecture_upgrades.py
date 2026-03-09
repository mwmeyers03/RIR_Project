"""Tests for architectural upgrades: SirenLayer, EarlyReflectionNet,
MultiResolutionSTFTLoss, autograd-based physics residuals, and DataLoader I/O."""

import sys
from pathlib import Path

import torch

# Ensure src/ layout is importable when tests run from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rir_project.loss import (
    MultiResolutionSTFTLoss,
    continuity_residual,
    momentum_residual,
)
from rir_project.models import EarlyReflectionNet, SirenLayer
from rir_project.synthesis import EarlyReflections


def test_siren_layer_output_shape() -> None:
    layer = SirenLayer(24, 64, is_first=True)
    x = torch.randn(4, 24)
    y = layer(x)
    assert y.shape == (4, 64)


def test_siren_layer_output_bounded() -> None:
    """SIREN output must be bounded in [-1, 1] due to sine activation."""
    layer = SirenLayer(8, 16, omega_0=30.0, is_first=True)
    x = torch.randn(32, 8)
    y = layer(x)
    assert y.min().item() >= -1.0
    assert y.max().item() <= 1.0


def test_early_reflection_net_shape_preserved() -> None:
    net = EarlyReflectionNet(n_taps=43)
    x = torch.randn(4, 256)
    y = net(x)
    assert y.shape == x.shape


def test_early_reflection_net_gradients_flow() -> None:
    net = EarlyReflectionNet(n_taps=43)
    x = torch.randn(4, 256, requires_grad=True)
    y = net(x)
    y.sum().backward()
    assert x.grad is not None
    assert net.coeffs.grad is not None


def test_early_reflections_synthesis_shape() -> None:
    er = EarlyReflections(n_taps=43)
    x = torch.randn(4, 256)
    y = er(x)
    assert y.shape == x.shape


def test_mr_stft_loss_positive() -> None:
    loss_fn = MultiResolutionSTFTLoss(window_lengths=[512, 1024, 2048])
    pred = torch.randn(2, 4096)
    target = torch.randn(2, 4096)
    loss = loss_fn(pred, target)
    assert loss.item() > 0


def test_mr_stft_loss_zero_for_identical() -> None:
    loss_fn = MultiResolutionSTFTLoss(window_lengths=[512])
    signal = torch.randn(2, 2048)
    loss = loss_fn(signal, signal.clone())
    assert loss.item() < 1e-4


def test_mr_stft_loss_backward() -> None:
    loss_fn = MultiResolutionSTFTLoss(window_lengths=[512])
    pred = torch.randn(2, 2048, requires_grad=True)
    target = torch.randn(2, 2048)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None


def test_continuity_residual_autograd() -> None:
    pred = torch.randn(2, 16, 6, requires_grad=True)
    res = continuity_residual(pred)
    assert res.item() > 0
    res.backward()
    assert pred.grad is not None


def test_momentum_residual_autograd() -> None:
    pred = torch.randn(2, 16, 6, requires_grad=True)
    res = momentum_residual(pred)
    assert res.item() > 0
    res.backward()
    assert pred.grad is not None


def test_continuity_residual_finite_diff_fallback() -> None:
    """When pred has no grad_fn, finite-diff fallback should work."""
    with torch.no_grad():
        pred = torch.randn(2, 16, 6)
    res = continuity_residual(pred)
    assert res.item() > 0


def test_momentum_residual_finite_diff_fallback() -> None:
    with torch.no_grad():
        pred = torch.randn(2, 16, 6)
    res = momentum_residual(pred)
    assert res.item() > 0
