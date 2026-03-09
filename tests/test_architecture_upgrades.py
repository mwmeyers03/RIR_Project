"""Tests for architectural upgrades: SirenLayer, EarlyReflectionNet,
MultiResolutionSTFTLoss, autograd-based physics residuals, and DataLoader I/O.
Also covers new features: SIRENCoordinateNet, CollocationPhysicsLoss,
acoustic residuals, MultibandSignStickyPhaseReconstructor, and UNetRefiner."""

import sys
from pathlib import Path

import torch

# Ensure src/ layout is importable when tests run from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rir_project.loss import (
    CollocationPhysicsLoss,
    MultiResolutionSTFTLoss,
    acoustic_continuity_residual,
    acoustic_momentum_residual,
    continuity_residual,
    momentum_residual,
)
from rir_project.models import EarlyReflectionNet, SIRENCoordinateNet, SirenLayer, UNetRefiner
from rir_project.synthesis import EarlyReflections, MultibandSignStickyPhaseReconstructor


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


# ---- New feature tests ----

def test_siren_coordinate_net_output_shape() -> None:
    """SIRENCoordinateNet should map [N, 4] -> [N, 4]."""
    net = SIRENCoordinateNet(hidden_dim=32, num_layers=2)
    xyzT = torch.randn(64, 4)
    pv = net(xyzT)
    assert pv.shape == (64, 4)


def test_siren_coordinate_net_gradients_flow() -> None:
    """Gradients must flow through SIRENCoordinateNet to inputs."""
    net = SIRENCoordinateNet(hidden_dim=16, num_layers=2)
    xyzT = torch.randn(32, 4, requires_grad=True)
    pv = net(xyzT)
    pv.sum().backward()
    assert xyzT.grad is not None


def test_acoustic_continuity_residual_shape() -> None:
    coords = torch.randn(64, 3, requires_grad=True)
    time = torch.randn(64, 1, requires_grad=True)
    pressure = torch.sin(coords.sum(dim=1, keepdim=True) + time)
    velocity = torch.cat([torch.sin(time)] * 3, dim=1)
    r = acoustic_continuity_residual(pressure, velocity, coords, time)
    assert r.shape == (64, 1)


def test_acoustic_continuity_residual_backward() -> None:
    coords = torch.randn(32, 3, requires_grad=True)
    time = torch.randn(32, 1, requires_grad=True)
    pressure = torch.sin(coords.sum(dim=1, keepdim=True) + time)
    velocity = torch.cat([torch.sin(time)] * 3, dim=1)
    r = acoustic_continuity_residual(pressure, velocity, coords, time)
    r.sum().backward()
    assert coords.grad is not None


def test_acoustic_momentum_residual_shape() -> None:
    coords = torch.randn(64, 3, requires_grad=True)
    time = torch.randn(64, 1, requires_grad=True)
    pressure = torch.sin(coords.sum(dim=1, keepdim=True) + time)
    velocity = torch.cat([torch.sin(time)] * 3, dim=1)
    r = acoustic_momentum_residual(pressure, velocity, coords, time)
    assert r.shape == (64, 3)


def test_collocation_physics_loss_positive() -> None:
    """CollocationPhysicsLoss should return a positive scalar when lambdas > 0."""
    coord_net = SIRENCoordinateNet(hidden_dim=16, num_layers=2)
    loss_fn = CollocationPhysicsLoss(coord_net=coord_net, lambda_cont=0.1, lambda_mom=0.1)
    room_dims = torch.tensor([[5.0, 4.0, 3.0], [6.0, 5.0, 4.0]])
    loss = loss_fn(room_dims, n_points=32)
    assert loss.item() > 0


def test_collocation_physics_loss_zero_when_disabled() -> None:
    """CollocationPhysicsLoss should return 0 when both lambdas are 0."""
    coord_net = SIRENCoordinateNet(hidden_dim=16, num_layers=2)
    loss_fn = CollocationPhysicsLoss(coord_net=coord_net, lambda_cont=0.0, lambda_mom=0.0)
    room_dims = torch.tensor([[5.0, 4.0, 3.0]])
    loss = loss_fn(room_dims, n_points=32)
    assert loss.item() == 0.0


def test_collocation_physics_loss_backward() -> None:
    """Gradients must flow through CollocationPhysicsLoss."""
    coord_net = SIRENCoordinateNet(hidden_dim=16, num_layers=2)
    loss_fn = CollocationPhysicsLoss(coord_net=coord_net, lambda_cont=0.1, lambda_mom=0.1)
    room_dims = torch.tensor([[5.0, 4.0, 3.0]])
    loss = loss_fn(room_dims, n_points=16)
    loss.backward()
    # Coordinate network parameters should have gradients
    for p in coord_net.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_multiband_sign_sticky_output_shape() -> None:
    """MultibandSignStickyPhaseReconstructor output shape should be [B, T-1]."""
    recon = MultibandSignStickyPhaseReconstructor()
    edc_mb = torch.rand(4, 256, 6)  # [B, T, bands]
    rir = recon(edc_mb)
    assert rir.shape == (4, 255)  # T-1


def test_multiband_sign_sticky_output_finite() -> None:
    """MultibandSignStickyPhaseReconstructor should produce finite values."""
    recon = MultibandSignStickyPhaseReconstructor()
    edc_mb = torch.rand(2, 64, 6).abs()
    rir = recon(edc_mb)
    assert torch.isfinite(rir).all()


def test_unet_refiner_shape_preserved() -> None:
    """UNetRefiner output length should match input length."""
    net = UNetRefiner(channels=1, base=8)
    x = torch.randn(2, 1, 512)
    y = net(x)
    assert y.shape == x.shape
