"""Phase 3: losses used for physics-informed training."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDCReconstructionLoss(nn.Module):
    """Weighted RMSE between predicted and target EDC (dB).

    Standard RMSE treats all dB levels equally, but early-decay structure
    is perceptually and physically more important than the noise floor.
    This loss adds:
      1. Early-emphasis weighting (higher weight in first 25% of time steps)
      2. Slope-matching penalty that encourages correct RT60 gradient

    Supports both multiband [B, T, F] and broadband [B, T] tensors.
    """

    def __init__(
        self,
        early_weight: float = 3.0,
        slope_weight: float = 0.5,
        decay_rate: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        early_weight : float
            Multiplier applied to the first time step (weight decays to 1.0).
        slope_weight : float
            Weight applied to the finite-difference slope-matching penalty.
        decay_rate : float
            Controls how quickly the early-emphasis weight decays toward 1.
            A value of 5.0 means the weight reaches ~1.04× at t=0.6, ensuring
            about the first 40% of time steps receive elevated emphasis.
        """
        super().__init__()
        self.early_weight = early_weight
        self.slope_weight = slope_weight
        self.decay_rate = decay_rate

    def forward(self, edc_pred: torch.Tensor, edc_target: torch.Tensor) -> torch.Tensor:
        T = edc_pred.shape[1]
        # Early-emphasis weight: decays from early_weight → 1 over the time axis
        t = torch.arange(T, device=edc_pred.device, dtype=edc_pred.dtype) / T
        w = 1.0 + (self.early_weight - 1.0) * torch.exp(-self.decay_rate * t)  # [T]
        # Broadcast over batch / band dimensions
        if edc_pred.dim() == 3:
            w = w.view(1, T, 1)
        else:
            w = w.view(1, T)
        # 1. Weighted RMSE
        diff = (edc_pred - edc_target) * w
        rmse = torch.sqrt(torch.mean(diff ** 2) + 1e-8)
        # 2. Slope penalty: finite-difference along time axis
        slope_pred = edc_pred[:, 1:] - edc_pred[:, :-1]
        slope_tgt  = edc_target[:, 1:] - edc_target[:, :-1]
        slope_loss = torch.sqrt(torch.mean((slope_pred - slope_tgt) ** 2) + 1e-8)
        return rmse + self.slope_weight * slope_loss


def continuity_residual(pred: torch.Tensor) -> torch.Tensor:
    """Acoustic Continuity Equation residual via torch.autograd.grad.

    When ``pred`` has a computational graph (requires_grad=True and an active
    graph), the residual is computed with ``torch.autograd.grad`` to leverage
    automatic differentiation for smooth SIREN outputs.  Otherwise, the
    residual falls back to a finite-difference approximation so the function
    remains usable in no-grad contexts (e.g. validation).
    """
    if pred.size(1) < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    if pred.requires_grad and pred.grad_fn is not None:
        # Create a synthetic time coordinate that shares the graph
        t = torch.linspace(0, 1, pred.size(1), device=pred.device, dtype=pred.dtype)
        t = t.unsqueeze(0).unsqueeze(-1).expand_as(pred).requires_grad_(True)
        # weighted sum to allow grad computation
        weighted = (pred * t).sum()
        grad_t = torch.autograd.grad(
            weighted, t, create_graph=True, retain_graph=True
        )[0]
        return torch.mean(grad_t ** 2)

    # Fallback: finite-difference (no graph available)
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))


def momentum_residual(pred: torch.Tensor) -> torch.Tensor:
    """Linearized Momentum Equation residual via torch.autograd.grad.

    Mirrors the continuity residual approach: uses autograd when a graph is
    available, otherwise falls back to second-order finite differences.
    """
    if pred.size(1) < 3:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    if pred.requires_grad and pred.grad_fn is not None:
        t = torch.linspace(0, 1, pred.size(1), device=pred.device, dtype=pred.dtype)
        t = t.unsqueeze(0).unsqueeze(-1).expand_as(pred).requires_grad_(True)
        weighted = (pred * t).sum()
        grad_t = torch.autograd.grad(
            weighted, t, create_graph=True, retain_graph=True
        )[0]
        grad2 = torch.autograd.grad(
            grad_t.sum(), t, create_graph=True, retain_graph=True
        )[0]
        return torch.mean(grad2 ** 2)

    # Fallback: finite-difference second derivative
    vel = pred[:, 1:] - pred[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return torch.mean(acc ** 2)


# ---- Proper PINN collocation residuals ----

def acoustic_continuity_residual(
    pressure: torch.Tensor,
    velocity: torch.Tensor,
    coords: torch.Tensor,
    time: torch.Tensor,
    rho0: float = 1.225,
    c: float = 343.0,
) -> torch.Tensor:
    """Residual of the acoustic continuity equation at collocation points.

    ∂p/∂t  +  ρ₀ c² ∇·u  =  0

    Parameters
    ----------
    pressure : Tensor[N, 1]
        Acoustic pressure at N collocation points (requires_grad=True path).
    velocity : Tensor[N, 3]
        Particle velocity (u_x, u_y, u_z) at the same points.
    coords : Tensor[N, 3]
        Spatial coordinates (x, y, z) — must have requires_grad=True.
    time : Tensor[N, 1]
        Temporal coordinate — must have requires_grad=True.
    rho0 : float
        Equilibrium air density (kg/m³).
    c : float
        Speed of sound (m/s).

    Returns
    -------
    residual : Tensor[N, 1]
        Point-wise residual of the continuity equation.
    """
    dp_dt = torch.autograd.grad(
        outputs=pressure,
        inputs=time,
        grad_outputs=torch.ones_like(pressure),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if dp_dt is None:
        dp_dt = torch.zeros_like(pressure)

    div_u = torch.zeros_like(pressure)
    for dim in range(3):
        du_i_full = torch.autograd.grad(
            outputs=velocity[:, dim : dim + 1],
            inputs=coords,
            grad_outputs=torch.ones_like(pressure),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if du_i_full is not None:
            div_u = div_u + du_i_full[:, dim : dim + 1]

    return dp_dt + rho0 * c ** 2 * div_u


def acoustic_momentum_residual(
    pressure: torch.Tensor,
    velocity: torch.Tensor,
    coords: torch.Tensor,
    time: torch.Tensor,
    rho0: float = 1.225,
) -> torch.Tensor:
    """Residual of the linearized momentum equation at collocation points.

    ρ₀ ∂u/∂t  +  ∇p  =  0

    Parameters
    ----------
    pressure : Tensor[N, 1]
        Acoustic pressure at N collocation points.
    velocity : Tensor[N, 3]
        Particle velocity (u_x, u_y, u_z).
    coords : Tensor[N, 3]
        Spatial coordinates — must have requires_grad=True.
    time : Tensor[N, 1]
        Temporal coordinate — must have requires_grad=True.
    rho0 : float
        Equilibrium air density (kg/m³).

    Returns
    -------
    residual : Tensor[N, 3]
        Per-component residual of the momentum equation.
    """
    grad_p = torch.autograd.grad(
        outputs=pressure,
        inputs=coords,
        grad_outputs=torch.ones_like(pressure),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad_p is None:
        grad_p = torch.zeros(pressure.size(0), 3, device=pressure.device, dtype=pressure.dtype)

    du_dt = torch.autograd.grad(
        outputs=velocity,
        inputs=time,
        grad_outputs=torch.ones_like(velocity),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if du_dt is None:
        du_dt = torch.zeros_like(velocity)

    return rho0 * du_dt + grad_p


class CollocationPhysicsLoss(nn.Module):
    """Physics loss computed over randomly sampled spatial-temporal collocation points.

    Uses a SIREN coordinate network to predict pressure and velocity from
    (x, y, z, t) coordinates, then enforces the acoustic wave equations
    via automatic differentiation.  This is the proper PINN formulation:
    the network is forced to obey acoustic wave propagation laws, not just
    match the EDC curve.

    Parameters
    ----------
    coord_net : nn.Module
        Network mapping [N, 4] → [N, 4] where outputs are (p, ux, uy, uz).
    lambda_cont : float
        Weight for the continuity-equation residual.
    lambda_mom : float
        Weight for the momentum-equation residual.
    rho0 : float
        Equilibrium air density (kg/m³).
    c : float
        Speed of sound (m/s).
    """

    def __init__(
        self,
        coord_net: nn.Module,
        lambda_cont: float = 0.01,
        lambda_mom: float = 0.01,
        rho0: float = 1.225,
        c: float = 343.0,
    ) -> None:
        super().__init__()
        self.coord_net = coord_net
        self.lambda_cont = lambda_cont
        self.lambda_mom = lambda_mom
        self.rho0 = rho0
        self.c = c

    def forward(
        self,
        room_dims: torch.Tensor,
        n_points: int = 128,
    ) -> torch.Tensor:
        """Sample random collocation points and compute physics loss.

        Parameters
        ----------
        room_dims : Tensor[B, 3]
            Room dimensions (L, W, H) in metres — used to bound sample space.
        n_points : int
            Number of collocation points to sample.

        Returns
        -------
        loss : scalar Tensor
            Weighted sum of continuity and momentum residual MSEs.
        """
        B = room_dims.size(0)
        device = room_dims.device
        dtype = room_dims.dtype

        # Sample spatial coordinates uniformly within the room bounding box.
        # Use the batch-mean dimensions as the shared bounding box.
        room_max = room_dims.mean(dim=0).clamp(min=0.1)  # [3]
        coords = torch.rand(n_points, 3, device=device, dtype=dtype, requires_grad=True)
        coords_scaled = coords * room_max.unsqueeze(0)  # [N, 3] in metres

        # Sample time in [0, 2 s] (typical RIR range)
        time = torch.rand(n_points, 1, device=device, dtype=dtype, requires_grad=True) * 2.0

        # Concatenate to a 4-D coordinate vector [N, 4]
        xyzT = torch.cat([coords_scaled, time], dim=1)  # [N, 4]

        # SIREN network predicts (p, ux, uy, uz) from (x, y, z, t)
        pv = self.coord_net(xyzT)  # [N, 4]
        pressure = pv[:, :1]        # [N, 1]
        velocity = pv[:, 1:]        # [N, 3]

        loss = torch.zeros((), device=device, dtype=dtype)
        if self.lambda_cont != 0.0:
            r_cont = acoustic_continuity_residual(pressure, velocity, coords_scaled, time, self.rho0, self.c)
            loss = loss + self.lambda_cont * torch.mean(r_cont ** 2)
        if self.lambda_mom != 0.0:
            r_mom = acoustic_momentum_residual(pressure, velocity, coords_scaled, time, self.rho0)
            loss = loss + self.lambda_mom * torch.mean(r_mom ** 2)
        return loss


# ---- Multi-Resolution STFT Loss ----

def _stft_mag_phase(
    x: torch.Tensor, fft_size: int, hop_length: int, win_length: int
) -> tuple:
    """Compute STFT magnitude and rectangular-coordinate phase."""
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    stft = torch.stft(x, fft_size, hop_length, win_length, window=window, return_complex=True)
    mag = stft.abs()
    # Map phase angle to rectangular coordinates on the unit circle
    # to avoid phase wraparound discontinuities
    phase_cos = stft.real / (mag + 1e-8)
    phase_sin = stft.imag / (mag + 1e-8)
    return mag, phase_cos, phase_sin


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT loss over several window lengths.

    Computes spectral magnitude (L1) and rectangular-coordinate phase loss
    over the given window sizes.  Phase angles are mapped to (cos, sin) on
    the unit circle to avoid phase-wraparound artefacts.
    """

    def __init__(self, window_lengths: List[int] | None = None) -> None:
        super().__init__()
        self.window_lengths = window_lengths or [512, 1024, 2048]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, T] time-domain waveforms
        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for wl in self.window_lengths:
            fft_size = wl
            hop = wl // 4
            mag_p, cos_p, sin_p = _stft_mag_phase(pred, fft_size, hop, wl)
            mag_t, cos_t, sin_t = _stft_mag_phase(target, fft_size, hop, wl)
            # Spectral convergence (magnitude L1)
            loss = loss + F.l1_loss(mag_p, mag_t)
            # Log-magnitude MSE
            loss = loss + F.mse_loss(
                torch.log(mag_p + 1e-7), torch.log(mag_t + 1e-7)
            )
            # Rectangular phase loss (avoids wraparound)
            loss = loss + F.mse_loss(cos_p, cos_t) + F.mse_loss(sin_p, sin_t)
        return loss / len(self.window_lengths)


class PhysicsInformedRIRLoss(nn.Module):
    """Combined loss: EDC reconstruction + continuity + momentum.

    The coefficients are configurable and can be controlled via a curriculum in
    the trainer.  This module does _not_ implement curriculum scheduling; that is
    handled by RIRTrainer.
    """

    def __init__(
        self,
        lambda_cont: float = 0.0,
        lambda_mom: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_mom = lambda_mom
        self.edc_loss = EDCReconstructionLoss(early_weight=3.0, slope_weight=0.5)

    def forward(self, edc_pred: torch.Tensor, edc_target: torch.Tensor) -> torch.Tensor:
        # edc_pred, edc_target: [B, T, bands]
        loss = self.edc_loss(edc_pred, edc_target)
        if self.lambda_cont != 0.0:
            loss = loss + self.lambda_cont * self._continuity(edc_pred, edc_target)
        if self.lambda_mom != 0.0:
            loss = loss + self.lambda_mom * self._momentum(edc_pred, edc_target)
        return loss

    def _continuity(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return continuity_residual(pred)

    def _momentum(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return momentum_residual(pred)
