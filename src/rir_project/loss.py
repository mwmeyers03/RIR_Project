"""Phase 3: losses used for physics-informed training."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDCReconstructionLoss(nn.Module):
    """Simple EDC reconstruction criterion."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


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

    def forward(self, edc_pred: torch.Tensor, edc_target: torch.Tensor) -> torch.Tensor:
        # edc_pred, edc_target: [B, T, bands]
        loss = F.mse_loss(edc_pred, edc_target)
        if self.lambda_cont != 0.0:
            loss = loss + self.lambda_cont * self._continuity(edc_pred, edc_target)
        if self.lambda_mom != 0.0:
            loss = loss + self.lambda_mom * self._momentum(edc_pred, edc_target)
        return loss

    def _continuity(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return continuity_residual(pred)

    def _momentum(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return momentum_residual(pred)
