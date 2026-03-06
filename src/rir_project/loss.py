"""Phase 3: losses used for physics-informed training."""

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
    if pred.size(1) < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))


def momentum_residual(pred: torch.Tensor) -> torch.Tensor:
    if pred.size(1) < 3:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    vel = pred[:, 1:] - pred[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return torch.mean(acc ** 2)


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
