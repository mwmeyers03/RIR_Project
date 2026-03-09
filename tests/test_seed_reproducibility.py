import sys
from pathlib import Path

import torch

# Ensure src/ layout is importable when tests run from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rir_project.loss import PhysicsInformedRIRLoss
from rir_project.synthesis import SignStickyPhaseReconstructor
from rir_project.utils import set_seed


def _run_once(seed: int) -> float:
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    edc = torch.rand((2, 33), generator=g)

    recon = SignStickyPhaseReconstructor(stickiness=0.85, seed=seed)
    phase = recon(edc)

    # PhysicsInformedRIRLoss expects [B, T, num_bands].
    pred = phase.unsqueeze(-1).repeat(1, 1, 2)
    target = torch.zeros_like(pred)
    loss = PhysicsInformedRIRLoss()(pred, target)
    return float(loss.item())


def test_same_seed_produces_identical_loss() -> None:
    loss_a = _run_once(1234)
    loss_b = _run_once(1234)
    assert loss_a == loss_b
