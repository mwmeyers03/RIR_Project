import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure src/ layout is importable when tests run from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rir_project.data import INPUT_DIM, METRICS_DIM, RIRMegaDataset
from rir_project.loss import PhysicsInformedRIRLoss
from rir_project.trainer import RIRTrainer, TrainingConfig


class _FakeHF(list):
    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "audio":
                return [row["audio"] for row in self]
            raise KeyError(item)
        return super().__getitem__(item)


def _make_dataset_for_tests(sample_id: str = "sample-001") -> RIRMegaDataset:
    ds = RIRMegaDataset.__new__(RIRMegaDataset)
    ds.split = "train"
    ds.max_rir_len = 128
    ds.num_time_steps = 16
    ds.sample_rate = 16_000

    rir = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    ds._hf_ds = _FakeHF(
        [
            {
                "audio": {
                    "path": f"/tmp/{sample_id}.wav",
                    "array": rir,
                }
            }
        ]
    )
    ds._meta = pd.DataFrame(
        [
            {
                "id": sample_id,
                "room_size": "[5.0, 4.0, 3.0]",
                "source": "[1.0, 1.0, 1.2]",
                "microphone": "[2.0, 2.0, 1.2]",
                "absorption": 0.2,
                "absorption_bands": "{'125':0.2,'250':0.2,'500':0.2,'1000':0.2,'2000':0.2,'4000':0.2}",
                "metrics": "{'rt60':0.5,'drr_db':-2,'c50_db':1.1,'c80_db':2.2,'band_rt60s':{'125':0.5,'250':0.5,'500':0.5,'1000':0.5,'2000':0.5,'4000':0.5}}",
            }
        ]
    )
    ds._meta_id_to_pos = {sample_id: 0}
    ds._index_map = ds._build_index()
    return ds


def test_dataset_sample_shapes_match_constants() -> None:
    ds = _make_dataset_for_tests()
    x, y = ds[0]
    assert x.shape[0] == INPUT_DIM
    assert y["metrics"].shape[0] == METRICS_DIM


def test_build_index_raises_when_hf_and_metadata_do_not_align() -> None:
    ds = _make_dataset_for_tests(sample_id="hf-id")
    ds._meta_id_to_pos = {"different-id": 0}
    with pytest.raises(AssertionError, match="alignment failed"):
        ds._build_index()


def test_invalid_room_dims_raise_value_error() -> None:
    ds = _make_dataset_for_tests()
    bad_row = pd.Series(
        {
            "room_size": "[-5.0, 4.0, 3.0]",
            "source": "[1.0, 1.0, 1.2]",
            "microphone": "[2.0, 2.0, 1.2]",
            "absorption": 0.2,
            "absorption_bands": "{}",
            "metrics": "{}",
        }
    )
    with pytest.raises(ValueError, match="Non-positive room dimensions"):
        ds._parse_meta_row(bad_row)


def test_physics_loss_respects_non_zero_lambdas() -> None:
    pred = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]], dtype=torch.float32)
    target = torch.zeros_like(pred)

    base = PhysicsInformedRIRLoss(lambda_cont=0.0, lambda_mom=0.0)(pred, target)
    with_terms = PhysicsInformedRIRLoss(lambda_cont=0.5, lambda_mom=0.5)(pred, target)

    assert with_terms.item() > base.item()


def test_trainer_fit_dry_run_without_full_data(monkeypatch: pytest.MonkeyPatch) -> None:
    import rir_project.trainer as trainer_module

    def _no_data_loader(*args, **kwargs):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(trainer_module, "get_dataloader", _no_data_loader)

    cfg = TrainingConfig(
        epochs=2,
        batch_size=2,
        num_workers=0,
        use_amp=False,
        hidden_dim=16,
        num_layers=1,
        num_time_steps=8,
        num_bands=2,
        log_every=1,
        seed=7,
        dry_run=True,
    )
    trainer = RIRTrainer(cfg)
    history = trainer.fit()

    assert "train_loss" in history and "val_loss" in history
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert isinstance(history["train_loss"][0], float)
