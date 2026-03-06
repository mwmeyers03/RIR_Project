"""Phase 5: training harness and configuration."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import optim

from .data import DEVICE, INPUT_DIM, get_dataloader
from .models import MultibandEDCPredictor
from .loss import PhysicsInformedRIRLoss


@dataclass
class TrainingConfig:
    # data
    batch_size: int = 8
    num_workers: int = 4
    max_rir_len: int = 32_000
    sample_rate: int = 16_000
    use_cache: bool = True

    # model
    hidden_dim: int = 256
    num_layers: int = 2
    num_time_steps: int = 256
    num_bands: int = 6
    model_dropout: float = 0.1

    # FDN
    train_fdn: bool = False
    fdn_num_delays: int = 16
    fdn_max_delay_ms: float = 50.0
    fdn_output_length: int = 4_000
    fdn_weight: float = 0.1

    # loss
    lambda_cont: float = 0.0
    lambda_mom: float = 0.0

    # optimiser
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # AMP
    use_amp: bool = True

    # scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # training
    epochs: int = 50
    log_every: int = 100
    val_every_epoch: int = 1


class RIRTrainer:
    def __init__(self, cfg: TrainingConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or DEVICE

        self._build_components()

    def _build_components(self):
        c = self.cfg
        self.train_loader = get_dataloader(
            split="train",
            batch_size=c.batch_size,
            num_workers=c.num_workers,
            max_rir_len=c.max_rir_len,
            num_time_steps=c.num_time_steps,
            sample_rate=c.sample_rate,
            use_cache=c.use_cache,
            shuffle=True,
        )
        self.val_loader = get_dataloader(
            split="val",
            batch_size=c.batch_size,
            num_workers=c.num_workers,
            max_rir_len=c.max_rir_len,
            num_time_steps=c.num_time_steps,
            sample_rate=c.sample_rate,
            use_cache=c.use_cache,
            shuffle=False,
        )
        self.lstm = MultibandEDCPredictor(
            input_dim=INPUT_DIM,
            hidden_dim=c.hidden_dim,
            num_layers=c.num_layers,
            num_time_steps=c.num_time_steps,
            num_bands=c.num_bands,
            dropout=c.model_dropout,
        ).to(self.device)
        self.criterion = PhysicsInformedRIRLoss(
            lambda_cont=c.lambda_cont,
            lambda_mom=c.lambda_mom,
        ).to(self.device)
        self.optimiser = optim.Adam(self.lstm.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, patience=c.scheduler_patience, factor=c.scheduler_factor
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp and self.device.type == "cuda")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.lstm.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            edc_target = y["edc_mb"].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                edc_pred = self.lstm(x)
                loss = self.criterion(edc_pred, edc_target)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.optimiser.zero_grad()
            total_loss += loss.item()
        avg = total_loss / len(self.train_loader)
        return {"total": avg}

    def validate(self) -> Dict[str, float]:
        self.lstm.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                edc_target = y["edc_mb"].to(self.device)
                edc_pred = self.lstm(x)
                loss = self.criterion(edc_pred, edc_target)
                total_loss += loss.item()
        avg = total_loss / len(self.val_loader)
        return {"total": avg}

    def fit(self) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            self.scheduler.step(val_metrics["total"])
            if (epoch + 1) % self.cfg.log_every == 0:
                print(f"Epoch {epoch+1}/{self.cfg.epochs} train={train_metrics['total']:.4f} val={val_metrics['total']:.4f}")
        return history
