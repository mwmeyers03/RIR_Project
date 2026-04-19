"""Phase 5: training harness and configuration."""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .data import DEVICE, INPUT_DIM, get_dataloader
from .loss import (
    CollocationPhysicsLoss,
    EDCDBLoss,
    MultiResolutionSTFTLoss,
    PhysicsInformedRIRLoss,
    late_tail_energy_growth_penalty,
)
from .models import (
    DifferentiableFDN,
    EarlyReflectionNet,
    MultibandEDCPredictor,
    SIRENCoordinateNet,
    UNetRefiner,
)
from .synthesis import MultibandSignStickyPhaseReconstructor, SignStickyPhaseReconstructor
from .utils import edc_rmse_db, estimate_rt60, log_spectral_distance, set_seed

SPEED_OF_SOUND_M_S = 343.0


@dataclass
class TrainingConfig:
    # data
    batch_size: int = 8
    num_workers: int = 4
    max_rir_len: int = 32_000
    sample_rate: int = 16_000
    use_cache: bool = True
    hf_cache_dir: Optional[str] = None

    # model
    hidden_dim: int = 512
    num_layers: int = 3
    num_time_steps: int = 256
    num_bands: int = 6
    model_dropout: float = 0.05

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
    seed: Optional[int] = 42
    dry_run: bool = False
    use_curriculum_ramp: bool = False
    physics_ramp_start_epoch: int = 0
    physics_ramp_end_epoch: int = 0
    lambda_cont_target: float = 0.0
    lambda_mom_target: float = 0.0
    early_late_split: bool = False
    metrics_eval_batches: int = 2
    save_metrics_path: str = ""
    fdn_plateau_grad_threshold: float = 1e-7
    auto_adjust_max_delay_ms: bool = True
    use_mr_stft: bool = False
    mr_stft_weight: float = 1.0
    mr_stft_windows: str = "512,1024,2048"
    use_composite_loss: bool = True
    loss_weight_time: float = 0.15
    loss_weight_mrstft: float = 0.55
    loss_weight_edc: float = 0.25
    loss_weight_direct: float = 0.05
    stage_a_ratio: float = 0.35
    stage_a_window_ms: float = 100.0
    direct_window_ms: float = 3.0
    direct_search_ms: float = 8.0
    use_smooth_l1_direct: bool = True
    use_late_tail_penalty: bool = False
    late_tail_start_ms: float = 50.0
    late_tail_penalty_weight: float = 0.02
    checkpoint_dir: str = "checkpoints"
    top_k_checkpoints: int = 3

    # collocation PINN
    use_collocation: bool = False
    collocation_n_points: int = 128
    collocation_lambda_cont: float = 0.01
    collocation_lambda_mom: float = 0.01
    siren_hidden_dim: int = 64
    siren_num_layers: int = 3

    # U-Net refiner
    use_unet: bool = False
    unet_weight: float = 1.0

    # curriculum FDN output length (shorter windows speed up early training)
    fdn_curriculum_length: int = 0  # 0 = disabled; if > 0, use this length until epoch fdn_curriculum_end_epoch
    fdn_curriculum_end_epoch: int = 10


class RIRTrainer:
    def __init__(self, cfg: TrainingConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or DEVICE
        self._build_exception: Optional[Exception] = None
        self._components_ready = False
        try:
            self._build_components()
        except Exception as exc:
            # Allows fit(dry_run=True) to run without full dataset availability.
            self._build_exception = exc

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
            cache_dir=c.hf_cache_dir,
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
            cache_dir=c.hf_cache_dir,
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
        self.phase_recon = SignStickyPhaseReconstructor(seed=c.seed)
        self.mb_phase_recon = MultibandSignStickyPhaseReconstructor()

        params = list(self.lstm.parameters())
        self.fdn = None
        self.early = None
        if c.train_fdn:
            self.fdn = DifferentiableFDN(
                num_delays=c.fdn_num_delays,
                max_delay_ms=c.fdn_max_delay_ms,
                sample_rate=c.sample_rate,
                output_length=c.fdn_output_length,
            ).to(self.device)
            params.extend(list(self.fdn.parameters()))
            if c.early_late_split:
                self.early = EarlyReflectionNet().to(self.device)
                params.extend(list(self.early.parameters()))

        self.unet_refiner = None
        if c.use_unet:
            self.unet_refiner = UNetRefiner(channels=1).to(self.device)
            params.extend(list(self.unet_refiner.parameters()))

        # Collocation-based PINN physics loss — build before optimizer so its
        # parameters are included in the optimizer's param groups.
        self.collocation_loss = None
        if c.use_collocation:
            coord_net = SIRENCoordinateNet(
                hidden_dim=c.siren_hidden_dim,
                num_layers=c.siren_num_layers,
            ).to(self.device)
            params.extend(list(coord_net.parameters()))
            self.collocation_loss = CollocationPhysicsLoss(
                coord_net=coord_net,
                lambda_cont=c.collocation_lambda_cont,
                lambda_mom=c.collocation_lambda_mom,
            ).to(self.device)

        self._optim_params = params
        self.optimiser = optim.Adam(self._optim_params, lr=c.lr, weight_decay=c.weight_decay)
        # CosineAnnealingWarmRestarts provides better convergence than ReduceLROnPlateau
        # for EDC regression; restarts help escape local minima during curriculum phases.
        # T_0 = max(10, epochs//5): restart every ~20% of total training, but at least
        # every 10 epochs so early phases get at least one full cosine cycle.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser,
            T_0=max(10, c.epochs // 5),
            T_mult=1,
            eta_min=c.lr * 1e-3,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=c.use_amp and self.device.type == "cuda")

        windows = [int(w) for w in c.mr_stft_windows.split(",")]
        self.mr_stft_loss = MultiResolutionSTFTLoss(window_lengths=windows).to(self.device)
        self.edc_db_loss = EDCDBLoss(use_smooth_l1=False).to(self.device)

        self._components_ready = True

    def _build_model_only_components(self) -> None:
        """Build minimal components needed for dry runs without data loaders."""
        c = self.cfg
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
        self.phase_recon = SignStickyPhaseReconstructor(seed=c.seed)
        self.mb_phase_recon = MultibandSignStickyPhaseReconstructor()
        self.fdn = None
        self.early = None
        self.unet_refiner = None
        windows = [int(w) for w in c.mr_stft_windows.split(",")]
        self.mr_stft_loss = MultiResolutionSTFTLoss(window_lengths=windows).to(self.device)
        self.edc_db_loss = EDCDBLoss(use_smooth_l1=False).to(self.device)
        self.collocation_loss = None
        self._optim_params = list(self.lstm.parameters())
        self.optimiser = optim.Adam(self._optim_params, lr=c.lr, weight_decay=c.weight_decay)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=c.use_amp and self.device.type == "cuda")

    @staticmethod
    def _git_commit_hash() -> str:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip() or "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _dataset_size(loader) -> int:
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return -1
        try:
            return len(dataset)
        except Exception:
            return -1

    def _log_training_start(self) -> None:
        train_size = self._dataset_size(getattr(self, "train_loader", None))
        val_size = self._dataset_size(getattr(self, "val_loader", None))
        print(f"[train-start] config={json.dumps(asdict(self.cfg), sort_keys=True)}")
        print(f"[train-start] seed={self.cfg.seed}")
        print(f"[train-start] git_commit={self._git_commit_hash()}")
        print(f"[train-start] dataset_size_train={train_size} dataset_size_val={val_size}")

    def _fit_dry_run(self) -> Dict[str, list]:
        if not hasattr(self, "lstm") or not hasattr(self, "criterion"):
            self._build_model_only_components()

        self.lstm.train()
        batch_size = max(1, min(int(self.cfg.batch_size), 2))
        x = torch.randn(batch_size, INPUT_DIM, device=self.device)
        edc_target = torch.randn(batch_size, self.cfg.num_time_steps, self.cfg.num_bands, device=self.device)

        self.optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
            edc_pred = self.lstm(x)
            loss = self.criterion(edc_pred, edc_target)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimiser)
        self.scaler.update()

        loss_value = float(loss.detach().item())
        print(f"[dry-run] completed single synthetic step loss={loss_value:.6f}")
        return {
            "train_loss": [loss_value],
            "val_loss": [loss_value],
            "train_time_loss": [loss_value],
            "train_mrstft_loss": [0.0],
            "train_edc_loss": [0.0],
            "train_direct_loss": [0.0],
            "val_time_loss": [loss_value],
            "val_mrstft_loss": [0.0],
            "val_edc_loss": [0.0],
            "val_direct_loss": [0.0],
            "val_composite_score": [float("inf")],
            "epoch_time_sec": [0.0],
            "rt60_error": [float("nan")],
            "lsd": [float("nan")],
            "edc_rmse": [float("nan")],
            "log_kappa_grad_norm": [0.0],
        }

    def _apply_curriculum(self, epoch: int) -> None:
        c = self.cfg
        if not c.use_curriculum_ramp:
            self.criterion.lambda_cont = c.lambda_cont
            self.criterion.lambda_mom = c.lambda_mom
            return

        if epoch <= c.physics_ramp_start_epoch:
            alpha = 0.0
        elif epoch >= c.physics_ramp_end_epoch:
            alpha = 1.0
        else:
            denom = max(1, c.physics_ramp_end_epoch - c.physics_ramp_start_epoch)
            alpha = float(epoch - c.physics_ramp_start_epoch) / float(denom)
        self.criterion.lambda_cont = c.lambda_cont_target * alpha
        self.criterion.lambda_mom = c.lambda_mom_target * alpha

    def _predict_rir_from_edc(
        self,
        edc_pred: torch.Tensor,
        apply_unet: bool = True,
    ) -> torch.Tensor:
        """Return a time-domain RIR from a multiband EDC prediction.

        The U‑Net refiner (if enabled) is applied **only** when
        ``apply_unet`` is True and ``self.unet_refiner`` is not None.  The
        flag simplifies loss computation in the training loop when we want
        to compare both pre- and post‑refinement signals.
        """
        edc_1d = edc_pred.mean(dim=2)
        if self.cfg.train_fdn and self.fdn is not None:
            late = self.fdn(edc_1d)
            if self.cfg.early_late_split and self.early is not None:
                rir = late + self.early(edc_1d)
            else:
                rir = late
        else:
            # Use multiband phase reconstruction (fixes metallic artefacts from single broadband)
            edc_mb_clamped = edc_pred.clamp(min=0.0)
            rir = self.mb_phase_recon(edc_mb_clamped)

        if apply_unet and self.unet_refiner is not None:
            # forward through refiner; shape [B, L] -> [B,1,L] -> [B,L]
            rir = self.unet_refiner(rir.unsqueeze(1)).squeeze(1)
        return rir

    @staticmethod
    def _acoustic_metrics(pred: np.ndarray, ref: np.ndarray, sample_rate: int) -> Dict[str, float]:
        n = min(len(pred), len(ref))
        if n < 4:
            return {"rt60_error": float("nan"), "lsd": float("nan"), "edc_rmse": float("nan")}
        p = pred[:n]
        r = ref[:n]
        return {
            "rt60_error": abs(estimate_rt60(p, sample_rate) - estimate_rt60(r, sample_rate)),
            "lsd": log_spectral_distance(p, r),
            "edc_rmse": edc_rmse_db(p, r),
        }

    def _effective_fdn_output_length(self, epoch: int) -> int:
        """Return the FDN output length, applying curriculum shortening if enabled."""
        c = self.cfg
        if c.fdn_curriculum_length > 0 and epoch < c.fdn_curriculum_end_epoch:
            return c.fdn_curriculum_length
        return c.fdn_output_length

    def _loss_window_samples(self, epoch: int, max_len: int) -> int:
        stage_a_epochs = max(1, int(np.ceil(self.cfg.epochs * self.cfg.stage_a_ratio)))
        if epoch < stage_a_epochs:
            short_len = int(self.cfg.sample_rate * (self.cfg.stage_a_window_ms / 1000.0))
            return max(1, min(max_len, short_len))
        return max_len

    def _direct_arrival_samples(self, x: torch.Tensor, rir_target: torch.Tensor) -> torch.Tensor:
        B, T = rir_target.shape
        search_len = max(1, min(T, int(self.cfg.sample_rate * (self.cfg.direct_search_ms / 1000.0))))
        fallback = torch.argmax(rir_target[:, :search_len].abs(), dim=1)

        if x.shape[1] < 9:
            return fallback
        room = x[:, :3]
        src = x[:, 3:6]
        mic = x[:, 6:9]
        valid_geom = (
            (room > 0.0).all(dim=1)
            & (src >= 0.0).all(dim=1)
            & (mic >= 0.0).all(dim=1)
            & (src <= room).all(dim=1)
            & (mic <= room).all(dim=1)
        )
        dist = torch.norm(src - mic, dim=1)
        geom = torch.clamp((dist / SPEED_OF_SOUND_M_S * self.cfg.sample_rate).long(), 0, max(0, T - 1))
        return torch.where(valid_geom, geom, fallback)

    def _direct_loss(self, rir_pred: torch.Tensor, rir_target: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        centers = self._direct_arrival_samples(x, rir_target)
        half_window = max(1, int(self.cfg.sample_rate * (self.cfg.direct_window_ms / 1000.0)))
        losses: List[torch.Tensor] = []
        for i in range(rir_pred.size(0)):
            c = int(centers[i].item())
            s = max(0, c - half_window)
            e = min(rir_pred.size(1), c + half_window + 1)
            p = rir_pred[i, s:e]
            t = rir_target[i, s:e]
            if self.cfg.use_smooth_l1_direct:
                losses.append(F.smooth_l1_loss(p, t))
            else:
                losses.append(F.l1_loss(p, t))
        if not losses:
            return torch.zeros((), device=rir_pred.device, dtype=rir_pred.dtype)
        return torch.stack(losses).mean()

    def _loss_components(
        self,
        epoch: int,
        x: torch.Tensor,
        rir_pred: torch.Tensor,
        rir_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        L = min(rir_pred.shape[1], rir_target.shape[1])
        L = self._loss_window_samples(epoch, L)
        rp = rir_pred[:, :L]
        rt = rir_target[:, :L]
        time_loss = F.mse_loss(rp, rt)
        mrstft_loss = self.mr_stft_loss(rp.float(), rt.float()) if self.mr_stft_loss is not None else torch.zeros((), device=self.device)
        edc_loss = self.edc_db_loss(rp, rt) if self.edc_db_loss is not None else torch.zeros((), device=self.device)
        direct_loss = self._direct_loss(rp, rt, x)
        tail_penalty = torch.zeros((), device=self.device)
        if self.cfg.use_late_tail_penalty:
            tail_penalty = late_tail_energy_growth_penalty(
                rp, sample_rate=self.cfg.sample_rate, start_ms=self.cfg.late_tail_start_ms
            )
        return {
            "time": time_loss,
            "mrstft": mrstft_loss,
            "edc": edc_loss,
            "direct": direct_loss,
            "tail_penalty": tail_penalty,
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.lstm.train()
        if self.fdn is not None:
            self.fdn.train()
        if self.early is not None:
            self.early.train()
        if self.unet_refiner is not None:
            self.unet_refiner.train()
        self._apply_curriculum(epoch)
        total_loss = 0.0
        total_time_loss = 0.0
        total_mrstft_loss = 0.0
        total_edc_loss = 0.0
        total_direct_loss = 0.0
        total_grad_norm = 0.0
        n_steps = 0
        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                edc_pred = self.lstm(x)
                rir_pred = self._predict_rir_from_edc(edc_pred, apply_unet=True)
                rir_target = y["rir"].to(self.device)
                if self.cfg.use_composite_loss:
                    comps = self._loss_components(epoch, x, rir_pred, rir_target)
                    loss = (
                        self.cfg.loss_weight_time * comps["time"]
                        + self.cfg.loss_weight_mrstft * comps["mrstft"]
                        + self.cfg.loss_weight_edc * comps["edc"]
                        + self.cfg.loss_weight_direct * comps["direct"]
                        + self.cfg.late_tail_penalty_weight * comps["tail_penalty"]
                    )
                else:
                    edc_target = y["edc_mb"].to(self.device)
                    loss = self.criterion(edc_pred, edc_target)
                    L = min(rir_pred.shape[1], rir_target.shape[1])
                    comps = {
                        "time": F.mse_loss(rir_pred[:, :L], rir_target[:, :L]),
                        "mrstft": torch.zeros((), device=self.device),
                        "edc": torch.zeros((), device=self.device),
                        "direct": torch.zeros((), device=self.device),
                    }
                    weight = self.cfg.unet_weight if self.unet_refiner is not None else 1.0
                    loss = loss + self.cfg.fdn_weight * weight * comps["time"]
                    if self.cfg.use_mr_stft and self.mr_stft_loss is not None:
                        loss = loss + self.cfg.mr_stft_weight * self.mr_stft_loss(
                            rir_pred[:, :L].float(), rir_target[:, :L].float()
                        )
                # Collocation-based PINN physics loss
                if self.collocation_loss is not None:
                    room_dims = x[:, :3].clamp(min=0.1)
                    coll_loss = self.collocation_loss(room_dims, n_points=self.cfg.collocation_n_points)
                    loss = loss + coll_loss
            self.scaler.scale(loss).backward()

            grad_norm = 0.0
            if self.cfg.train_fdn and self.fdn is not None and self.fdn.log_kappa.grad is not None:
                grad_norm = float(self.fdn.log_kappa.grad.detach().norm().item())

            torch.nn.utils.clip_grad_norm_(self._optim_params, self.cfg.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item()
            total_time_loss += float(comps["time"].item())
            total_mrstft_loss += float(comps["mrstft"].item())
            total_edc_loss += float(comps["edc"].item())
            total_direct_loss += float(comps["direct"].item())
            total_grad_norm += grad_norm
            n_steps += 1

        denom = max(1, n_steps)
        avg = total_loss / denom
        return {
            "total": avg,
            "time": total_time_loss / denom,
            "mrstft": total_mrstft_loss / denom,
            "edc": total_edc_loss / denom,
            "direct": total_direct_loss / denom,
            "log_kappa_grad_norm": total_grad_norm / denom,
            "lambda_cont": float(self.criterion.lambda_cont),
            "lambda_mom": float(self.criterion.lambda_mom),
        }

    def validate(self, epoch: Optional[int] = None) -> Dict[str, float]:
        self.lstm.eval()
        if self.fdn is not None:
            self.fdn.eval()
        if self.early is not None:
            self.early.eval()
        if self.unet_refiner is not None:
            self.unet_refiner.eval()
        epoch_for_mask = self.cfg.epochs if epoch is None else epoch
        total_loss = 0.0
        total_time_loss = 0.0
        total_mrstft_loss = 0.0
        total_edc_loss = 0.0
        total_direct_loss = 0.0
        metric_count = 0
        rt60_vals, lsd_vals, edc_vals = [], [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                edc_pred = self.lstm(x)
                rir_pred = self._predict_rir_from_edc(edc_pred, apply_unet=True)
                rir_target = y["rir"].to(self.device)
                if self.cfg.use_composite_loss:
                    comps = self._loss_components(epoch=epoch_for_mask, x=x, rir_pred=rir_pred, rir_target=rir_target)
                    loss = (
                        self.cfg.loss_weight_time * comps["time"]
                        + self.cfg.loss_weight_mrstft * comps["mrstft"]
                        + self.cfg.loss_weight_edc * comps["edc"]
                        + self.cfg.loss_weight_direct * comps["direct"]
                        + self.cfg.late_tail_penalty_weight * comps["tail_penalty"]
                    )
                else:
                    edc_target = y["edc_mb"].to(self.device)
                    loss = self.criterion(edc_pred, edc_target)
                    L = min(rir_pred.shape[1], rir_target.shape[1])
                    comps = {
                        "time": F.mse_loss(rir_pred[:, :L], rir_target[:, :L]),
                        "mrstft": torch.zeros((), device=self.device),
                        "edc": torch.zeros((), device=self.device),
                        "direct": torch.zeros((), device=self.device),
                    }
                    weight = self.cfg.unet_weight if self.unet_refiner is not None else 1.0
                    loss = loss + self.cfg.fdn_weight * weight * comps["time"]

                if batch_idx < max(1, self.cfg.metrics_eval_batches):
                    rir_ref = y["rir"].cpu().numpy()
                    pred_np = rir_pred.detach().cpu().numpy()
                    for i in range(pred_np.shape[0]):
                        m = self._acoustic_metrics(pred_np[i], rir_ref[i], sample_rate=self.cfg.sample_rate)
                        if not np.isnan(m["rt60_error"]):
                            rt60_vals.append(m["rt60_error"])
                        if not np.isnan(m["lsd"]):
                            lsd_vals.append(m["lsd"])
                        if not np.isnan(m["edc_rmse"]):
                            edc_vals.append(m["edc_rmse"])
                        metric_count += 1

                total_loss += loss.item()
                total_time_loss += float(comps["time"].item())
                total_mrstft_loss += float(comps["mrstft"].item())
                total_edc_loss += float(comps["edc"].item())
                total_direct_loss += float(comps["direct"].item())

        denom = max(1, len(self.val_loader))
        avg = total_loss / denom
        return {
            "total": avg,
            "time": total_time_loss / denom,
            "mrstft": total_mrstft_loss / denom,
            "edc": total_edc_loss / denom,
            "direct": total_direct_loss / denom,
            "rt60_error": float(np.nanmean(rt60_vals)) if rt60_vals else float("nan"),
            "lsd": float(np.nanmean(lsd_vals)) if lsd_vals else float("nan"),
            "edc_rmse": float(np.nanmean(edc_vals)) if edc_vals else float("nan"),
            "metrics_samples": metric_count,
        }

    def fit(self) -> Dict[str, list]:
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)
        self._log_training_start()

        if self.cfg.dry_run:
            return self._fit_dry_run()

        if not self._components_ready:
            if self._build_exception is not None:
                raise RuntimeError("Failed to build training components") from self._build_exception
            self._build_components()

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_time_loss": [],
            "train_mrstft_loss": [],
            "train_edc_loss": [],
            "train_direct_loss": [],
            "val_time_loss": [],
            "val_mrstft_loss": [],
            "val_edc_loss": [],
            "val_direct_loss": [],
            "val_composite_score": [],
            "epoch_time_sec": [],
            "rt60_error": [],
            "lsd": [],
            "edc_rmse": [],
            "log_kappa_grad_norm": [],
        }
        top_ckpts: List[Dict[str, object]] = []
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.cfg.epochs):
            t0 = time.perf_counter()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch=epoch)
            elapsed = time.perf_counter() - t0

            score = float("inf")
            if (
                np.isfinite(val_metrics["lsd"])
                and np.isfinite(val_metrics["edc_rmse"])
                and np.isfinite(val_metrics["rt60_error"])
            ):
                score = (
                    0.5 * float(val_metrics["lsd"])
                    + 0.3 * float(val_metrics["edc_rmse"])
                    + 0.2 * float(val_metrics["rt60_error"])
                )

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["train_time_loss"].append(train_metrics["time"])
            history["train_mrstft_loss"].append(train_metrics["mrstft"])
            history["train_edc_loss"].append(train_metrics["edc"])
            history["train_direct_loss"].append(train_metrics["direct"])
            history["val_time_loss"].append(val_metrics["time"])
            history["val_mrstft_loss"].append(val_metrics["mrstft"])
            history["val_edc_loss"].append(val_metrics["edc"])
            history["val_direct_loss"].append(val_metrics["direct"])
            history["val_composite_score"].append(score)
            history["epoch_time_sec"].append(elapsed)
            history["rt60_error"].append(val_metrics["rt60_error"])
            history["lsd"].append(val_metrics["lsd"])
            history["edc_rmse"].append(val_metrics["edc_rmse"])
            history["log_kappa_grad_norm"].append(train_metrics.get("log_kappa_grad_norm", 0.0))

            self.scheduler.step()
            print(
                f"Epoch {epoch+1}/{self.cfg.epochs} "
                f"train={train_metrics['total']:.4f} "
                f"(time={train_metrics['time']:.4f}, mrstft={train_metrics['mrstft']:.4f}, "
                f"edc={train_metrics['edc']:.4f}, direct={train_metrics['direct']:.4f}) "
                f"val={val_metrics['total']:.4f} "
                f"(time={val_metrics['time']:.4f}, mrstft={val_metrics['mrstft']:.4f}, "
                f"edc={val_metrics['edc']:.4f}, direct={val_metrics['direct']:.4f}) "
                f"score={score:.4f} rt60={val_metrics['rt60_error']:.4f}s "
                f"lsd={val_metrics['lsd']:.4f}dB edc_rmse={val_metrics['edc_rmse']:.4f}dB "
                f"time={elapsed:.2f}s"
            )

            if np.isfinite(score):
                ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}_score_{score:.6f}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "score": score,
                        "config": asdict(self.cfg),
                        "lstm": self.lstm.state_dict(),
                        "fdn": self.fdn.state_dict() if self.fdn is not None else None,
                        "early": self.early.state_dict() if self.early is not None else None,
                        "unet_refiner": self.unet_refiner.state_dict() if self.unet_refiner is not None else None,
                    },
                    ckpt_path,
                )
                top_ckpts.append({"score": score, "path": str(ckpt_path)})
                top_ckpts = sorted(top_ckpts, key=lambda x: float(x["score"]))[: max(1, self.cfg.top_k_checkpoints)]
                keep = {entry["path"] for entry in top_ckpts}
                for p in ckpt_dir.glob("epoch_*_score_*.pt"):
                    if str(p) not in keep:
                        p.unlink(missing_ok=True)

        if self.cfg.train_fdn and self.fdn is not None:
            grads = [g for g in history["log_kappa_grad_norm"] if not np.isnan(g)]
            if grads and max(grads) < self.cfg.fdn_plateau_grad_threshold:
                print(
                    "[fdn-check] log_kappa gradients appear plateaued "
                    f"(max={max(grads):.3e}, threshold={self.cfg.fdn_plateau_grad_threshold:.3e})."
                )
                if self.cfg.auto_adjust_max_delay_ms:
                    old = self.fdn.max_delay_ms
                    self.fdn.max_delay_ms = old * 1.5
                    print(
                        "[fdn-check] Adjusted max_delay_ms mapping "
                        f"from {old:.2f} to {self.fdn.max_delay_ms:.2f}."
                    )

        if self.cfg.save_metrics_path:
            payload = {
                "config": asdict(self.cfg),
                "final": {
                    "rt60_error": history["rt60_error"][-1] if history["rt60_error"] else float("nan"),
                    "lsd": history["lsd"][-1] if history["lsd"] else float("nan"),
                    "edc_rmse": history["edc_rmse"][-1] if history["edc_rmse"] else float("nan"),
                    "epoch_time_sec_mean": float(np.nanmean(history["epoch_time_sec"])) if history["epoch_time_sec"] else float("nan"),
                    "time_loss": history["val_time_loss"][-1] if history["val_time_loss"] else float("nan"),
                    "mrstft_loss": history["val_mrstft_loss"][-1] if history["val_mrstft_loss"] else float("nan"),
                    "edc_loss": history["val_edc_loss"][-1] if history["val_edc_loss"] else float("nan"),
                    "direct_loss": history["val_direct_loss"][-1] if history["val_direct_loss"] else float("nan"),
                    "val_composite_score": history["val_composite_score"][-1] if history["val_composite_score"] else float("inf"),
                    "log_kappa_grad_norm": history["log_kappa_grad_norm"][-1] if history["log_kappa_grad_norm"] else 0.0,
                },
                "history": history,
            }
            with open(self.cfg.save_metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[metrics] Saved run metrics to {self.cfg.save_metrics_path}")
        if top_ckpts:
            best = top_ckpts[0]
            print(f"[checkpoint] best={best['path']} score={float(best['score']):.6f}")
        return history
