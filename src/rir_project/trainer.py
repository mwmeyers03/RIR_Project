"""Phase 5: training harness and configuration."""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .data import DEVICE, INPUT_DIM, get_dataloader
from .loss import CollocationPhysicsLoss, MultiResolutionSTFTLoss, PhysicsInformedRIRLoss
from .models import (
    DifferentiableFDN,
    EarlyReflectionNet,
    MultibandEDCPredictor,
    SIRENCoordinateNet,
    UNetRefiner,
)
from .synthesis import MultibandSignStickyPhaseReconstructor, SignStickyPhaseReconstructor
from .utils import edc_rmse_db, estimate_rt60, log_spectral_distance, set_seed


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

        self.optimiser = optim.Adam(params, lr=c.lr, weight_decay=c.weight_decay)
        # CosineAnnealingWarmRestarts provides better convergence than ReduceLROnPlateau
        # for EDC regression; restarts help escape local minima during curriculum phases.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser,
            T_0=max(10, c.epochs // 5),
            T_mult=1,
            eta_min=c.lr * 1e-3,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=c.use_amp and self.device.type == "cuda")

        self.mr_stft_loss = None
        if c.use_mr_stft:
            windows = [int(w) for w in c.mr_stft_windows.split(",")]
            self.mr_stft_loss = MultiResolutionSTFTLoss(window_lengths=windows).to(self.device)

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
        self.mr_stft_loss = None
        self.collocation_loss = None
        self.optimiser = optim.Adam(self.lstm.parameters(), lr=c.lr, weight_decay=c.weight_decay)
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
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
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
            "epoch_time_sec": [0.0],
            "rt60_error": [float("nan")],
            "lsd": [float("nan")],
            "edc_rmse": [float("nan")],
            "fdn_loss": [0.0],
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

    def _predict_rir_from_edc(self, edc_pred: torch.Tensor) -> torch.Tensor:
        edc_1d = edc_pred.mean(dim=2)
        if self.cfg.train_fdn and self.fdn is not None:
            late = self.fdn(edc_1d)
            if self.cfg.early_late_split and self.early is not None:
                return late + self.early(edc_1d)
            return late
        # Use multiband phase reconstruction (fixes metallic artefacts from single broadband)
        edc_mb_clamped = edc_pred.clamp(min=0.0)
        return self.mb_phase_recon(edc_mb_clamped)

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
        total_fdn_loss = 0.0
        total_grad_norm = 0.0
        n_steps = 0
        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            edc_target = y["edc_mb"].to(self.device)
            self.optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                edc_pred = self.lstm(x)
                loss = self.criterion(edc_pred, edc_target)
                fdn_loss = torch.zeros((), device=self.device)
                if self.cfg.train_fdn and self.fdn is not None:
                    rir_pred = self._predict_rir_from_edc(edc_pred)
                    rir_target = y["rir"].to(self.device)
                    L = min(rir_pred.shape[1], rir_target.shape[1])
                    fdn_loss = F.mse_loss(rir_pred[:, :L], rir_target[:, :L])
                    loss = loss + self.cfg.fdn_weight * fdn_loss
                    # MR-STFT loss applied to time-domain RIR when FDN is active
                    if self.mr_stft_loss is not None:
                        mr_loss = self.mr_stft_loss(rir_pred[:, :L].float(), rir_target[:, :L].float())
                        loss = loss + self.cfg.mr_stft_weight * mr_loss
                # Collocation-based PINN physics loss
                if self.collocation_loss is not None:
                    room_dims = x[:, :3].clamp(min=0.1)
                    coll_loss = self.collocation_loss(room_dims, n_points=self.cfg.collocation_n_points)
                    loss = loss + coll_loss
            self.scaler.scale(loss).backward()

            grad_norm = 0.0
            if self.cfg.train_fdn and self.fdn is not None and self.fdn.log_kappa.grad is not None:
                grad_norm = float(self.fdn.log_kappa.grad.detach().norm().item())

            torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item()
            total_fdn_loss += float(fdn_loss.item())
            total_grad_norm += grad_norm
            n_steps += 1

        denom = max(1, n_steps)
        avg = total_loss / denom
        return {
            "total": avg,
            "fdn": total_fdn_loss / denom,
            "log_kappa_grad_norm": total_grad_norm / denom,
            "lambda_cont": float(self.criterion.lambda_cont),
            "lambda_mom": float(self.criterion.lambda_mom),
        }

    def validate(self) -> Dict[str, float]:
        self.lstm.eval()
        if self.fdn is not None:
            self.fdn.eval()
        if self.early is not None:
            self.early.eval()
        if self.unet_refiner is not None:
            self.unet_refiner.eval()
        total_loss = 0.0
        total_fdn_loss = 0.0
        metric_count = 0
        rt60_vals, lsd_vals, edc_vals = [], [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                edc_target = y["edc_mb"].to(self.device)
                edc_pred = self.lstm(x)
                loss = self.criterion(edc_pred, edc_target)
                fdn_loss = torch.zeros((), device=self.device)
                rir_pred = self._predict_rir_from_edc(edc_pred)
                if self.cfg.train_fdn and self.fdn is not None:
                    rir_target = y["rir"].to(self.device)
                    L = min(rir_pred.shape[1], rir_target.shape[1])
                    fdn_loss = F.mse_loss(rir_pred[:, :L], rir_target[:, :L])
                    loss = loss + self.cfg.fdn_weight * fdn_loss

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
                total_fdn_loss += float(fdn_loss.item())

        denom = max(1, len(self.val_loader))
        avg = total_loss / denom
        return {
            "total": avg,
            "fdn": total_fdn_loss / denom,
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
            "epoch_time_sec": [],
            "rt60_error": [],
            "lsd": [],
            "edc_rmse": [],
            "fdn_loss": [],
            "log_kappa_grad_norm": [],
        }
        for epoch in range(self.cfg.epochs):
            t0 = time.perf_counter()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.perf_counter() - t0

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["epoch_time_sec"].append(elapsed)
            history["rt60_error"].append(val_metrics["rt60_error"])
            history["lsd"].append(val_metrics["lsd"])
            history["edc_rmse"].append(val_metrics["edc_rmse"])
            history["fdn_loss"].append(val_metrics.get("fdn", 0.0))
            history["log_kappa_grad_norm"].append(train_metrics.get("log_kappa_grad_norm", 0.0))

            self.scheduler.step()
            if (epoch + 1) % self.cfg.log_every == 0:
                print(
                    f"Epoch {epoch+1}/{self.cfg.epochs} "
                    f"train={train_metrics['total']:.4f} val={val_metrics['total']:.4f} "
                    f"rt60={val_metrics['rt60_error']:.4f}s lsd={val_metrics['lsd']:.4f}dB "
                    f"edc={val_metrics['edc_rmse']:.4f}dB time={elapsed:.2f}s"
                )

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
                    "fdn_loss": history["fdn_loss"][-1] if history["fdn_loss"] else 0.0,
                    "log_kappa_grad_norm": history["log_kappa_grad_norm"][-1] if history["log_kappa_grad_norm"] else 0.0,
                },
                "history": history,
            }
            with open(self.cfg.save_metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[metrics] Saved run metrics to {self.cfg.save_metrics_path}")
        return history
