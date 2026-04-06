"""Phase 6: synthesis components (bridge, phase reconstructor, full RIR)."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import INPUT_DIM, OCTAVE_BANDS
from .models import DifferentiableFDN, UNetRefiner


class ConditionedFDN(nn.Module):
    """FDN wrapper that accepts conditioning parameters from the mapper."""

    def __init__(self, num_delays: int = 16, sample_rate: int = 16_000, output_length: int = 32_000) -> None:
        super().__init__()
        self.fdn = DifferentiableFDN(
            num_delays=num_delays,
            sample_rate=sample_rate,
            output_length=output_length,
        )

    def forward(self, edc_1d: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # Parameter-conditioning hook can be expanded later.
        return self.fdn(edc_1d)


class EarlyReflections(nn.Module):
    """Simple delayed-sum early reflection generator (43 taps)."""

    def __init__(self, n_taps: int = 43):
        super().__init__()
        self.n_taps = n_taps
        self.gains = nn.Parameter(torch.zeros(n_taps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        B, L = x.shape
        kernel = self.gains.flip(0).view(1, 1, self.n_taps)
        out = F.conv1d(x.unsqueeze(1), kernel, padding=self.n_taps - 1)
        return out.squeeze(1)[:, :L]


class EDCToFDNMapper(nn.Module):
    def __init__(
        self,
        num_bands: int = len(OCTAVE_BANDS),
        num_time_steps: int = 256,
        num_delays: int = 16,
        room_dim: int = 3,
    ) -> None:
        super().__init__()
        edc_feat_dim = num_bands * 2
        self.edc_encoder = nn.Sequential(
            nn.Linear(edc_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.delay_head = nn.Sequential(
            nn.Linear(room_dim + 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_delays),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_delays),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_delays),
        )

    def forward(self, edc_pred: torch.Tensor, room_dims: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = edc_pred.size(0)
        edc_mean = edc_pred.mean(dim=1)
        T = edc_pred.size(1)
        q1 = edc_pred[:, : T // 4, :].mean(dim=1)
        q4 = edc_pred[:, 3 * T // 4 :, :].mean(dim=1)
        edc_slope = q4 - q1
        edc_feat = torch.cat([edc_mean, edc_slope], dim=1)
        h = self.edc_encoder(edc_feat)
        delay_in = torch.cat([room_dims, h], dim=1)
        log_kappa = self.delay_head(delay_in)
        alpha_raw = self.alpha_head(h)
        beta_raw = self.beta_head(h)
        return {"log_kappa": log_kappa, "alpha_raw": alpha_raw, "beta_raw": beta_raw}


class SignStickyPhaseReconstructor(nn.Module):
    """Reconstructs time-domain waveform from EDC using random sign-sticky scheme."""

    def __init__(self, stickiness: float = 0.90, seed: Optional[int] = None) -> None:
        super().__init__()
        assert 0.0 <= stickiness <= 1.0
        self.stickiness = stickiness
        self.seed = seed
        self._generators: Dict[str, torch.Generator] = {}

    def forward(self, edc: torch.Tensor) -> torch.Tensor:
        # edc: [B, T]
        B, T = edc.shape
        # reverse-diff
        diff = edc[:, :-1] - edc[:, 1:]
        amp = torch.sqrt(torch.clamp(diff, min=0.0))
        signs = self._sticky_signs(B, amp.size(1), device=edc.device)
        return amp * signs

    def _generator_for(self, device: torch.device) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        key = str(device)
        if key not in self._generators:
            gen_device = "cuda" if device.type == "cuda" else "cpu"
            gen = torch.Generator(device=gen_device)
            gen.manual_seed(self.seed)
            self._generators[key] = gen
        return self._generators[key]

    def _sticky_signs(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        # probability of sign flip at each step = 1 - stickiness
        flip_prob = 1.0 - self.stickiness
        generator = self._generator_for(device)
        if generator is None:
            flips = torch.rand((B, T), device=device) < flip_prob
        else:
            flips = torch.rand((B, T), generator=generator, device=device) < flip_prob
        # vectorised: cumulative XOR via cumsum mod 2
        flip_cumsum = flips.float().cumsum(dim=1)
        signs = 1.0 - 2.0 * (flip_cumsum % 2)
        return signs


class MultibandSignStickyPhaseReconstructor(nn.Module):
    """Applies the Sign-Sticky algorithm per frequency band, then sums bands.

    Applying phase reconstruction on a per-band basis preserves the
    individual decay rates of each octave band, fixing the metallic /
    spiky artefacts that result from a single broadband envelope.

    Parameters
    ----------
    stickiness : float
        Probability of sign *staying* the same at each step (0-1).
    seed : int, optional
        Random seed for reproducible sign sequences.
    """

    def __init__(self, stickiness: float = 0.90, seed: Optional[int] = None) -> None:
        super().__init__()
        self.band_recon = SignStickyPhaseReconstructor(stickiness=stickiness, seed=seed)

    def forward(self, edc_mb: torch.Tensor) -> torch.Tensor:
        """Reconstruct per-band waveforms and sum.

        Parameters
        ----------
        edc_mb : Tensor[B, T, num_bands]
            Multiband EDC (linear scale, positive, monotonically decreasing).

        Returns
        -------
        rir : Tensor[B, T-1]
            Reconstructed time-domain waveform (sum of band waveforms).
        """
        B, T, num_bands = edc_mb.shape
        band_waveforms: List[torch.Tensor] = []
        for b in range(num_bands):
            band_edc = edc_mb[:, :, b]  # [B, T]
            band_rir = self.band_recon(band_edc)  # [B, T-1]
            band_waveforms.append(band_rir)
        # Stack and sum across bands, normalised by number of bands
        stacked = torch.stack(band_waveforms, dim=1)  # [B, num_bands, T-1]
        return stacked.mean(dim=1)


class RIRSynthesiser(nn.Module):
    """End-to-end: room parameters -> complete RIR waveform.

    Chains LSTM -> FDN mapper -> Conditioned FDN -> per-band phase reconstruction.
    Optionally applies the U-Net refiner as a neural post-processor.

    Parameters
    ----------
    lstm : nn.Module
        Trained MultibandEDCPredictor.
    num_delays : int
        Number of FDN delay lines.
    sample_rate : int
        Audio sample rate in Hz.
    output_length : int
        Number of output samples.
    use_unet : bool
        If True, appends the UNetRefiner to the synthesis pipeline.
    stickiness : float
        Sign-sticky stickiness parameter.
    """

    def __init__(
        self,
        lstm: nn.Module,
        num_delays: int = 16,
        sample_rate: int = 16_000,
        output_length: int = 32_000,
        use_unet: bool = False,
        stickiness: float = 0.90,
        train_fdn: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = lstm
        self.train_fdn = train_fdn
        self.mapper = EDCToFDNMapper(num_delays=num_delays)
        self.fdn = ConditionedFDN(num_delays=num_delays, sample_rate=sample_rate, output_length=output_length)
        self.early = EarlyReflections()
        # Per-band phase reconstruction (fallback when train_fdn=False)
        self.mb_phase_recon = MultibandSignStickyPhaseReconstructor(stickiness=stickiness)
        # Optional U-Net refiner
        self.unet: Optional[nn.Module] = UNetRefiner(channels=1) if use_unet else None

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        edc_pred = self.lstm(x)                      # [B, T, bands]
        edc_1d = edc_pred.mean(dim=2)                # [B, T] broadband
        params = self.mapper(edc_pred, x[:, :3])
        late = self.fdn(edc_1d, params=params)
        early = self.early(edc_1d)
        phase: Optional[torch.Tensor] = None
        if self.train_fdn:
            # FDN path: early reflections + FDN late reverberation, both in
            # time domain already — do NOT apply phase_recon (it randomises
            # signs and blocks FDN gradients).
            L = min(late.size(1), early.size(1))
            rir_out = early[:, :L] + late[:, :L]
        else:
            # Fallback: per-band phase reconstruction from EDC
            edc_mb_clamped = edc_pred.clamp(min=0.0)
            phase = self.mb_phase_recon(edc_mb_clamped)  # [B, T-1]
            L = min(late.size(1), early.size(1), phase.size(1))
            rir_out = (early[:, :L] + late[:, :L]) * phase[:, :L]
        # Optional U-Net post-processing
        if self.unet is not None:
            rir_out = self.unet(rir_out.unsqueeze(1)).squeeze(1)
        # Peak normalisation
        rir_out = rir_out / (torch.amax(rir_out.abs(), dim=-1, keepdim=True) + 1e-8)
        out: Dict[str, torch.Tensor] = {"rir": rir_out, "edc_pred": edc_pred, "fdn_params": params}
        if return_intermediates and phase is not None:
            out["phase"] = phase
        return out
