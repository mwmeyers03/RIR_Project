"""Phase 2 & 4: neural architectures for EDC prediction and FDN.

This module defines the LSTM predictor and a differentiable feedback delay
network whose delay lengths are constrained in log-space, addressing the
plateau issue described in the prior analysis.  To facilitate early/late
bifurcation the FDN is built with a companion `EarlyReflectionNet` stub.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import INPUT_DIM, OCTAVE_BANDS


def _hadamard_matrix(n: int) -> torch.Tensor:
    """Construct a Hadamard matrix for power-of-two n."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard size must be a positive power of 2")
    H = torch.tensor([[1.0]])
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / (n ** 0.5)


class SirenLayer(nn.Module):
    """Sinusoidal representation layer (SIREN) with sine activation.

    Uses ``sin(omega_0 * linear(x))`` to ensure smooth, infinitely
    differentiable outputs — critical for ``torch.autograd.grad``-based
    physics residuals.
    """

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        # SIREN-style initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = (6.0 / in_features) ** 0.5 / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENCoordinateNet(nn.Module):
    """SIREN MLP that maps (x, y, z, t) coordinates → (p, u_x, u_y, u_z).

    Used as the coordinate network in collocation-based PINN training.
    All layers use the sinusoidal activation which guarantees smooth,
    infinitely differentiable outputs — essential for
    ``torch.autograd.grad``-based physics residuals.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden SIREN layer.
    num_layers : int
        Number of hidden layers (≥ 1).
    omega_0 : float
        Frequency multiplier for the first layer.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [SirenLayer(4, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0, is_first=False))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 4)  # p, ux, uy, uz

    def forward(self, xyzT: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        xyzT : Tensor[N, 4]
            Concatenated spatial (x, y, z) and temporal (t) coordinates.

        Returns
        -------
        pv : Tensor[N, 4]
            Predicted pressure (col 0) and velocity (cols 1-3).
        """
        return self.out(self.net(xyzT))


class MultibandEDCPredictor(nn.Module):
    """LSTM model that maps room features -> multiband EDC.

    Uses LayerNorm (not BatchNorm1d) so it works correctly when
    batch_size == 1 (which would otherwise crash BatchNorm).
    The output is constrained to be monotonically decreasing via
    softplus cumsum, ensuring physically realistic EDC curves.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_time_steps: int = 256,
        num_bands: int = len(OCTAVE_BANDS),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_time_steps = num_time_steps
        self.num_bands = num_bands
        self.dropout = dropout

        # LayerNorm normalises over the feature dimension and works for any batch
        # size (including B=1), unlike BatchNorm1d which requires B>1 in training.
        self.input_norm = nn.LayerNorm(input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.h0_proj = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self.c0_proj = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self.time_embed = nn.Parameter(
            torch.randn(1, num_time_steps, hidden_dim) * 0.02
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Two-layer head: produces log-decrements for monotonic EDC
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_bands),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        B = x.size(0)
        # LayerNorm normalises over the last dim (input_dim), works for any B
        x = self.input_norm(x)                        # [B, input_dim]
        ctx = self.encoder(x)                         # [B, hidden]
        h0 = self.h0_proj(ctx).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(ctx).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        lstm_in = self.time_embed.expand(B, -1, -1)
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))   # [B, T, hidden]
        # Monotonically-decreasing dB EDC via softplus + cumsum
        log_dec = self.head(lstm_out)                 # [B, T, num_bands]
        decrements = F.softplus(log_dec) * 0.5
        edc_pred = -torch.cumsum(decrements, dim=1)   # [B, T, num_bands]
        return edc_pred


class DifferentiableFDN(nn.Module):
    """Feedback delay network with log-space constrained delays.

    The parameter self.log_kappa stores unbounded values; a sigmoid mapping
    ensures the actual delays lie within (0, max_delay_ms) milliseconds and
    gradients flow properly.  This prevents integer-programming behaviour
    that previously caused the FDN loss to plateau.
    """

    def __init__(
        self,
        num_delays: int = 16,
        max_delay_ms: float = 50.0,
        sample_rate: float = 16_000,
        output_length: int = 4_000,
    ) -> None:
        super().__init__()
        self.num_delays = num_delays
        self.max_delay_ms = max_delay_ms
        self.sample_rate = sample_rate
        self.output_length = output_length

        # unbounded parameters; mapped through sigmoid
        self.log_kappa = nn.Parameter(torch.zeros(num_delays))
        self.alpha_raw = nn.Parameter(torch.zeros(num_delays))
        self.beta_raw = nn.Parameter(torch.zeros(num_delays))

        # fixed orthonormal mixing matrix
        h_size = 1
        while h_size < num_delays:
            h_size *= 2
        H = _hadamard_matrix(h_size)[:num_delays, :num_delays]
        self.register_buffer("H", H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] input to FDN
        B, T = x.shape
        D = self.num_delays

        # Map unbounded params to stable physical ranges.
        max_delay_samples = max(1.0, self.max_delay_ms * self.sample_rate / 1000.0)
        kappa = 1.0 + torch.sigmoid(self.log_kappa) * (max_delay_samples - 1.0)
        alpha = torch.sigmoid(self.alpha_raw)
        beta = torch.sigmoid(self.beta_raw)

        # Each delay line is an exponential smoother whose decay is controlled by kappa.
        decays = torch.exp(-1.0 / kappa).clamp(0.0, 0.9999)
        states = []
        for d in range(D):
            decay = decays[d]
            prev = torch.zeros(B, device=x.device, dtype=x.dtype)
            seq = []
            for t in range(T):
                prev = decay * prev + x[:, t]
                seq.append(prev)
            states.append(torch.stack(seq, dim=1))

        delay_bank = torch.stack(states, dim=1)  # [B, D, T]
        mixed = torch.einsum("ij,bjt->bit", self.H, delay_bank)
        out = (alpha.view(1, D, 1) * mixed + beta.view(1, D, 1) * delay_bank).mean(dim=1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EarlyReflectionNet(nn.Module):
    """Simple delayed-sum network for the first few milliseconds (43 taps).

    Produces the early-reflection portion of the RIR via a 1-D convolution
    with learnable tap gains.
    """

    def __init__(self, n_taps: int = 43):
        super().__init__()
        self.n_taps = n_taps
        self.coeffs = nn.Parameter(torch.randn(n_taps) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] — broadband EDC / excitation signal
        B, L = x.shape
        # 1-D convolution (delayed-sum): each tap is a weighted, delayed copy
        kernel = self.coeffs.flip(0).view(1, 1, self.n_taps)
        out = F.conv1d(x.unsqueeze(1), kernel, padding=self.n_taps - 1)
        return out.squeeze(1)[:, :L]


class ConvBlock1D(nn.Module):
    """Conv1D → GroupNorm → ReLU (× 2).

    GroupNorm instead of BatchNorm so this works with any batch size, including B=1.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        _grp = min(out_ch, 8) if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.GroupNorm(_grp, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.GroupNorm(_grp, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock1D(in_ch, out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock1D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SinusoidalPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttentionBottleneck(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x)
        return self.norm(x + y)


class UNetRefiner(nn.Module):
    """Compact U-Net refiner for optional post-processing."""

    def __init__(self, channels: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = EncoderBlock(channels, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.bottleneck = ConvBlock1D(base * 2, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.dec1 = DecoderBlock(base * 2, base, base)
        self.out = nn.Conv1d(base, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        b = self.bottleneck(p2)
        d2 = self.dec2(b, s2)
        d1 = self.dec1(d2, s1)
        return self.out(d1)
