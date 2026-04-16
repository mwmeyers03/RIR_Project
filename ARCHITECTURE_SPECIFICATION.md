# Physics-Informed RIR Generation — Architecture Specification

> **Scope:** White-box specification generated from source code in `src/rir_project/`.  
> **Format:** Senior Software Architect review — system topology, granular function-level logic, mathematical specification (LaTeX), cell-level state management, and validation contracts.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [System Topology](#2-system-topology)
3. [Global Constants & Tensor Schema](#3-global-constants--tensor-schema)
4. [Phase 1 — Data Pipeline (`data.py`)](#4-phase-1--data-pipeline-datapy)
5. [Phase 2 — Neural Architectures (`models.py`)](#5-phase-2--neural-architectures-modelspy)
6. [Phase 3 — Physics-Informed Loss (`loss.py`)](#6-phase-3--physics-informed-loss-losspy)
7. [Phase 4–6 — Synthesis Pipeline (`synthesis.py`)](#7-phase-46--synthesis-pipeline-synthesispy)
8. [Phase 5 — Training Harness (`trainer.py`)](#8-phase-5--training-harness-trainerpy)
9. [Utilities & Evaluation (`utils.py`, `train.py`)](#9-utilities--evaluation-utilspy-trainpy)
10. [Mathematical Specification](#10-mathematical-specification)
11. [Cell-Level State Management](#11-cell-level-state-management)
12. [Validation & System Integrity](#12-validation--system-integrity)

---

## 1. Project Summary

**Goal:** Build an end-to-end differentiable pipeline that maps compact macroscopic room
descriptors (geometry, absorption, source/mic positions) to full-length Room Impulse Response
(RIR) waveforms, guided by acoustic physics priors.

**Dataset:** [`mandipgoswami/rirmega`](https://huggingface.co/datasets/mandipgoswami/rirmega) —
~50,000 measured RIRs with companion metadata CSV (room geometry, source/mic positions,
per-octave-band absorption, RT60/DRR/C50/C80 metrics).

**Key design choices:**
- Physics-informed loss via `torch.autograd.grad`-based acoustic PDE residuals.
- Analytical room-mode feature extraction (Schroeder frequency, first axial mode, etc.).
- Differentiable Feedback Delay Network (FDN) with sigmoid-constrained, log-space delay parameters.
- Monotonically-decreasing EDC prediction enforced through softplus cumsum.
- Optional collocation PINN, multi-resolution STFT loss, and U-Net residual refiner.

---

## 2. System Topology

```
                     ┌──────────────────────────────────────────────┐
                     │                  train.py (CLI)               │
                     └────────────────────┬─────────────────────────┘
                                          │ TrainingConfig
                     ┌────────────────────▼─────────────────────────┐
                     │              RIRTrainer (trainer.py)          │
                     │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
                     │  │DataLoader│  │Optimizer │  │ AMP Scaler │ │
                     │  └────┬─────┘  └──────────┘  └────────────┘ │
                     └───────┼──────────────────────────────────────┘
                             │ (x, y) batches
            ┌────────────────▼──────────────────────────────────────┐
            │                   data.py                             │
            │  RIRMegaDataset ──▶ CachedRIRDataset ──▶ rir_collate │
            └────────────────────────────────────────────────────────┘
                             │
                  x: [B,24]  │  y: {edc_mb, rir, edc, metrics, rir_length}
                             │
            ┌────────────────▼──────────────────────────────────────┐
            │            MultibandEDCPredictor (models.py)          │
            │    Encoder (MLP) ──▶ LSTM ──▶ Head ──▶ softplus cumsum│
            └────────────────┬───────────────────────────────────────┘
                             │ edc_pred: [B, T, 6]
          ┌──────────────────┼──────────────────────────────────────┐
          │                  │                                       │
          ▼                  ▼                                       ▼
  PhysicsInformedRIRLoss  EDCToFDNMapper          CollocationPhysicsLoss
  (loss.py)               (synthesis.py)          + SIRENCoordinateNet
                               │                   (loss.py / models.py)
                    ┌──────────▼─────────────┐
                    │  ConditionedFDN        │
                    │  EarlyReflections      │
                    │  MultibandSignSticky   │
                    │  [Optional] UNetRefiner│
                    └──────────┬─────────────┘
                               │ rir_out: [B, L], peak-normalised
                    ┌──────────▼─────────────┐
                    │   utils.py: metrics,   │
                    │   evaluation, plots,   │
                    │   checkpointing        │
                    └────────────────────────┘
```

**Training path (summary):**
1. `RIRMegaDataset.__getitem__` → decoded RIR + computed EDC/features → `(x, y)`.
2. `MultibandEDCPredictor` → `edc_pred [B,T,bands]`.
3. `PhysicsInformedRIRLoss` → `loss (EDC + optional PDE residuals)`.
4. Optional: FDN time-domain `F.mse_loss`, MR-STFT loss, collocation PINN loss.
5. Backprop + gradient clip + Adam step.

**Inference path (summary):**
```
room_params (x) → LSTM → edc_pred → EDCToFDNMapper
                → ConditionedFDN (late) + EarlyReflections (early)
                → [optional] phase_recon, U-Net
                → peak-normalised RIR waveform
```

---

## 3. Global Constants & Tensor Schema

```python
INPUT_DIM       = 24      # total feature vector dimension
MODAL_FEAT_DIM  = 8       # contribution of room-mode features
METRICS_DIM     = 10      # RT60 + DRR + C50 + C80 + 6×band_RT60
OCTAVE_BANDS    = ["125","250","500","1000","2000","4000"]
DEFAULT_MAX_RIR_LEN = 32_000   # samples at 16 kHz = 2 s
DEVICE          = "cuda" if available else "cpu"
```

### Input vector `x` — `[B, 24]`

| Indices | Description |
|---------|-------------|
| 0–2 | Room dimensions L, W, H (metres) |
| 3–5 | Source position x, y, z (metres) |
| 6–8 | Microphone position x, y, z (metres) |
| 9 | Broadband absorption coefficient |
| 10–15 | Per-octave-band absorption (125 Hz … 4 kHz) |
| 16 | `n_below_300` — axial mode count below 300 Hz |
| 17 | `f_schroeder` — Schroeder frequency (Hz) |
| 18 | `f_first_axial` — first axial mode frequency (Hz) |
| 19 | `mean_spacing` — mean axial-mode spacing (Hz) |
| 20 | `std_spacing` — std-dev of axial-mode spacing (Hz) |
| 21 | `modal_overlap` — f_schroeder / f_first_axial |
| 22 | `tang_ax_ratio` — tangential-to-axial ratio (fixed 1.0) |
| 23 | `log_volume` — log₁₀(L·W·H) |

### Target dict `y`

| Key | Shape | Description |
|-----|-------|-------------|
| `metrics` | `[B, 10]` | RT60, DRR, C50, C80, 6× band RT60 |
| `rir` | `[B, max_rir_len]` | Padded/truncated raw RIR waveform |
| `edc` | `[B, max_rir_len]` | Broadband EDC (dB) |
| `edc_mb` | `[B, T, 6]` | Multiband EDC; T = num_time_steps (default 256) |
| `rir_length` | `[B]` long | Original unpadded RIR length in samples |

---

## 4. Phase 1 — Data Pipeline (`data.py`)

### 4.1 Module-Level Functions

#### `compute_edc(rir: np.ndarray) → np.ndarray`

- **Purpose:** Compute the Schroeder Energy Decay Curve in dB.
- **Input:** `rir` — mono float32 waveform array of arbitrary length.
- **Output:** `edc` — same-length float32 array in dB.
- **Side effects:** None.
- **Algorithm:**

$$E[n] = \sum_{k=n}^{N-1} rir[k]^2$$
$$EDC_{dB}[n] = 10\log_{10}\!\left(\frac{E[n]}{\max_n(E[n])+\epsilon}+\epsilon\right), \quad \epsilon=10^{-12}$$

---

#### `downsample_edc_tensor(edc: np.ndarray, num_time_steps: int = 256) → np.ndarray`

- **Purpose:** Reduce EDC to fixed temporal resolution via integer linear interpolation.
- **Input:** `edc` — any-length array; `num_time_steps` — target resolution.
- **Output:** `edc[idx]` — downsampled array with shape `[num_time_steps]`.
- **Algorithm:**
$$idx_i = \left\lfloor \frac{(N-1)\,i}{T-1} \right\rfloor, \quad i=0\ldots T-1$$

---

#### `compute_multiband_edc(rir, sr=16000, num_time_steps=256) → np.ndarray`

- **Purpose:** Build `[T, 6]` multiband EDC for the six octave bands.
- **Output:** `[T, 6]` float32 (current implementation duplicates broadband across all 6 bands — stub for per-band filtering).
- **Algorithm:**
$$EDC_{mb}[t,b] = EDC_{ds}[t] \quad \forall b$$

---

#### `compute_room_modes(L, W, H) → np.ndarray`

- **Purpose:** Analytically derive 8-dimensional modal feature vector.
- **Input:** Room dimensions in metres. Raises `ValueError` if any dimension ≤ 0.
- **Output:** `[8]` float32 vector.
- **Algorithm:**

Axial mode frequencies:
$$f_{axial}(n,d) = \frac{c}{2}\cdot\frac{n}{d}, \quad n=1\ldots 5,\; d\in\{L,W,H\},\; c=343\;\text{m/s}$$

Schroeder frequency:
$$f_s = 2000\sqrt{\frac{0.161\,V}{S}}, \quad V=LWH,\; S=LW+WH+LH$$

Returned vector:
$$\mathbf{m} = [n_{<300},\; f_s,\; f_{ax,1},\; \bar{\Delta}f,\; \sigma_{\Delta f},\; f_s/f_{ax,1},\; 1.0,\; \log_{10}V]$$

---

#### `_decode_audio(audio: Dict) → np.ndarray`

- **Purpose:** Normalise HuggingFace audio dict to float32 waveform.
- **Three paths:** `array` key (already decoded) → direct cast; `path` key → WAV file read; `bytes` key → in-memory BytesIO WAV.
- **Side effects:** File/IO reads.
- **PCM normalisation:**
$$x_{f32} = x_{i16} / 32768.0 \quad \text{or} \quad x_{i32} / 2^{31}$$

---

#### `_pad_or_truncate(arr, length) → np.ndarray`

Truncates or zero-pads to exactly `length` samples. No side effects.

---

#### `_safe_spacing(xs) → (float, float)`

Returns `(mean, std)` of first-difference of sorted values. Returns `(0, 0)` when `len(xs) < 2`.

---

### 4.2 `RIRMegaDataset(Dataset)`

**State:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `split` | str | `"train"`, `"val"`, or `"test"` |
| `_hf_ds` | HF Dataset | HuggingFace dataset handle (audio column lazy, decode=False) |
| `_meta` | DataFrame | Filtered metadata CSV rows for this split |
| `_meta_id_to_pos` | Dict[str→int] | Lookup from sample ID to metadata row position |
| `_index_map` | List[(int,int)] | Aligned `(hf_idx, meta_idx)` pairs |
| `max_rir_len` | int | Truncation/pad length (default 32,000) |
| `num_time_steps` | int | EDC temporal resolution (default 256) |
| `sample_rate` | int | Audio sample rate in Hz (default 16,000) |

**`_build_index()` — alignment algorithm:**  
For each audio entry in the HF dataset, extract the sample ID from the file-path basename. Look up the ID in `_meta_id_to_pos`. Collect matched `(hf_idx, meta_idx)` pairs. Asserts at least one match.

**`__getitem__(idx)` — per-sample pipeline:**
1. Decode raw audio → float32 via `_decode_audio`.
2. Pad/truncate to `max_rir_len`.
3. Compute `edc` (broadband), `edc_mb` (multiband `[T,6]`).
4. Parse metadata row → room_size, source, microphone, absorption, band_abs.
5. Compute `modal_feats` from `compute_room_modes`.
6. Concatenate into `x ∈ ℝ^{24}` and assert shape.
7. Build `y` dict: `metrics [10]`, `rir`, `edc`, `edc_mb`, `rir_length`.

---

### 4.3 `CachedRIRDataset(Dataset)`

**State:** `base` (wrapped dataset) + `_cache: Dict[int, sample]`.

`__getitem__` checks dict on cache miss; reads from `base` and stores. Memory grows linearly with distinct sample access.

---

### 4.4 `rir_collate_fn`

Stacks `x` tensors along dim-0; for each key in `y`, stacks matching tensors. Returns `(x_batch [B,24], y_batch)`.

---

### 4.5 `get_dataloader`

Factory that constructs `RIRMegaDataset` (optionally wrapped in `CachedRIRDataset`) and configures a `DataLoader`:
- `shuffle=True` only for train split.
- If `num_workers > 0`, sets `prefetch_factor=2` and `persistent_workers=True`.
- `pin_memory=True` when CUDA is available.

---

## 5. Phase 2 — Neural Architectures (`models.py`)

### 5.1 `_hadamard_matrix(n: int) → Tensor`

Recursively constructs a normalized Hadamard matrix for power-of-two `n`:

$$H_1 = [1],\quad H_{2k} = \frac{1}{\sqrt{2k}}\begin{bmatrix}H_k & H_k \\ H_k & -H_k\end{bmatrix}$$

Final result is divided by $\sqrt{n}$ to be orthonormal. Raises `ValueError` for non-power-of-two input.

---

### 5.2 `SirenLayer(nn.Module)`

**Purpose:** Sinusoidal representation (SIREN) layer with smooth, infinitely differentiable outputs — required for `torch.autograd.grad`-based physics residuals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_features` | — | Input width |
| `out_features` | — | Output width |
| `omega_0` | 30.0 | Frequency multiplier |
| `is_first` | False | Controls weight initialization scale |

**Forward:**
$$y = \sin(\omega_0 \cdot (Wx + b))$$

**Weight initialization (SIREN-specific):**
- First layer: $W \sim \mathcal{U}(-1/n_{in}, 1/n_{in})$
- Hidden layers: $W \sim \mathcal{U}\!\left(-\sqrt{6/n_{in}}/\omega_0,\; \sqrt{6/n_{in}}/\omega_0\right)$

---

### 5.3 `SIRENCoordinateNet(nn.Module)`

**Purpose:** SIREN MLP mapping $(x,y,z,t) \in \mathbb{R}^4 \to (p, u_x, u_y, u_z) \in \mathbb{R}^4$.  
Used as the coordinate network in collocation PINN training.

| Param | Default | Description |
|-------|---------|-------------|
| `hidden_dim` | 64 | Width per hidden layer |
| `num_layers` | 3 | Number of hidden SIREN layers |
| `omega_0` | 30.0 | Frequency multiplier |

**State:** Sequential `SirenLayer` stack + final `nn.Linear(hidden_dim, 4)`.

**Forward:** `[N, 4] → stack of SIREN layers → linear out → [N, 4]`.

---

### 5.4 `MultibandEDCPredictor(nn.Module)`

**Purpose:** Main model — maps room parameters to monotonically-decreasing multiband EDC.

| Param | Default | Description |
|-------|---------|-------------|
| `input_dim` | 24 | Feature vector dimension |
| `hidden_dim` | 512 | Encoder and LSTM width |
| `num_layers` | 3 | LSTM depth |
| `num_time_steps` | 256 | Output temporal resolution |
| `num_bands` | 6 | Output frequency bands |
| `dropout` | 0.05 | Dropout probability |

**Internal state:**
| Component | Description |
|-----------|-------------|
| `input_norm` | `LayerNorm(24)` — works for any batch size |
| `encoder` | 2-layer MLP with LayerNorm + ReLU |
| `h0_proj` / `c0_proj` | Linear projections from encoder output to LSTM initial state |
| `time_embed` | Learnable `[1, T, hidden_dim]` positional tensor |
| `lstm` | `nn.LSTM(hidden_dim, hidden_dim, num_layers)` |
| `head` | 2-layer MLP → `[hidden//2, num_bands]` |

**Forward algorithm:**
```
x [B,24] → LayerNorm → encoder [B,H]
  → h0_proj, c0_proj → reshape to [num_layers, B, H]
  → LSTM(time_embed.expand(B,-1,-1), (h0, c0)) → out [B,T,H]
  → head(out) → log_dec [B,T,bands]
  → decrements = softplus(log_dec) * 0.5
  → edc_pred = -cumsum(decrements, dim=1)  [B,T,bands]
```

**Monotonicity guarantee:**

$$\hat{e}_{t,b} = -\sum_{\tau=0}^{t}\underbrace{\text{softplus}(z_{\tau,b})}_{\geq 0}\cdot 0.5 \quad \Rightarrow \quad \hat{e}_{0,b} \geq \hat{e}_{1,b} \geq \cdots$$

---

### 5.5 `DifferentiableFDN(nn.Module)`

**Purpose:** Differentiable approximation of a Feedback Delay Network via per-delay exponential smoothers with sigmoid-constrained, log-space delay parameters.

| Param | Default | Description |
|-------|---------|-------------|
| `num_delays` | 16 | Number of FDN channels |
| `max_delay_ms` | 50.0 | Upper bound on delay in ms |
| `sample_rate` | 16,000 | Audio sample rate (Hz) |
| `output_length` | 4,000 | Samples in output |

**Learnable parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `log_kappa` | `[D]` | Unbounded; mapped via sigmoid to delay samples |
| `alpha_raw` | `[D]` | Unbounded; mixed contribution weight |
| `beta_raw` | `[D]` | Unbounded; direct contribution weight |

**Buffer:** `H [D,D]` — truncated normalized Hadamard feedback matrix (fixed, non-learnable).

**Forward algorithm:**

$$\kappa_d = 1 + \sigma(\log\kappa_d)\cdot(\kappa_{max}-1), \quad \kappa_{max} = \frac{max\_delay\_ms \cdot sr}{1000}$$
$$\delta_d = \min(e^{-1/\kappa_d},\; 0.9999)$$

Per-delay recurrence:
$$s_t^{(d)} = \delta_d\,s_{t-1}^{(d)} + x_t$$

Hadamard mixing:
$$\tilde{s}^{(d)} = \sum_{j=0}^{D-1} H_{dj}\, s^{(j)}$$

Output blend:
$$y_t = \frac{1}{D}\sum_{d=0}^{D-1}\left(\alpha_d\,\tilde{s}_t^{(d)} + \beta_d\,s_t^{(d)}\right)$$

---

### 5.6 `EarlyReflectionNet(nn.Module)`

**Purpose:** Models the first ≈2.7 ms (43 taps @ 16 kHz) via a learnable 1-D convolution (delayed-sum).

**Forward:**  
Kernel `[1,1,43]` applied via `F.conv1d` with `padding=n_taps-1`, then trimmed to input length.

---

### 5.7 U-Net Components

#### `ConvBlock1D(nn.Module)`

`Conv1d → GroupNorm → ReLU → Conv1d → GroupNorm → ReLU`  
GroupNorm (`num_groups = min(out_ch, 8)`) is used instead of BatchNorm to support batch size 1.

#### `EncoderBlock(nn.Module)`

`ConvBlock1D` → `MaxPool1d(2)`. Returns `(feat, pooled)` for skip connections.

#### `DecoderBlock(nn.Module)`

`ConvTranspose1d(stride=2)` upsampling + optional `F.interpolate` to match skip size → concatenate → `ConvBlock1D`.

#### `SinusoidalPosEncoding(nn.Module)`

Standard sinusoidal positional encoding added to the sequence:
$$PE_{t,2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right),\quad PE_{t,2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

Registered as a buffer `pe [1, max_len, d_model]`.

#### `MultiHeadAttentionBottleneck(nn.Module)`

`nn.MultiheadAttention` (self-attention) with residual + LayerNorm:
$$y = \text{LayerNorm}(x + \text{MHA}(x, x, x))$$

#### `UNetRefiner(nn.Module)`

**Purpose:** Compact U-Net residual refiner applied after FDN synthesis; makes micro-adjustments without overwriting macroscopic decay structure.

**Architecture:**
```
enc1(1→base) → enc2(base→2b) → bottleneck(2b→4b) → dec2(4b+2b→2b) → dec1(2b+b→b) → out(b→1)
```
Residual output: $y = x + \gamma \cdot \text{UNet}(x)$, where $\gamma$ is a scalar `nn.Parameter` initialized to `0.01`.

Runs in `torch.autocast("cuda")` context for mixed-precision.

---

## 6. Phase 3 — Physics-Informed Loss (`loss.py`)

### 6.1 `EDCReconstructionLoss(nn.Module)`

**Purpose:** Weighted RMSE between predicted and target EDC (dB), with early-emphasis and slope-matching.

| Param | Default | Description |
|-------|---------|-------------|
| `early_weight` | 3.0 | Weight multiplier at t=0 |
| `slope_weight` | 0.5 | Weight for slope-matching penalty |
| `decay_rate` | 5.0 | Controls speed of weight decay toward 1 |

**Forward:**

Time-dependent weighting:
$$w_t = 1 + (w_0 - 1)\exp\!\left(-\lambda \frac{t}{T}\right), \quad w_0=3,\; \lambda=5$$

Main loss:
$$\mathcal{L}_{RMSE} = \sqrt{\mathbb{E}\!\left[(w_t(\hat{e}_t - e_t))^2\right] + \epsilon}$$

Slope-matching penalty:
$$\mathcal{L}_{slope} = \sqrt{\mathbb{E}\!\left[(\Delta\hat{e}_t - \Delta e_t)^2\right] + \epsilon}, \quad \Delta e_t = e_{t+1}-e_t$$

Combined:
$$\mathcal{L}_{edc} = \mathcal{L}_{RMSE} + \lambda_s\,\mathcal{L}_{slope}$$

Supports both `[B, T]` broadband and `[B, T, F]` multiband tensors via automatic weight broadcasting.

---

### 6.2 `continuity_residual(pred) → scalar`

**Purpose:** Approximate acoustic continuity residual over a temporal EDC sequence.

**Autograd path (when `requires_grad=True`):**

Constructs synthetic time coordinate sharing the computation graph, then:
$$r_c = \mathbb{E}\!\left[\left(\frac{\partial (\hat{e} \cdot t)}{\partial t}\right)^2\right]$$

**Fallback (no grad_fn):** Finite-difference approximation:
$$r_c \approx \mathbb{E}[|\hat{e}_{t+1} - \hat{e}_t|]$$

---

### 6.3 `momentum_residual(pred) → scalar`

**Purpose:** Approximate linearized momentum residual (second-order temporal smoothness).

**Autograd path:** Second-order autograd derivative:
$$r_m = \mathbb{E}\!\left[\left(\frac{\partial^2(\hat{e}\cdot t)}{\partial t^2}\right)^2\right]$$

**Fallback:** Second-order finite difference:
$$r_m \approx \mathbb{E}[(\hat{e}_{t+2} - 2\hat{e}_{t+1} + \hat{e}_t)^2]$$

---

### 6.4 `acoustic_continuity_residual(pressure, velocity, coords, time, ρ₀, c) → [N,1]`

**Purpose:** Point-wise residual of the full acoustic continuity PDE at collocation points.

**Equation:**
$$\frac{\partial p}{\partial t} + \rho_0 c^2 \nabla \cdot \mathbf{u} = 0$$

**Implementation:**
$$\nabla \cdot \mathbf{u} = \sum_{i=1}^{3}\frac{\partial u_i}{\partial x_i}$$

Both derivatives computed via `torch.autograd.grad` with `create_graph=True`. Falls back to zeros when `allow_unused=True` triggers.

**Input/Output:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `pressure` | `[N,1]` | Acoustic pressure at N points |
| `velocity` | `[N,3]` | Particle velocity $(u_x, u_y, u_z)$ |
| `coords` | `[N,3]` | Spatial coords — must have `requires_grad=True` |
| `time` | `[N,1]` | Temporal coord — must have `requires_grad=True` |
| **Returns** | `[N,1]` | Point-wise residual |

---

### 6.5 `acoustic_momentum_residual(pressure, velocity, coords, time, ρ₀) → [N,3]`

**Purpose:** Point-wise residual of the linearized Euler momentum equation.

**Equation:**
$$\rho_0\frac{\partial \mathbf{u}}{\partial t} + \nabla p = 0$$

**Returns:** `[N,3]` per-component residual.

---

### 6.6 `CollocationPhysicsLoss(nn.Module)`

**Purpose:** Proper PINN formulation — evaluates PDE residuals at randomly sampled spatial-temporal collocation points.

| Param | Default | Description |
|-------|---------|-------------|
| `coord_net` | — | `SIRENCoordinateNet` mapping `[N,4]→[N,4]` |
| `lambda_cont` | 0.01 | Continuity residual weight |
| `lambda_mom` | 0.01 | Momentum residual weight |
| `rho0` | 1.225 | Air density (kg/m³) |
| `c` | 343.0 | Speed of sound (m/s) |

**Forward algorithm:**
1. Compute batch-mean room dimensions as bounding box.
2. Sample `N` spatial collocation points: $\mathbf{x} \sim \mathcal{U}([0, L] \times [0, W] \times [0, H])$.
3. Sample time: $t \sim \mathcal{U}([0, 2])$ s.
4. Query SIREN net: $(p, \mathbf{u}) = \text{coord\_net}(\mathbf{x}, t)$.
5. Evaluate PDE residuals via `acoustic_continuity_residual` and `acoustic_momentum_residual`.
6. Return:

$$\mathcal{L}_{coll} = \lambda_{cont}\,\mathbb{E}[r_{cont}^2] + \lambda_{mom}\,\mathbb{E}[r_{mom}^2]$$

---

### 6.7 `MultiResolutionSTFTLoss(nn.Module)`

**Purpose:** Multi-resolution spectral loss over multiple STFT window lengths; avoids phase-wraparound via rectangular-coordinate phase.

**Window lengths:** default `[512, 1024, 2048]`.

**`_stft_mag_phase(x, fft_size, hop, win_length)`:**
- Computes Hann-windowed STFT → complex tensor.
- Returns magnitude, and phase in rectangular form: $(\cos\theta, \sin\theta)$ where $\theta$ is the STFT phase angle.

**Forward (per window length $w$, hop $= w/4$):**
$$\mathcal{L}_w = L_1(|X_w|, |Y_w|) + \text{MSE}(\log|X_w|, \log|Y_w|) + \text{MSE}(\cos\phi_X, \cos\phi_Y) + \text{MSE}(\sin\phi_X, \sin\phi_Y)$$

$$\mathcal{L}_{STFT} = \frac{1}{|W|}\sum_{w\in W}\mathcal{L}_w$$

---

### 6.8 `PhysicsInformedRIRLoss(nn.Module)`

**Purpose:** Combined loss; delegates to `EDCReconstructionLoss` + optional temporal physics residuals.

$$\mathcal{L}_{total} = \mathcal{L}_{edc} + \lambda_{cont}\,r_c(\hat{e}) + \lambda_{mom}\,r_m(\hat{e})$$

The coefficients `lambda_cont` and `lambda_mom` are mutable at runtime — mutated by curriculum scheduling in `RIRTrainer._apply_curriculum`.

---

## 7. Phase 4–6 — Synthesis Pipeline (`synthesis.py`)

### 7.1 `ConditionedFDN(nn.Module)`

Thin wrapper over `DifferentiableFDN`. Accepts `edc_1d [B,T]` and optional `params` dict. Currently passes through to FDN without applying params (conditioning hook for future extension).

---

### 7.2 `EarlyReflections(nn.Module)`

Learnable 43-tap delayed-sum convolution for the early reflection portion (first ~2.7 ms). Identical to `EarlyReflectionNet` in `models.py` but parametrized by `gains` (initialized to zero rather than small random values).

---

### 7.3 `EDCToFDNMapper(nn.Module)`

**Purpose:** Encode multiband EDC statistics into FDN delay, alpha, beta parameters.

**Internal components:**
| Sub-module | Input → Output | Description |
|------------|----------------|-------------|
| `edc_encoder` | `[B, 12]` → `[B, 64]` | Encodes `[edc_mean ‖ edc_slope]` |
| `delay_head` | `[B, 67]` → `[B, D]` | Outputs `log_kappa` |
| `alpha_head` | `[B, 64]` → `[B, D]` | Outputs `alpha_raw` |
| `beta_head` | `[B, 64]` → `[B, D]` | Outputs `beta_raw` |

**Feature extraction:**
$$\text{edc\_mean} = \frac{1}{T}\sum_t \hat{e}_{t,:}, \quad \text{edc\_slope} = \bar{e}_{3T/4:} - \bar{e}_{:T/4}$$

**Forward:** Returns `{"log_kappa": [B,D], "alpha_raw": [B,D], "beta_raw": [B,D]}`.

---

### 7.4 `SignStickyPhaseReconstructor(nn.Module)`

**Purpose:** Reconstruct time-domain waveform from a single-channel EDC by assigning signs with temporal correlation (high stickiness → fewer sign flips → physically realistic oscillation).

| Param | Default | Description |
|-------|---------|-------------|
| `stickiness` | 0.90 | Prob. sign *stays the same* at each step |
| `seed` | None | Optional per-device reproducible seed |

**Forward:**

Amplitude from EDC differences:
$$a_t = \sqrt{\max(\hat{e}_t - \hat{e}_{t+1},\; 0)}$$

Sign generation (`_sticky_signs`):
$$f_t \sim \text{Bernoulli}(1-\text{stickiness}), \quad c_t = \sum_{\tau\le t}f_\tau$$
$$s_t = 1 - 2(c_t \bmod 2)$$

$$\hat{rir}_t = a_t \cdot s_t$$

Stickiness effectively controls $q$-factor — high stickiness produces fewer sign flips and more resonant sounding decay.

---

### 7.5 `MultibandSignStickyPhaseReconstructor(nn.Module)`

**Purpose:** Apply sign-sticky reconstruction independently per octave band, then average across bands. Preserves individual band decay rates; fixes metallic artefacts from single broadband envelope.

**Forward:**

For band $b$: $\hat{rir}_{:,b} = \text{SignSticky}(\hat{e}_{:,b})$

$$\hat{rir} = \frac{1}{B_{bands}}\sum_{b=0}^{B_{bands}-1}\hat{rir}_{:,b}$$

---

### 7.6 `RIRSynthesiser(nn.Module)`

**Purpose:** End-to-end synthesis chain; room parameters → peak-normalised RIR waveform.

| Param | Default | Description |
|-------|---------|-------------|
| `lstm` | — | Trained `MultibandEDCPredictor` |
| `num_delays` | 16 | FDN channels |
| `sample_rate` | 16,000 | Hz |
| `output_length` | 32,000 | Output samples |
| `use_unet` | False | Enable U-Net refiner |
| `stickiness` | 0.90 | Sign-sticky parameter |
| `train_fdn` | True | Whether FDN path (vs phase-recon fallback) is used |

**Sub-modules:** `lstm`, `mapper` (EDCToFDNMapper), `fdn` (ConditionedFDN), `early` (EarlyReflections), `mb_phase_recon` (MultibandSignStickyPhaseReconstructor), optional `unet`.

**Forward (when `train_fdn=True`):**
```
edc_pred = lstm(x)              [B, T, 6]
edc_1d   = edc_pred.mean(dim=2) [B, T]
params   = mapper(edc_pred, x[:,:3])
late     = fdn(edc_1d, params)  [B, L_late]
early    = early(edc_1d)        [B, L_early]
rir_out  = early[:,:L] + late[:,:L]
```

**Forward (when `train_fdn=False` — phase-recon fallback):**
```
edc_mb_clamped = edc_pred.clamp(min=0)
phase  = mb_phase_recon(edc_mb_clamped)  [B, T-1]
rir_out = (early[:,:L] + late[:,:L]) * phase[:,:L]
```

**Post-processing:**
- Optional U-Net: `rir_out = unet(rir_out.unsqueeze(1)).squeeze(1)`.
- Peak normalisation:
$$\hat{rir}_{norm} = \frac{\hat{rir}}{\max_t|\hat{rir}_t|+\epsilon}$$

**Returns:** `{"rir": [B,L], "edc_pred": [B,T,6], "fdn_params": dict, "phase": [B,T-1] if requested}`.

---

## 8. Phase 5 — Training Harness (`trainer.py`)

### 8.1 `TrainingConfig` (dataclass)

All hyperparameters stored as a flat dataclass. Serializable via `asdict()`.

**Groups:**

| Group | Key Fields |
|-------|-----------|
| Data | `batch_size`, `num_workers`, `max_rir_len`, `sample_rate`, `use_cache`, `hf_cache_dir` |
| Model | `hidden_dim`, `num_layers`, `num_time_steps`, `num_bands`, `model_dropout` |
| FDN | `train_fdn`, `fdn_num_delays`, `fdn_max_delay_ms`, `fdn_output_length`, `fdn_weight` |
| Loss | `lambda_cont`, `lambda_mom`, `use_mr_stft`, `mr_stft_weight`, `mr_stft_windows` |
| Optimizer | `lr`, `weight_decay`, `grad_clip` |
| Scheduler | `scheduler_patience`, `scheduler_factor` |
| Curriculum | `use_curriculum_ramp`, `physics_ramp_start_epoch`, `physics_ramp_end_epoch`, `lambda_cont_target`, `lambda_mom_target` |
| FDN curriculum | `fdn_curriculum_length`, `fdn_curriculum_end_epoch` |
| Collocation PINN | `use_collocation`, `collocation_n_points`, `collocation_lambda_cont`, `collocation_lambda_mom`, `siren_hidden_dim`, `siren_num_layers` |
| U-Net | `use_unet`, `unet_weight` |
| Misc | `epochs`, `seed`, `dry_run`, `save_metrics_path`, `use_amp` |

---

### 8.2 `RIRTrainer`

**State after `_build_components()`:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `train_loader` / `val_loader` | `DataLoader` | Training/validation data |
| `lstm` | `MultibandEDCPredictor` | Main predictor |
| `criterion` | `PhysicsInformedRIRLoss` | Primary loss |
| `fdn` | `DifferentiableFDN` or None | Optional FDN |
| `early` | `EarlyReflectionNet` or None | Optional early reflections |
| `unet_refiner` | `UNetRefiner` or None | Optional refiner |
| `collocation_loss` | `CollocationPhysicsLoss` or None | Optional PINN loss |
| `phase_recon` | `SignStickyPhaseReconstructor` | Broadband phase recon |
| `mb_phase_recon` | `MultibandSignStickyPhaseReconstructor` | Multiband phase recon |
| `optimiser` | `Adam` | All trainable parameters |
| `scheduler` | `CosineAnnealingWarmRestarts` | LR scheduler (T₀ = max(10, epochs//5)) |
| `scaler` | `GradScaler` | AMP gradient scaler |
| `mr_stft_loss` | `MultiResolutionSTFTLoss` or None | Optional MR-STFT |

---

### 8.3 `_apply_curriculum(epoch)`

Linear curriculum ramp for physics loss weights:

$$\alpha = \begin{cases}0 & \text{if } epoch \le e_{start} \\ \frac{epoch - e_{start}}{e_{end} - e_{start}} & \text{if } e_{start} < epoch < e_{end} \\ 1 & \text{if } epoch \ge e_{end}\end{cases}$$

$$\lambda_{cont}^{(e)} = \lambda_{cont}^{target}\cdot\alpha, \quad \lambda_{mom}^{(e)} = \lambda_{mom}^{target}\cdot\alpha$$

---

### 8.4 `train_one_epoch(epoch) → Dict`

**Algorithm per batch:**
1. Forward `lstm(x)` → `edc_pred`.
2. Compute `criterion(edc_pred, edc_target)`.
3. If `train_fdn`: synthesize `rir_pred` → `F.mse_loss(rir_pred, rir_target)` weighted by `fdn_weight`; optional MR-STFT loss.
4. If `use_collocation`: compute `collocation_loss(room_dims)`.
5. Total loss: `loss_edc + fdn_weight*unet_weight*fdn_loss + mr_stft_weight*mr_loss + coll_loss`.
6. `scaler.scale(loss).backward()`.
7. Record `fdn.log_kappa.grad` norm.
8. `clip_grad_norm_(lstm.parameters(), grad_clip)`.
9. `scaler.step(optimiser); scaler.update()`.
10. Return averaged `{total, fdn, log_kappa_grad_norm, lambda_cont, lambda_mom}`.

---

### 8.5 `validate() → Dict`

Forward-only pass. Computes `criterion` + `fdn_loss` for all validation batches. For first `metrics_eval_batches` batches also computes:
- `rt60_error` — |RT60(pred) − RT60(ref)|
- `lsd` — Log Spectral Distance
- `edc_rmse` — EDC RMSE in dB

Returns averaged metrics.

---

### 8.6 `fit() → Dict`

Full training loop:
1. Seed RNG if `cfg.seed` is set.
2. Dry-run shortcut: one synthetic step, returns stub history.
3. Epoch loop: `train_one_epoch` → `validate` → `scheduler.step()`.
4. Post-training FDN plateau detection: if `max(log_kappa_grad_norm) < threshold`, warn and optionally multiply `max_delay_ms` by 1.5.
5. Optional JSON metrics dump.
6. Returns `history` dict with per-epoch arrays.

---

## 9. Utilities & Evaluation (`utils.py`, `train.py`)

### 9.1 Acoustic Metrics

#### `estimate_rt60(rir, sample_rate=16000) → float`

Extracts RT60 via T20 linear regression on EDC:
1. Find time at −5 dB and −25 dB.
2. $RT60 = 3 \times (t_{-25} - t_{-5})$ (Sabine-style extrapolation from T20).
Returns 0.0 if regression range is degenerate.

#### `log_spectral_distance(rir_pred, rir_ref) → float`

$$LSD = \sqrt{\frac{1}{N}\sum_{k=0}^{N/2}\left(20\log_{10}|X[k]| - 20\log_{10}|Y[k]|\right)^2}$$

Uses `np.fft.rfft` on zero-padded (to common length) signals.

#### `edc_rmse_db(rir_pred, rir_ref) → float`

$$EDC\text{-}RMSE = \sqrt{\frac{1}{N}\sum_t(EDC_{pred}[t]-EDC_{ref}[t])^2}$$

#### `compute_drr(rir, sample_rate, direct_ms=2.5) → float`

$$DRR = 10\log_{10}\!\frac{\sum_{n < n_{direct}}rir[n]^2 + \epsilon}{\sum_{n \geq n_{direct}}rir[n]^2 + \epsilon}$$

where $n_{direct} = \lceil (2.5\times10^{-3})\cdot sr \rceil$.

---

### 9.2 Inference & Loading

#### `load_synthesiser(checkpoint_dir, ...)`

Constructs `MultibandEDCPredictor` + `RIRSynthesiser`. Loads `best_lstm.pt` and `best_fdn.pt` if found. Uses `weights_only=True` for safe deserialization.

#### `generate_rir_from_params(synth, x, device)`

Validates `x.shape[-1] == 24`. Runs `synth.eval()` in `torch.no_grad()`. Returns detached CPU tensor dict.

#### `evaluate_on_test_set(synth, loader, sample_rate, device)`

Full-dataset evaluation: RT60 error, LSD, EDC RMSE, DRR averaged over the test set.

---

### 9.3 CLI Entrypoint (`train.py`)

`build_parser()` dynamically creates CLI args from `TrainingConfig` fields via `dataclasses.fields`.  
`_str_to_bool` handles boolean flags from strings.  
`_coerce_optional_seed` converts string `"none"` / `"null"` to Python `None`.  
`main()` constructs config, trainer, calls `trainer.fit()`, and JSON-prints history.

---

### 9.4 Visualization Functions

| Function | Inputs | Plot type |
|----------|--------|-----------|
| `plot_training_curves` | history dict | Train/val loss vs epoch |
| `plot_multiband_edc` | `[T,6]` edc | Per-band EDC curves |
| `plot_rir_waveform` | pred + optional ref | Time-domain overlay |
| `plot_edc_with_rt60` | RIR | EDC with RT60 annotation |
| `plot_spectrogram_comparison` | pred + ref | Side-by-side spectrograms |
| `plot_results_table` | metrics dict | Tabular display |
| `plot_per_band_rt60` | band→RT60 dict | Bar chart |
| `visualise_demo` | demo output dict | RIR waveform + EDC |

All functions call `_save_or_show(save_path)` — saves to file if path given, else `plt.show()`.

### 9.5 Persistence Helpers

| Function | Action |
|----------|--------|
| `save_checkpoint(state_dict, name)` | `torch.save` + return absolute path |
| `save_metrics(dict, name)` | JSON dump |
| `save_history(dict, name)` | JSON dump |
| `save_rir_audio(rir, sr, name)` | Peak-normalize → int16 → WAV |
| `save_figure(fig_or_path, name)` | matplotlib save or file copy |
| `backup_notebook(name)` | Byte-copy `*.ipynb` to `*_backup.ipynb` |

### 9.6 `set_seed(seed, deterministic=True)`

Sets RNG state for Python `random`, NumPy, and all PyTorch devices. When `deterministic=True`, enables `torch.use_deterministic_algorithms(True)` and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`. Silently ignores operators that lack deterministic implementations.

---

## 10. Mathematical Specification (Summary)

| Symbol | Definition |
|--------|-----------|
| $E[n]$ | Cumulative energy $\sum_{k=n}^{N-1}rir[k]^2$ |
| $EDC_{dB}$ | $10\log_{10}(E/\max E + \epsilon)$ |
| $f_s$ | Schroeder frequency $2000\sqrt{0.161\,V/S}$ |
| $\hat{e}_{t,b}$ | Predicted EDC at time $t$, band $b$ (monotonic) |
| $\delta_d$ | FDN per-channel decay $e^{-1/\kappa_d}$ |
| $\kappa_d$ | Effective delay samples via sigmoid mapping |
| $w_t$ | Time-dependent EDC loss weight $1+(w_0-1)e^{-\lambda t/T}$ |
| $\mathcal{L}_{edc}$ | Weighted EDC RMSE + slope-matching penalty |
| $\mathcal{L}_{coll}$ | Collocation PINN loss (continuity + momentum MSE) |
| $\mathcal{L}_{STFT}$ | Multi-resolution STFT spectral/phase loss |
| $\mathcal{L}_{total}$ | $\mathcal{L}_{edc} + w_{fdn}\mathcal{L}_{time} + w_{stft}\mathcal{L}_{STFT} + \mathcal{L}_{coll}$ |

---

## 11. Cell-Level State Management

| Module | Mutable State | Scope |
|--------|--------------|-------|
| `RIRMegaDataset` | None (after init) | Dataset lifetime |
| `CachedRIRDataset` | `_cache: Dict[int, sample]` | Grows with access |
| `MultibandEDCPredictor` | `encoder`, `lstm`, `head` weights; `time_embed` | Gradient-updated |
| `DifferentiableFDN` | `log_kappa`, `alpha_raw`, `beta_raw`; FDN recurrent state (local) | Gradient-updated; recurrence cleared per forward |
| `UNetRefiner` | `enc1..dec1` weights; `gamma` scalar | Gradient-updated |
| `SIRENCoordinateNet` | SIREN layer weights | Gradient-updated |
| `SignStickyPhaseReconstructor` | `_generators: Dict[str, Generator]` | Per-device, created lazily |
| `PhysicsInformedRIRLoss` | `lambda_cont`, `lambda_mom` | Mutated by curriculum |
| `RIRTrainer` | All above + `history dict`, LR scheduler state | Training lifetime |
| `train.py/main` | None (stateless after return) | Single process |

---

## 12. Validation & System Integrity

### 12.1 Input Contracts
- `INPUT_DIM = 24` asserted at sample build time in `RIRMegaDataset.__getitem__`.
- Room dimensions validated positive before modal feature extraction.
- Dataset-metadata alignment asserted — zero matches triggers `AssertionError` at startup.
- FDN Hadamard matrix size must be power-of-two.

### 12.2 Numerical Guards
| Location | Guard | Purpose |
|----------|-------|---------|
| `compute_edc` | `ε = 1e-12` in denominator and `log` | Prevents log(0) |
| `compute_room_modes` | `max(…, 1e-9)` in Schroeder | Prevents divide-by-zero |
| `DifferentiableFDN` | `decay.clamp(0, 0.9999)` | Prevents marginally stable FDN |
| `RIRSynthesiser` | `peak / (max + 1e-8)` normalisation | Prevents inf after peak normalisation |
| MR-STFT | `1e-7` in log-magnitude | Prevents log(0) in spectral loss |
| `compute_drr` | `+ 1e-8` in numerator and denominator | Prevents NaN for silent windows |

### 12.3 Physical Consistency
- EDC predictions are monotonically non-increasing by construction (softplus cumsum).
- FDN delays are bounded in `(0, max_delay_ms)` via sigmoid — prevents integer-programming plateau.
- Physics PDE residuals (`acoustic_continuity_residual`, `acoustic_momentum_residual`) enforce wave equation compliance at collocation points.

### 12.4 Training Stability
- Gradient clipping: `clip_grad_norm_(lstm.parameters(), grad_clip=1.0)`.
- AMP (automatic mixed precision) via `torch.amp.GradScaler`.
- Cosine Annealing with Warm Restarts avoids LR plateau.
- FDN plateau detection: post-training check on `log_kappa` gradient norms; auto-expands delay range if stalled.
- Curriculum ramp prevents physics loss from dominating early training before EDC prediction converges.

### 12.5 Automated Tests (`tests/`)

| Test file | Coverage |
|-----------|----------|
| `test_architecture_upgrades.py` | SIREN layer bounds, EarlyReflectionNet shape/grad, MR-STFT correctness, autograd residuals, CollocationPhysicsLoss, MultibandSignSticky shape/finiteness, UNetRefiner residual correctness/gamma gradients, `compute_drr` NaN guards, `RIRSynthesiser` FDN/fallback paths |
| `test_data_loss_trainer_contracts.py` | Dataset/collate shapes, loss backward, trainer dry-run |
| `test_seed_reproducibility.py` | Deterministic output given fixed seed |

---

*Generated from source: `src/rir_project/{data,models,loss,synthesis,trainer,utils}.py` + `train.py`.*
