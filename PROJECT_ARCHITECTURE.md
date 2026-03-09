# Physics-Informed RIR Generation Framework — Full Architecture & Functionality Document

> **Version:** Post-rebuild (RIR_Project.ipynb — 29 cells, 3,790 lines of inline code)
> **Purpose:** This document is intended for AI-assisted code review (e.g., feeding to Gemini/GPT-4) to identify further improvements. It describes every component, data flow, design decision, known limitation, and performance target.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Notebook Structure (29 cells)](#2-notebook-structure)
3. [Phase 1 — Data Pipeline](#3-phase-1--data-pipeline)
4. [Phase 2 — LSTM EDC Predictor](#4-phase-2--lstm-edc-predictor)
5. [Phase 3 — Physics-Informed Loss](#5-phase-3--physics-informed-loss)
6. [Phase 4 — Differentiable FDN](#6-phase-4--differentiable-fdn)
7. [Phase 5 — Curriculum Training](#7-phase-5--curriculum-training)
8. [Phase 6 — Synthesis & Phase Reconstruction](#8-phase-6--synthesis--phase-reconstruction)
9. [Evaluation & Visualisation](#9-evaluation--visualisation)
10. [Optional UNet Refiner](#10-optional-unet-refiner)
11. [Infrastructure — Drive, HF Token, Checkpointing](#11-infrastructure)
12. [Tensor Schema (Full)](#12-tensor-schema)
13. [Hyperparameter Reference](#13-hyperparameter-reference)
14. [Known Limitations & Suggested Improvements](#14-known-limitations--suggested-improvements)
15. [Performance Targets vs Current Results](#15-performance-targets-vs-current-results)
16. [File Structure](#16-file-structure)

---

## 1. Project Overview

**Goal:** Build an end-to-end differentiable pipeline that maps compact macroscopic room descriptors (geometry, absorption, source/mic positions) directly to full-length Room Impulse Response (RIR) waveforms, guided by acoustic physics priors.

**Why synthetic RIR generation?** Measuring RIRs requires specialised equipment and hours of setup per room configuration. This model replaces costly measurements with learned synthesis conditioned on room parameters.

**Key innovations:**
1. **Physics-Informed Loss** — acoustic continuity equation (∂p/∂t + ρ₀c²∇·u = 0) and linearised Euler momentum equation (ρ₀∂u/∂t + ∇p = 0) residuals added to training loss via `torch.autograd.grad`
2. **Room Acoustic Mode Features** — 8-dimensional modal feature vector derived analytically from room geometry (axial mode count, Schroeder frequency, first axial mode, mode spacing, modal overlap, tangential/axial ratio, log volume)
3. **Differentiable FDN** — 16-channel Feedback Delay Network with sigmoid-constrained delays, learnable per-channel absorption, and Hadamard orthonormal feedback matrix, trained end-to-end
4. **4-Phase Curriculum Training** — warm-up (EDC only) → physics ramp → FDN co-training → fine-tune

**Dataset:** [`mandipgoswami/rirmega`](https://huggingface.co/datasets/mandipgoswami/rirmega) on HuggingFace — ~50 000 measured RIRs with room geometry, source/mic positions, absorption coefficients, and acoustic metrics (RT60, DRR, C50, C80) in a companion metadata CSV.

---

## 2. Notebook Structure

| Cell | Type | Phase | Description |
|------|------|-------|-------------|
| 0 | Code | Infra | **Google Drive mount + HF token + save helpers** |
| 1 | Markdown | — | Project title & Phase overview table |
| 2 | Markdown | — | Related Works (FAST-RIR, Schlecht FDN, Steinmetz, PINN) |
| 3 | Code | Infra | `pip install` dependencies |
| 4 | Code | P1 | Imports & constants (`INPUT_DIM=24`, `METRICS_DIM=10`, etc.) |
| 5 | Code | P1 | Helper functions (EDC, multiband EDC, room modes, audio utils) |
| 6 | Code | P1 | `RIRMegaDataset` class (HF + CSV alignment, HF token) |
| 7 | Code | P1 | `rir_collate_fn` + `get_dataloader` factory |
| 8 | Code | P1 | **Unit test — data pipeline shapes** |
| 9 | Markdown | P2 | Phase 2 overview |
| 10 | Code | P2 | `MultibandEDCPredictor` (MLP encoder + 2-layer LSTM) |
| 11 | Code | P2 | **Unit test — LSTM model shapes + gradient flow** |
| 12 | Markdown | P3 | Phase 3 overview |
| 13 | Code | P3 | `EDCReconstructionLoss`, `continuity_residual`, `momentum_residual`, `PhysicsInformedRIRLoss` |
| 14 | Code | P3 | **Unit test — loss functions + PDE residuals** |
| 15 | Markdown | P4 | Phase 4 overview |
| 16 | Code | P4 | `DifferentiableFDN` (16 delays, Hadamard, fractional delay read) |
| 17 | Code | P4 | **Unit test — FDN shape, delay constraints, gradient flow** |
| 18 | Markdown | P5 | Phase 5 overview |
| 19 | Code | P5 | `TrainingConfig` dataclass + `RIRTrainer` class |
| 20 | Code | P5 | **Launch cell — `DRY_RUN=True` smoke test** |
| 21 | Markdown | P6 | Phase 6 overview |
| 22 | Code | P6 | `EDCToFDNMapper`, `ConditionedFDN`, `SignStickyPhaseReconstructor`, `RIRSynthesiser` |
| 23 | Code | P6 | `generate_rir_from_params`, `load_synthesiser`, `demo_inference` |
| 24 | Code | Eval | `estimate_rt60`, `log_spectral_distance`, `edc_rmse_db`, `compute_drr`, `evaluate_on_test_set` |
| 25 | Code | Viz | Conference-quality plotting helpers (7 plot functions) |
| 26 | Code | P5+ | `CurriculumConfig`, `CurriculumTrainer` (4-phase schedule) |
| 27 | Code | P5+ | **Full pipeline execution** (curriculum training + eval + Drive save, `DRY_RUN=False`) |
| 28 | Code | Opt | `UNetRefiner` (optional 1D U-Net post-processor) |

---

## 3. Phase 1 — Data Pipeline

### Constants
```python
OCTAVE_BANDS      = ["125", "250", "500", "1000", "2000", "4000"]  # 6 bands
DEFAULT_MAX_RIR_LEN = 32_000   # 2 s @ 16 kHz
MODAL_FEAT_DIM    = 8          # room acoustic mode features
INPUT_DIM         = 24         # 16 base + 8 modal
METRICS_DIM       = 10         # RT60, DRR, C50, C80, 6×band-RT60
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### `compute_edc(rir)` → `ndarray[L]`
Schroeder backward integration: `EDC(t) = Σ_{τ=t}^∞ h²(τ)`, normalised to 0 dB at t=0, returned in dB.

### `compute_multiband_edc(rir, sr, bands, num_time_steps)` → `ndarray[T, 6]`
1. 4th-order Butterworth bandpass filter per octave band (lo = fc/√2, hi = fc·√2, clamped to Nyquist)
2. Schroeder EDC per band
3. Strided downsampling `L → num_time_steps` (default 256)

### `compute_room_modes(L, W, H)` → `ndarray[8]`
Enumerates all rectangular room standing-wave modes up to order 6:
`f(nx,ny,nz) = (c/2)·√((nx/L)² + (ny/W)² + (nz/H)²)`

Features:
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `n_below_300` | Mode count below 300 Hz (low-frequency density) |
| 1 | `f_schroeder` | Schroeder frequency: 2000·√(T60_est/V) |
| 2 | `f_first_axial` | Lowest axial mode (Hz) |
| 3 | `mean_axial_spacing` | Mean spacing between axial modes |
| 4 | `std_axial_spacing` | Std of axial mode spacings (irregularity) |
| 5 | `modal_overlap` | Modal overlap factor at 500 Hz |
| 6 | `tang_axial_ratio` | Tangential/axial mode count ratio below 500 Hz |
| 7 | `log10_volume` | log₁₀(L·W·H) for scale normalisation |

### `RIRMegaDataset`
- Loads HF audio with `load_dataset("mandipgoswami/rirmega", token=HF_TOKEN)`
- Downloads `metadata/metadata.csv` with `hf_hub_download(token=HF_TOKEN)`
- Aligns audio (by `rir_NNNNNN` ID extracted from file path) with metadata rows
- Falls back to positional matching if ID extraction fails
- Parses JSON-encoded fields: `room_size`, `source`, `microphone`, `absorption_bands`, `metrics`
- Returns per-sample `(x [INPUT_DIM], y dict)`

**Input vector x [24]:**
`[L, W, H, src_x, src_y, src_z, mic_x, mic_y, mic_z, a_broad, a_125, a_250, a_500, a_1k, a_2k, a_4k, mode_0, …, mode_7]`

**Target dict y:**
```python
{
  "metrics":    Tensor[10],       # RT60, DRR, C50, C80, 6×band-RT60
  "rir":        Tensor[32000],    # padded/truncated RIR
  "edc":        Tensor[32000],    # broadband Schroeder EDC (dB)
  "edc_mb":     Tensor[256, 6],   # multiband EDC (dB)
  "rir_length": LongTensor,       # original sample count
}
```

### `get_dataloader` factory
- Wraps `RIRMegaDataset` with `rir_collate_fn` (stacks tensors, collects y-dicts)
- `pin_memory=True` when CUDA available (10–20% faster transfer)
- Optional `persistent_workers`, `prefetch_factor` for further speed

---

## 4. Phase 2 — LSTM EDC Predictor

### `MultibandEDCPredictor(nn.Module)`

**Architecture:** `x[B,24] → BatchNorm → MLP encoder → LSTM initial states + temporal embeddings → LSTM → output head → edc_pred[B,256,6]`

```
Input [B, 24]
  └─ BatchNorm1d(24)          ← normalise wildly-scaled features
  └─ Linear(24, 256) + LN + ReLU + Dropout
  └─ Linear(256, 256) + LN + ReLU     ← context vector [B, 256]
       ├─ h0_proj → [num_layers, B, 256]   ← LSTM initial hidden state
       ├─ c0_proj → [num_layers, B, 256]   ← LSTM initial cell state
       └─ time_embed [1, T, 256] → expand [B, T, 256]   ← LSTM input
  └─ LSTM(256→256, 2 layers, batch_first)
  └─ Linear(256, 128) + ReLU + Linear(128, 6)
Output [B, T=256, 6]
```

**Weight initialisation:**
- LSTM weights: orthogonal init
- LSTM forget-gate bias: set to 1.0 (better gradient flow through time)
- Linear layers: Xavier uniform

**Key design rationale:** Context vector derived from room parameters conditions the LSTM by setting its initial hidden/cell states, then learned temporal embeddings (`nn.Parameter [1, T, hidden]`) drive time-step generation — this separates spatial conditioning from temporal generation.

**Parameter count:** ~950 000 (at `hidden_dim=256, num_layers=2`)

---

## 5. Phase 3 — Physics-Informed Loss

### `EDCReconstructionLoss`
RMSE between predicted and target multiband EDC (dB):
`L_edc = √(mean((edc_pred - edc_target)²) + ε)`

Supports both `[B, T, F]` (multiband) and `[B, T]` (broadband) tensors.

### `continuity_residual(pressure, velocity, coords, time)`
Residual of acoustic continuity equation: **∂p/∂t + ρ₀c²∇·u = 0**

Computed via `torch.autograd.grad` for exact automatic differentiation (falls back to finite differences in no-grad contexts). Requires `coords` and `time` with `requires_grad=True`.

### `momentum_residual(pressure, velocity, coords, time)`
Residual of linearised Euler momentum equation: **ρ₀∂u/∂t + ∇p = 0**

### `PhysicsInformedRIRLoss`
```
L_total = L_edc + λ_cont · mean(r_cont²) + λ_mom · mean(r_mom²)
```
Physics terms are **optional** — only computed when `pressure/velocity/coords/time` are supplied. Returns dict: `{total, edc, continuity, momentum}`.

**Curriculum usage:** λ_cont and λ_mom start at 0 (Phase A), ramp linearly to targets during Phase B (e.g., 0.01 / 0.005), then stay fixed for Phases C & D.

---

## 6. Phase 4 — Differentiable FDN

### `DifferentiableFDN(nn.Module)`

16-channel Feedback Delay Network synthesising **late reverberation tail**.

**Architecture:**
```
Impulse [B, T]
  └─ Input bus: x_in[t] = impulse[t] * b_in   [B, N]
  └─ Circular delay buffers [B, N, max_delay+2]
     ├─ Fractional delay read (linear interpolation between int taps)
     ├─ Per-channel decay: dl_out * g_i where g_i = α_i·exp(-β_i·m_i/fs)
     ├─ Hadamard mixing: fb = dl_out @ H.T   (unitary, energy-preserving)
     └─ Scatter write (out-of-place for autograd compatibility)
  └─ Output bus: sum(dl_out * c_out, dim=N)   [B, T]
```

**Learnable parameters (all unconstrained, mapped via activations):**
| Parameter | Shape | Constraint | Init |
|-----------|-------|-----------|------|
| `log_delays` | [16] | `sigmoid → (1, max_delay_samples)` | linspace(-2, 2) |
| `alpha_raw` | [16] | `sigmoid → (0, 1)` | 2.94 (≈ 0.95) |
| `beta_raw` | [16] | `softplus → (0, ∞)` | 0.5 |
| `input_gains` | [16] | unconstrained | 1/√16 |
| `output_gains` | [16] | unconstrained | 1/√16 |

**Fixed buffer:** Hadamard matrix `H[16,16]` (normalised, `H @ H.T = I`).

**Fractional delay:** Linear interpolation between `floor(delay)` and `floor(delay)+1` tap — keeps operation differentiable w.r.t. `log_delays`.

**Out-of-place scatter writes:** Uses `buf.scatter(2, idx, new_val)` instead of in-place `buf[:,:,idx] = val` to preserve autograd graph across T steps.

---

## 7. Phase 5 — Curriculum Training

### `TrainingConfig` (dataclass)
All ~30 hyperparameters in one place. Key groups:

| Group | Parameters |
|-------|-----------|
| Data | `batch_size=8, num_workers=2, max_rir_len=32000, sample_rate=16000` |
| LSTM | `hidden_dim=256, num_layers=2, num_time_steps=256, num_bands=6, model_dropout=0.1` |
| FDN | `train_fdn=False, fdn_num_delays=16, fdn_max_delay_ms=50.0, fdn_output_length=3200, fdn_weight=0.1` |
| Loss | `lambda_cont=0.0, lambda_mom=0.0` (off during warm-up) |
| Optimiser | `lr=1e-3, weight_decay=1e-5, grad_clip=1.0` |
| Scheduler | `patience=5, factor=0.5` (ReduceLROnPlateau) |
| Training | `epochs=50, log_every=100, val_every_epoch=1` |

### `RIRTrainer`
Basic trainer — builds dataloaders, LSTM, optional FDN, loss, Adam optimiser, ReduceLROnPlateau scheduler. Used for quick runs or unit tests.

### `CurriculumConfig` / `CurriculumTrainer`
Recommended for full training. 4-phase schedule:

| Phase | Epochs | What's active | λ_cont | λ_mom |
|-------|--------|--------------|--------|-------|
| A — Warm-up | 1–12 | EDC RMSE only | 0.0 | 0.0 |
| B — Physics ramp | 13–30 | EDC + physics (linear ramp) | 0 → 0.01 | 0 → 0.005 |
| C — FDN co-train | 31–42 | EDC + physics + FDN | 0.01 | 0.005 |
| D — Fine-tune | 43–50 | All losses, LR × 0.1 | 0.01 | 0.005 |

**FDN activation (Phase C):** FDN is lazily built and added to the optimiser as a new param group at epoch `fdn_phase_start`. This avoids wasting memory during Phase A/B.

**Checkpoint saving:** Best model (lowest val loss) saved as `best_lstm.pt` / `best_fdn.pt` and copied to Google Drive via `save_checkpoint()`.

**Recommended full-training config:**
```python
cfg = CurriculumConfig(
    batch_size=16, epochs=50,
    warmup_epochs=12, physics_ramp_end=30,
    fdn_phase_start=30, finetune_start=42,
    lambda_cont_target=0.01, lambda_mom_target=0.005,
    train_fdn=True, fdn_weight=0.05,
    fdn_output_length=3200,
    lr=1e-3, grad_clip=1.0,
)
```

---

## 8. Phase 6 — Synthesis & Phase Reconstruction

### `EDCToFDNMapper(nn.Module)`
Maps predicted multiband EDC `[B, T, 6]` + room dims `[B, 3]` → per-sample FDN parameters.

```
edc_pred [B, T, 6]
  └─ mean over T → [B, 6]
  └─ slope (q4 - q1 averages) → [B, 6]
  └─ concat → [B, 12]
  └─ edc_encoder (Linear 12→64 → ReLU → Linear 64→64 → ReLU) → [B, 64]

room_dims [B, 3]
  └─ concat with edc features → [B, 67]
  └─ delay_head → log_delays [B, 16]

h [B, 64]
  └─ alpha_head → alpha_raw [B, 16]
  └─ beta_head  → beta_raw  [B, 16]
```

### `ConditionedFDN(nn.Module)`
Like `DifferentiableFDN` but accepts **per-sample** parameters from `EDCToFDNMapper` instead of shared `nn.Parameter`. Each sample in the batch gets its own delay/decay configuration.

### `SignStickyPhaseReconstructor(nn.Module)`
Converts a broadband EDC (dB) → time-domain waveform:
1. dB → linear energy: `E(t) = 10^(EDC(t)/10)`
2. Reverse-diff: `e_inst(t) = max(0, E(t) - E(t+1))`
3. Amplitude: `a(t) = √(e_inst(t))`
4. Sticky ±1 signs: flip probability = `1/mean_run` (default `mean_run=8`), computed via cumsum mod 2
5. `rir(t) = a(t) × sign(t)`

### `RIRSynthesiser(nn.Module)` — Full pipeline
```
x [B, 24]
  → LSTM → edc_pred [B, 256, 6]
  → EDCToFDNMapper(edc_pred, x[:, :3]) → {log_delays, alpha_raw, beta_raw}
  → ConditionedFDN → fdn_rir [B, T]
  → compute broadband EDC of fdn_rir → edc_fdn_db [B, T]
  → SignStickyPhaseReconstructor(edc_fdn_db) → rir [B, T]

Output dict: {"rir": [B,T], "edc_pred": [B,256,6], optionally "fdn_rir", "edc_fdn_db"}
```

### `generate_rir_from_params()`
Standalone inference: takes Python lists for room/source/mic/absorption, builds the 24-dim input tensor (including computing modal features), returns full output dict.

---

## 9. Evaluation & Visualisation

### Acoustic Metrics

| Function | Description | Unit |
|----------|-------------|------|
| `estimate_rt60(rir, sr)` | Schroeder T20 method: −5 dB to −25 dB, extrapolate × 3 | s |
| `log_spectral_distance(pred, ref, n_fft=2048)` | RMS of log-magnitude difference: `√mean((20·log₁₀|S_pred| - 20·log₁₀|S_ref|)²)` | dB |
| `edc_rmse_db(pred, ref)` | RMSE between Schroeder EDC curves | dB |
| `compute_drr(rir, sr, direct_ms=2.5)` | Direct-to-reverberant ratio: energy in first `direct_ms` vs tail | dB |

### `evaluate_on_test_set(synthesiser, test_loader, ...)`
Runs full test-set evaluation, computes all 4 metrics per sample, returns aggregated `{mean, std, median, n}` per metric.

### Visualisation Functions (7 total)

| Function | Output |
|----------|--------|
| `plot_training_curves(history)` | 3-panel: total loss, component breakdown, smoothed convergence |
| `plot_multiband_edc(edc_pred, edc_ref)` | Predicted (solid) vs reference (dashed) per band, viridis colormap |
| `plot_rir_waveform(pred, ref)` | 1 or 2 panel waveform comparison |
| `plot_edc_with_rt60(rir)` | Broadband EDC with −5/−25/−60 dB markers and RT60 annotation |
| `plot_spectrogram_comparison(pred, ref)` | Side-by-side spectrograms (scipy, gouraud shading) |
| `plot_results_table(metrics)` | Conference-style table with colour-coded pass/fail vs targets |
| `plot_per_band_rt60(pred, ref)` | Grouped bar chart comparing per-band RT60 |

All functions accept `save_path` and save at 300 DPI for conference papers.

---

## 10. Optional UNet Refiner

### `UNetRefiner(nn.Module)`
Post-processing module to refine generated RIR fine spectral detail.

```
Input [B, 1, L]
  └─ Encoder 1: ConvBlock1D(1→16) → skip1 + MaxPool → [B, 16, L/2]
  └─ Encoder 2: ConvBlock1D(16→32) → skip2 + MaxPool → [B, 32, L/4]
  └─ Encoder 3: ConvBlock1D(32→64) → skip3 + MaxPool → [B, 64, L/8]
  └─ Encoder 4: ConvBlock1D(64→128) → skip4 + MaxPool → [B, 128, L/16]
  └─ SinusoidalPosEncoding(128) → [B, 128, L/16]
  └─ Bottleneck: ConvBlock1D(128→128) + MultiHeadAttention(128, 4 heads) → [B, 128, L/16]
  └─ Decoder 3: Upsample + concat skip3 + ConvBlock1D → [B, 64, L/8]
  └─ Decoder 2: Upsample + concat skip2 + ConvBlock1D → [B, 32, L/4]
  └─ Decoder 1: Upsample + concat skip1 + ConvBlock1D → [B, 16, L/2]
  └─ head: Conv1d(16→1) → [B, 1, L]
Output [B, 1, L]
```

`ConvBlock1D` = 2 × (Conv1D + BatchNorm1D + ReLU). Skip connections recover fine structure lost by pooling.

**Note:** Currently not integrated into the training loop — activate by post-processing `RIRSynthesiser` output.

---

## 11. Infrastructure

### Cell 0 — Google Drive + HF Token
```python
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Colab Secrets → "HF_TOKEN"
drive.mount("/content/drive")
SAVE_DIR = "/content/drive/MyDrive/RIR_Project_outputs"
```

**Save helpers available globally after Cell 0:**
| Function | Saves |
|----------|-------|
| `save_checkpoint(state_dict, name)` | `.pt` model weights → Drive |
| `save_figure(fig_or_path, name)` | matplotlib figure or file → Drive |
| `save_metrics(metrics_dict, name)` | JSON metrics → Drive |
| `save_history(history, name)` | JSON training history → Drive |
| `save_rir_audio(rir, sr, name)` | Normalised `.wav` file → Drive |
| `backup_notebook(name)` | Copies `.ipynb` → Drive |

### HF Token (`HF_TOKEN`)
- Set via **Colab → Runtime → Manage secrets → Add "HF_TOKEN"**
- Passed as `token=HF_TOKEN if HF_TOKEN else None` to both `load_dataset()` and `hf_hub_download()`
- Without token: anonymous access → HuggingFace rate limiting (especially problematic for 50k files)
- With token: authenticated access → full download speed

### Checkpointing
After each epoch where validation loss improves: `best_lstm.pt` and `best_fdn.pt` are saved locally and to Drive. `training_history.json` and `test_metrics.json` are also saved.

---

## 12. Tensor Schema (Full)

```
Input x:  [B, 24]
  Indices 0-2:   room_size [L, W, H] in metres
  Indices 3-5:   source position [x, y, z] in metres
  Indices 6-8:   microphone position [x, y, z] in metres
  Index   9:     broadband absorption coefficient
  Indices 10-15: per-band absorption [125, 250, 500, 1000, 2000, 4000 Hz]
  Indices 16-23: room acoustic mode features [8-dim]

Target y:
  y["metrics"]    [B, 10]       RT60, DRR_dB, C50_dB, C80_dB, band_RT60 × 6
  y["rir"]        [B, 32000]    padded/truncated RIR @ 16 kHz
  y["edc"]        [B, 32000]    broadband Schroeder EDC (dB)
  y["edc_mb"]     [B, 256, 6]   multiband EDC (dB), T=256 time steps × 6 bands
  y["rir_length"] [B]           original sample count (scalar, LongTensor)

Intermediate:
  edc_pred        [B, 256, 6]   LSTM output — predicted multiband EDC
  fdn_rir         [B, T_fdn]    DifferentiableFDN output (T_fdn ≤ 32000)
  rir_out         [B, T]        final synthesised RIR
```

---

## 13. Hyperparameter Reference

```python
# Full recommended curriculum config (Colab T4/L4 GPU):
CurriculumConfig(
    # Data
    batch_size        = 16,       # 16 on T4; reduce to 8 if OOM
    num_workers       = 2,
    max_rir_len       = 32_000,   # 2s @ 16kHz
    sample_rate       = 16_000,

    # LSTM model
    hidden_dim        = 256,
    num_layers        = 2,
    num_time_steps    = 256,
    num_bands         = 6,
    model_dropout     = 0.1,

    # FDN
    train_fdn         = True,
    fdn_num_delays    = 16,
    fdn_max_delay_ms  = 50.0,
    fdn_output_length = 3_200,    # 0.2s; shorter = faster
    fdn_weight        = 0.05,

    # Curriculum schedule
    epochs            = 50,
    warmup_epochs     = 12,       # Phase A: EDC only
    physics_ramp_end  = 30,       # Phase B: gentle ramp
    fdn_phase_start   = 30,       # Phase C: add FDN
    finetune_start    = 42,       # Phase D: reduce LR

    # Physics loss targets
    lambda_cont_target = 0.01,
    lambda_mom_target  = 0.005,

    # Optimiser
    lr                = 1e-3,
    weight_decay      = 1e-5,
    grad_clip         = 1.0,
    scheduler_patience= 5,
    scheduler_factor  = 0.5,
    finetune_lr_factor= 0.1,
)
```

---

## 14. Known Limitations & Suggested Improvements

### Data
- **Limited dataset diversity:** `mandipgoswami/rirmega` contains ~50k RIRs but they may represent a narrow range of room types. Consider augmenting with synthetic RIRs from the Image Source Method.
- **No data augmentation:** Could add speed-of-sound perturbation, small absorption noise, or room-dimension jitter to improve generalization.
- **Fixed sample rate (16 kHz):** Restricts high-frequency content above 8 kHz. Consider 48 kHz support.
- **Static collation:** The `rir_collate_fn` always pads to `max_rir_len=32000`. Dynamic padding per batch would reduce memory waste for shorter RIRs.

### Model Architecture
- **Broadband EDC used in synthesis:** `RIRSynthesiser` computes a single broadband EDC from the FDN output, then applies phase reconstruction. The predicted **multiband** EDC is richer — consider using band-specific phase reconstruction.
- **LSTM temporal resolution:** 256 time steps map to 0 → 2s. Consider non-uniform time steps (denser early, sparser late) to better capture early reflections.
- **No explicit early-reflection model:** The FDN primarily models late reverberation. Consider separating early (<80 ms) and late (>80 ms) synthesis with a dedicated early-reflection module (43-tap convolution is available in the codebase but not used in `RIRSynthesiser`).
- **ConditionedFDN is not shared:** Parameters come from `EDCToFDNMapper` per sample, but the mapper is a small MLP. A larger mapper (or cross-attention over the full EDC sequence) could produce better FDN conditioning.
- **UNetRefiner not in training loop:** The optional U-Net is defined but never used. Integrate as a post-processing stage trained with a combined perceptual + EDC loss.

### Loss Functions
- **Physics terms require collocation points:** `continuity_residual` and `momentum_residual` need explicit pressure/velocity/coords/time tensors — currently the trainer calls the loss with EDC tensors only (physics terms are always 0). A proper PINN formulation should sample collocation points in the (x,y,z,t) domain and evaluate a neural field.
- **No perceptual loss:** Multi-resolution STFT loss (MRSTFT) is defined in the previous codebase version but not used here. Adding MRSTFT on the synthesised RIR waveform would encourage perceptual quality.
- **EDC RMSE in dB space:** Large values at the tail (very negative dB) dominate the loss. Consider truncating the EDC comparison to the range [0, −60 dB] and using a weighted MSE that emphasises the early decay.

### Training
- **FDN forward is slow (sequential T-step loop):** The `DifferentiableFDN.forward()` loop runs T=32000 steps sequentially. This is the bottleneck. A parallelised FDN implementation using convolution or matrix exponentiation of the state transition would be orders of magnitude faster.
- **No mixed-precision (AMP):** `torch.amp.autocast` + `GradScaler` would give ~1.5–2× speedup on modern GPUs with minimal accuracy loss. Previously in the codebase but removed.
- **No gradient accumulation:** Useful if memory limits force small batch sizes. Accumulate over N steps before `optimizer.step()`.
- **Validation metrics only track loss:** Add acoustic metric tracking (RT60 error, LSD) during validation for early stopping based on perceptual quality, not just loss.

### Evaluation
- **DRR always N/A:** The `compute_drr` function finds the peak and takes a 2.5 ms window. This often fails for synthesised RIRs (which may lack a sharp direct sound peak). Fix by using the first sample as the direct sound for synthesised RIRs.
- **No statistical significance testing:** Evaluation reports mean/std but no confidence intervals. Add bootstrap CIs for conference-quality reporting.

### Infrastructure
- **HF token falls back to anonymous:** If `HF_TOKEN` env var is not set, downloads proceed without auth. This is correct but could confuse users who set the wrong secret name. Add a warning if download rate is throttled.
- **No reproducibility seed:** `torch.manual_seed` / `numpy.seed` not set globally. Add `seed_everything(42)` call before dataset loading.
- **Checkpoint format:** Saves raw `state_dict`. Consider saving a full config dict alongside to avoid architecture mismatch on load.

---

## 15. Performance Targets vs Current Results

| Metric | Current (diverged) | Baseline (50 epochs) | Target |
|--------|-------------------|---------------------|--------|
| Train loss trend | ↑ 920 → 1090 (increasing) | ↓ converging | Decreasing |
| Val loss | ↑ 1030 → 1205 (increasing) | ↓ converging | Decreasing |
| RT60 Error | 0.224 s | ≲ 0.10 s | < 0.10 s |
| RT60 Std | 0.127 s | — | < 0.05 s |
| LSD | 11.48 dB | — | < 5.0 dB |
| EDC RMSE | 16.01 dB | — | < 5.0 dB |
| DRR Error | N/A | — | — |

**Root cause of divergence (fixed in rebuilt notebook):** The previous version used SIREN activations (`sin(ω₀·Wx + b)`) in `MultibandEDCPredictor`. SIREN layers amplify gradients during backpropagation and require very specific per-layer frequency initialisation. Without this, the network diverges. The rebuilt notebook restores the baseline `MLP encoder (Linear+LayerNorm+ReLU) + 2-layer LSTM` architecture.

---

## 16. File Structure

```
RIR_Project/
├── RIR_Project.ipynb               ← MAIN NOTEBOOK (29 cells, 3790 lines)
│                                     All code is inline — no .py imports
├── baseline/
│   ├── RIR_Project_baseline.ipynb  ← Original 28-cell reference
│   └── README.md
├── src/rir_project/                ← Python package (supplementary)
│   ├── __init__.py
│   ├── data.py                     (data pipeline, mirrors Cell 5-7)
│   ├── models.py                   (LSTM + FDN, mirrors Cell 10 + 16)
│   ├── loss.py                     (loss functions, mirrors Cell 13)
│   ├── trainer.py                  (trainer, mirrors Cell 19 + 26)
│   ├── synthesis.py                (synthesis, mirrors Cell 22)
│   └── utils.py                    (metrics + viz, mirrors Cell 23-25)
├── tests/
│   ├── test_architecture_upgrades.py
│   ├── test_data_loss_trainer_contracts.py
│   └── test_seed_reproducibility.py
├── train.py                        ← CLI training entry point
├── pyproject.toml
├── ARCHITECTURE.md                 ← High-level phase overview
└── PROJECT_ARCHITECTURE.md         ← THIS FILE (full AI-friendly doc)
```

### Relationship between notebook and .py files

The `src/rir_project/` package mirrors the notebook cells but is **not imported by the notebook**. All notebook cells are self-contained. The .py files exist for:
- CLI training via `train.py`
- Unit testing via `pytest tests/`
- Potential import by downstream projects

Changes to the notebook do **not** automatically update the .py files and vice versa. They must be kept in sync manually.

---

*Document generated: 2026-03-09. Covers RIR_Project.ipynb v2 (post-rebuild with HF token + Google Drive + curriculum training restored).*
