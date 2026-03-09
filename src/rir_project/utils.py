"""Utility and evaluation helpers migrated from the notebook."""

from __future__ import annotations

import json
import os
import random
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import DEVICE, INPUT_DIM, OCTAVE_BANDS, compute_edc
from .models import MultibandEDCPredictor
from .synthesis import RIRSynthesiser


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNG state.

    When ``deterministic`` is True, this also requests deterministic PyTorch
    kernels where possible to improve reproducibility between runs.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operators may not support deterministic algorithms.
            pass


seed = set_seed


def estimate_rt60(rir: np.ndarray, sample_rate: int = 16_000) -> float:
    edc = compute_edc(rir)
    t = np.arange(len(edc), dtype=np.float32) / float(sample_rate)
    i5 = np.argmax(edc <= -5.0)
    i25 = np.argmax(edc <= -25.0)
    if i25 <= i5:
        return 0.0
    t5, t25 = t[i5], t[i25]
    return float(3.0 * max(t25 - t5, 0.0))


def log_spectral_distance(rir_pred: np.ndarray, rir_ref: np.ndarray) -> float:
    n = min(len(rir_pred), len(rir_ref))
    if n == 0:
        return float("nan")
    a = np.fft.rfft(rir_pred[:n])
    b = np.fft.rfft(rir_ref[:n])
    la = 20 * np.log10(np.abs(a) + 1e-9)
    lb = 20 * np.log10(np.abs(b) + 1e-9)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def edc_rmse_db(rir_pred: np.ndarray, rir_ref: np.ndarray) -> float:
    n = min(len(rir_pred), len(rir_ref))
    if n == 0:
        return float("nan")
    e1 = compute_edc(rir_pred[:n])
    e2 = compute_edc(rir_ref[:n])
    return float(np.sqrt(np.mean((e1 - e2) ** 2)))


def compute_drr(rir: np.ndarray, sample_rate: int = 16_000, direct_ms: float = 2.5) -> float:
    if len(rir) == 0:
        return float("nan")
    n_direct = max(1, int((direct_ms / 1000.0) * sample_rate))
    direct = np.sum(rir[:n_direct] ** 2)
    reverb = np.sum(rir[n_direct:] ** 2)
    return float(10.0 * np.log10((direct + 1e-12) / (reverb + 1e-12)))


def generate_rir_from_params(synth: RIRSynthesiser, x: torch.Tensor, device: str = str(DEVICE)) -> Dict[str, Any]:
    synth = synth.to(device)
    synth.eval()
    with torch.no_grad():
        out = synth(x.to(device), return_intermediates=True)
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}


def load_synthesiser(
    checkpoint_dir: str = ".",
    hidden_dim: int = 256,
    num_layers: int = 2,
    sample_rate: int = 16_000,
    device: str = str(DEVICE),
) -> RIRSynthesiser:
    lstm = MultibandEDCPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_bands=len(OCTAVE_BANDS),
        num_time_steps=256,
    ).to(device)
    synth = RIRSynthesiser(lstm=lstm, sample_rate=sample_rate).to(device)

    ckpt = Path(checkpoint_dir)
    lstm_path = ckpt / "best_lstm.pt"
    if lstm_path.exists():
        synth.lstm.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
    fdn_path = ckpt / "best_fdn.pt"
    if fdn_path.exists():
        state = torch.load(fdn_path, map_location=device, weights_only=True)
        synth.fdn.load_state_dict(state, strict=False)
    return synth


def demo_inference(synth: RIRSynthesiser, x: torch.Tensor, device: str = str(DEVICE)) -> Dict[str, Any]:
    return generate_rir_from_params(synth, x, device=device)


def evaluate_on_test_set(
    synth: RIRSynthesiser,
    loader,
    sample_rate: int = 16_000,
    device: str = str(DEVICE),
) -> Dict[str, float]:
    rt60_err, lsd_vals, edc_vals, drr_vals = [], [], [], []
    synth.eval()
    with torch.no_grad():
        for x, y in loader:
            out = synth(x.to(device))
            pred = out["rir"].detach().cpu().numpy()
            ref = y["rir"].numpy()
            for i in range(pred.shape[0]):
                p = pred[i]
                r = ref[i]
                rt60_err.append(abs(estimate_rt60(p, sample_rate) - estimate_rt60(r, sample_rate)))
                lsd_vals.append(log_spectral_distance(p, r))
                edc_vals.append(edc_rmse_db(p, r))
                drr_vals.append(compute_drr(p, sample_rate))
    return {
        "rt60_error": float(np.nanmean(rt60_err)) if rt60_err else float("nan"),
        "lsd": float(np.nanmean(lsd_vals)) if lsd_vals else float("nan"),
        "edc_rmse": float(np.nanmean(edc_vals)) if edc_vals else float("nan"),
        "drr": float(np.nanmean(drr_vals)) if drr_vals else float("nan"),
    }


def _save_or_show(save_path: Optional[str] = None) -> None:
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_training_curves(history: Dict[str, Any], title: str = "Training", save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.get("train_loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    _save_or_show(save_path)


def plot_multiband_edc(edc: np.ndarray, title: str = "Multiband EDC", save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(8, 4))
    for i, b in enumerate(OCTAVE_BANDS):
        plt.plot(edc[:, i], label=f"{b} Hz")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("EDC (dB)")
    plt.legend()
    _save_or_show(save_path)


def plot_rir_waveform(rir_pred: np.ndarray, rir_ref: Optional[np.ndarray] = None, title: str = "RIR", save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(rir_pred, label="pred")
    if rir_ref is not None:
        plt.plot(rir_ref, label="ref", alpha=0.7)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    _save_or_show(save_path)


def plot_edc_with_rt60(rir: np.ndarray, sample_rate: int = 16_000, title: str = "EDC", save_path: Optional[str] = None) -> None:
    edc = compute_edc(rir)
    t = np.arange(len(edc)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(t, edc)
    plt.title(f"{title} | RT60~{estimate_rt60(rir, sample_rate):.3f}s")
    plt.xlabel("Time (s)")
    plt.ylabel("EDC (dB)")
    _save_or_show(save_path)


def plot_spectrogram_comparison(rir_pred: np.ndarray, rir_ref: np.ndarray, sample_rate: int = 16_000, title: str = "Spectrogram", save_path: Optional[str] = None) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axs[0].specgram(rir_pred, Fs=sample_rate)
    axs[0].set_title("Pred")
    axs[1].specgram(rir_ref, Fs=sample_rate)
    axs[1].set_title("Ref")
    fig.suptitle(title)
    _save_or_show(save_path)


def plot_results_table(metrics: Dict[str, Any], title: str = "Results", save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    rows = [[k, f"{v:.4f}" if isinstance(v, (float, int, np.floating)) else str(v)] for k, v in metrics.items()]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title(title)
    _save_or_show(save_path)


def plot_per_band_rt60(values: Dict[str, float], title: str = "Per-band RT60", save_path: Optional[str] = None) -> None:
    bands = list(values.keys())
    vals = [values[b] for b in bands]
    plt.figure(figsize=(8, 4))
    plt.bar(bands, vals)
    plt.title(title)
    plt.xlabel("Band")
    plt.ylabel("RT60 (s)")
    _save_or_show(save_path)


def visualise_demo(demo_result: Dict[str, Any], sample_rate: int = 16_000) -> None:
    rir = demo_result.get("rir")
    if torch.is_tensor(rir):
        rir = rir[0].cpu().numpy()
    if rir is not None:
        plot_rir_waveform(rir, title="Demo RIR")
        plot_edc_with_rt60(rir, sample_rate=sample_rate, title="Demo EDC")


def save_checkpoint(state_dict: dict, name: str) -> str:
    torch.save(state_dict, name)
    return os.path.abspath(name)


def save_figure(fig_or_path, name: str) -> str:
    if isinstance(fig_or_path, str):
        src = fig_or_path
        with open(src, "rb") as fsrc, open(name, "wb") as fdst:
            fdst.write(fsrc.read())
    else:
        fig_or_path.savefig(name, dpi=200, bbox_inches="tight")
    return os.path.abspath(name)


def save_metrics(metrics_dict: dict, name: str = "test_metrics.json") -> str:
    with open(name, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    return os.path.abspath(name)


def save_history(history: dict, name: str = "training_history.json") -> str:
    with open(name, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return os.path.abspath(name)


def save_rir_audio(rir: np.ndarray, sample_rate: int, name: str) -> str:
    rir = np.asarray(rir, dtype=np.float32)
    peak = np.max(np.abs(rir)) + 1e-9
    wav = (rir / peak * 32767.0).astype(np.int16)
    with wave.open(name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav.tobytes())
    return os.path.abspath(name)


def backup_notebook(notebook_name: str = "RIR_Project.ipynb") -> str:
    src = Path(notebook_name)
    dst = src.with_name(src.stem + "_backup" + src.suffix)
    dst.write_bytes(src.read_bytes())
    return str(dst.resolve())
