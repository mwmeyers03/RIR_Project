"""
dataset.py — Phase 1: Data Pipeline for Physics-Informed RIR Generation
========================================================================

PyTorch Dataset + DataLoader wrapping the HuggingFace ``mandipgoswami/rirmega``
dataset.  Metadata (room geometry, source/mic positions, absorption, acoustic
metrics) is read from the companion ``metadata/metadata.csv`` file shipped
inside the same HF repo.

Input tensor X  (16-d):
    [room_L, room_W, room_H,           # 3  — room dimensions (m)
     src_x, src_y, src_z,              # 3  — source position (m)
     mic_x, mic_y, mic_z,              # 3  — receiver position (m)
     abs_broadband,                    # 1  — broadband absorption coeff
     abs_125, abs_250, abs_500,        # 6  — octave-band absorption coeffs
     abs_1000, abs_2000, abs_4000]

Target dict Y:
    'metrics'    : Tensor[10]           — [RT60, DRR_dB, C50_dB, C80_dB,
                                           band_RT60_125 … band_RT60_4000]
    'rir'        : Tensor[max_rir_len]  — padded / truncated RIR waveform
    'edc'        : Tensor[max_rir_len]  — Schroeder Energy Decay Curve (dB)
    'rir_length' : Tensor (scalar)      — original sample count before padding
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OCTAVE_BANDS: List[str] = ["125", "250", "500", "1000", "2000", "4000"]
DEFAULT_MAX_RIR_LEN: int = 32_000          # 2 s @ 16 kHz
INPUT_DIM: int = 16                        # 3+3+3+1+6
METRICS_DIM: int = 10                      # 4 scalar + 6 band RT60s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json_field(val):
    """Robustly parse a CSV cell that stores JSON / Python-literal data."""
    if isinstance(val, (dict, list)):
        return val
    if pd.isna(val) or val is None:
        return None
    try:
        return json.loads(val.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        pass
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None


def compute_edc(rir: np.ndarray) -> np.ndarray:
    """Schroeder backward-integration Energy Decay Curve (dB).

    EDC(t) = ∫_t^∞  h²(τ) dτ   (normalised so EDC[0] = 0 dB)
    """
    h2 = rir.astype(np.float64) ** 2
    edc = np.cumsum(h2[::-1])[::-1].copy()
    edc /= edc[0] + 1e-30                    # normalise
    edc_db = 10.0 * np.log10(edc + 1e-30)
    return edc_db.astype(np.float32)


def _extract_sample_id(path: str) -> Optional[str]:
    """Pull ``rir_NNNNNN`` out of an audio file path."""
    if not path:
        return None
    m = re.search(r"(rir_\d+)", os.path.basename(str(path)))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RIRMegaDataset(Dataset):
    """PyTorch Dataset for the RIRmega corpus.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"`` (or ``"validation"``), ``"test"``.
    max_rir_len : int
        Fixed waveform length in samples (pad / truncate).
    cache_dir : str | None
        Optional HuggingFace cache directory.
    """

    # Mapping between the HF dataset split name and the CSV's split tag
    _HF_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test",
                      "val": "val"}

    def __init__(
        self,
        split: str = "train",
        max_rir_len: int = DEFAULT_MAX_RIR_LEN,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.max_rir_len = max_rir_len

        hf_split = "validation" if split == "val" else split

        # ---- 1. Load HF audio dataset ----------------------------------
        from datasets import load_dataset, Audio

        print(f"[RIRMegaDataset] Loading HF audio (split='{hf_split}') …")
        self._hf_ds = load_dataset(
            "mandipgoswami/rirmega",
            split=hf_split,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # ---- 2. Download & parse metadata CSV --------------------------
        from huggingface_hub import hf_hub_download

        print("[RIRMegaDataset] Downloading metadata CSV …")
        meta_path = hf_hub_download(
            repo_id="mandipgoswami/rirmega",
            filename="metadata/metadata.csv",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        full_meta = pd.read_csv(meta_path)
        full_meta["id"] = full_meta["id"].astype(str).str.strip()

        # Filter metadata to requested split
        csv_split = self._HF_SPLIT_MAP.get(hf_split, split)
        self._meta = (
            full_meta[full_meta["split"].str.strip() == csv_split]
            .reset_index(drop=True)
        )
        self._meta_id_to_pos = {
            row["id"]: i for i, row in self._meta.iterrows()
        }

        # ---- 3. Build alignment (HF index ↔ metadata row) -------------
        self._index_map = self._build_index()
        print(f"[RIRMegaDataset] Ready — {len(self)} aligned samples "
              f"(split='{split}')")

    # ------------------------------------------------------------------
    def _build_index(self) -> List[Tuple[int, int]]:
        """Align HF audio items with metadata rows by sample-ID.

        Falls back to positional matching when paths are unavailable.
        """
        from datasets import Audio

        # Get file paths *without* decoding (fast)
        raw = self._hf_ds.cast_column("audio", Audio(decode=False))
        paths = raw["audio"]                       # list[dict]

        index_map: List[Tuple[int, int]] = []
        for hf_idx, item in enumerate(paths):
            path = item.get("path", "") if isinstance(item, dict) else ""
            sid = _extract_sample_id(path)

            if sid and sid in self._meta_id_to_pos:
                index_map.append((hf_idx, self._meta_id_to_pos[sid]))

        # Positional fallback when path matching yields nothing
        if not index_map:
            n = min(len(self._hf_ds), len(self._meta))
            index_map = [(i, i) for i in range(n)]

        return index_map

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index_map)

    # ------------------------------------------------------------------
    def _parse_meta_row(self, row: pd.Series) -> Dict:
        """Unpack all JSON-encoded fields for a single metadata row."""
        return {
            "room_size":        _parse_json_field(row["room_size"])       or [0., 0., 0.],
            "source":           _parse_json_field(row["source"])          or [0., 0., 0.],
            "microphone":       _parse_json_field(row["microphone"])      or [0., 0., 0.],
            "absorption":       float(row.get("absorption", 0.0)),
            "absorption_bands": _parse_json_field(row["absorption_bands"]) or {},
            "metrics":          _parse_json_field(row["metrics"])          or {},
        }

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hf_idx, meta_idx = self._index_map[idx]

        # ---- Audio (decoded on-the-fly by HF datasets) ----------------
        audio = self._hf_ds[hf_idx]["audio"]
        rir_np = np.asarray(audio["array"], dtype=np.float32)
        original_len = len(rir_np)

        # Pad / truncate to fixed length
        if len(rir_np) >= self.max_rir_len:
            rir_np = rir_np[: self.max_rir_len]
        else:
            rir_np = np.pad(rir_np, (0, self.max_rir_len - len(rir_np)))

        # Schroeder EDC
        edc_np = compute_edc(rir_np)

        # ---- Metadata -------------------------------------------------
        p = self._parse_meta_row(self._meta.iloc[meta_idx])

        band_abs = [float(p["absorption_bands"].get(b, 0.0))
                    for b in OCTAVE_BANDS]

        x = torch.tensor(
            p["room_size"]                          # 3  room dims
            + p["source"]                           # 3  source xyz
            + p["microphone"]                       # 3  mic xyz
            + [p["absorption"]]                     # 1  broadband abs
            + band_abs,                             # 6  band abs
            dtype=torch.float32,
        )  # → shape [16]

        m = p["metrics"]
        band_rt60 = m.get("band_rt60s", {})
        metrics_vec = (
            [float(m.get("rt60", 0.0)),
             float(m.get("drr_db", 0.0)),
             float(m.get("c50_db", 0.0)),
             float(m.get("c80_db", 0.0))]
            + [float(band_rt60.get(b, 0.0)) for b in OCTAVE_BANDS]
        )

        y = {
            "metrics":    torch.tensor(metrics_vec, dtype=torch.float32),       # [10]
            "rir":        torch.from_numpy(rir_np),                             # [max_rir_len]
            "edc":        torch.from_numpy(edc_np),                             # [max_rir_len]
            "rir_length": torch.tensor(original_len, dtype=torch.long),         # scalar
        }

        return x, y


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def rir_collate_fn(batch):
    """Stack X tensors and collate Y dicts into batched tensors."""
    xs, ys = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        {
            "metrics":    torch.stack([y["metrics"]    for y in ys], dim=0),
            "rir":        torch.stack([y["rir"]        for y in ys], dim=0),
            "edc":        torch.stack([y["edc"]        for y in ys], dim=0),
            "rir_length": torch.stack([y["rir_length"] for y in ys], dim=0),
        },
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloader(
    split: str = "train",
    batch_size: int = 8,
    max_rir_len: int = DEFAULT_MAX_RIR_LEN,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    **loader_kwargs,
) -> DataLoader:
    """Convenience factory — returns a ready-to-iterate DataLoader."""
    if shuffle is None:
        shuffle = split == "train"

    ds = RIRMegaDataset(split=split, max_rir_len=max_rir_len,
                        cache_dir=cache_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=rir_collate_fn,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SEP = "=" * 64
    MAX_LEN = DEFAULT_MAX_RIR_LEN
    BATCH = 4

    print(f"\n{SEP}")
    print("  Unit Test — RIRMegaDataset  (Phase 1: Data Pipeline)")
    print(SEP)

    # ---- Single-sample test -------------------------------------------
    ds = RIRMegaDataset(split="train", max_rir_len=MAX_LEN)

    x, y = ds[0]
    print(f"\n[Single sample — index 0]")
    print(f"  X shape      : {x.shape}           (expect [{INPUT_DIM}])")
    print(f"  X dtype      : {x.dtype}")
    print(f"  X values     : {x.tolist()[:6]} …")
    print(f"  Y keys       : {sorted(y.keys())}")
    print(f"  metrics shape: {y['metrics'].shape}  (expect [{METRICS_DIM}])")
    print(f"  rir shape    : {y['rir'].shape}  (expect [{MAX_LEN}])")
    print(f"  edc shape    : {y['edc'].shape}  (expect [{MAX_LEN}])")
    print(f"  rir_length   : {y['rir_length'].item()}")

    assert x.shape == (INPUT_DIM,),          f"X shape {x.shape}"
    assert y["metrics"].shape == (METRICS_DIM,), f"metrics shape {y['metrics'].shape}"
    assert y["rir"].shape == (MAX_LEN,),     f"rir shape {y['rir'].shape}"
    assert y["edc"].shape == (MAX_LEN,),     f"edc shape {y['edc'].shape}"
    assert y["rir_length"].dtype == torch.long

    # ---- Batch / DataLoader test --------------------------------------
    loader = get_dataloader(
        split="train", batch_size=BATCH,
        max_rir_len=MAX_LEN, num_workers=0,
    )
    xb, yb = next(iter(loader))

    print(f"\n[Batch — batch_size={BATCH}]")
    print(f"  X batch      : {xb.shape}        (expect [{BATCH}, {INPUT_DIM}])")
    print(f"  metrics batch: {yb['metrics'].shape}  (expect [{BATCH}, {METRICS_DIM}])")
    print(f"  rir batch    : {yb['rir'].shape}  (expect [{BATCH}, {MAX_LEN}])")
    print(f"  edc batch    : {yb['edc'].shape}  (expect [{BATCH}, {MAX_LEN}])")
    print(f"  rir_lengths  : {yb['rir_length'].tolist()}")

    assert xb.shape == (BATCH, INPUT_DIM)
    assert yb["metrics"].shape == (BATCH, METRICS_DIM)
    assert yb["rir"].shape == (BATCH, MAX_LEN)
    assert yb["edc"].shape == (BATCH, MAX_LEN)

    print(f"\n{'─' * 64}")
    print("  ✓  All assertions passed — data pipeline is operational.")
    print(SEP + "\n")
