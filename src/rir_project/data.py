"""Phase 1: data pipeline helpers and datasets."""

from __future__ import annotations

import ast
import io
import os
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

INPUT_DIM = 24
MODAL_FEAT_DIM = 8
METRICS_DIM = 10
OCTAVE_BANDS = ["125", "250", "500", "1000", "2000", "4000"]
DEFAULT_MAX_RIR_LEN = 32_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_json_field(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    return val


def compute_edc(rir: np.ndarray) -> np.ndarray:
    energy = np.cumsum(rir[::-1] ** 2, dtype=np.float64)[::-1]
    energy = energy / (float(np.max(energy)) + 1e-12)
    return 10.0 * np.log10(energy + 1e-12).astype(np.float32)


def downsample_edc_tensor(edc: np.ndarray, num_time_steps: int = 256) -> np.ndarray:
    idx = np.linspace(0, len(edc) - 1, num_time_steps).astype(np.int64)
    return edc[idx]


def compute_multiband_edc(
    rir: np.ndarray, sr: int = 16_000, num_time_steps: int = 256
) -> np.ndarray:
    # Lightweight default: duplicate broadband EDC into six bands.
    # The filtering-heavy variant can be swapped in later without API changes.
    edc = compute_edc(rir)
    edc_ds = downsample_edc_tensor(edc, num_time_steps=num_time_steps)
    return np.repeat(edc_ds[:, None], len(OCTAVE_BANDS), axis=1).astype(np.float32)


def _safe_spacing(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return 0.0, 0.0
    diffs = np.diff(np.array(sorted(xs), dtype=np.float32))
    return float(np.mean(diffs)), float(np.std(diffs))


def compute_room_modes(L: float, W: float, H: float) -> np.ndarray:
    if min(L, W, H) <= 0:
        raise ValueError("Room dimensions must be positive")

    c = 343.0
    axial = [c / 2.0 * n / dim for dim in (L, W, H) for n in range(1, 6)]
    axial = sorted(axial)
    n_below_300 = float(sum(1 for f in axial if f < 300.0))
    f_schroeder = float(2000.0 * np.sqrt(0.161 * (L * W * H) / max(L * W + W * H + L * H, 1e-9)))
    f_first_axial = float(axial[0]) if axial else 0.0
    mean_spacing, std_spacing = _safe_spacing(axial)
    modal_overlap = float(f_schroeder / max(f_first_axial, 1e-6))
    tang_ax_ratio = 1.0
    log_volume = float(np.log10(max(L * W * H, 1e-9)))
    return np.array(
        [
            n_below_300,
            f_schroeder,
            f_first_axial,
            mean_spacing,
            std_spacing,
            modal_overlap,
            tang_ax_ratio,
            log_volume,
        ],
        dtype=np.float32,
    )


def _extract_sample_id(path: str) -> Optional[str]:
    if not path:
        return None
    norm = path.replace("\\", "/")
    base = os.path.basename(norm)
    sid, _ = os.path.splitext(base)
    return sid.strip() or None


def _decode_audio(audio: Dict[str, Any]) -> np.ndarray:
    if isinstance(audio, dict) and "array" in audio:
        return np.asarray(audio["array"], dtype=np.float32).ravel()
    if isinstance(audio, dict) and audio.get("path") and os.path.exists(audio["path"]):
        with wave.open(audio["path"], "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
    elif isinstance(audio, dict) and audio.get("bytes"):
        with wave.open(io.BytesIO(audio["bytes"]), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
    else:
        raise ValueError("Audio sample cannot be decoded")

    data = np.asarray(data)
    if data.dtype == np.int16:
        return (data.astype(np.float32) / 32768.0).ravel()
    if data.dtype == np.int32:
        return (data.astype(np.float32) / 2147483648.0).ravel()
    return data.astype(np.float32).ravel()


def _pad_or_truncate(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)))


class RIRMegaDataset(Dataset):
    _HF_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test", "val": "val"}

    def __init__(
        self,
        split: str = "train",
        max_rir_len: int = DEFAULT_MAX_RIR_LEN,
        num_time_steps: int = 256,
        sample_rate: int = 16_000,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.max_rir_len = max_rir_len
        self.num_time_steps = num_time_steps
        self.sample_rate = sample_rate

        hf_split = "validation" if split == "val" else split
        self._hf_ds = load_dataset("mandipgoswami/rirmega", split=hf_split, cache_dir=cache_dir)
        self._hf_ds = self._hf_ds.cast_column("audio", Audio(decode=False))

        meta_path = hf_hub_download(
            repo_id="mandipgoswami/rirmega",
            filename="metadata/metadata.csv",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        full_meta = pd.read_csv(meta_path)
        full_meta["id"] = full_meta["id"].astype(str).str.strip()
        csv_split = self._HF_SPLIT_MAP.get(hf_split, split)
        self._meta = full_meta[full_meta["split"].astype(str).str.strip() == csv_split].reset_index(drop=True)
        self._meta_id_to_pos = {row["id"]: i for i, row in self._meta.iterrows()}

        self._index_map = self._build_index()

    def _build_index(self) -> List[Tuple[int, int]]:
        paths = self._hf_ds["audio"]
        index_map: List[Tuple[int, int]] = []
        for hf_idx, item in enumerate(paths):
            sid = _extract_sample_id(item.get("path", "") if isinstance(item, dict) else "")
            if sid and sid in self._meta_id_to_pos:
                index_map.append((hf_idx, self._meta_id_to_pos[sid]))

        assert len(index_map) > 0, "Dataset/metadata alignment failed: no matching sample IDs"
        return index_map

    def __len__(self) -> int:
        return len(self._index_map)

    @staticmethod
    def _ensure_len3(val: Any) -> List[float]:
        seq = list(val) if isinstance(val, (list, tuple)) else []
        if len(seq) != 3:
            raise ValueError("Expected a length-3 geometry/position field")
        out = [float(x) for x in seq]
        return out

    def _parse_meta_row(self, row: pd.Series) -> Dict[str, Any]:
        room_size = self._ensure_len3(_parse_json_field(row.get("room_size")))
        source = self._ensure_len3(_parse_json_field(row.get("source")))
        microphone = self._ensure_len3(_parse_json_field(row.get("microphone")))
        if min(room_size) <= 0.0:
            raise ValueError("Non-positive room dimensions in metadata")
        absorption = float(row.get("absorption", 0.0))
        return {
            "room_size": room_size,
            "source": source,
            "microphone": microphone,
            "absorption": absorption,
            "absorption_bands": _parse_json_field(row.get("absorption_bands")) or {},
            "metrics": _parse_json_field(row.get("metrics")) or {},
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hf_idx, meta_idx = self._index_map[idx]
        rir_np = _decode_audio(self._hf_ds[hf_idx]["audio"])
        original_len = len(rir_np)
        rir_np = _pad_or_truncate(rir_np, self.max_rir_len)
        edc_np = compute_edc(rir_np)
        edc_mb_np = compute_multiband_edc(rir_np, sr=self.sample_rate, num_time_steps=self.num_time_steps)

        p = self._parse_meta_row(self._meta.iloc[meta_idx])
        band_abs = [float(p["absorption_bands"].get(b, 0.0)) for b in OCTAVE_BANDS]
        modal_feats = compute_room_modes(*p["room_size"])

        x = torch.tensor(
            p["room_size"] + p["source"] + p["microphone"] + [p["absorption"]] + band_abs + modal_feats.tolist(),
            dtype=torch.float32,
        )
        assert x.shape[0] == INPUT_DIM, f"Expected input dim {INPUT_DIM}, got {x.shape[0]}"

        m = p["metrics"]
        band_rt60 = m.get("band_rt60s", {})
        metrics_vec = [
            float(m.get("rt60", 0.0)),
            float(m.get("drr_db", 0.0)),
            float(m.get("c50_db", 0.0)),
            float(m.get("c80_db", 0.0)),
            *[float(band_rt60.get(b, 0.0)) for b in OCTAVE_BANDS],
        ]

        y = {
            "metrics": torch.tensor(metrics_vec, dtype=torch.float32),
            "rir": torch.from_numpy(rir_np.astype(np.float32)),
            "edc": torch.from_numpy(edc_np.astype(np.float32)),
            "edc_mb": torch.from_numpy(edc_mb_np.astype(np.float32)),
            "rir_length": torch.tensor(original_len, dtype=torch.long),
        }
        return x, y


class CachedRIRDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset
        self._cache: Dict[int, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        if idx not in self._cache:
            self._cache[idx] = self.base[idx]
        return self._cache[idx]


def rir_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    x = torch.stack([b[0] for b in batch], dim=0)
    keys = batch[0][1].keys()
    y = {k: torch.stack([b[1][k] for b in batch], dim=0) for k in keys}
    return x, y


def get_dataloader(
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 2,
    max_rir_len: int = DEFAULT_MAX_RIR_LEN,
    num_time_steps: int = 256,
    sample_rate: int = 16_000,
    use_cache: bool = True,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    ds: Dataset = RIRMegaDataset(
        split=split,
        max_rir_len=max_rir_len,
        num_time_steps=num_time_steps,
        sample_rate=sample_rate,
        cache_dir=cache_dir,
    )
    if use_cache:
        ds = CachedRIRDataset(ds)
    loader_kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=(shuffle and split == "train"),
        num_workers=num_workers,
        collate_fn=rir_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(ds, **loader_kwargs)
