"""NetCDF data loading, leave-one-out splits, and FAST-volume normalization."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from microct_denoising.constants import EPS, RAW_MAX, RAW_MIN


StatsDict = dict[str, float]


@dataclass(frozen=True)
class ExperimentSplit:
    """File paths for one train/test or train/validation/test split."""

    train_fast: list[Path]
    train_long: list[Path]
    test_fast: Path
    test_long: Path
    test_idx: int
    val_fast: Path | None = None
    val_long: Path | None = None
    val_idx: int | None = None

    @property
    def test_sample_id(self) -> str:
        return self.test_fast.stem

    @property
    def val_sample_id(self) -> str | None:
        return self.val_fast.stem if self.val_fast is not None else None


def list_paired_netcdf_files(fast_dir: Path, long_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return sorted FAST and LONG paired NetCDF files."""

    fast_files = sorted(Path(fast_dir).glob("*.nc"))
    long_files = sorted(Path(long_dir).glob("*.nc"))

    if not fast_files:
        raise FileNotFoundError(f"No .nc files found in {fast_dir}")
    if not long_files:
        raise FileNotFoundError(f"No .nc files found in {long_dir}")
    if len(fast_files) != len(long_files):
        raise ValueError(
            f"Different FAST/LONG file counts: {len(fast_files)} vs {len(long_files)}"
        )

    return fast_files, long_files


def make_leave_one_out_split(
    fast_files: list[Path],
    long_files: list[Path],
    exclude_idx: int,
    validation_mode: str = "none",
) -> ExperimentSplit:
    """Create the split used in the article experiments.

    validation_mode="previous" reserves the previous sample as validation:
    val_idx = (exclude_idx - 1) % n_samples.
    """

    n_samples = len(fast_files)
    if n_samples != len(long_files):
        raise ValueError("FAST and LONG lists must have the same length.")
    if exclude_idx < 0 or exclude_idx >= n_samples:
        raise IndexError(f"exclude_idx must be in 0..{n_samples - 1}")

    test_idx = exclude_idx
    reserved = {test_idx}
    val_idx: int | None = None

    if validation_mode == "previous":
        if n_samples < 3:
            raise ValueError("At least 3 paired volumes are needed for validation_mode=previous.")
        val_idx = (exclude_idx - 1) % n_samples
        reserved.add(val_idx)
    elif validation_mode != "none":
        raise ValueError("validation_mode must be 'none' or 'previous'.")

    train_fast = [p for i, p in enumerate(fast_files) if i not in reserved]
    train_long = [p for i, p in enumerate(long_files) if i not in reserved]

    return ExperimentSplit(
        train_fast=train_fast,
        train_long=train_long,
        test_fast=fast_files[test_idx],
        test_long=long_files[test_idx],
        test_idx=test_idx,
        val_fast=fast_files[val_idx] if val_idx is not None else None,
        val_long=long_files[val_idx] if val_idx is not None else None,
        val_idx=val_idx,
    )


def load_volume(path: Path, var_name: str) -> np.ndarray:
    """Load one NetCDF variable as float32."""

    with xr.open_dataset(path) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name!r} not found in {path}")
        return ds[var_name].values.astype(np.float32)


def _stats_key(fast_path: Path, suffix: str) -> str:
    return f"{Path(fast_path).name}_{suffix}"


def compute_fast_mean_std_stats(
    fast_files: list[Path],
    var_name: str,
    stats_path: Path | None = None,
    force: bool = False,
) -> StatsDict:
    """Compute or load per-volume FAST mean/std statistics.

    The LONG target is normalized with the matching FAST statistics. This keeps
    the target in the same intensity space as the input and avoids using LONG
    statistics during training or inference.
    """

    if stats_path is not None:
        stats_path = Path(stats_path)
        if stats_path.exists() and not force:
            with np.load(stats_path) as loaded:
                return {key: float(loaded[key]) for key in loaded.files}

    stats: StatsDict = {}
    for fast_path in fast_files:
        data = load_volume(fast_path, var_name)
        stats[_stats_key(fast_path, "mean")] = float(np.nanmean(data))
        stats[_stats_key(fast_path, "std")] = float(np.nanstd(data))

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = stats_path.with_name(f"{stats_path.stem}_{os.getpid()}.tmp.npz")
        np.savez(tmp_path, **stats)
        os.replace(tmp_path, stats_path)

    return stats


def zscore_with_fast_stats(arr: np.ndarray, fast_path: Path, stats: StatsDict) -> np.ndarray:
    """Normalize an array using the matching FAST volume mean/std."""

    mean = float(stats[_stats_key(fast_path, "mean")])
    std = float(stats[_stats_key(fast_path, "std")])
    arr = arr.astype(np.float32)
    if std > EPS:
        return ((arr - mean) / std).astype(np.float32)
    return (arr - mean).astype(np.float32)


def inverse_zscore_with_fast_stats(
    arr: np.ndarray,
    fast_path: Path,
    stats: StatsDict,
) -> np.ndarray:
    """Map a normalized prediction back to raw intensity."""

    mean = float(stats[_stats_key(fast_path, "mean")])
    std = float(stats[_stats_key(fast_path, "std")])
    arr = arr.astype(np.float32)
    if std > EPS:
        return (arr * std + mean).astype(np.float32)
    return (arr + mean).astype(np.float32)


def raw_bounds_in_fast_zscore_space(fast_path: Path, stats: StatsDict) -> tuple[np.float32, np.float32]:
    """Return valid raw intensity bounds in the FAST-normalized space."""

    mean = float(stats[_stats_key(fast_path, "mean")])
    std = float(stats[_stats_key(fast_path, "std")])
    if std > EPS:
        norm_min = (RAW_MIN - mean) / std
        norm_max = (RAW_MAX - mean) / std
    else:
        norm_min = RAW_MIN - mean
        norm_max = RAW_MAX - mean
    return np.float32(norm_min), np.float32(norm_max)


def write_split_info(split: ExperimentSplit, output_path: Path, model_name: str, run_kind: str) -> None:
    """Write a compact text record of the files used in a run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"run_kind: {run_kind}\n")
        f.write("split_type: leave_one_out\n")
        f.write(f"test_idx: {split.test_idx}\n")
        f.write(f"test_fast: {split.test_fast}\n")
        f.write(f"test_long: {split.test_long}\n")
        if split.val_idx is not None:
            f.write("validation_rule: val_idx = (exclude_idx - 1) % n_samples\n")
            f.write(f"val_idx: {split.val_idx}\n")
            f.write(f"val_fast: {split.val_fast}\n")
            f.write(f"val_long: {split.val_long}\n")
        f.write(f"n_train_samples: {len(split.train_fast)}\n")
        f.write("train_fast:\n")
        for path in split.train_fast:
            f.write(f"  {path}\n")
        f.write("train_long:\n")
        for path in split.train_long:
            f.write(f"  {path}\n")


class PairedNCDataset(Dataset):
    """Random paired 2D crops from matching FAST/LONG NetCDF volumes."""

    def __init__(
        self,
        fast_files: list[Path],
        long_files: list[Path],
        var_name: str,
        stats: StatsDict,
        crop_size: int,
        channels: int,
    ) -> None:
        if len(fast_files) != len(long_files):
            raise ValueError("FAST and LONG file lists must have the same length.")
        if channels < 1:
            raise ValueError("channels must be >= 1")

        self.samples: list[tuple[Path, Path, int]] = []
        self.var_name = var_name
        self.stats = stats
        self.crop_size = crop_size
        self.channels = channels

        for fast_path, long_path in zip(fast_files, long_files):
            with xr.open_dataset(fast_path) as ds:
                if var_name not in ds:
                    raise KeyError(f"{var_name!r} not found in {fast_path}")
                z_len = int(ds.sizes["z"])
            for z_idx in range(z_len):
                self.samples.append((fast_path, long_path, z_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast_path, long_path, z_idx = self.samples[idx]

        with xr.open_dataset(fast_path) as ds_fast, xr.open_dataset(long_path) as ds_long:
            noisy_full = ds_fast[self.var_name].isel(z=z_idx).values.astype(np.float32)
            clean_full = ds_long[self.var_name].isel(z=z_idx).values.astype(np.float32)

        h, w = noisy_full.shape
        if h < self.crop_size or w < self.crop_size:
            raise ValueError(f"crop_size={self.crop_size} is larger than slice shape {(h, w)}")

        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        noisy = noisy_full[top : top + self.crop_size, left : left + self.crop_size]
        clean = clean_full[top : top + self.crop_size, left : left + self.crop_size]

        noisy = zscore_with_fast_stats(noisy, fast_path, self.stats)
        clean = zscore_with_fast_stats(clean, fast_path, self.stats)
        norm_min, norm_max = raw_bounds_in_fast_zscore_space(fast_path, self.stats)

        return (
            torch.from_numpy(_as_channels(noisy, self.channels)),
            torch.from_numpy(_as_channels(clean, self.channels)),
            torch.tensor(norm_min, dtype=torch.float32),
            torch.tensor(norm_max, dtype=torch.float32),
        )


def _as_channels(arr: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return arr[np.newaxis].astype(np.float32)
    return np.stack([arr] * channels).astype(np.float32)
