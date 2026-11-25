"""Epoch selection and test prediction of the nested leave-one-plug-out protocol.

After `microct-train --stage select` and `--stage final` have run for a test plug, this CLI
  1. reads select/metrics.csv (3D PSNR of the validation plug after every epoch),
  2. chooses the epoch with the highest validation PSNR,
  3. loads that epoch of the final model (trained on every plug except the test plug),
  4. writes the prediction of the test plug and its 3D PSNR/SSIM against the long scan (selected.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from microct_denoising.checkpoints import load_model_state
from microct_denoising.data import compute_fast_mean_std_stats, list_paired_netcdf_files, load_volume
from microct_denoising.inference import predict_and_save_volume
from microct_denoising.metrics import compute_full_volume_psnr_ssim
from microct_denoising.train import build_model_from_args, model_run_name, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["redcnn", "restormer"], required=True)
    parser.add_argument("--use-pretrained", action="store_true", help="Select the fine-tuned Restormer run.")
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--long-dir", type=Path, required=True)
    parser.add_argument("--test-idx", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--var-name", default="microtom")
    parser.add_argument("--criterion", choices=["psnr", "ssim"], default="psnr")
    parser.add_argument("--clip-output", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--feature-channels", type=int, default=96)
    parser.add_argument("--final-relu", action="store_true")
    parser.add_argument("--restormer-root", type=Path, default=None)
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def choose_epoch(metrics_csv: Path, criterion: str = "psnr") -> int:
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) > 0]
    if not rows:
        raise RuntimeError(f"No trained epochs in {metrics_csv}")
    best = max(rows, key=lambda r: float(r[criterion]))
    return int(best["epoch"])


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    fast_files, long_files = list_paired_netcdf_files(args.fast_dir, args.long_dir)
    fast_path, long_path = fast_files[args.test_idx], long_files[args.test_idx]
    run_kind = model_run_name(args)
    base = args.output_root / run_kind / f"test_{fast_path.stem}"

    epoch = choose_epoch(base / "select" / "metrics.csv", args.criterion)
    ckpt = base / "final" / "checkpoints" / f"{args.model}_ep{epoch}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Final-stage checkpoint not found: {ckpt}")

    stats_path = args.stats_path or (args.output_root / "fast_volume_mean_std_stats.npz")
    stats = compute_fast_mean_std_stats(fast_files, args.var_name, stats_path=stats_path)
    model, channels, pad_multiple = build_model_from_args(args)
    model = model.to(device)
    load_model_state(ckpt, model, device=device)

    out_path = base / "final" / f"{fast_path.stem}_{run_kind}_selected_epoch_{epoch:02d}.nc"
    t0 = time.time()
    pred = predict_and_save_volume(
        model=model, fast_path=fast_path, var_name=args.var_name, stats=stats, output_path=out_path,
        model_name=run_kind, device=device, channels=channels, pad_multiple=pad_multiple,
        tile_size=args.tile_size, overlap=args.tile_overlap, clip_output=args.clip_output,
        extra_attrs=dict(selected_epoch=epoch, selection=f"max validation {args.criterion.upper()}"),
    )
    seconds = time.time() - t0
    psnr, ssim = compute_full_volume_psnr_ssim(pred, load_volume(long_path, args.var_name))
    info = dict(test_plug=fast_path.stem, model=run_kind, selected_epoch=epoch,
                criterion=f"max validation {args.criterion.upper()}", prediction=out_path.name,
                test_psnr=psnr, test_ssim=ssim, prediction_seconds=seconds)
    (base / "final" / "selected.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"{fast_path.stem}: epoch {epoch} | PSNR {psnr:.4f} dB | SSIM {ssim:.5f} | {seconds:.0f} s -> {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
