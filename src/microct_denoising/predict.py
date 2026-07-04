"""Inference CLI for trained or pretrained micro-CT denoising models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from microct_denoising.checkpoints import load_model_state
from microct_denoising.data import compute_fast_mean_std_stats
from microct_denoising.inference import predict_and_save_volume, save_middle_slice_preview
from microct_denoising.models.redcnn import build_redcnn
from microct_denoising.models.restormer import build_restormer, load_restormer_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run micro-CT denoising inference.")
    parser.add_argument("--model", choices=["redcnn", "restormer"], required=True)
    parser.add_argument("--input-nc", type=Path, required=True)
    parser.add_argument("--output-nc", type=Path, required=True)
    parser.add_argument("--var-name", default="microtom")
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument(
        "--fit-stats-from-input",
        action="store_true",
        help="Compute FAST mean/std from --input-nc when no training stats file is available.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-preview", type=Path, default=None)

    parser.add_argument("--feature-channels", type=int, default=96)
    parser.add_argument("--final-relu", action="store_true")
    parser.add_argument("--restormer-root", type=Path, default=None)
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    model, channels, pad_multiple = build_model(args)
    model = model.to(device)
    if args.checkpoint is not None:
        load_model_state(args.checkpoint, model, device=device)
    elif args.pretrained_weights is not None:
        if args.model != "restormer":
            raise ValueError("--pretrained-weights is only supported for Restormer.")
        load_restormer_weights(model, args.pretrained_weights, device=device)
    else:
        raise ValueError("Provide --checkpoint or --pretrained-weights.")

    if args.stats_path and args.stats_path.exists():
        stats = compute_fast_mean_std_stats([args.input_nc], args.var_name, args.stats_path)
    elif args.fit_stats_from_input:
        stats = compute_fast_mean_std_stats([args.input_nc], args.var_name, args.stats_path, force=True)
    else:
        raise ValueError("Provide an existing --stats-path or use --fit-stats-from-input.")

    pred_vol = predict_and_save_volume(
        model=model,
        fast_path=args.input_nc,
        var_name=args.var_name,
        stats=stats,
        output_path=args.output_nc,
        model_name=args.model,
        device=device,
        channels=channels,
        pad_multiple=pad_multiple,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
    )

    if args.save_preview is not None:
        save_middle_slice_preview(pred_vol, args.save_preview, f"{args.model} prediction")

    print(f"Saved prediction: {args.output_nc}", flush=True)
    return 0


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def build_model(args: argparse.Namespace):
    if args.model == "redcnn":
        return build_redcnn(args.feature_channels, args.final_relu), 1, None
    if args.restormer_root is None:
        raise ValueError("--restormer-root is required for Restormer.")
    return build_restormer(args.restormer_root), 3, 8


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

