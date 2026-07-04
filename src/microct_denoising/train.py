"""Training CLI for RED-CNN and Restormer micro-CT denoising models."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from microct_denoising.checkpoints import (
    find_latest_checkpoint,
    load_model_state,
    save_training_checkpoint,
)
from microct_denoising.constants import RANGE_LOSS_WEIGHT
from microct_denoising.data import (
    PairedNCDataset,
    compute_fast_mean_std_stats,
    list_paired_netcdf_files,
    load_volume,
    make_leave_one_out_split,
    write_split_info,
)
from microct_denoising.inference import predict_and_save_volume, save_middle_slice_preview
from microct_denoising.metrics import compute_full_volume_psnr_ssim, upsert_metrics_row
from microct_denoising.models.redcnn import build_redcnn
from microct_denoising.models.restormer import build_restormer, load_restormer_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train micro-CT denoising models.")
    parser.add_argument("--model", choices=["redcnn", "restormer"], required=True)
    parser.add_argument("--fast-dir", type=Path, required=True, help="Directory with fast/2min .nc files.")
    parser.add_argument("--long-dir", type=Path, required=True, help="Directory with long/60min .nc files.")
    parser.add_argument("--exclude-idx", type=int, required=True, help="Leave-one-out test sample index.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--var-name", default="microtom")
    parser.add_argument("--validation-mode", choices=["none", "previous"], default="none")

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--range-loss-weight", type=float, default=RANGE_LOSS_WEIGHT)
    parser.add_argument("--predict-every", type=int, default=1, help="Save full-volume prediction every N epochs.")
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--feature-channels", type=int, default=96, help="RED-CNN feature channels.")
    parser.add_argument("--final-relu", action="store_true", help="Use RED-CNN final ReLU.")

    parser.add_argument("--restormer-root", type=Path, default=None, help="Path to official swz30/Restormer checkout.")
    parser.add_argument("--pretrained-weights", type=Path, default=None, help="Restormer real_denoising.pth path.")
    parser.add_argument("--use-pretrained", action="store_true", help="Initialize Restormer with pretrained weights.")
    parser.add_argument("--no-predict-epoch-zero", action="store_true")
    parser.add_argument("--tile-size", type=int, default=0, help="Use tiled inference when > 0.")
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.model == "redcnn" and args.crop_size <= 20:
        raise ValueError("RED-CNN needs crop_size > 20 because it uses valid 5x5 convolutions.")
    if args.model == "restormer" and args.crop_size % 8 != 0:
        raise ValueError("Restormer crop_size must be divisible by 8.")
    if args.model == "restormer" and args.restormer_root is None:
        raise ValueError("--restormer-root is required for --model restormer.")
    if args.use_pretrained and args.pretrained_weights is None:
        raise ValueError("--pretrained-weights is required when --use-pretrained is set.")

    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    fast_files, long_files = list_paired_netcdf_files(args.fast_dir, args.long_dir)
    split = make_leave_one_out_split(
        fast_files=fast_files,
        long_files=long_files,
        exclude_idx=args.exclude_idx,
        validation_mode=args.validation_mode,
    )

    run_kind = "pretrained" if args.use_pretrained else "scratch"
    run_dir = make_run_dir(args.output_root, args.model, run_kind, split)
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / ("validation_nc" if split.val_fast else "test_nc")
    preview_dir = run_dir / "previews"
    metrics_path = run_dir / "metrics.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    write_split_info(split, run_dir / "split_info.txt", args.model, run_kind)

    stats_path = args.stats_path or (args.output_root / "fast_volume_mean_std_stats.npz")
    stats = compute_fast_mean_std_stats(fast_files, args.var_name, stats_path=stats_path)

    model, channels, pad_multiple = build_model_from_args(args)
    model = model.to(device)
    if args.use_pretrained:
        load_restormer_weights(model, args.pretrained_weights, device=device)

    optimizer = build_optimizer(args.model, model, args.learning_rate)
    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = maybe_resume(checkpoint_dir, args.model, model, optimizer, device)
    prediction_fast, prediction_long, prediction_id = prediction_target(split)

    if args.use_pretrained and not args.no_predict_epoch_zero:
        run_prediction_if_needed(
            args=args,
            epoch=0,
            model=model,
            fast_path=prediction_fast,
            long_path=prediction_long,
            sample_id=prediction_id,
            stats=stats,
            prediction_dir=prediction_dir,
            preview_dir=preview_dir,
            metrics_path=metrics_path,
            device=device,
            channels=channels,
            pad_multiple=pad_multiple,
            train_losses=None,
        )

    if args.epochs == 0:
        print(f"epochs=0; pretrained/scratch pre-run complete. Outputs: {run_dir}", flush=True)
        return 0

    dataset = PairedNCDataset(
        fast_files=split.train_fast,
        long_files=split.train_long,
        var_name=args.var_name,
        stats=stats,
        crop_size=args.crop_size,
        channels=channels,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(
        f"Starting {args.model} fold {args.exclude_idx} | run={run_kind} | "
        f"device={device} | train_slices={len(dataset)} | outputs={run_dir}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        train_losses = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            range_loss_weight=args.range_loss_weight,
            epoch=epoch + 1,
            epochs=args.epochs,
        )

        ckpt_epoch = epoch + 1
        ckpt_path = checkpoint_dir / f"{args.model}_ep{ckpt_epoch}.pth"
        save_training_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=ckpt_epoch,
            model_name=args.model,
            run_kind=run_kind,
        )

        should_predict = args.predict_every > 0 and (
            ckpt_epoch % args.predict_every == 0 or ckpt_epoch == args.epochs
        )
        if should_predict:
            run_prediction_if_needed(
                args=args,
                epoch=ckpt_epoch,
                model=model,
                fast_path=prediction_fast,
                long_path=prediction_long,
                sample_id=prediction_id,
                stats=stats,
                prediction_dir=prediction_dir,
                preview_dir=preview_dir,
                metrics_path=metrics_path,
                device=device,
                channels=channels,
                pad_multiple=pad_multiple,
                train_losses=train_losses,
            )

    print(f"Complete. Outputs: {run_dir}", flush=True)
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def build_model_from_args(args: argparse.Namespace) -> tuple[nn.Module, int, int | None]:
    if args.model == "redcnn":
        return build_redcnn(args.feature_channels, args.final_relu), 1, None
    model = build_restormer(args.restormer_root)
    return model, 3, 8


def build_optimizer(model_name: str, model: nn.Module, lr: float) -> optim.Optimizer:
    if model_name == "restormer":
        return optim.AdamW(model.parameters(), lr=lr)
    return optim.Adam(model.parameters(), lr=lr)


def maybe_resume(
    checkpoint_dir: Path,
    model_name: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> int:
    latest = find_latest_checkpoint(checkpoint_dir, model_name)
    if latest is None:
        return 0

    checkpoint = load_model_state(latest, model, device=device)
    if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = int(checkpoint.get("epoch") or latest.stem.replace(f"{model_name}_ep", ""))
    print(f"Resuming from {latest} at epoch {epoch + 1}", flush=True)
    return epoch


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    range_loss_weight: float,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_range = 0.0

    for batch_idx, (noisy, clean, norm_min, norm_max) in enumerate(loader, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        norm_min = norm_min.to(device, non_blocking=True).view(-1, 1, 1, 1)
        norm_max = norm_max.to(device, non_blocking=True).view(-1, 1, 1, 1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(noisy)
            recon_loss = criterion(pred, clean)
            range_loss = torch.relu(norm_min - pred).mean() + torch.relu(pred - norm_max).mean()
            loss = recon_loss + range_loss_weight * range_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_recon += float(recon_loss.item())
        total_range += float(range_loss.item())

        if batch_idx % 50 == 0 or batch_idx == len(loader):
            print(
                f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(loader)}] | "
                f"Loss {loss.item():.8f} | Recon {recon_loss.item():.8f} | "
                f"Range {range_loss.item():.8f}",
                flush=True,
            )

    denom = max(1, len(loader))
    losses = {
        "train_loss": total_loss / denom,
        "train_recon_loss": total_recon / denom,
        "train_range_loss": total_range / denom,
    }
    print(
        f"Epoch {epoch} finished | Avg loss {losses['train_loss']:.8f} | "
        f"Avg recon {losses['train_recon_loss']:.8f} | "
        f"Avg range {losses['train_range_loss']:.8f}",
        flush=True,
    )
    return losses


def make_run_dir(output_root: Path, model_name: str, run_kind: str, split) -> Path:
    if split.val_fast is not None:
        return (
            output_root
            / model_name
            / run_kind
            / f"test_{split.test_sample_id}__val_{split.val_sample_id}"
        )
    return output_root / model_name / run_kind / f"test_{split.test_sample_id}"


def prediction_target(split) -> tuple[Path, Path, str]:
    if split.val_fast is not None and split.val_long is not None:
        return split.val_fast, split.val_long, split.val_sample_id or split.val_fast.stem
    return split.test_fast, split.test_long, split.test_sample_id


def run_prediction_if_needed(
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    fast_path: Path,
    long_path: Path,
    sample_id: str,
    stats: dict,
    prediction_dir: Path,
    preview_dir: Path,
    metrics_path: Path,
    device: torch.device,
    channels: int,
    pad_multiple: int | None,
    train_losses: dict[str, float] | None,
) -> None:
    pred_path = prediction_dir / f"{sample_id}_{args.model}_epoch_{epoch:02d}.nc"
    if pred_path.exists() and not args.force_predictions:
        print(f"Prediction exists, skipping: {pred_path}", flush=True)
        return

    pred_vol = predict_and_save_volume(
        model=model,
        fast_path=fast_path,
        var_name=args.var_name,
        stats=stats,
        output_path=pred_path,
        model_name=args.model,
        device=device,
        channels=channels,
        pad_multiple=pad_multiple,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
    )

    ref_vol = load_volume(long_path, args.var_name)
    psnr, ssim = compute_full_volume_psnr_ssim(pred_vol, ref_vol)
    row = {
        "epoch": epoch,
        "model": args.model,
        "run_kind": "pretrained" if args.use_pretrained else "scratch",
        "sample_id": sample_id,
        "psnr": f"{psnr:.10f}",
        "ssim": f"{ssim:.10f}",
    }
    if train_losses:
        row.update({key: f"{value:.10f}" for key, value in train_losses.items()})
    upsert_metrics_row(metrics_path, row=row, key_fields=("epoch", "model", "sample_id"))
    print(f"Metrics epoch {epoch} | PSNR {psnr:.6f} dB | SSIM {ssim:.6f}", flush=True)

    if args.save_preview:
        save_middle_slice_preview(
            pred_vol=pred_vol,
            output_path=preview_dir / f"{sample_id}_{args.model}_epoch_{epoch:02d}.png",
            title=f"{args.model} epoch {epoch}",
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
