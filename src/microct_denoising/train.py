"""Training CLI for RED-CNN and Restormer, nested leave-one-plug-out protocol of the article.

For a test plug k of n paired plugs (validation plug v = k - 1, cyclic):

  --stage select : train on the n - 2 plugs other than k and v; after every epoch predict plug v and record its
                   3D PSNR/SSIM in select/metrics.csv (used only to choose the epoch);
  --stage final  : train on the n - 1 plugs other than k and save the model after every epoch; the 3D PSNR/SSIM of
                   plug k is recorded for information (final/metrics.csv), never used for any choice.

Then `microct-select-predict` picks the epoch with the highest validation PSNR and writes the prediction of plug k
with that epoch of the final model. Every decision for plug k therefore uses only the other plugs.

Training (both architectures): one random crop of every axial slice of every training plug per epoch, batch 2,
AdamW (learning rate 1e-4, weight decay 0.01, constant), L1 loss, mixed precision on GPU, 4 epochs, seed 42.
Input and target are z-score normalized with the mean/std of the fast volume of the same plug; predictions are
mapped back with the same numbers and are not clipped.
"""

from __future__ import annotations

import argparse
import json
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
from microct_denoising import constants as C
from microct_denoising.data import (
    PairedNCDataset,
    compute_fast_mean_std_stats,
    list_paired_netcdf_files,
    load_volume,
    make_leave_one_out_split,
    write_split_info,
)
from microct_denoising.inference import predict_volume, save_middle_slice_preview, save_prediction_volume
from microct_denoising.metrics import compute_full_volume_psnr_ssim, upsert_metrics_row
from microct_denoising.models.redcnn import build_redcnn
from microct_denoising.models.restormer import build_restormer, load_restormer_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train micro-CT denoising models (nested leave-one-plug-out).")
    parser.add_argument("--model", choices=["redcnn", "restormer"], required=True)
    parser.add_argument("--fast-dir", type=Path, required=True, help="Directory with the fast .nc volumes.")
    parser.add_argument("--long-dir", type=Path, required=True, help="Directory with the long (reference) .nc volumes.")
    parser.add_argument("--test-idx", "--exclude-idx", dest="test_idx", type=int, required=True,
                        help="0-based index of the test plug (files are paired by sorted name).")
    parser.add_argument("--stage", choices=["select", "final"], default="final",
                        help="select: hold out the previous plug for epoch selection; final: train on all but test.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--var-name", default="microtom")

    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    parser.add_argument("--crop-size", type=int, default=C.CROP_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--range-loss-weight", type=float, default=C.RANGE_LOSS_WEIGHT,
                        help="Optional penalty outside [0, 65535] (0 in the article).")
    parser.add_argument("--clip-output", action="store_true", help="Clip predictions to [0, 65535] (off in the article).")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write the prediction volume of the evaluated plug after every epoch.")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-batches", type=int, default=0, help="Stop each epoch after N batches (quick tests).")
    parser.add_argument("--max-eval-slices", type=int, default=0, help="Evaluate only the first N slices (quick tests).")

    parser.add_argument("--feature-channels", type=int, default=96, help="RED-CNN feature channels.")
    parser.add_argument("--final-relu", action="store_true", help="Use RED-CNN final ReLU.")

    parser.add_argument("--restormer-root", type=Path, default=None, help="Path to the official swz30/Restormer checkout.")
    parser.add_argument("--pretrained-weights", type=Path, default=None, help="Restormer real_denoising.pth path.")
    parser.add_argument("--use-pretrained", action="store_true", help="Initialize Restormer with the pretrained weights.")
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
        exclude_idx=args.test_idx,
        validation_mode="previous" if args.stage == "select" else "none",
    )

    run_kind = model_run_name(args)
    base_dir = args.output_root / run_kind / f"test_{split.test_sample_id}"
    run_dir = base_dir / args.stage
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_split_info(split, run_dir / "split_info.txt", args.model, run_kind)

    stats_path = args.stats_path or (args.output_root / "fast_volume_mean_std_stats.npz")
    stats = compute_fast_mean_std_stats(fast_files, args.var_name, stats_path=stats_path)

    eval_fast, eval_long = (split.val_fast, split.val_long) if args.stage == "select" else (split.test_fast, split.test_long)
    run_info = dict(
        model=run_kind, stage=args.stage, seed=args.seed, test_plug=split.test_sample_id,
        validation_plug=split.val_sample_id, eval_plug=eval_fast.stem,
        train_plugs=[p.stem for p in split.train_fast], epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, crop_size=args.crop_size,
        loss=args.loss, range_loss_weight=args.range_loss_weight, clipping=bool(args.clip_output),
        normalization="z-score with the mean/std of the fast volume of each plug",
    )
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")

    model, channels, pad_multiple = build_model_from_args(args)
    model = model.to(device)
    if args.use_pretrained:
        load_restormer_weights(model, args.pretrained_weights, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = maybe_resume(checkpoint_dir, args.model, model, optimizer, device)

    if args.epochs == 0:
        # epoch-0 evaluation (e.g. direct transfer of the pretrained Restormer)
        evaluate_epoch(args, 0, model, eval_fast, eval_long, stats, run_dir, metrics_path, device, channels,
                       pad_multiple, None)
        print(f"epochs=0; evaluation complete. Outputs: {run_dir}", flush=True)
        return 0

    dataset = PairedNCDataset(
        fast_files=split.train_fast,
        long_files=split.train_long,
        var_name=args.var_name,
        stats=stats,
        crop_size=args.crop_size,
        channels=channels,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init,
        generator=generator,
    )
    for _ in range(start_epoch):  # keep the shuffling sequence when resuming
        list(iter(torch.utils.data.RandomSampler(dataset, generator=generator)))

    print(
        f"Starting {run_kind} stage={args.stage} test={split.test_sample_id} eval={eval_fast.stem} | "
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
            max_batches=args.max_batches,
        )
        ckpt_epoch = epoch + 1
        save_training_checkpoint(
            path=checkpoint_dir / f"{args.model}_ep{ckpt_epoch}.pth",
            model=model,
            optimizer=optimizer,
            epoch=ckpt_epoch,
            model_name=args.model,
            run_kind=run_kind,
        )
        evaluate_epoch(args, ckpt_epoch, model, eval_fast, eval_long, stats, run_dir, metrics_path, device,
                       channels, pad_multiple, train_losses)

    print(f"Complete. Outputs: {run_dir}", flush=True)
    return 0


def model_run_name(args: argparse.Namespace) -> str:
    if args.model == "redcnn":
        return "redcnn"
    return "restormer_pretrained" if args.use_pretrained else "restormer_scratch"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def build_model_from_args(args: argparse.Namespace) -> tuple[nn.Module, int, int | None]:
    """Return (model, input channels, padding multiple). Restormer: gray slice replicated to 3 channels."""
    if args.model == "redcnn":
        return build_redcnn(args.feature_channels, args.final_relu), 1, None
    return build_restormer(args.restormer_root), 3, 8


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
    scaler: torch.amp.GradScaler,
    device: torch.device,
    range_loss_weight: float,
    epoch: int,
    epochs: int,
    max_batches: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = total_recon = total_range = 0.0
    n_batches = 0

    for batch_idx, (noisy, clean, norm_min, norm_max) in enumerate(loader, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(noisy)
            recon_loss = criterion(pred, clean)
            loss = recon_loss
            range_loss = torch.zeros((), device=device)
            if range_loss_weight > 0:
                lo = norm_min.to(device).view(-1, 1, 1, 1)
                hi = norm_max.to(device).view(-1, 1, 1, 1)
                range_loss = torch.relu(lo - pred).mean() + torch.relu(pred - hi).mean()
                loss = recon_loss + range_loss_weight * range_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_recon += float(recon_loss.item())
        total_range += float(range_loss.item())
        n_batches += 1
        if batch_idx % 50 == 0 or batch_idx == len(loader):
            print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(loader)}] | Loss {loss.item():.6f}", flush=True)
        if max_batches and batch_idx >= max_batches:
            break

    denom = max(1, n_batches)
    losses = {
        "train_loss": total_loss / denom,
        "train_recon_loss": total_recon / denom,
        "train_range_loss": total_range / denom,
    }
    print(f"Epoch {epoch} finished | mean loss {losses['train_loss']:.6f}", flush=True)
    return losses


def evaluate_epoch(
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    fast_path: Path,
    long_path: Path,
    stats: dict,
    run_dir: Path,
    metrics_path: Path,
    device: torch.device,
    channels: int,
    pad_multiple: int | None,
    train_losses: dict[str, float] | None,
) -> None:
    """Predict the evaluated plug (validation plug in `select`, test plug in `final`) and record 3D PSNR/SSIM."""
    pred_vol = predict_volume(
        model=model, fast_path=fast_path, var_name=args.var_name, stats=stats, device=device, channels=channels,
        pad_multiple=pad_multiple, tile_size=args.tile_size, overlap=args.tile_overlap, clip_output=args.clip_output,
        max_slices=args.max_eval_slices or None, progress=False,
    )
    ref_vol = load_volume(long_path, args.var_name)[: pred_vol.shape[0]]
    psnr, ssim = compute_full_volume_psnr_ssim(pred_vol, ref_vol)
    row = {"epoch": epoch, "stage": args.stage, "eval_plug": fast_path.stem, "psnr": f"{psnr:.10f}",
           "ssim": f"{ssim:.10f}"}
    if train_losses:
        row.update({key: f"{value:.10f}" for key, value in train_losses.items()})
    upsert_metrics_row(metrics_path, row=row, key_fields=("epoch",))
    print(f"Epoch {epoch} | {fast_path.stem} ({args.stage}) | PSNR {psnr:.4f} dB | SSIM {ssim:.5f}", flush=True)

    if args.save_predictions:
        save_prediction_volume(pred_vol, fast_path, args.var_name, stats,
                               run_dir / "predictions" / f"{fast_path.stem}_epoch_{epoch:02d}.nc",
                               model_run_name(args), clip_output=args.clip_output)
    if args.save_preview:
        save_middle_slice_preview(pred_vol, run_dir / "previews" / f"{fast_path.stem}_epoch_{epoch:02d}.png",
                                  f"{model_run_name(args)} epoch {epoch}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
