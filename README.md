# Micro-CT Image Denoising Models

Clean model code for the fast micro-CT denoising experiments described in:

**Article:** `ARTICLE_LINK_PLACEHOLDER`

This repository contains only model and workflow code. It does not include the article PDF, datasets, generated outputs, or pretrained weights.

## What Is Included

- RED-CNN baseline trained from scratch on paired NetCDF volumes.
- Restormer training and inference using the official upstream Restormer code.
- Leave-one-out splitting for paired fast/long scans.
- Optional validation split where `val_idx = (exclude_idx - 1) % n_samples`.
- FAST-only normalization: both FAST input and LONG target are normalized with the matching FAST volume mean/std.
- Full-volume NetCDF prediction export, optional PNG preview, and PSNR/SSIM metrics.
- A notebook pre-run example for official Restormer `real_denoising.pth`.

## Repository Layout

```text
image-denoising/
  configs/
    example_commands.md
  notebooks/
    restormer_pretrained_prerun.ipynb
  scripts/
    download_restormer_assets.py
  src/microct_denoising/
    data.py
    inference.py
    metrics.py
    train.py
    predict.py
    models/
      redcnn.py
      restormer.py
```

## Data Layout Expected By The Commands

Place paired NetCDF volumes outside Git or under ignored local folders:

```text
data/
  crops_2min/
    sample_01.nc
    sample_02.nc
  crops_60min/
    sample_01.nc
    sample_02.nc
```

The default variable name is `microtom`. Use `--var-name` if your NetCDF variable differs.

## Install

```bash
python -m pip install -e ".[dev]"
```

For GPU training, install the PyTorch build that matches your CUDA version before installing this package.

## Restormer Assets

Restormer is intentionally not vendored here. Clone the official implementation and download the official real-denoising checkpoint:

```bash
python scripts/download_restormer_assets.py \
  --repo-dir external/Restormer \
  --download-real-denoising
```

The script downloads:

- official code: https://github.com/swz30/Restormer
- official checkpoint: `real_denoising.pth` from the `v1.0` release

## Train RED-CNN

```bash
python -m microct_denoising.train \
  --model redcnn \
  --fast-dir data/crops_2min \
  --long-dir data/crops_60min \
  --exclude-idx 0 \
  --output-root outputs \
  --epochs 4 \
  --batch-size 2 \
  --crop-size 400
```

## Train Restormer From Official Pretrained Weights

```bash
python -m microct_denoising.train \
  --model restormer \
  --restormer-root external/Restormer \
  --pretrained-weights external/Restormer/Denoising/pretrained_models/real_denoising.pth \
  --use-pretrained \
  --fast-dir data/crops_2min \
  --long-dir data/crops_60min \
  --exclude-idx 0 \
  --output-root outputs \
  --epochs 4 \
  --batch-size 2 \
  --crop-size 400
```

Use `--epochs 0 --use-pretrained` to save an epoch-0 pretrained prediction before fine-tuning.

## Train With Validation

```bash
python -m microct_denoising.train \
  --model restormer \
  --restormer-root external/Restormer \
  --pretrained-weights external/Restormer/Denoising/pretrained_models/real_denoising.pth \
  --use-pretrained \
  --fast-dir data/crops_2min \
  --long-dir data/crops_60min \
  --exclude-idx 0 \
  --validation-mode previous \
  --output-root outputs \
  --epochs 4
```

With `--validation-mode previous`, the excluded sample remains the held-out test sample and the previous sample is used for validation.

## Run Inference From A Checkpoint

```bash
python -m microct_denoising.predict \
  --model redcnn \
  --checkpoint outputs/redcnn/scratch/test_sample_01/checkpoints/redcnn_ep4.pth \
  --input-nc data/crops_2min/sample_01.nc \
  --output-nc outputs/sample_01_redcnn.nc \
  --stats-path outputs/fast_volume_mean_std_stats.npz
```

For a standalone Restormer pretrained pre-run on one volume:

```bash
python -m microct_denoising.predict \
  --model restormer \
  --restormer-root external/Restormer \
  --pretrained-weights external/Restormer/Denoising/pretrained_models/real_denoising.pth \
  --input-nc data/crops_2min/sample_01.nc \
  --output-nc outputs/sample_01_restormer_pretrained.nc \
  --fit-stats-from-input \
  --save-preview outputs/sample_01_restormer_pretrained.png
```

## Notes For Reproducibility

- File pairing is sorted by filename. Keep FAST and LONG folders aligned.
- The stats file stores one FAST mean/std pair per FAST filename.
- Predictions are saved in raw intensity space and clipped to `[0, 65535]`.
- RED-CNN uses one input/output channel.
- Restormer uses three replicated channels because the official real-denoising weights are RGB.
- Restormer training crops must be divisible by 8; the default `--crop-size 400` satisfies this.
- Use `--tile-size` if full-slice Restormer inference exceeds GPU memory.
