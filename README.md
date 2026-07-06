# Micro-CT Image Denoising Models

**Article:**:[PDF](./artigo_fast_microCT.pdf)

This repository contains only model and workflow code. It does not include the article PDF, datasets, generated outputs, or pretrained weights.

## Model References

- Restormer: https://arxiv.org/abs/2111.09881
- RED-CNN: https://ieeexplore.ieee.org/document/7947200

## Repository Scope

This repository provides the code for the models implemented in the article: RED-CNN and Restormer. It includes training with the article-style data splits, optional prediction export, and PSNR/SSIM metric calculation.

## Repository Layout

```text
image-denoising/
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
python -m pip install -e .
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

## Google Colab Restormer Example

Open the Colab example here:

https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing

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

## Train Restormer From Scratch

Use the same Restormer command, but remove `--use-pretrained` and remove `--pretrained-weights`:

```bash
python -m microct_denoising.train \
  --model restormer \
  --restormer-root external/Restormer \
  --fast-dir data/crops_2min \
  --long-dir data/crops_60min \
  --exclude-idx 0 \
  --output-root outputs \
  --epochs 4 \
  --batch-size 2 \
  --crop-size 400
```

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

For standalone Restormer pretrained inference on one volume:

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
