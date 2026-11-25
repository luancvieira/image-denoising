# Micro-CT Image Denoising: Fast-to-Long Acquisition Mapping

**Article:** [PDF](./artigo_fast_microCT.pdf)

**Sample viewer:** [luancvieira.github.io/image-denoising](https://luancvieira.github.io/image-denoising/)

Code for the article *Learning Fast-to-Long Acquisition Mapping for Denoising Micro-CT*: RED-CNN and Restormer models that map fast (2 min 35 s) micro-CT scans of carbonate plugs to their long (60 min) acquisitions, together with the full workflow of the article:

1. aluminium standardization of the gray scale;
2. nested leave-one-plug-out training, with the training epoch of every test plug chosen on another plug;
3. prediction of a new fast scan;
4. evaluation: 3D PSNR/SSIM, sensitivity to the SSIM data range, region-wise errors and consistency between slices;
5. prespecified automatic segmentation and segmentation metrics against the manual segmentation;
6. the sample viewer published from `docs/`.

The repository does not include the datasets, generated outputs or trained weights. The dataset is available upon request with permission from the Brazilian National Agency of Petroleum, Natural Gas and Biofuels (ANP).

## Repository Layout

```text
image-denoising/
  docs/                          sample viewer (GitHub Pages); data/ holds its PNG previews and manifest.json
  scripts/
    download_restormer_assets.py official Restormer code (pinned commit) and real-denoising weights
    export_viewer.py             exports the viewer images from the volumes of each sample
  src/microct_denoising/
    standardize.py               aluminium standardization          (microct-standardize)
    train.py                     nested leave-one-plug-out training  (microct-train)
    select_predict.py            epoch selection + test prediction   (microct-select-predict)
    predict.py                   prediction of any fast volume       (microct-predict)
    metrics.py                   image metrics                       (microct-evaluate)
    segmentation.py              automatic segmentation and metrics  (microct-segment)
    data.py, inference.py, checkpoints.py, constants.py
    models/redcnn.py, models/restormer.py
  notebooks/restormer_pretrained_prerun.ipynb
```

## Install

```bash
python -m pip install -e .
```

For GPU training, install the PyTorch build that matches your CUDA version before installing this package. The article used scikit-image 0.24 for the metrics.

## Restormer Assets

Restormer is not vendored here. The script clones the official implementation, checks out the commit used in the article (`68dc6ac`), and downloads the official real-denoising checkpoint (`real_denoising.pth`, release `v1.0`):

```bash
python scripts/download_restormer_assets.py --repo-dir external/Restormer --download-real-denoising
```

The architecture is used unchanged: the gray slice is replicated to the three input channels and the prediction is the mean of the three output channels.

## Data Layout

Each plug is a NetCDF volume with a 3D variable (default `microtom`, dimensions `z, x, y`; change it with `--var-name`). The fast, long and (optional) intermediate scans of a plug must be aligned voxel by voxel. Files are paired by sorted file name, so use the same name for the scans of one plug:

```text
data/
  fast/          plug_01.nc  plug_02.nc  ...    fast scans (rock crops)
  long/          plug_01.nc  plug_02.nc  ...    long reference scans
  intermediate/  plug_01.nc  ...                optional, for comparison only
  seg/           plug_01.nc  ...                manual segmentation of the long scans (labelled volume)
```

## 1. Aluminium Standardization

Each scan is shifted so that the aluminium standard scanned with the plug reads T = 6500:

X_al = X + (T - A)

where A is the median gray value of the aluminium ring, measured on the uncropped reconstruction over the slices of the rock crop, between 1.01 x the inner radius and 0.99 x the outer radius of the ring. The output is float32 and is not clipped. Every scan uses only its own aluminium value, so a new plug needs only its fast scan.

```bash
microct-standardize \
  --input data/fast/plug_01.nc --output data_al/fast/plug_01.nc \
  --reconstruction recon/plug_01_fast_full.nc --z-range 120 1612 \
  --rings recon/plug_01_rings.csv
```

`--rings` is a CSV with one row per slice and circle (`slice, region, center_x, center_y, radii`; region 1 = inner and 2 = outer wall of the ring). For a ring that does not move between slices use `--center-x --center-y --r-inner --r-outer` instead, and with a known value use `--aluminium-value A`. Repeat for the long and intermediate scans. The value of A and the shift are written next to the output (`*.aluminium.json`).

The article reports results both with standardization (`data_al/`) and without it (`data/`); the training below is the same in both cases.

## 2. Nested Leave-One-Plug-Out Training

For a test plug k (0-based `--test-idx`), the validation plug is the previous one, (k - 1) mod n:

- `--stage select` trains on the n - 2 remaining plugs and records the 3D PSNR/SSIM of the validation plug after every epoch;
- `--stage final` trains on the n - 1 plugs other than the test plug and saves the model after every epoch;
- `microct-select-predict` takes the epoch with the highest validation PSNR and writes the prediction of the test plug with that epoch of the final model.

No decision about a test plug uses the test plug itself.

Training settings: one random 400 x 400 crop of every axial slice of every training plug per epoch, batch 2, AdamW (learning rate 1e-4, weight decay 0.01, constant), L1 loss, mixed precision on GPU, 4 epochs, seed 42. Input and target are z-score normalized with the mean and standard deviation of the fast volume of the same plug; the prediction is mapped back with the same numbers and is not clipped.

RED-CNN, one test plug:

```bash
for STAGE in select final; do
  microct-train --model redcnn --fast-dir data_al/fast --long-dir data_al/long \
    --test-idx 0 --stage $STAGE --output-root outputs
done
microct-select-predict --model redcnn --fast-dir data_al/fast --long-dir data_al/long \
  --test-idx 0 --output-root outputs
```

Restormer, fine-tuned from the real-denoising weights (omit `--use-pretrained --pretrained-weights ...` in both commands for random initialization):

```bash
R=external/Restormer
for STAGE in select final; do
  microct-train --model restormer --restormer-root $R \
    --use-pretrained --pretrained-weights $R/Denoising/pretrained_models/real_denoising.pth \
    --fast-dir data_al/fast --long-dir data_al/long --test-idx 0 --stage $STAGE --output-root outputs
done
microct-select-predict --model restormer --restormer-root $R --use-pretrained \
  --fast-dir data_al/fast --long-dir data_al/long --test-idx 0 --output-root outputs
```

Repeat for every `--test-idx` (0 to n - 1); the stages of different plugs are independent and can run in parallel. Outputs:

```text
outputs/
  fast_volume_mean_std_stats.npz                     per-plug normalization statistics
  <model>/test_<plug>/select/metrics.csv             validation PSNR/SSIM per epoch
  <model>/test_<plug>/final/checkpoints/<arch>_epN.pth
  <model>/test_<plug>/final/<plug>_<model>_selected_epoch_NN.nc
  <model>/test_<plug>/final/selected.json            chosen epoch, test PSNR/SSIM, prediction time
```

`<model>` is `redcnn`, `restormer_pretrained` or `restormer_scratch`. Direct transfer of the pretrained Restormer (no training) is `microct-predict` with `--pretrained-weights` (below), or `microct-train ... --use-pretrained --epochs 0`, which evaluates epoch 0.

## 3. Prediction of a New Fast Scan

```bash
microct-predict --model redcnn \
  --checkpoint outputs/redcnn/test_plug_01/final/checkpoints/redcnn_ep2.pth \
  --input-nc new_fast_al.nc --output-nc new_restored.nc --fit-stats-from-input
```

The new scan is standardized first (step 1) and normalized with its own mean and standard deviation (`--fit-stats-from-input`). Restormer needs `--restormer-root`; use `--pretrained-weights` instead of `--checkpoint` for direct transfer, and `--tile-size` if a full slice does not fit in GPU memory. The prediction time is printed.

## 4. Evaluation

```bash
microct-evaluate --reference data_al/long/plug_01.nc \
  --volume fast=data_al/fast/plug_01.nc \
  --volume intermediate=data_al/intermediate/plug_01.nc \
  --volume redcnn=outputs/redcnn/test_plug_01/final/plug_01_redcnn_selected_epoch_02.nc \
  --manual-seg data/seg/plug_01.nc --output-dir evaluation/plug_01
```

Writes three CSV files:

- `images.csv`: PSNR and 3D SSIM (7 x 7 x 7 uniform window, scikit-image definition) with L = 65535 and with L_ref = p99.9 - p0.1 of the reference, and 2D SSIM with 7 x 7 windows in the axial plane and the two longitudinal planes;
- `regions.csv`: MAE, mean signed residual and RMSE in the pore interior, the solid interior and the pore-solid boundary (voxels with a voxel of the other phase in their 3 x 3 x 3 neighbourhood), from the manual segmentation;
- `consistency.csv`: mean absolute difference between neighbouring voxels along each axis, for each volume and its residual.

## 5. Segmentation

```bash
microct-segment --reference data_al/long/plug_01.nc --manual-seg data/seg/plug_01.nc \
  --volume fast=data_al/fast/plug_01.nc \
  --volume redcnn=outputs/redcnn/test_plug_01/final/plug_01_redcnn_selected_epoch_02.nc \
  --output segmentation/plug_01.csv
```

Every volume, including the reference, is segmented with the four prespecified pipelines of the article, each threshold computed on that volume alone:

| Pipeline | Filter | Threshold |
|---|---|---|
| `otsu2_raw` | none | two-class Otsu, darker class = pore |
| `otsu2_med3` | 3 x 3 x 3 median | two-class Otsu |
| `otsu3_raw` | none | three-class multi-Otsu, darkest class = pore |
| `otsu3_med3` | 3 x 3 x 3 median | three-class multi-Otsu |

Thresholds come from a 1024-bin histogram over the 0.01-99.99 percentile range, and pore components smaller than 27 voxels (6-connectivity) are removed. Each mask is compared with the manual segmentation (`*_man` columns) and with the reference segmented by the same pipeline: porosity and porosity error, pore Dice and IoU, boundary F1 (1-voxel tolerance) and percolation along z.

## 6. Sample Viewer

`docs/` is a static website, published with GitHub Pages, with the central axial, coronal and sagittal slices of the 12 samples, without and with aluminium standardization: gray values of the acquisitions, the restored volumes (including Restormer direct transfer) and the manual segmentation; error maps (volume minus reference); and the automatic segmentations of the four pipelines (pore and solid), with PSNR, SSIM, porosity and Dice against the manual segmentation. The images are 8-bit PNG renderings of each sample's gray window at full resolution (the volumes themselves are 16-bit).

To regenerate the images, list the volumes of each sample in a CSV (`sample, voxel_um, reference, fast, intermediate, restormer_direct, redcnn, restormer_finetuned, restormer_scratch, manual_seg`), one CSV with the aluminium-standardized volumes and one with the volumes as acquired, and run:

```bash
python scripts/export_viewer.py --samples samples_al.csv --output docs/data
python scripts/export_viewer.py --samples samples_raw.csv --output docs/data/raw --dir-prefix raw/
```

To publish, open the repository **Settings > Pages**, choose **Deploy from a branch**, branch `main`, folder `/docs`. To preview locally:

```bash
python -m http.server --directory docs 8000
```

and open http://localhost:8000.

## Google Colab Restormer Example

https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing

## Notes for Reproducibility

- Files are paired by sorted name; keep the fast, long, intermediate and segmentation folders aligned.
- The normalization statistics file stores one fast mean/std per fast file name.
- Predictions are saved in gray-value space as float32, without clipping (`--clip-output` restores clipping to [0, 65535]).
- No range penalty is used (`--range-loss-weight 0`, the default).
- Restormer training crops must be divisible by 8; inference pads each slice to a multiple of 8 by reflection. RED-CNN is fully convolutional and predicts whole slices.
- `--max-batches` and `--max-eval-slices` shorten a run for quick tests.

## Model References

- Restormer: https://arxiv.org/abs/2111.09881 (code: https://github.com/swz30/Restormer, commit `68dc6ac`)
- RED-CNN: https://ieeexplore.ieee.org/document/7947200
