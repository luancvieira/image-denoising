# Example Commands

Replace the paths below with your local NetCDF folders.

```bash
python -m microct_denoising.train \
  --model redcnn \
  --fast-dir data/crops_2min \
  --long-dir data/crops_60min \
  --exclude-idx 0 \
  --output-root outputs \
  --epochs 4
```

```bash
python scripts/download_restormer_assets.py --repo-dir external/Restormer --download-real-denoising

python -m microct_denoising.train \
  --model restormer \
  --restormer-root external/Restormer \
  --pretrained-weights external/Restormer/Denoising/pretrained_models/real_denoising.pth \
  --use-pretrained \
  --fast-dir data/crops_2min \
  --long-dir data/crops_60min \
  --exclude-idx 0 \
  --output-root outputs \
  --epochs 4
```

