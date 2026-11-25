"""Shared numeric constants used by the micro-CT denoising workflow."""

EPS = 1e-8
RAW_MIN = 0.0
RAW_MAX = 65535.0
DATA_RANGE = RAW_MAX - RAW_MIN  # 16-bit range used for PSNR/SSIM (L = 65535)

# Training protocol of the article (revised version)
EPOCHS = 4
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
CROP_SIZE = 400
SEED = 42

# Optional penalty on predictions outside [RAW_MIN, RAW_MAX]; not used in the article (weight 0).
RANGE_LOSS_WEIGHT = 0.0

# Aluminium standardization: every volume is shifted so that its aluminium standard reads this value.
ALUMINIUM_TARGET = 6500.0
