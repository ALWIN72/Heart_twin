"""
src/config.py

Central configuration for the MSCardio Digital Heart Twin project.
Every script imports constants from here so that shapes and hyperparameters
stay consistent across data loading, modeling, training, and evaluation.
"""
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MANIFESTS = PROJECT_ROOT / "data" / "manifests"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_MANIFESTS, CHECKPOINT_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Signal / windowing parameters
# ---------------------------------------------------------------------------
SAMPLING_RATE_HZ = 250          # target resample rate for all channels
WINDOW_SECONDS = 10             # length of a training window
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_SECONDS    # 2500 samples
PATCH_SECONDS = 0.5
PATCH_SIZE = int(SAMPLING_RATE_HZ * PATCH_SECONDS)    # 125 samples/patch
NUM_PATCHES = WINDOW_SAMPLES // PATCH_SIZE            # 20 patches/window

# 3 SCG axes (x, y, z accelerometer) + 3 Gyro axes (x, y, z) = 6 channels.
# Real GCG is not yet released; see src/utils/preprocessing.py for how the
# 3 "gyro" channels are currently derived from the uncalibrated SCG stream
# as an explicit, clearly-labeled proxy.
N_SCG_CHANNELS = 3
N_GYRO_CHANNELS = 3
N_CHANNELS = N_SCG_CHANNELS + N_GYRO_CHANNELS         # 6

PLACEMENT_CLASSES = ["Sternum", "Left", "Right"]
DEVICE_PLATFORMS = ["iOS", "Android"]


# ---------------------------------------------------------------------------
# Model dimensions
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    in_channels: int = N_CHANNELS
    patch_size: int = PATCH_SIZE
    num_patches: int = NUM_PATCHES
    embed_dim: int = 256
    num_heads: int = 8
    encoder_depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 128
    decoder_depth: int = 2


@dataclass
class TwinConfig:
    global_feat_dim: int = 256
    personal_hidden_dim: int = 32
    latent_dim: int = 8
    anomaly_threshold: float = 2.0   # KL-divergence "2-sigma" alert threshold


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"  # auto-falls-back to cpu at runtime, see utils.get_device


MODEL_CFG = ModelConfig()
TWIN_CFG = TwinConfig()
TRAIN_CFG = TrainConfig()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> "torch.device":
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
