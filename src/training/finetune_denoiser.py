"""
src/training/finetune_denoiser.py

Phase 2 training: fine-tune the MotionArtifactCanceller to reconstruct
clean SCG from motion-corrupted SCG + gyro/gyro-proxy.

Supervision strategy
---------------------
We don't have ground-truth "clean vs motion-corrupted" pairs in the raw
dataset. Instead we use the standard self-supervised trick for this kind
of denoising task: take naturally low-motion windows (low gyro-proxy
energy) as pseudo-clean targets, synthetically inject motion bursts into
the SCG channels to create the noisy input, and train the model to
recover the original clean window. This lets the model learn a general
motion-cancellation function that transfers to real motion artifact at
inference time.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, ModelConfig, TrainConfig, get_device, set_seed
from src.data_loader import build_multimodal_manifest, create_splits
from src.dataset import SCGWindowDataset, collate_scg
from src.models.denoiser import MotionArtifactCanceller
from src.models.encoder import SCGEncoder


def inject_motion_artifact(x: torch.Tensor, burst_prob: float = 0.7,
                            burst_frac: float = 0.25, noise_scale: float = 1.5) -> torch.Tensor:
    """
    x: (B, 6, T) clean window. Returns a corrupted copy where the SCG
    channels (0:3) have a synthetic motion burst added, and the gyro
    channels (3:6) get a correlated burst (since real motion shows up in
    both streams -- that correlation is exactly what the model should
    learn to exploit for cancellation).
    """
    x_noisy = x.clone()
    b, c, t = x.shape
    burst_len = max(1, int(t * burst_frac))

    for i in range(b):
        if torch.rand(1).item() > burst_prob:
            continue
        start = torch.randint(0, max(1, t - burst_len), (1,)).item()
        burst_shape = torch.randn(1, burst_len, device=x.device) * noise_scale
        # correlated low-frequency burst across SCG axes
        x_noisy[i, 0:3, start:start + burst_len] += burst_shape
        # gyro proxy picks up a scaled, slightly delayed version of the same motion
        x_noisy[i, 3:6, start:start + burst_len] += burst_shape * 1.3

    return x_noisy


def patchify_3ch_target(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Match MotionArtifactCanceller's flattened-patch output layout for the 3 SCG channels."""
    b, c, t = x.shape  # c should be 3 (SCG only)
    n = t // patch_size
    x = x[:, :, : n * patch_size].reshape(b, c, n, patch_size)
    x = x.permute(0, 2, 1, 3).reshape(b, n, c * patch_size)
    return x


def run_denoiser_finetuning(data_path: str = "data/raw", epochs: int = 50, batch_size: int = 32,
                             lr: float = 5e-5, encoder_ckpt: str = "encoder_foundation.pt",
                             out_ckpt: str = "denoiser.pt", max_steps: int | None = None):
    set_seed(42)
    device = get_device()
    cfg = ModelConfig()

    build_multimodal_manifest(raw_data_path=data_path)
    train_df, val_df, _ = create_splits()
    train_ds = SCGWindowDataset(train_df)
    val_ds = SCGWindowDataset(val_df) if len(val_df) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_scg,
                               drop_last=len(train_ds) >= batch_size)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_scg)
                  if val_ds and len(val_ds) > 0 else None)

    foundation = SCGEncoder(cfg)
    ckpt_path = CHECKPOINT_DIR / encoder_ckpt
    if ckpt_path.exists():
        foundation.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        print(f"[finetune_denoiser] loaded pretrained foundation encoder from {ckpt_path}")
    else:
        print(f"[finetune_denoiser] WARNING: {ckpt_path} not found, using randomly initialized "
              f"foundation encoder. Run pretrain.py first for best results.")

    model = MotionArtifactCanceller(foundation, cfg).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"denoiser epoch {epoch+1}/{epochs}", leave=False)
        for x_clean, _meta in pbar:
            x_clean = x_clean.to(device)
            x_noisy = inject_motion_artifact(x_clean)

            pred_patches = model(x_noisy)                                   # (B, N, 3*P)
            target_patches = patchify_3ch_target(x_clean[:, :3, :], cfg.patch_size)

            loss = criterion(pred_patches, target_patches)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = epoch_loss / max(1, n_batches)

        if val_loader is not None:
            model.eval()
            val_loss, vb = 0.0, 0
            with torch.no_grad():
                for x_clean, _ in val_loader:
                    x_clean = x_clean.to(device)
                    x_noisy = inject_motion_artifact(x_clean)
                    pred = model(x_noisy)
                    target = patchify_3ch_target(x_clean[:, :3, :], cfg.patch_size)
                    val_loss += criterion(pred, target).item()
                    vb += 1
            val_loss /= max(1, vb)
        else:
            val_loss = train_loss

        print(f"[finetune_denoiser] epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / out_ckpt)

        if max_steps is not None and global_step >= max_steps:
            print(f"[finetune_denoiser] reached max_steps={max_steps}, stopping early.")
            break

    print(f"[finetune_denoiser] done. best_val_loss={best_val:.4f}. Saved to {CHECKPOINT_DIR / out_ckpt}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 2: motion artifact canceller fine-tuning")
    parser.add_argument("--data_path", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    run_denoiser_finetuning(data_path=args.data_path, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
