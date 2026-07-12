"""
src/training/finetune_harmonizer.py

Phase 3 training: Domain-Adversarial fine-tuning of the DeviceHarmonizer.

The 'device' label comes straight from the manifest's `platform` column
(iOS/Android), built from each subject's general_metadata.json. We jointly
optimize:
  - reconstruction loss (keeps features informative)
  - domain classification loss, through a gradient-reversal layer (makes
    features device-invariant)
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DEVICE_PLATFORMS, ModelConfig, get_device, set_seed
from src.data_loader import build_multimodal_manifest, create_splits
from src.dataset import SCGWindowDataset
from src.models.dann import DeviceHarmonizer
from src.models.encoder import SCGEncoder


def platform_to_idx(platform: str) -> int:
    try:
        return DEVICE_PLATFORMS.index(platform)
    except ValueError:
        return 0  # unknown platforms default to class 0


def collate_with_platform(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    platform_idx = torch.tensor([platform_to_idx(b[1]["platform"]) for b in batch], dtype=torch.long)
    return xs, platform_idx


def patchify_target(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, t = x.shape
    n = t // patch_size
    x = x[:, :, : n * patch_size].reshape(b, c, n, patch_size)
    x = x.permute(0, 2, 1, 3).reshape(b, n, c * patch_size)
    return x


def run_harmonizer_finetuning(data_path: str = "data/raw", epochs: int = 50, batch_size: int = 32,
                               lr: float = 5e-5, lambda_domain: float = 0.5,
                               encoder_ckpt: str = "encoder_foundation.pt",
                               out_ckpt: str = "harmonizer.pt", max_steps: int | None = None):
    set_seed(42)
    device = get_device()
    cfg = ModelConfig()

    build_multimodal_manifest(raw_data_path=data_path)
    train_df, val_df, _ = create_splits()

    n_platforms = max(2, train_df["platform"].nunique())
    train_ds = SCGWindowDataset(train_df)
    val_ds = SCGWindowDataset(val_df) if len(val_df) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_with_platform, drop_last=len(train_ds) >= batch_size)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_with_platform)
                  if val_ds and len(val_ds) > 0 else None)

    foundation = SCGEncoder(cfg)
    ckpt_path = CHECKPOINT_DIR / encoder_ckpt
    if ckpt_path.exists():
        foundation.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        print(f"[finetune_harmonizer] loaded pretrained encoder from {ckpt_path}")
    else:
        print(f"[finetune_harmonizer] WARNING: {ckpt_path} not found, using random init.")

    model = DeviceHarmonizer(foundation, num_devices=len(DEVICE_PLATFORMS), cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    global_step = 0
    total_steps = epochs * max(1, len(train_loader))

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_domain_acc, n_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"harmonizer epoch {epoch+1}/{epochs}", leave=False)
        for x, platform_idx in pbar:
            x, platform_idx = x.to(device), platform_idx.to(device)

            # Ramp up the adversarial strength over training (standard DANN schedule)
            p = global_step / max(1, total_steps)
            alpha = (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0).item() * lambda_domain

            recon, domain_pred = model(x, alpha=alpha)
            target = patchify_target(x, cfg.patch_size)

            recon_loss = recon_criterion(recon, target)
            domain_loss = domain_criterion(domain_pred, platform_idx)
            loss = recon_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            domain_acc = (domain_pred.argmax(-1) == platform_idx).float().mean().item()
            epoch_loss += loss.item()
            epoch_domain_acc += domain_acc
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", domain_acc=f"{domain_acc:.2f}", alpha=f"{alpha:.2f}")
            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = epoch_loss / max(1, n_batches)
        train_domain_acc = epoch_domain_acc / max(1, n_batches)
        # NOTE: as training succeeds, domain_acc should drift TOWARD chance
        # level (1/n_platforms) -- that's the adversarial objective working,
        # not a sign of failure.
        print(f"[finetune_harmonizer] epoch {epoch+1}/{epochs} loss={train_loss:.4f} "
              f"domain_acc={train_domain_acc:.3f} (chance={1/len(DEVICE_PLATFORMS):.2f})")

        if val_loader is not None:
            model.eval()
            val_loss, vb = 0.0, 0
            with torch.no_grad():
                for x, platform_idx in val_loader:
                    x, platform_idx = x.to(device), platform_idx.to(device)
                    recon, domain_pred = model(x, alpha=0.0)
                    target = patchify_target(x, cfg.patch_size)
                    val_loss += recon_criterion(recon, target).item()
                    vb += 1
            val_loss /= max(1, vb)
        else:
            val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / out_ckpt)

        if max_steps is not None and global_step >= max_steps:
            print(f"[finetune_harmonizer] reached max_steps={max_steps}, stopping.")
            break

    print(f"[finetune_harmonizer] done. best_val_recon_loss={best_val:.4f}. Saved to {CHECKPOINT_DIR / out_ckpt}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 3: device harmonizer (DANN) fine-tuning")
    parser.add_argument("--data_path", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    run_harmonizer_finetuning(data_path=args.data_path, epochs=args.epochs,
                               batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
