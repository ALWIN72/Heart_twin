"""
src/training/pretrain.py

Phase 1: Self-supervised masked-autoencoder pretraining of the foundation
encoder (SCGEncoder) on all available recordings (no labels needed).

Usage:
    python -m src.training.pretrain --data_path data/raw --epochs 200 --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, ModelConfig, TrainConfig, get_device, set_seed
from src.data_loader import build_multimodal_manifest, create_splits
from src.dataset import SCGWindowDataset, collate_scg
from src.models.mae import MaskedAutoencoder


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: MaskedAutoencoder, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        total_loss += out["loss"].item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def run_pretraining(data_path: str = "data/raw", epochs: int = 200, batch_size: int = 64,
                     lr: float = 1.5e-4, checkpoint_name: str = "encoder_foundation.pt",
                     log_every: int = 10, max_steps: int | None = None):
    set_seed(42)
    device = get_device()
    print(f"[pretrain] device={device}")

    manifest = build_multimodal_manifest(raw_data_path=data_path)
    if manifest.empty:
        raise RuntimeError(
            f"No recordings found under {data_path}. Generate the synthetic dev "
            f"dataset (src/utils/synthetic_data.py) or point --data_path at the "
            f"real MSCardio dataset before pretraining."
        )
    train_df, val_df, _test_df = create_splits()
    print(f"[pretrain] train recordings={len(train_df)} val recordings={len(val_df)}")

    train_ds = SCGWindowDataset(train_df)
    val_ds = SCGWindowDataset(val_df) if len(val_df) > 0 else None
    print(f"[pretrain] train windows={len(train_ds)} val windows={len(val_ds) if val_ds else 0}")

    train_cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)
    model_cfg = ModelConfig()

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                               collate_fn=collate_scg, num_workers=0, drop_last=len(train_ds) >= train_cfg.batch_size)
    val_loader = (DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_scg)
                  if val_ds and len(val_ds) > 0 else None)

    model = MaskedAutoencoder(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, steps_per_epoch=max(1, len(train_loader)))

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{train_cfg.epochs}", leave=False)
        for x, _meta in pbar:
            x = x.to(device)
            out = model(x)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = epoch_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            history["val_loss"].append(val_loss)
        else:
            val_loss = train_loss  # no held-out subjects available; fall back to train loss

        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == train_cfg.epochs - 1:
            elapsed = time.time() - start_time
            print(f"[pretrain] epoch {epoch+1}/{train_cfg.epochs} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed={elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            encoder = model.export_encoder()
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), CHECKPOINT_DIR / checkpoint_name)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"mae_full_{checkpoint_name}")

        if max_steps is not None and global_step >= max_steps:
            print(f"[pretrain] reached max_steps={max_steps}, stopping early.")
            break

    with open(CHECKPOINT_DIR / "pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[pretrain] done. best_val_loss={best_val:.4f}. "
          f"Encoder saved to {CHECKPOINT_DIR / checkpoint_name}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Phase 1: MAE foundation model pretraining")
    parser.add_argument("--data_path", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Optional cap on total optimizer steps, useful for smoke tests.")
    args = parser.parse_args()
    run_pretraining(data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
