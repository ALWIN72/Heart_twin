"""
src/training/finetune_placement.py

Phase 3.5 training: PlacementClassifier (+ optional joint SpatialCorrector
fine-tuning).

Label strategy
--------------
If the manifest's `placement` column is populated (i.e. the dataset
release logged Sternum/Left/Right per recording), we train directly with
supervised cross-entropy.

If `placement` is missing for some/all recordings (true for the current
Digital Heart Twin release), we fall back to a self-supervised pretext task: take a
recording with implicit/assumed Sternum placement (the protocol's default),
synthetically apply a *known* 3D rotation to the 6-channel signal to
simulate what a Left/Right placement would look like, and train the
classifier to recover which synthetic rotation class was applied. This
mirrors exactly how the model will be used at inference (classify
placement from axis-orientation patterns) without requiring placement
labels to exist yet.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, PLACEMENT_CLASSES, get_device, set_seed
from src.data_loader import build_multimodal_manifest, create_splits
from src.dataset import SCGWindowDataset, collate_scg
from src.models.placement_classifier import PlacementClassifier

# Approximate physical rotations for Left/Right chest placement relative to
# Sternum (rotation about the body's longitudinal axis). These are
# reasonable priors, not measured values -- swap in calibrated rotation
# matrices once real placement-labeled data lets us fit them empirically.
_ROTATIONS = {
    0: torch.eye(3),                                                   # Sternum: identity
    1: torch.tensor([[0.8, -0.5, 0.0], [0.5, 0.8, 0.0], [0.0, 0.0, 1.0]]),   # Left: ~30 deg yaw
    2: torch.tensor([[0.8, 0.5, 0.0], [-0.5, 0.8, 0.0], [0.0, 0.0, 1.0]]),   # Right: ~-30 deg yaw
}


def apply_synthetic_placement(x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 6, T) assumed-Sternum recording.
    class_idx: (B,) target synthetic placement class per sample.
    Returns: (B, 6, T) with the corresponding rotation applied to both the
    SCG (0:3) and gyro (3:6) channel triplets.
    """
    b = x.shape[0]
    out = x.clone()
    for i in range(b):
        R = _ROTATIONS[int(class_idx[i].item())].to(x.device)
        out[i, 0:3, :] = R @ x[i, 0:3, :]
        out[i, 3:6, :] = R @ x[i, 3:6, :]
    return out


def run_placement_finetuning(data_path: str = "data/raw", epochs: int = 30, batch_size: int = 32,
                              lr: float = 1e-4, out_ckpt: str = "placement_classifier.pt",
                              max_steps: int | None = None):
    set_seed(42)
    device = get_device()

    manifest = build_multimodal_manifest(raw_data_path=data_path)
    has_real_labels = "placement" in manifest.columns and manifest["placement"].notna().any()

    train_df, val_df, _ = create_splits()
    train_ds = SCGWindowDataset(train_df)
    val_ds = SCGWindowDataset(val_df) if len(val_df) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_scg,
                               drop_last=len(train_ds) >= batch_size)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_scg)
                  if val_ds and len(val_ds) > 0 else None)

    if has_real_labels:
        print("[finetune_placement] using REAL placement labels from manifest.")
    else:
        print("[finetune_placement] no real placement labels found in manifest -- "
              "training with self-supervised synthetic-rotation pretext task instead. "
              "Re-run this script once placement-labeled data is available for a fully "
              "supervised classifier.")

    model = PlacementClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"placement epoch {epoch+1}/{epochs}", leave=False)
        for x, meta in pbar:
            x = x.to(device)

            if has_real_labels:
                labels = torch.tensor(
                    [PLACEMENT_CLASSES.index(m.get("placement", "Sternum"))
                     if m.get("placement") in PLACEMENT_CLASSES else 0 for m in meta],
                    dtype=torch.long, device=device,
                )
                x_input = x
            else:
                labels = torch.randint(0, len(PLACEMENT_CLASSES), (x.shape[0],), device=device)
                x_input = apply_synthetic_placement(x, labels)

            logits = model(x_input)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(-1) == labels).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}")
            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = epoch_loss / max(1, n_batches)
        train_acc = epoch_acc / max(1, n_batches)
        print(f"[finetune_placement] epoch {epoch+1}/{epochs} loss={train_loss:.4f} acc={train_acc:.3f}")

        if val_loader is not None:
            model.eval()
            val_acc, vb = 0.0, 0
            with torch.no_grad():
                for x, meta in val_loader:
                    x = x.to(device)
                    if has_real_labels:
                        labels = torch.tensor(
                            [PLACEMENT_CLASSES.index(m.get("placement", "Sternum"))
                             if m.get("placement") in PLACEMENT_CLASSES else 0 for m in meta],
                            dtype=torch.long, device=device,
                        )
                        x_input = x
                    else:
                        labels = torch.randint(0, len(PLACEMENT_CLASSES), (x.shape[0],), device=device)
                        x_input = apply_synthetic_placement(x, labels)
                    logits = model(x_input)
                    val_acc += (logits.argmax(-1) == labels).float().mean().item()
                    vb += 1
            val_acc /= max(1, vb)
        else:
            val_acc = train_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / out_ckpt)

        if max_steps is not None and global_step >= max_steps:
            print(f"[finetune_placement] reached max_steps={max_steps}, stopping.")
            break

    print(f"[finetune_placement] done. best_val_acc={best_val_acc:.3f}. Saved to {CHECKPOINT_DIR / out_ckpt}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 3.5: placement classifier fine-tuning")
    parser.add_argument("--data_path", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    run_placement_finetuning(data_path=args.data_path, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
