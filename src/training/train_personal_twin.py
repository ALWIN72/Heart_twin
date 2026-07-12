"""
src/training/train_personal_twin.py

Phase 4 training: fit a PersonalHeartTwin for one user.

Workflow per user:
  1. Load the frozen pretrained foundation encoder.
  2. Split this user's recordings into a "baseline" set (their earliest
     N recordings, assumed healthy/reference) and a "finetune" set used to
     train the personal VAE head with the standard VAE loss
     (reconstruction + KL-to-standard-normal).
  3. Call set_baseline() on the baseline set to lock in their personal
     "normal" distribution for anomaly scoring.
  4. Save the per-user twin checkpoint.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, ModelConfig, TwinConfig, get_device, set_seed
from src.data_loader import build_multimodal_manifest
from src.dataset import SCGWindowDataset, collate_scg
from src.models.encoder import SCGEncoder
from src.models.personal_twin import PersonalHeartTwin


def vae_loss(out: dict, kl_weight: float = 1e-3) -> tuple[torch.Tensor, dict]:
    recon_loss = F.mse_loss(out["recon"], out["target"])
    kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    total = recon_loss + kl_weight * kl_loss
    return total, {"recon_loss": recon_loss.item(), "kl_loss": kl_loss.item(), "total": total.item()}


def load_foundation_encoder(cfg: ModelConfig, encoder_ckpt: str = "encoder_foundation.pt") -> SCGEncoder:
    encoder = SCGEncoder(cfg)
    ckpt_path = CHECKPOINT_DIR / encoder_ckpt
    if ckpt_path.exists():
        encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    else:
        print(f"[train_personal_twin] WARNING: {ckpt_path} not found, using random-init encoder.")
    return encoder


def train_user_twin(user_id: str, manifest_df, foundation: SCGEncoder, n_baseline_recordings: int = 5,
                     epochs: int = 50, lr: float = 1e-4, kl_weight: float = 1e-3,
                     device: torch.device | None = None) -> PersonalHeartTwin | None:
    """
    manifest_df: full manifest (or a pre-filtered subset); will be filtered
    to this user's rows internally, sorted by recording_id so the first
    n_baseline_recordings are treated as the reference baseline.
    """
    device = device or get_device()
    cfg = ModelConfig()
    twin_cfg = TwinConfig()

    user_df = manifest_df[manifest_df["subject_id"].astype(str) == str(user_id)].copy()
    user_df = user_df.sort_values("recording_id").reset_index(drop=True)

    if len(user_df) < n_baseline_recordings + 1:
        print(f"[train_personal_twin] user {user_id} has only {len(user_df)} recordings "
              f"(<{n_baseline_recordings + 1} needed for baseline+finetune split). Skipping.")
        return None

    baseline_df = user_df.iloc[:n_baseline_recordings]
    finetune_df = user_df.iloc[n_baseline_recordings:]

    baseline_ds = SCGWindowDataset(baseline_df)
    finetune_ds = SCGWindowDataset(finetune_df)
    if len(baseline_ds) == 0 or len(finetune_ds) == 0:
        print(f"[train_personal_twin] user {user_id}: empty baseline or finetune window set. Skipping.")
        return None

    twin = PersonalHeartTwin(foundation, user_id=user_id, model_cfg=cfg, twin_cfg=twin_cfg).to(device)
    optimizer = torch.optim.Adam(
        list(twin.personal_encoder.parameters()) + list(twin.fc_mu.parameters())
        + list(twin.fc_logvar.parameters()) + list(twin.decoder.parameters()),
        lr=lr,
    )

    finetune_loader = DataLoader(finetune_ds, batch_size=min(16, len(finetune_ds)), shuffle=True,
                                  collate_fn=collate_scg)

    # Step 1: fine-tune the personal VAE head on this user's finetune split.
    twin.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, _meta in finetune_loader:
            x = x.to(device)
            out = twin(x)
            loss, parts = vae_loss(out, kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"[train_personal_twin] user={user_id} epoch {epoch+1}/{epochs} "
                  f"loss={epoch_loss / max(1, n_batches):.4f}")

    # Step 2: lock in the baseline ("what normal looks like for this person").
    baseline_loader = DataLoader(baseline_ds, batch_size=len(baseline_ds), shuffle=False, collate_fn=collate_scg)
    baseline_x, _ = next(iter(baseline_loader))
    twin.set_baseline(baseline_x.to(device))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / f"twin_user_{user_id}.pt"
    torch.save(twin.state_dict(), out_path)
    print(f"[train_personal_twin] saved twin for user {user_id} -> {out_path}")
    return twin


def train_all_users(data_path: str = "data/raw", n_baseline_recordings: int = 5, epochs: int = 50,
                     min_recordings: int = 6):
    set_seed(42)
    device = get_device()
    cfg = ModelConfig()

    manifest = build_multimodal_manifest(raw_data_path=data_path)
    foundation = load_foundation_encoder(cfg).to(device)
    for p in foundation.parameters():
        p.requires_grad = False

    counts = manifest.groupby("subject_id").size()
    eligible = counts[counts >= min_recordings].index.tolist()
    print(f"[train_personal_twin] {len(eligible)}/{manifest['subject_id'].nunique()} subjects "
          f"have >= {min_recordings} recordings and are eligible for a personal twin.")

    twins = {}
    for user_id in eligible:
        twin = train_user_twin(str(user_id), manifest, foundation,
                                n_baseline_recordings=n_baseline_recordings, epochs=epochs, device=device)
        if twin is not None:
            twins[str(user_id)] = twin
    return twins


def main():
    parser = argparse.ArgumentParser(description="Phase 4: personal heart twin training")
    parser.add_argument("--data_path", type=str, default="data/raw")
    parser.add_argument("--user_id", type=str, default=None, help="Train a single user; omit to train all eligible users.")
    parser.add_argument("--n_baseline_recordings", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.user_id:
        set_seed(42)
        device = get_device()
        cfg = ModelConfig()
        manifest = build_multimodal_manifest(raw_data_path=args.data_path)
        foundation = load_foundation_encoder(cfg).to(device)
        train_user_twin(args.user_id, manifest, foundation,
                         n_baseline_recordings=args.n_baseline_recordings, epochs=args.epochs, device=device)
    else:
        train_all_users(data_path=args.data_path, n_baseline_recordings=args.n_baseline_recordings,
                         epochs=args.epochs)


if __name__ == "__main__":
    main()
