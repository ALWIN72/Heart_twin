"""
run_pipeline.py

Top-level orchestration script: runs every training phase in sequence.
Each phase is independently re-runnable (see src/training/*.py), but this
script gives a single command to go from raw data to a fully trained
system + benchmark report.

Usage:
    # Quick smoke test on synthetic data (few epochs, runs in ~1 minute)
    python run_pipeline.py --smoke_test

    # Full run on real data
    python run_pipeline.py --data_path data/raw --epochs 200
"""
from __future__ import annotations

import argparse

from src.config import get_device
from src.data_loader import build_multimodal_manifest
from src.evaluation.benchmark import run_full_benchmark
from src.training.finetune_denoiser import run_denoiser_finetuning
from src.training.finetune_harmonizer import run_harmonizer_finetuning
from src.training.finetune_placement import run_placement_finetuning
from src.training.pretrain import run_pretraining
from src.training.train_personal_twin import train_all_users
from src.utils.synthetic_data import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Run the full Digital Heart Twin Digital Heart Twin pipeline")
    parser.add_argument("--data_path", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs for Phase 1 pretraining")
    parser.add_argument("--finetune_epochs", type=int, default=50, help="Epochs for Phases 2/3/3.5")
    parser.add_argument("--twin_epochs", type=int, default=50, help="Epochs for Phase 4 personal twins")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--smoke_test", action="store_true",
                         help="Generate a tiny synthetic dataset and run every phase for a "
                              "handful of steps, to verify the full pipeline executes end-to-end.")
    parser.add_argument("--skip_synthetic_gen", action="store_true",
                         help="Don't (re)generate synthetic data even with --smoke_test "
                              "(use existing data/raw contents).")
    args = parser.parse_args()

    print(f"[run_pipeline] device={get_device()}")

    if args.smoke_test:
        if not args.skip_synthetic_gen:
            print("[run_pipeline] === Generating synthetic dev dataset ===")
            generate_synthetic_dataset(n_subjects=6, recordings_per_subject=8, duration_seconds=20)
        epochs, finetune_epochs, twin_epochs, batch_size, max_steps = 2, 2, 3, 8, 6
    else:
        epochs, finetune_epochs, twin_epochs = args.epochs, args.finetune_epochs, args.twin_epochs
        batch_size, max_steps = args.batch_size, None

    print("[run_pipeline] === Building manifest ===")
    manifest = build_multimodal_manifest(raw_data_path=args.data_path)
    print(f"[run_pipeline] {len(manifest)} recordings across {manifest['subject_id'].nunique()} subjects.")
    if manifest.empty:
        raise RuntimeError(f"No data found at {args.data_path}. Use --smoke_test for synthetic "
                            f"data, or point --data_path at the real Digital Heart Twin dataset.")

    print("\n[run_pipeline] === Phase 1: Foundation model pretraining (MAE) ===")
    run_pretraining(data_path=args.data_path, epochs=epochs, batch_size=batch_size, max_steps=max_steps)

    print("\n[run_pipeline] === Phase 2: Motion artifact denoiser ===")
    run_denoiser_finetuning(data_path=args.data_path, epochs=finetune_epochs, batch_size=batch_size,
                             max_steps=max_steps)

    print("\n[run_pipeline] === Phase 3: Device harmonizer (DANN) ===")
    run_harmonizer_finetuning(data_path=args.data_path, epochs=finetune_epochs, batch_size=batch_size,
                               max_steps=max_steps)

    print("\n[run_pipeline] === Phase 3.5: Placement classifier ===")
    run_placement_finetuning(data_path=args.data_path, epochs=finetune_epochs, batch_size=batch_size,
                              max_steps=max_steps)

    print("\n[run_pipeline] === Phase 4: Personal heart twins (per user) ===")
    train_all_users(data_path=args.data_path, epochs=twin_epochs)

    print("\n[run_pipeline] === Phase 5: Benchmark evaluation ===")
    run_full_benchmark(data_path=args.data_path)

    print("\n[run_pipeline] === Done ===")


if __name__ == "__main__":
    main()
