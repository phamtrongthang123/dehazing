"""Generate synthetic training data from a trained ZEA score model.

Pipeline validation: generate samples from the known-good ZEA tissue model,
denormalize them back to raw RF scale, and save as NPZ files that ZeaDataset
can re-load. Then train a fresh model on this synthetic data to verify the
training pipeline is sound.

Usage:
    cd joint_diffusion
    conda run -n dehazing python generate_training_data.py
    conda run -n dehazing python generate_training_data.py --n_train 400 --n_val 50
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from generate_samples import load_model_and_sampler
from utils.bmode import undo_normalization


def load_original_stats(data_root, subdir="zea_synth", kind="tissue"):
    """Load original dataset to get data_min/data_max for denormalization."""
    path = Path(data_root) / subdir / kind / "train.npz"
    data = np.load(path)["rf"].astype(np.float32)
    return float(data.min()), float(data.max())


def generate_batch(sampler, batch_size, image_shape, device):
    """Generate a batch of unconditional samples."""
    shape = [batch_size, *image_shape]
    with torch.no_grad():
        samples = sampler(y=None, z=None, shape=shape, progress_bar=True)
    return torch.clamp(samples, -1, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from ZEA tissue model"
    )
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--data_root", type=str,
        default="/scrfs/storage/tp030/home/f2f_ldm/data",
    )
    parser.add_argument(
        "--tissue_run", type=str,
        default="wandb/run--jm8ljbms/files",
    )
    parser.add_argument("--output_subdir", type=str, default="zea_synth_generated")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────
    print("=== Loading ZEA tissue model ===")
    model, sampler, config, device = load_model_and_sampler(
        args.tissue_run, use_ema=False,
    )
    image_shape = config.image_shape
    image_range = tuple(config.get("image_range", [-1, 1]))
    print(f"  image_shape: {image_shape}, image_range: {image_range}")

    # ── Get original data stats for denormalization ───────────────────
    print("=== Loading original data stats ===")
    data_min, data_max = load_original_stats(args.data_root)
    print(f"  data_min={data_min:.4f}, data_max={data_max:.4f}")

    # ── Generate samples ──────────────────────────────────────────────
    for split, n_samples in [("train", args.n_train), ("val", args.n_val)]:
        print(f"\n=== Generating {n_samples} {split} samples ===")
        n_batches = math.ceil(n_samples / args.batch_size)
        all_samples = []

        for i in range(n_batches):
            bs = min(args.batch_size, n_samples - len(all_samples))
            print(f"  Batch {i+1}/{n_batches} (size {bs})...")
            samples = generate_batch(sampler, bs, image_shape, device)
            all_samples.append(samples.cpu().numpy())

        data = np.concatenate(all_samples, axis=0)[:n_samples]
        print(f"  Generated shape: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]")

        # ── Denormalize: image_range → mu-law expand → min-max restore ──
        data = undo_normalization(
            data, image_range=image_range, mu=255,
            data_min=data_min, data_max=data_max,
        )
        print(f"  Denormalized range: [{data.min():.3f}, {data.max():.3f}]")

        # ── Save ──────────────────────────────────────────────────────
        out_dir = Path(args.data_root) / args.output_subdir / "tissue"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.npz"
        np.savez_compressed(out_path, rf=data.astype(np.float32))
        print(f"  Saved to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
