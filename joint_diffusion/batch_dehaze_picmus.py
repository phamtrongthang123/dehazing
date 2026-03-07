"""Batch dehazing of all PICMUS tissue val frames.

Processes all 43 val samples in batches, saves dehazed RF and B-mode images.

Usage:
  python batch_dehaze_picmus.py --inf_config configs/inference/paper/picmus_dehaze_v4.yaml
  python batch_dehaze_picmus.py --inf_config configs/inference/paper/picmus_dehaze_v4.yaml --n_batch 4
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, str(Path(__file__).parent))

from datasets import ZeaDataset
from utils.inverse import SGMDenoiser
from utils.runs import init_config
from utils.utils import load_config_from_yaml, set_random_seed


def batch_dehaze(inf_config_path: str, n_batch: int = 4, seed: int = 1234,
                 dynamic_range: tuple = (-50, 0), output_dir: str = None):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    inf_cfg = load_config_from_yaml(Path(inf_config_path))
    run_ids = inf_cfg.run_id
    cfg = init_config(run_ids["sgm"], inf_cfg)
    cfg["num_img"] = n_batch
    cfg["keep_track"] = False
    cfg["show_noise_priors"] = False

    # Load val dataset
    data_root = Path(cfg.data_root)
    val_path = data_root / "picmus" / "tissue" / "val.npz"
    print(f"Loading val dataset from: {val_path}")
    val_ds = ZeaDataset(
        val_path,
        npz_key=cfg.get("npz_key", "rf"),
        image_range=tuple(cfg.get("image_range", [0, 1])),
        training=False,
    )
    n_total = len(val_ds)
    print(f"Val dataset: {n_total} samples")

    # Set image_shape from dataset
    sample = val_ds[0]
    cfg.image_shape = list(sample.shape)
    cfg["num_img"] = n_batch

    # Output directory
    if output_dir is None:
        config_name = Path(inf_config_path).stem
        output_dir = f"results/batch_dehaze_{config_name}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Create denoiser once
    print("\nLoading score model...")
    denoiser = SGMDenoiser(config=cfg, dataset=None)

    # B-mode converter
    from utils.bmode_picmus import rf_to_bmode, undo_normalization

    def to_bmode(rf_tensor):
        if isinstance(rf_tensor, torch.Tensor):
            x = rf_tensor.detach().cpu().numpy()
        else:
            x = rf_tensor.copy()
        vmin, vmax = val_ds.image_range
        x = np.clip(x, vmin, vmax)
        x = undo_normalization(x, image_range=(vmin, vmax),
                               data_min=val_ds.data_min, data_max=val_ds.data_max)
        return rf_to_bmode(x, dynamic_range=dynamic_range)

    # Process all frames in batches
    all_denoised_rf = []
    all_tissue_rf = []
    all_hazy_rf = []

    for batch_start in range(0, n_total, n_batch):
        batch_end = min(batch_start + n_batch, n_total)
        batch_idx = list(range(batch_start, batch_end))
        print(f"\n--- Batch {batch_start // n_batch + 1}: frames {batch_idx[0]}-{batch_idx[-1]} ({len(batch_idx)} frames) ---")

        frames = torch.stack([val_ds[i] for i in batch_idx])
        denoised = denoiser(target_samples=frames, plot=False, save=False)

        if isinstance(denoised, (list, tuple)):
            denoised = denoised[-1]

        noisy = denoiser.noisy_samples
        target = denoiser.target_samples

        # Store RF tensors
        all_tissue_rf.append(target.detach().cpu())
        all_hazy_rf.append(noisy.detach().cpu())
        all_denoised_rf.append(denoised.detach().cpu())

        # Convert to B-mode and save individual images
        tissue_bmodes = to_bmode(target)
        hazy_bmodes = to_bmode(noisy)
        denoised_bmodes = to_bmode(denoised)

        for i, idx in enumerate(batch_idx):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(tissue_bmodes[i], cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Tissue (GT)")
            axes[0].axis("off")
            axes[1].imshow(hazy_bmodes[i], cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Hazy")
            axes[1].axis("off")
            axes[2].imshow(denoised_bmodes[i], cmap="gray", vmin=0, vmax=255)
            axes[2].set_title("Dehazed")
            axes[2].axis("off")
            fig.suptitle(f"Val frame {idx}", fontsize=14)
            fig.tight_layout()
            fig.savefig(output_dir / f"frame_{idx:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Saved {len(batch_idx)} frame images")

    # Concatenate all results
    all_tissue_rf = torch.cat(all_tissue_rf, dim=0)
    all_hazy_rf = torch.cat(all_hazy_rf, dim=0)
    all_denoised_rf = torch.cat(all_denoised_rf, dim=0)

    # Save summary grid (first 10 frames)
    n_show = min(10, n_total)
    tissue_bmodes = to_bmode(all_tissue_rf[:n_show])
    hazy_bmodes = to_bmode(all_hazy_rf[:n_show])
    denoised_bmodes = to_bmode(all_denoised_rf[:n_show])

    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4 * n_show))
    for i in range(n_show):
        axes[i, 0].imshow(tissue_bmodes[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(hazy_bmodes[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(denoised_bmodes[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 0].set_title("Tissue (GT)", fontsize=14)
            axes[i, 1].set_title("Hazy", fontsize=14)
            axes[i, 2].set_title("Dehazed", fontsize=14)
        axes[i, 0].set_ylabel(f"Frame {i}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary grid to {output_dir / 'summary_grid.png'}")

    # Save dehazed RF as npz for further analysis
    np.savez_compressed(
        output_dir / "dehazed_rf.npz",
        tissue=all_tissue_rf.numpy(),
        hazy=all_hazy_rf.numpy(),
        dehazed=all_denoised_rf.numpy(),
    )
    print(f"Saved dehazed RF data to {output_dir / 'dehazed_rf.npz'}")
    print(f"\nDone! {n_total} frames dehazed. Results in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Batch dehaze all PICMUS val frames")
    parser.add_argument("--inf_config", type=str, required=True,
                        help="Path to inference config yaml")
    parser.add_argument("--n_batch", type=int, default=4,
                        help="Frames per batch (default 4)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/batch_dehaze_<config_name>)")
    args = parser.parse_args()
    batch_dehaze(args.inf_config, n_batch=args.n_batch, seed=args.seed,
                 output_dir=args.output_dir)


if __name__ == "__main__":
    main()
