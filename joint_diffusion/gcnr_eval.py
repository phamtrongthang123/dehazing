"""Phase 5: gCNR evaluation for PICMUS contrast_speckle frames.

Computes generalized contrast-to-noise ratio (gCNR) for:
  1. Ground-truth tissue (clean RF)
  2. Hazy input (tissue + synthetic haze)
  3. Dehazed output (score model reconstruction)

Uses the known cyst location from the PICMUS contrast_speckle phantom:
  - Cyst 1: X=-0.1mm, Z=14.9mm, diameter=4.5mm
  - Mapped to pixel coords (cx=134, cz=67) in 271x237 B-mode image
  - B-mode FOV: X=[-20,+20]mm, Z=[5,40]mm

Val frames from contrast_speckle (seed=42 split):
  - Positions [6, 14, 17, 19, 20, 21, 22, 26, 29, 32, 35, 36, 38]
  - 13 frames total (6 experimental + 7 simulation)

Usage:
  python gcnr_eval.py --inf_config configs/inference/paper/picmus_dehaze_v4.yaml
  python gcnr_eval.py --inf_config configs/inference/paper/picmus_dehaze_v5.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, str(Path(__file__).parent))

from datasets import ZeaDataset
from utils.inverse import SGMDenoiser
from utils.runs import init_config
from utils.utils import load_config_from_yaml, set_random_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Val indices (0-indexed in val.npz) that come from contrast_speckle datasets.
# Computed by tracing data_conversion/convert_picmus.py with seed=42.
# 6 from contrast_speckle_expe + 7 from contrast_speckle_simu = 13 total.
CONTRAST_VAL_INDICES = [6, 14, 17, 19, 20, 21, 22, 26, 29, 32, 35, 36, 38]

# Cyst 1 pixel coordinates in 271x237 B-mode image (cx=x-pixel, cz=z-pixel from top)
# Physical: X=-0.1mm, Z=14.9mm, diameter=4.5mm (radius=2.25mm)
# FOV: X=[-20,+20]mm => 271 pixels, Z=[5,40]mm => 237 pixels
CYST_CX = 134   # x pixel (column)
CYST_CZ = 67    # z pixel (row)
R_SIGNAL = 13   # signal ROI radius (pixels) — slightly inside cyst boundary
R_BG_INNER = 17  # background annulus inner radius
R_BG_OUTER = 30  # background annulus outer radius


def make_roi_masks(img_h, img_w, cz, cx, r_sig, r_bg_inner, r_bg_outer):
    """Create signal (cyst) and background (annulus) boolean masks.

    Args:
        img_h: image height (z dimension)
        img_w: image width (x dimension)
        cz, cx: cyst center pixel coordinates
        r_sig: signal ROI radius
        r_bg_inner, r_bg_outer: background annulus radii

    Returns:
        signal_mask, background_mask — boolean arrays of shape (img_h, img_w)
    """
    zz, xx = np.mgrid[:img_h, :img_w]
    dist2 = (zz - cz) ** 2 + (xx - cx) ** 2
    signal_mask = dist2 <= r_sig ** 2
    bg_mask = (dist2 >= r_bg_inner ** 2) & (dist2 <= r_bg_outer ** 2)
    return signal_mask, bg_mask


def gcnr(x, y, bins=256):
    """Generalized contrast-to-noise ratio (from processing.py)."""
    x = x.flatten()
    y = y.flatten()
    _, bin_edges = np.histogram(np.concatenate((x, y)), bins=bins)
    f, _ = np.histogram(x, bins=bin_edges, density=True)
    g, _ = np.histogram(y, bins=bin_edges, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))


def rf_to_bmode_normalized(rf_tensor, dataset, dynamic_range=(-50, 0)):
    """Convert normalized RF tensor to B-mode uint8 image.

    Applies undo_normalization (reverse of ZeaDataset) then beamforms.

    Args:
        rf_tensor: (N, C, H, W) tensor or (C, H, W) tensor in [vmin, vmax] range
        dataset: ZeaDataset instance (for data_min, data_max, image_range)
        dynamic_range: dB range for log compression

    Returns:
        List of B-mode uint8 numpy arrays, each shape (grid_z, grid_x)
    """
    from utils.bmode_picmus import rf_to_bmode, undo_normalization

    if isinstance(rf_tensor, torch.Tensor):
        x = rf_tensor.detach().cpu().numpy()
    else:
        x = rf_tensor.copy()

    vmin, vmax = dataset.image_range
    x = np.clip(x, vmin, vmax)
    x = undo_normalization(
        x,
        image_range=(vmin, vmax),
        data_min=dataset.data_min,
        data_max=dataset.data_max,
    )
    return rf_to_bmode(x, dynamic_range=dynamic_range)


def evaluate_gcnr(inf_config_path: str, n_batch: int = 4, seed: int = 1234,
                  dynamic_range: tuple = (-50, 0)):
    """Run gCNR evaluation on contrast_speckle val frames.

    Args:
        inf_config_path: Path to inference config yaml (e.g. picmus_dehaze_v4.yaml)
        n_batch: Number of frames to process per batch (memory constraint)
        seed: Random seed
        dynamic_range: dB range for B-mode display

    Returns:
        dict with gCNR means for tissue, hazy, dehazed
    """
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load inference config and merge with model training config
    inf_cfg = load_config_from_yaml(Path(inf_config_path))
    run_ids = inf_cfg.run_id
    cfg = init_config(run_ids["sgm"], inf_cfg)
    cfg["num_img"] = n_batch
    cfg["keep_track"] = False
    cfg["show_noise_priors"] = False

    # Load val dataset (full, for normalization constants)
    data_root = Path(cfg.data_root)
    val_path = data_root / "picmus" / "tissue" / "val.npz"
    print(f"Loading val dataset from: {val_path}")
    val_ds = ZeaDataset(
        val_path,
        npz_key=cfg.get("npz_key", "rf"),
        image_range=tuple(cfg.get("image_range", [0, 1])),
        training=False,
    )
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Contrast val indices: {CONTRAST_VAL_INDICES}")
    print(f"Will evaluate {len(CONTRAST_VAL_INDICES)} contrast_speckle frames")

    # Precompute ROI masks (B-mode image shape: grid_z=237, grid_x=271)
    bmode_h, bmode_w = 237, 271
    signal_mask, bg_mask = make_roi_masks(
        bmode_h, bmode_w, CYST_CZ, CYST_CX,
        R_SIGNAL, R_BG_INNER, R_BG_OUTER,
    )
    print(f"Signal ROI pixels: {signal_mask.sum()}, Background ROI pixels: {bg_mask.sum()}")

    gcnr_tissue_list = []
    gcnr_hazy_list = []
    gcnr_denoised_list = []

    # Set image_shape from the dataset (3, 1024, 128 for PICMUS)
    sample = val_ds[0]
    cfg.image_shape = list(sample.shape)
    cfg["num_img"] = n_batch

    # Create denoiser once (loads model checkpoint once)
    print("\nLoading score model...")
    denoiser = SGMDenoiser(config=cfg, dataset=None)

    # Process frames in batches
    indices = CONTRAST_VAL_INDICES
    for batch_start in range(0, len(indices), n_batch):
        batch_idx = indices[batch_start:batch_start + n_batch]
        print(f"\n--- Batch {batch_start // n_batch + 1}: val indices {batch_idx} ---")

        # Extract frames from dataset
        frames = torch.stack([val_ds[i] for i in batch_idx])  # (B, C, H, W)

        # Run denoiser with explicit target_samples (bypasses random val loader)
        denoised = denoiser(target_samples=frames, plot=False, save=False)

        if isinstance(denoised, (list, tuple)):
            denoised = denoised[-1]

        noisy = denoiser.noisy_samples
        target = denoiser.target_samples

        # Convert to B-mode
        print("  Converting to B-mode...")
        tissue_bmodes = rf_to_bmode_normalized(target, val_ds, dynamic_range)
        hazy_bmodes = rf_to_bmode_normalized(noisy, val_ds, dynamic_range)
        denoised_bmodes = rf_to_bmode_normalized(denoised, val_ds, dynamic_range)

        # Compute gCNR for each frame
        for i in range(len(batch_idx)):
            t_bmode = np.asarray(tissue_bmodes[i]).astype(np.float32)
            h_bmode = np.asarray(hazy_bmodes[i]).astype(np.float32)
            d_bmode = np.asarray(denoised_bmodes[i]).astype(np.float32)

            # Verify B-mode shape
            if t_bmode.shape != (bmode_h, bmode_w):
                print(f"  WARNING: unexpected bmode shape {t_bmode.shape}, "
                      f"expected ({bmode_h}, {bmode_w})")

            g_tissue = gcnr(t_bmode[signal_mask], t_bmode[bg_mask])
            g_hazy = gcnr(h_bmode[signal_mask], h_bmode[bg_mask])
            g_denoised = gcnr(d_bmode[signal_mask], d_bmode[bg_mask])

            gcnr_tissue_list.append(g_tissue)
            gcnr_hazy_list.append(g_hazy)
            gcnr_denoised_list.append(g_denoised)

            print(f"  Frame {batch_idx[i]:2d}: gCNR tissue={g_tissue:.3f}, "
                  f"hazy={g_hazy:.3f}, denoised={g_denoised:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"gCNR Summary ({len(indices)} contrast_speckle frames)")
    print(f"{'='*60}")
    print(f"  Tissue (clean ground truth): {np.mean(gcnr_tissue_list):.3f} "
          f"± {np.std(gcnr_tissue_list):.3f}")
    print(f"  Hazy input (+ noise):        {np.mean(gcnr_hazy_list):.3f} "
          f"± {np.std(gcnr_hazy_list):.3f}")
    print(f"  Dehazed output:              {np.mean(gcnr_denoised_list):.3f} "
          f"± {np.std(gcnr_denoised_list):.3f}")

    improvement = np.mean(gcnr_denoised_list) - np.mean(gcnr_hazy_list)
    print(f"\n  gCNR improvement (dehazed - hazy): {improvement:+.3f}")
    if improvement > 0:
        print("  => PASS: dehazing improves gCNR")
    else:
        print("  => FAIL: dehazing does not improve gCNR")

    return {
        "gcnr_tissue": float(np.mean(gcnr_tissue_list)),
        "gcnr_hazy": float(np.mean(gcnr_hazy_list)),
        "gcnr_denoised": float(np.mean(gcnr_denoised_list)),
        "gcnr_improvement": float(improvement),
        "n_frames": len(indices),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: gCNR evaluation on PICMUS contrast_speckle val frames"
    )
    parser.add_argument(
        "--inf_config",
        type=str,
        required=True,
        help="Path to inference config yaml (e.g. configs/inference/paper/picmus_dehaze_v4.yaml)",
    )
    parser.add_argument(
        "--n_batch", type=int, default=4,
        help="Frames per batch (memory). Default 4 (~16GB on A100).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    results = evaluate_gcnr(args.inf_config, n_batch=args.n_batch, seed=args.seed)

    print("\nFinal results (for SCORES.md):")
    print(f"  gCNR tissue:   {results['gcnr_tissue']:.3f}")
    print(f"  gCNR hazy:     {results['gcnr_hazy']:.3f}")
    print(f"  gCNR dehazed:  {results['gcnr_denoised']:.3f}")
    print(f"  Improvement:   {results['gcnr_improvement']:+.3f}")


if __name__ == "__main__":
    main()
