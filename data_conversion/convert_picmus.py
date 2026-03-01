"""Convert PICMUS HDF5 RF data to NPZ format for training.

Reads all 6 PICMUS datasets, selects 3 plane-wave angles closest to -5, 0, +5 deg,
resamples axially to 1024 samples, and creates training frames by sliding windows
across the 75 available angles.

Output: /scrfs/storage/tp030/home/f2f_ldm/data/picmus/tissue/{train,val}.npz
Shape: (N, 3, 1024, 128) with key 'rf'
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample


PICMUS_ROOT = Path("/scrfs/storage/tp030/home/f2f_ldm/data/picmus_raw")
OUTPUT_ROOT = Path("/scrfs/storage/tp030/home/f2f_ldm/data/picmus")

# All RF dataset files (experimental + simulation + in-vivo)
RF_FILES = [
    # Experimental
    "archive_to_download/database/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
    "archive_to_download/database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
    # Simulation
    "archive_to_download/database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
    "archive_to_download/database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
    # In-vivo
    "in_vivo/carotid_cross/carotid_cross_expe_dataset_rf.hdf5",
    "in_vivo/carotid_long/carotid_long_expe_dataset_rf.hdf5",
]

N_AX_TARGET = 1024  # Target axial samples
N_TX = 3            # Number of transmit angles per frame


def load_rf_data(path):
    """Load RF data and angles from a PICMUS HDF5 file.

    Returns:
        data: array of shape (n_angles, n_el, n_ax)
        angles: array of shape (n_angles,) in radians
    """
    with h5py.File(path, "r") as f:
        ds = f["US/US_DATASET0000"]
        data = np.array(ds["data/real"])  # (n_angles, n_el, n_ax)
        angles = np.array(ds["angles"])   # (n_angles,) in radians
    return data, angles


def extract_frames(data, angles, n_tx=3, stride=1):
    """Extract overlapping frames of n_tx consecutive angles.

    Creates many training frames by sliding a window of `n_tx` angles across
    the 75 available angles with a given stride.

    Args:
        data: (n_angles, n_el, n_ax) RF data
        angles: (n_angles,) angle values
        n_tx: number of angles per frame
        stride: sliding window stride

    Returns:
        frames: (N, n_tx, n_ax_resampled, n_el) ready for training
    """
    n_angles, n_el, n_ax = data.shape
    frames = []

    for start in range(0, n_angles - n_tx + 1, stride):
        # Select n_tx consecutive angles
        frame = data[start:start + n_tx]  # (n_tx, n_el, n_ax)
        # Transpose to (n_tx, n_ax, n_el)
        frame = frame.transpose(0, 2, 1)
        # Resample axially to N_AX_TARGET
        if n_ax != N_AX_TARGET:
            frame = resample(frame, N_AX_TARGET, axis=1)
        frames.append(frame.astype(np.float32))

    return np.stack(frames)  # (N, n_tx, n_ax, n_el)


def main():
    parser = argparse.ArgumentParser(description="Convert PICMUS data to NPZ")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sliding window stride across angles (default: 1)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    all_frames = []

    for rf_file in RF_FILES:
        path = PICMUS_ROOT / rf_file
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            continue

        data, angles = load_rf_data(path)
        print(f"  {rf_file.split('/')[-1]}: data shape {data.shape}, "
              f"{len(angles)} angles")

        frames = extract_frames(data, angles, n_tx=N_TX, stride=args.stride)
        print(f"    -> {len(frames)} frames of shape {frames.shape[1:]}")
        all_frames.append(frames)

    all_frames = np.concatenate(all_frames, axis=0)
    print(f"\nTotal frames: {len(all_frames)}, shape: {all_frames.shape}")

    # Shuffle and split
    indices = rng.permutation(len(all_frames))
    n_val = max(1, int(len(all_frames) * args.val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_data = all_frames[train_idx]
    val_data = all_frames[val_idx]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Train stats: min={train_data.min():.6f}, max={train_data.max():.6f}")
    print(f"Val stats:   min={val_data.min():.6f}, max={val_data.max():.6f}")

    # Save
    out_dir = OUTPUT_ROOT / "tissue"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "train.npz", rf=train_data)
    np.savez(out_dir / "val.npz", rf=val_data)
    print(f"\nSaved to {out_dir}/{{train,val}}.npz")


if __name__ == "__main__":
    main()
