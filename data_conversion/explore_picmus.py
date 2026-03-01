"""Explore PICMUS HDF5 dataset structure.

Prints key hierarchy, array shapes, and probe metadata for all RF datasets.
"""
import h5py
import numpy as np
from pathlib import Path


PICMUS_ROOT = Path("/scrfs/storage/tp030/home/f2f_ldm/data/picmus_raw")


def print_hdf5_tree(group, prefix=""):
    """Recursively print HDF5 group structure."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            print_hdf5_tree(item, prefix + "  ")
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"{prefix}{key}: {type(item)}")


def explore_file(path):
    """Explore a single HDF5 file."""
    print(f"\n{'='*80}")
    print(f"FILE: {path}")
    print(f"{'='*80}")
    with h5py.File(path, "r") as f:
        print_hdf5_tree(f)
        # Print attributes at root level
        if f.attrs:
            print("\nRoot attributes:")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")


def explore_rf_data(path):
    """Deep exploration of RF dataset file."""
    print(f"\n{'='*80}")
    print(f"RF DATA: {path.name}")
    print(f"{'='*80}")
    with h5py.File(path, "r") as f:
        print_hdf5_tree(f)

        # Try to find the actual RF data arrays
        def find_datasets(group, path=""):
            datasets = []
            for key in group.keys():
                item = group[key]
                full_path = f"{path}/{key}"
                if isinstance(item, h5py.Dataset):
                    datasets.append((full_path, item.shape, item.dtype))
                elif isinstance(item, h5py.Group):
                    datasets.extend(find_datasets(item, full_path))
            return datasets

        datasets = find_datasets(f)
        print(f"\nAll datasets:")
        for dpath, shape, dtype in datasets:
            print(f"  {dpath}: shape={shape}, dtype={dtype}")

        # Try to read a small sample to understand value range
        for dpath, shape, dtype in datasets:
            if len(shape) >= 2 and np.prod(shape) > 100:
                data = f[dpath]
                sample = data[..., :min(10, shape[-1])] if len(shape) > 1 else data[:10]
                sample = np.array(sample)
                print(f"\n  Sample from {dpath}:")
                print(f"    min={sample.min():.6f}, max={sample.max():.6f}, "
                      f"mean={sample.mean():.6f}, std={sample.std():.6f}")


def explore_scan(path):
    """Explore scan metadata file."""
    print(f"\n{'='*80}")
    print(f"SCAN: {path.name}")
    print(f"{'='*80}")
    with h5py.File(path, "r") as f:
        print_hdf5_tree(f)

        # Try to extract probe/scan parameters
        def print_all_scalars(group, prefix=""):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    if item.shape == () or (len(item.shape) == 1 and item.shape[0] <= 200):
                        val = np.array(item)
                        print(f"  {prefix}{key} = {val}")
                elif isinstance(item, h5py.Group):
                    print(f"  {prefix}{key}/")
                    print_all_scalars(item, prefix + "  ")

        print("\nScalar/small values:")
        print_all_scalars(f)


if __name__ == "__main__":
    # Find all RF dataset files
    rf_files = sorted(PICMUS_ROOT.rglob("*dataset_rf.hdf5"))
    scan_files = sorted(PICMUS_ROOT.rglob("*scan.hdf5"))

    print(f"Found {len(rf_files)} RF dataset files")
    print(f"Found {len(scan_files)} scan files")

    # Explore first scan file for probe metadata
    if scan_files:
        explore_scan(scan_files[0])

    # Explore all RF files
    for rf_file in rf_files:
        explore_rf_data(rf_file)
