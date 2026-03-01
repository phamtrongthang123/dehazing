"""Generate synthetic haze RF data for 128-element probe.

Creates bandpass-filtered random noise matching the spectral characteristics
of ultrasound RF data (L11-4v probe: fc=5.2 MHz, fs=20.832 MHz, 128 elements).

Output: /scrfs/storage/tp030/home/f2f_ldm/data/picmus/haze/{train,val}.npz
Shape: (N, 3, 1024, 128) with key 'rf'
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


OUTPUT_ROOT = Path("/scrfs/storage/tp030/home/f2f_ldm/data/picmus")

# L11-4v probe parameters
FC = 5.2e6       # Center frequency (Hz)
FS = 20.832e6    # Sampling frequency (Hz)
BW_FRAC = 0.65   # Fractional bandwidth (typical for L11-4v)

N_TX = 3         # Number of transmit angles
N_AX = 1024      # Axial samples
N_EL = 128       # Number of elements


def generate_haze_frame(rng, n_tx=N_TX, n_ax=N_AX, n_el=N_EL):
    """Generate a single haze frame as bandpass-filtered noise.

    The noise is filtered to match the transducer bandwidth, creating
    realistic-looking acoustic clutter/haze patterns.

    Args:
        rng: numpy RandomState
        n_tx: number of transmit angles
        n_ax: number of axial samples
        n_el: number of elements

    Returns:
        frame: (n_tx, n_ax, n_el) float32 array
    """
    # Bandpass filter design matching transducer bandwidth
    f_low = FC * (1 - BW_FRAC / 2)
    f_high = FC * (1 + BW_FRAC / 2)
    # Clamp to Nyquist
    f_nyq = FS / 2
    f_low = max(f_low, 100e3)  # avoid DC
    f_high = min(f_high, f_nyq * 0.95)

    sos = butter(4, [f_low / f_nyq, f_high / f_nyq], btype="band", output="sos")

    frame = np.zeros((n_tx, n_ax, n_el), dtype=np.float32)
    for tx in range(n_tx):
        # Generate white noise
        noise = rng.randn(n_ax, n_el).astype(np.float64)
        # Apply bandpass filter along axial dimension (per element)
        filtered = sosfiltfilt(sos, noise, axis=0).astype(np.float32)
        frame[tx] = filtered

    return frame


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic haze for 128-element probe")
    parser.add_argument("--n-train", type=int, default=400,
                        help="Number of training haze frames (default: 400)")
    parser.add_argument("--n-val", type=int, default=50,
                        help="Number of validation haze frames (default: 50)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    for split, n_frames in [("train", args.n_train), ("val", args.n_val)]:
        print(f"Generating {n_frames} {split} haze frames...")
        frames = []
        for i in range(n_frames):
            frame = generate_haze_frame(rng)
            frames.append(frame)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_frames}")

        data = np.stack(frames)  # (N, 3, 1024, 128)
        print(f"  Shape: {data.shape}")
        print(f"  Stats: min={data.min():.6f}, max={data.max():.6f}, "
              f"mean={data.mean():.6f}, std={data.std():.6f}")

        out_dir = OUTPUT_ROOT / "haze"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"{split}.npz", rf=data)
        print(f"  Saved to {out_dir / f'{split}.npz'}")


if __name__ == "__main__":
    main()
