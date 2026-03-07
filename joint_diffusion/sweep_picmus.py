"""PICMUS inference hyperparameter sweep.

Sweeps noise_stddev (gamma), lambda/kappa guidance strengths, and ccdf
to find optimal dehazing parameters for the PICMUS dataset.

Usage:
    python sweep_picmus.py --phase 1   # gamma sweep
    python sweep_picmus.py --phase 2 --best_gamma 0.1
    python sweep_picmus.py --phase 3 --best_gamma 0.1 --best_lambda 0.1 --best_kappa 0.1
    python sweep_picmus.py --phase 4   # asymmetric kappa sweep (kappa >> lambda)
"""
import argparse
import csv
import os
import time
from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from datasets import get_dataset
from utils.inverse import SGMDenoiser
from utils.runs import init_config
from utils.utils import load_config_from_yaml, set_random_seed, update_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("sweep_results/picmus")
INF_CONFIG_PATH = Path("configs/inference/paper/picmus_dehaze_pigdm.yaml")

# Module-level override — set by main() from --inf_config arg
_inf_config_path_override = None


def compute_metrics(target, denoised, image_range=(0, 1)):
    """Compute PSNR and SSIM between target and denoised arrays.

    Args:
        target: (B, C, H, W) numpy array
        denoised: (B, C, H, W) numpy array
        image_range: tuple of (min, max) used for data_range

    Returns:
        dict with mean psnr and ssim
    """
    lo, hi = image_range
    data_range = hi - lo

    psnr_vals = []
    ssim_vals = []
    for i in range(len(target)):
        # Move to (H, W, C) for skimage
        t = target[i].transpose(1, 2, 0)
        d = denoised[i].transpose(1, 2, 0)
        psnr_vals.append(skimage_psnr(t, d, data_range=data_range))
        ssim_vals.append(skimage_ssim(t, d, data_range=data_range, channel_axis=-1))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }


def to_np(x):
    """Convert tensor to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def build_config(overrides: dict, image_shape=None):
    """Load base PICMUS inference config and apply overrides.

    Args:
        overrides: dict of parameter overrides
        image_shape: pre-computed image_shape from dataset loading (avoids
            the need to call get_dataset on every config)

    Returns merged config ready for SGMDenoiser.
    """
    cfg_path = Path(_inf_config_path_override) if _inf_config_path_override else INF_CONFIG_PATH
    inf_cfg = load_config_from_yaml(cfg_path)

    # Apply overrides to top-level and sgm sub-dict
    for k, v in overrides.items():
        if k in ("lambda_coeff", "kappa_coeff", "ccdf"):
            inf_cfg.sgm[k] = v
        else:
            inf_cfg[k] = v

    # Speed settings for sweep
    inf_cfg["num_img"] = inf_cfg.get("num_img", 2)
    inf_cfg["keep_track"] = False
    inf_cfg["show_noise_priors"] = False

    # Merge with full training config (same as inference.py:denoise flow)
    run_ids = inf_cfg.run_id
    cfg = init_config(run_ids["sgm"], inf_cfg)

    # image_shape is normally set as side-effect of get_dataset();
    # apply it here so we don't need to reload data for every run.
    if image_shape is not None:
        cfg.image_shape = image_shape

    return cfg, run_ids


def load_dataset_once(overrides: dict):
    """Build config, load dataset, and capture image_shape.

    Returns (cfg, dataset, image_shape) where image_shape can be reused
    with build_config(..., image_shape=...) for subsequent runs.
    """
    cfg, _ = build_config(overrides)
    _, dataset = get_dataset(cfg)  # sets cfg.image_shape as side-effect
    return cfg, dataset, list(cfg.image_shape)


def run_single(cfg, dataset, run_label: str, save_fig: bool = True):
    """Run a single inference and return metrics dict.

    Args:
        cfg: merged config (edict)
        dataset: validation DataLoader
        run_label: string label for this run (used in filenames)
        save_fig: whether to save the comparison figure

    Returns:
        dict with psnr, ssim, and timing info
    """
    t0 = time.time()

    denoiser = SGMDenoiser(config=cfg, dataset=dataset)

    # Run denoising without auto-plot (we'll save our own)
    denoised = denoiser(plot=False, save=False)

    elapsed = time.time() - t0

    # Extract final denoised tensor
    d = denoised
    if isinstance(d, tuple):
        d = d[0]
    if isinstance(d, list):
        d = d[-1]

    target_np = to_np(denoiser.target_samples)
    denoised_np = to_np(d)

    metrics = compute_metrics(target_np, denoised_np, tuple(cfg.image_range))
    metrics["time_s"] = round(elapsed, 1)

    # Save comparison figure
    if save_fig:
        fig_path = RESULTS_DIR / f"{run_label}.png"
        denoiser.plot(save=str(fig_path))
        plt.close("all")

    return metrics


def append_csv(row: dict, csv_path: Path):
    """Append a row dict to a CSV file (creates header if needed)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Sweep phases
# ---------------------------------------------------------------------------


def phase1_gamma(args):
    """Sweep noise_stddev (gamma)."""
    gammas = [0.05, 0.1, 0.15, 0.2, 0.3]
    fixed = dict(lambda_coeff=0.1, kappa_coeff=0.1, ccdf=0.8)

    csv_path = RESULTS_DIR / "results.csv"

    # Load dataset once and capture image_shape
    cfg0, dataset, image_shape = load_dataset_once(
        {**fixed, "noise_stddev": gammas[0], "num_img": args.num_img})

    best_psnr = -float("inf")
    best_gamma = gammas[0]

    for gamma in gammas:
        label = f"phase1_gamma_{gamma}"
        print(f"\n{'='*60}")
        print(f"Phase 1: gamma={gamma}  (lambda={fixed['lambda_coeff']}, "
              f"kappa={fixed['kappa_coeff']}, ccdf={fixed['ccdf']})")
        print(f"{'='*60}")

        cfg, _ = build_config({**fixed, "noise_stddev": gamma, "num_img": args.num_img},
                              image_shape=image_shape)
        metrics = run_single(cfg, dataset, label)

        row = {"phase": 1, "gamma": gamma, **fixed, **metrics}
        append_csv(row, csv_path)
        print(f"  => PSNR={metrics['psnr']:.2f} dB,  SSIM={metrics['ssim']:.4f},  "
              f"time={metrics['time_s']}s")

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            best_gamma = gamma

    print(f"\nPhase 1 best: gamma={best_gamma}  (PSNR={best_psnr:.2f} dB)")
    return best_gamma


def phase2_guidance(args, best_gamma: float):
    """Sweep lambda_coeff x kappa_coeff."""
    lambdas = [0.01, 0.1, 0.5]
    kappas = [0.01, 0.1, 0.5]
    fixed = dict(noise_stddev=best_gamma, ccdf=0.8)

    csv_path = RESULTS_DIR / "results.csv"

    cfg0, dataset, image_shape = load_dataset_once(
        {**fixed, "lambda_coeff": lambdas[0], "kappa_coeff": kappas[0],
         "num_img": args.num_img})

    best_psnr = -float("inf")
    best_lam, best_kap = lambdas[0], kappas[0]

    for lam, kap in product(lambdas, kappas):
        label = f"phase2_lam_{lam}_kap_{kap}"
        print(f"\n{'='*60}")
        print(f"Phase 2: lambda={lam}, kappa={kap}  "
              f"(gamma={best_gamma}, ccdf={fixed['ccdf']})")
        print(f"{'='*60}")

        cfg, _ = build_config({**fixed, "lambda_coeff": lam, "kappa_coeff": kap,
                               "num_img": args.num_img}, image_shape=image_shape)
        metrics = run_single(cfg, dataset, label)

        row = {"phase": 2, "gamma": best_gamma, "lambda_coeff": lam,
               "kappa_coeff": kap, "ccdf": fixed["ccdf"], **metrics}
        append_csv(row, csv_path)
        print(f"  => PSNR={metrics['psnr']:.2f} dB,  SSIM={metrics['ssim']:.4f},  "
              f"time={metrics['time_s']}s")

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            best_lam, best_kap = lam, kap

    print(f"\nPhase 2 best: lambda={best_lam}, kappa={best_kap}  "
          f"(PSNR={best_psnr:.2f} dB)")
    return best_lam, best_kap


def phase3_ccdf(args, best_gamma: float, best_lam: float, best_kap: float):
    """Sweep ccdf (SDE start time)."""
    ccdfs = [0.7, 0.8, 0.85, 0.9, 0.95]
    fixed = dict(noise_stddev=best_gamma, lambda_coeff=best_lam, kappa_coeff=best_kap)

    csv_path = RESULTS_DIR / "results.csv"

    cfg0, dataset, image_shape = load_dataset_once(
        {**fixed, "ccdf": ccdfs[0], "num_img": args.num_img})

    best_psnr = -float("inf")
    best_ccdf = ccdfs[0]

    for ccdf in ccdfs:
        label = f"phase3_ccdf_{ccdf}"
        print(f"\n{'='*60}")
        print(f"Phase 3: ccdf={ccdf}  (gamma={best_gamma}, "
              f"lambda={best_lam}, kappa={best_kap})")
        print(f"{'='*60}")

        cfg, _ = build_config({**fixed, "ccdf": ccdf, "num_img": args.num_img},
                              image_shape=image_shape)
        metrics = run_single(cfg, dataset, label)

        row = {"phase": 3, "gamma": best_gamma, **fixed, "ccdf": ccdf, **metrics}
        append_csv(row, csv_path)
        print(f"  => PSNR={metrics['psnr']:.2f} dB,  SSIM={metrics['ssim']:.4f},  "
              f"time={metrics['time_s']}s")

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            best_ccdf = ccdf

    print(f"\nPhase 3 best: ccdf={best_ccdf}  (PSNR={best_psnr:.2f} dB)")
    return best_ccdf


def phase4_asymmetric_kappa(args):
    """Sweep kappa >> lambda to compensate for weak gamma gradient.

    With gamma=0.05, the gradient w.r.t. noise (h) includes a factor of gamma,
    making it ~10-20x weaker than the gradient w.r.t. tissue (x). The previous
    sweep used symmetric lambda=kappa which meant the noise model got almost
    no guidance signal. This phase tests asymmetric kappa >> lambda.

    Also tests gamma=0.1 with scaled kappa for comparison.
    """
    csv_path = RESULTS_DIR / "results.csv"

    # Configurations to sweep: (gamma, lambda, kappa, ccdf)
    configs = [
        # gamma=0.05: kappa needs ~10x lambda to match ZEA's x:n ratio
        (0.05, 0.05, 1.0, 0.8),
        (0.05, 0.05, 2.0, 0.8),
        (0.05, 0.05, 5.0, 0.8),
        (0.05, 0.05, 10.0, 0.8),
        (0.05, 0.1, 1.0, 0.8),
        (0.05, 0.1, 2.0, 0.8),
        (0.05, 0.1, 5.0, 0.8),
        (0.05, 0.1, 10.0, 0.8),
        # gamma=0.1: weaker asymmetry needed
        (0.1, 0.1, 0.5, 0.8),
        (0.1, 0.1, 1.0, 0.8),
        (0.1, 0.1, 2.0, 0.8),
        (0.1, 0.1, 5.0, 0.8),
    ]

    # Load dataset once
    g0, l0, k0, c0 = configs[0]
    cfg0, dataset, image_shape = load_dataset_once(
        {"noise_stddev": g0, "lambda_coeff": l0, "kappa_coeff": k0,
         "ccdf": c0, "num_img": args.num_img})

    best_psnr = -float("inf")
    best_cfg = configs[0]

    for gamma, lam, kap, ccdf in configs:
        label = f"phase4_g{gamma}_l{lam}_k{kap}"
        print(f"\n{'='*60}")
        print(f"Phase 4: gamma={gamma}, lambda={lam}, kappa={kap}, ccdf={ccdf}")
        print(f"  (effective x:n ratio ≈ {lam/(kap*gamma):.1f}:1)")
        print(f"{'='*60}")

        cfg, _ = build_config(
            {"noise_stddev": gamma, "lambda_coeff": lam, "kappa_coeff": kap,
             "ccdf": ccdf, "num_img": args.num_img},
            image_shape=image_shape)
        metrics = run_single(cfg, dataset, label)

        row = {"phase": 4, "gamma": gamma, "lambda_coeff": lam,
               "kappa_coeff": kap, "ccdf": ccdf, **metrics}
        append_csv(row, csv_path)
        print(f"  => PSNR={metrics['psnr']:.2f} dB,  SSIM={metrics['ssim']:.4f},  "
              f"time={metrics['time_s']}s")

        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            best_cfg = (gamma, lam, kap, ccdf)

    g, l, k, c = best_cfg
    print(f"\nPhase 4 best: gamma={g}, lambda={l}, kappa={k}, ccdf={c}  "
          f"(PSNR={best_psnr:.2f} dB)")
    return best_cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PICMUS inference hyperparameter sweep")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Sweep phase: 1=gamma, 2=lambda/kappa, 3=ccdf, 4=asymmetric kappa")
    parser.add_argument("--num_img", type=int, default=2,
                        help="Number of images per run (default 2 for speed)")
    parser.add_argument("--best_gamma", type=float, default=None,
                        help="Best gamma from Phase 1 (required for phases 2-3)")
    parser.add_argument("--best_lambda", type=float, default=None,
                        help="Best lambda from Phase 2 (required for phase 3)")
    parser.add_argument("--best_kappa", type=float, default=None,
                        help="Best kappa from Phase 2 (required for phase 3)")
    parser.add_argument("--inf_config", type=str, default=None,
                        help="Override inference config YAML path (default: configs/inference/paper/picmus_dehaze_pigdm.yaml)")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"
    set_random_seed(1234)

    global _inf_config_path_override
    if args.inf_config:
        _inf_config_path_override = args.inf_config

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == 1:
        phase1_gamma(args)
    elif args.phase == 2:
        if args.best_gamma is None:
            parser.error("--best_gamma required for phase 2")
        phase2_guidance(args, args.best_gamma)
    elif args.phase == 3:
        if args.best_gamma is None or args.best_lambda is None or args.best_kappa is None:
            parser.error("--best_gamma, --best_lambda, --best_kappa required for phase 3")
        phase3_ccdf(args, args.best_gamma, args.best_lambda, args.best_kappa)
    elif args.phase == 4:
        phase4_asymmetric_kappa(args)


if __name__ == "__main__":
    main()
