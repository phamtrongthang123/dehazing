"""Generate unconditional samples from trained score models.

Uses the reverse SDE to generate samples from pure noise, showing what
the tissue and haze priors have learned. Also loads real data for comparison.

Usage:
    cd joint_diffusion
    conda run -n dehazing python generate_samples.py
    conda run -n dehazing python generate_samples.py --dataset zea
    conda run -n dehazing python generate_samples.py --num_samples 8
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from generators.models import get_model
from generators.SGM.sampling import ScoreSampler
from utils.checkpoints import ModelCheckpoint
from utils.runs import init_config


# ── config ───────────────────────────────────────────────────────────────
CONFIGS = {
    "picmus": {
        "tissue_run": "wandb/run--xnhklvch/files",
        "haze_run": "wandb/run--w1lwubg5/files",
        "data_root": "/scrfs/storage/tp030/home/f2f_ldm/data",
        "tissue_data": "picmus/tissue",
        "haze_data": "picmus/haze",
        "bmode_module": "picmus",
    },
    "zea": {
        "tissue_run": "wandb/run--jm8ljbms/files",
        "haze_run": "wandb/run--1bj42atz/files",
        "data_root": "/scrfs/storage/tp030/home/f2f_ldm/data",
        "tissue_data": "zea_synth/tissue",
        "haze_data": "zea_synth/haze",
        "bmode_module": "zea",
    },
}


def load_model_and_sampler(run_dir, use_ema=False):
    """Load a trained score model and build an unconditional sampler."""
    config = init_config(run_id=run_dir, verbose=False)

    # image_shape from config
    if "image_shape" not in config and "image_size" in config:
        channels = 3  # RF data always has 3 channels (transmits)
        config.image_shape = [channels, *config.image_size]

    model = get_model(config, training=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load checkpoint
    ckpt = ModelCheckpoint(model, config=config)
    checkpoint = ckpt.restore(config.get("checkpoint_file"))

    # Optionally load EMA weights
    if use_ema:
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state and isinstance(ema_state.get("shadow"), dict):
            model_sd = model.state_dict()
            updated = 0
            for k, v in ema_state["shadow"].items():
                if k in model_sd:
                    model_sd[k] = v
                    updated += 1
            if updated > 0:
                model.load_state_dict(model_sd)
                print(f"  Loaded EMA weights ({updated} params)")

    # Build sampler (unconditional — no guidance, no corruptor)
    sampler = ScoreSampler(
        model=model,
        sde=model.sde,
        image_shape=config.image_shape,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
        guidance=None,
        corruptor=None,
        keep_track=False,
    )

    return model, sampler, config, device


def load_real_data(data_root, subdir, num_samples):
    """Load real data for comparison."""
    path = Path(data_root) / subdir / "val.npz"
    if not path.exists():
        path = Path(data_root) / subdir / "train.npz"
    data = np.load(path)["rf"]
    # Normalize to [-1, 1]
    dmin, dmax = data.min(), data.max()
    data = 2 * (data - dmin) / (dmax - dmin) - 1
    return torch.from_numpy(data[:num_samples].astype(np.float32))


def generate_unconditional(sampler, num_samples, image_shape, device):
    """Generate unconditional samples from the prior."""
    shape = [num_samples, *image_shape]
    with torch.no_grad():
        samples = sampler(y=None, z=None, shape=shape, progress_bar=True)
    # Clamp to [-1, 1]
    samples = torch.clamp(samples, -1, 1)
    return samples


def to_bmode(rf_data, bmode_module="picmus", dynamic_range=(-50, 0)):
    """Convert RF data to B-mode images."""
    if bmode_module == "picmus":
        from utils.bmode_picmus import rf_to_bmode
    else:
        from utils.bmode import rf_to_bmode
    return rf_to_bmode(rf_data, dynamic_range=dynamic_range)


def plot_samples(real_tissue, gen_tissue, real_haze, gen_haze,
                 bmode_module="picmus", save_path=None):
    """Plot real vs generated samples for both tissue and haze, in B-mode."""
    n = min(4, len(gen_tissue))
    dynamic_range = (-50, 0)

    if bmode_module == "picmus":
        from utils.bmode_picmus import extent_mm
    else:
        extent_mm = None

    fig, axes = plt.subplots(n, 4, figsize=(14, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Real Tissue", "Generated Tissue", "Real Haze", "Generated Haze"]

    for row in range(n):
        samples = [real_tissue, gen_tissue, real_haze, gen_haze]
        for col, data in enumerate(samples):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            rf = data[row:row+1]  # (1, C, H, W)
            bmode = to_bmode(rf, bmode_module=bmode_module, dynamic_range=dynamic_range)
            img = np.asarray(bmode[0])
            if extent_mm is not None:
                axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=255, extent=extent_mm)
                if row == n - 1:
                    axes[row, col].set_xlabel("X (mm)")
                if col == 0:
                    axes[row, col].set_ylabel("Z (mm)")
            else:
                axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=255)
                axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11)

    fig.suptitle("Score Model Priors: Real Data vs Unconditional Samples", fontsize=13)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate unconditional samples from score models")
    parser.add_argument("--dataset", choices=["picmus", "zea"], default="picmus")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--tissue_run", type=str, default=None,
                        help="Override tissue model run dir")
    parser.add_argument("--haze_run", type=str, default=None,
                        help="Override haze model run dir")
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    if args.tissue_run:
        cfg["tissue_run"] = args.tissue_run
    if args.haze_run:
        cfg["haze_run"] = args.haze_run

    # ── Load tissue model ────────────────────────────────────────────
    print(f"\n=== Loading {args.dataset.upper()} tissue model ===")
    tissue_model, tissue_sampler, tissue_cfg, device = load_model_and_sampler(
        cfg["tissue_run"], use_ema=args.use_ema,
    )
    print(f"  image_shape: {tissue_cfg.image_shape}")

    # ── Load haze model ──────────────────────────────────────────────
    print(f"\n=== Loading {args.dataset.upper()} haze model ===")
    haze_model, haze_sampler, haze_cfg, _ = load_model_and_sampler(
        cfg["haze_run"], use_ema=args.use_ema,
    )
    print(f"  image_shape: {haze_cfg.image_shape}")

    # ── Load real data ───────────────────────────────────────────────
    print(f"\n=== Loading real data ===")
    real_tissue = load_real_data(cfg["data_root"], cfg["tissue_data"], args.num_samples)
    real_haze = load_real_data(cfg["data_root"], cfg["haze_data"], args.num_samples)
    print(f"  tissue: {real_tissue.shape}, haze: {real_haze.shape}")

    # ── Generate unconditional samples ───────────────────────────────
    print(f"\n=== Generating {args.num_samples} tissue samples ===")
    gen_tissue = generate_unconditional(
        tissue_sampler, args.num_samples, tissue_cfg.image_shape, device
    )
    print(f"  range: [{gen_tissue.min():.3f}, {gen_tissue.max():.3f}]")

    print(f"\n=== Generating {args.num_samples} haze samples ===")
    gen_haze = generate_unconditional(
        haze_sampler, args.num_samples, haze_cfg.image_shape, device
    )
    print(f"  range: [{gen_haze.min():.3f}, {gen_haze.max():.3f}]")

    # ── Plot ─────────────────────────────────────────────────────────
    save_path = args.save_path or f"figures/unconditional_samples_{args.dataset}.png"
    plot_samples(
        real_tissue, gen_tissue, real_haze, gen_haze,
        bmode_module=cfg["bmode_module"],
        save_path=save_path,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
