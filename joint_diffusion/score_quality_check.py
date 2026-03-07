"""Compute score model quality at a specific fixed t value (default t=0.01).

This gives the benchmark metric reported in SCORES.md:
  "val loss at t=0.01"

Usage:
  python score_quality_check.py --run_dir wandb/run--ez1mqfie/files
  python score_quality_check.py --run_dir wandb/run--xnhklvch/files --t 0.01
"""
import argparse
import os
import sys
from pathlib import Path

import torch

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, str(Path(__file__).parent))

from datasets import get_dataset
from generators.models import get_model
from utils.checkpoints import ModelCheckpoint
from utils.runs import init_config
from utils.utils import set_random_seed


def score_loss_at_t(model, batch, t_val, device):
    """Compute score loss at a specific fixed t value."""
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        B = batch.shape[0]
        t = torch.full((B,), t_val, device=device)
        z = torch.randn_like(batch)
        mean, std = model.sde.marginal_prob(batch, t)
        while std.dim() < batch.dim():
            std = std.unsqueeze(-1)
        perturbed = mean + std * z
        score = model.get_score(perturbed, t)
        losses = torch.square(score * std + z)
        # Use reduce_mean (same as training)
        losses = torch.mean(losses.reshape(B, -1), dim=-1)
        return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to wandb run files dir")
    parser.add_argument("--data_root", type=str, default="/scrfs/storage/tp030/home/f2f_ldm/data")
    parser.add_argument("--t", type=float, default=0.01, help="Fixed t value for evaluation")
    parser.add_argument("--n_batches", type=int, default=20, help="Number of val batches to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file (default: latest)")
    args = parser.parse_args()

    set_random_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config and model
    run_dir = Path(args.run_dir)
    print(f"Loading config from: {run_dir}")
    config = init_config(str(run_dir))
    config["data_root"] = args.data_root

    # Load val dataset
    _, val_loader = get_dataset(config)
    print(f"Val batches: {len(val_loader)}")

    # Build model
    model = get_model(config, plot_summary=False, training=False)
    model = model.to(device)

    # Load checkpoint
    ckpt_manager = ModelCheckpoint(model, config)
    checkpoint = ckpt_manager.restore(args.checkpoint, load_optimizer=False, load_ema=False)
    epoch = checkpoint.get("epoch", -1)
    print(f"Loaded checkpoint: epoch {epoch + 1}")

    # Evaluate
    print(f"\nComputing score loss at t={args.t} over {args.n_batches} val batches...")
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= args.n_batches:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        loss = score_loss_at_t(model, batch, args.t, device)
        total_loss += loss
        n += 1
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{args.n_batches}: running mean = {total_loss / n:.4f}")

    mean_loss = total_loss / max(n, 1)
    print(f"\n=== Score loss at t={args.t}: {mean_loss:.4f} ===")
    print(f"    Run: {run_dir}")
    print(f"    Epoch: {epoch + 1}")
    print(f"    Reference: PICMUS v2 (ep480) = 28.6, ZEA tissue (ep350) = 8.2")
    print(f"    Target: < 20")


if __name__ == "__main__":
    main()
