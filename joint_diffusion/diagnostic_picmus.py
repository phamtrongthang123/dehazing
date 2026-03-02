"""Quick diagnostic: is the tissue model or the guidance the problem?

Tests:
1. lambda=0, kappa=0: pure tissue score model (no guidance)
2. lambda=0.1, kappa=0, no haze model: single-model guidance
3. lambda=0.1, kappa=0.1, gamma=0.5: ZEA-like params on PICMUS
"""
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from datasets import get_dataset
from utils.inverse import SGMDenoiser
from utils.runs import init_config
from utils.utils import load_config_from_yaml, set_random_seed

RESULTS_DIR = Path("sweep_results/picmus/diagnostic")
INF_CONFIG_PATH = Path("configs/inference/paper/picmus_dehaze_pigdm.yaml")

os.environ["WANDB_MODE"] = "disabled"
set_random_seed(1234)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_config(overrides, image_shape=None):
    inf_cfg = load_config_from_yaml(INF_CONFIG_PATH)
    for k, v in overrides.items():
        if k in ("lambda_coeff", "kappa_coeff", "ccdf"):
            inf_cfg.sgm[k] = v
        elif k == "corruptor_run_id":
            inf_cfg.sgm[k] = v
        else:
            inf_cfg[k] = v
    inf_cfg["num_img"] = inf_cfg.get("num_img", 2)
    inf_cfg["keep_track"] = False
    inf_cfg["show_noise_priors"] = False
    run_ids = inf_cfg.run_id
    cfg = init_config(run_ids["sgm"], inf_cfg)
    if image_shape is not None:
        cfg.image_shape = image_shape
    return cfg, run_ids


# Load dataset once
cfg0, _ = build_config({"noise_stddev": 0.05, "lambda_coeff": 0.1, "kappa_coeff": 0.1, "ccdf": 0.8})
_, dataset = get_dataset(cfg0)
image_shape = list(cfg0.image_shape)


tests = [
    # Test 1: Pure tissue model, no guidance
    ("no_guidance", {"noise_stddev": 0.05, "lambda_coeff": 0.0, "kappa_coeff": 0.0, "ccdf": 0.8}),
    # Test 2: Single-model guidance (no haze model)
    ("single_model_lam0.1", {"noise_stddev": 0.05, "lambda_coeff": 0.1, "kappa_coeff": 0.0,
                              "ccdf": 0.8, "corruptor_run_id": None}),
    # Test 3: ZEA-like params (gamma=0.5)
    ("zea_params", {"noise_stddev": 0.5, "lambda_coeff": 0.1, "kappa_coeff": 0.1, "ccdf": 0.8}),
    # Test 4: ZEA params with lower gamma (0.2)
    ("gamma0.2_zea", {"noise_stddev": 0.2, "lambda_coeff": 0.1, "kappa_coeff": 0.1, "ccdf": 0.8}),
]

for label, overrides in tests:
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"  Params: {overrides}")
    print(f"{'='*60}")

    cfg, _ = build_config(overrides, image_shape=image_shape)

    t0 = time.time()
    denoiser = SGMDenoiser(config=cfg, dataset=dataset)
    denoised = denoiser(plot=False, save=False)
    elapsed = time.time() - t0

    fig_path = RESULTS_DIR / f"{label}.png"
    denoiser.plot(save=str(fig_path))
    plt.close("all")
    print(f"  Time: {elapsed:.1f}s, saved to {fig_path}")
