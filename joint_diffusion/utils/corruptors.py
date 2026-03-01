"""Corruptors — PyTorch version.
Author(s): Tristan Stevens
"""
import abc
from pathlib import Path

import numpy as np
import torch

_CORRUPTORS = {}


def register_corruptor(cls=None, *, name=None):
    """A decorator for registering corruptor classes."""

    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _CORRUPTORS:
            raise ValueError(f"Already registered corruptor with name: {local_name}")
        _CORRUPTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_corruptor(name):
    """Get corruptor class for a given name."""
    return _CORRUPTORS[name]


class Corruptor(abc.ABC):
    """Corruptor abstract class."""

    def __init__(self, config, dataset_name=None, task=None, model=None,
                 verbose=True, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.task = "denoising" if task is None else task
        self.name = config.corruptor
        self.model = model
        self.verbose = verbose
        self.batch_size = config.batch_size
        self.image_shape = config.image_shape
        self.A = None  # measurement matrix
        self.noise = None
        self.blend_factor = getattr(config, "blend_factor", 1.0)

    def corrupt(self, images):
        """Corrupt input images with noise."""
        raise NotImplementedError


@register_corruptor(name="gaussian")
class GaussianCorruptor(Corruptor):
    """Gaussian corruptor, adds gaussian noise."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.noise_stddev = config.noise_stddev

    def corrupt(self, images):
        noise = torch.randn_like(images) * self.noise_stddev
        noisy_images = images + noise
        self.noise = noise
        return noisy_images


@register_corruptor(name="cs")
class CSCorruptor(Corruptor):
    """Compressed sensing corruptor."""

    def __init__(self, config, **kwargs):
        super().__init__(config, task="compressive-sensing", **kwargs)
        self.noise_stddev = config.noise_stddev
        self.subsample_factor = config.subsample_factor
        self.image_shape = config.image_shape

        self.n = int(np.prod(self.image_shape))
        self.m = int(self.n * (1 / self.subsample_factor))
        self.A = self.get_sensing_matrix()

    def corrupt(self, images):
        noise = torch.randn_like(images) * self.noise_stddev
        noisy_images = images + noise
        noisy_flat = noisy_images.reshape(-1, self.n)
        A_T = torch.from_numpy(self.A.T).float().to(images.device)
        return noisy_flat @ A_T

    def get_sensing_matrix(self):
        A = np.random.normal(0, 1 / np.sqrt(self.m), size=(self.m, self.n))
        return A.astype(np.float32)


@register_corruptor(name="haze")
class HazeCorruptor(Corruptor):
    """Haze corruptor for ultrasound dehazing (additive model y = x + h)."""

    def __init__(self, config, **kwargs):
        super().__init__(config, task="dehazing", **kwargs)
        self.noise_stddev = getattr(config, "noise_stddev", 0.5)
        self.blend_factor = getattr(config, "blend_factor", 1.0)
        self._haze_data = None
        self._haze_idx = 0
        
        # Haze normalization stats (set after loading)
        self._haze_data_min = None
        self._haze_data_max = None

        # Try to load haze dataset for inference mode
        data_root = getattr(config, "data_root", None)
        haze_data_subdir = getattr(config, "haze_data_subdir", "zea_synth")
        if data_root:
            haze_path = Path(data_root) / haze_data_subdir / "haze" / "val.npz"
            if haze_path.exists():
                import numpy as np
                npz_key = getattr(config, "npz_key", "rf")
                data = np.load(haze_path)[npz_key].astype(np.float32)
                # Ensure channel-first format (N, C, H, W)
                if data.ndim == 3:
                    data = data[:, np.newaxis, :, :]

                # Apply ZeaDataset-style normalization so corrupt() sees
                # properly companded data (same pipeline as training).
                mu = 255
                # Step 1: min-max normalize to [-1, 1]
                self._haze_data_min = float(data.min())
                self._haze_data_max = float(data.max())
                if self._haze_data_max > self._haze_data_min:
                    data = 2.0 * (data - self._haze_data_min) / (self._haze_data_max - self._haze_data_min) - 1.0
                # Step 2: mu-law compress
                data = np.sign(data) * np.log1p(mu * np.abs(data)) / np.log1p(mu)
                # Step 3: rescale to image_range
                lo, hi = getattr(config, "image_range", [0, 1])
                data = (data + 1.0) / 2.0 * (hi - lo) + lo

                self._haze_data = data
                print(f"Loaded haze data from {haze_path}: shape {self._haze_data.shape}, "
                      f"normalized to [{data.min():.3f}, {data.max():.3f}]")

    @staticmethod
    def mu_law_compress(x, mu=255):
        """Mu-law companding compression: C(x) = sign(x) * log1p(mu * |x|) / log1p(mu)"""
        return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, dtype=x.dtype, device=x.device))

    @staticmethod
    def mu_law_expand(x, mu=255):
        """Mu-law companding expansion: C^{-1}(x) = sign(x) * ((1 + mu)^|x| - 1) / mu"""
        return torch.sign(x) * ((1 + mu) ** torch.abs(x) - 1) / mu

    def corrupt(self, tissue, haze=None):
        """Create hazy measurement: y = C(C^{-1}(tissue) + gamma * C^{-1}(haze)).

        Linear addition in RF domain, then recompand. This matches the
        CompandedProjection guidance forward model in guidance.py.

        Args:
            tissue: clean tissue RF data in companded domain (B, C, H, W)
            haze: haze RF data in companded domain (B, C, H, W). If None, samples from loaded haze dataset.

        Returns:
            y: hazy measurement in companded domain
        """
        import numpy as np

        # If haze not provided, sample from loaded haze dataset
        if haze is None:
            if self._haze_data is None:
                raise ValueError(
                    "HazeCorruptor requires haze data. Either provide haze argument "
                    "or ensure data_root config points to zea_synth with haze/val.npz"
                )
            # Get batch size
            batch_size = tissue.shape[0]

            # Sample haze (cycle through if needed)
            haze_indices = np.arange(self._haze_idx, self._haze_idx + batch_size) % len(self._haze_data)
            self._haze_idx = (self._haze_idx + batch_size) % len(self._haze_data)

            haze = self._haze_data[haze_indices]
            haze = torch.from_numpy(haze).to(tissue.device).float()

        # Additive haze model in RF domain: y_rf = tissue_rf + gamma * haze_rf
        # Then recompand: y = C(y_rf)
        gamma = self.noise_stddev
        mu = 255
        tissue_rf = self.mu_law_expand(tissue, mu)
        haze_rf = self.mu_law_expand(haze, mu)
        y = self.mu_law_compress(tissue_rf + gamma * haze_rf, mu)
        self.noise = haze
        return y
