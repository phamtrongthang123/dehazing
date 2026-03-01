"""B-mode conversion utilities for PICMUS RF data (L11-4v probe, 128 elements).

Same pipeline as bmode.py but configured for the PICMUS probe geometry.
"""

import os
os.environ["KERAS_BACKEND"] = "numpy"
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")

import numpy as np

import zea
from zea import init_device
from zea.probes import Probe
from zea.scan import Scan
from zea.beamform.delays import compute_t0_delays_planewave

init_device(verbose=False)

# L11-4v probe parameters (from PICMUS metadata)
n_el = 128
pitch = 0.3e-3  # 0.3 mm element pitch
aperture = (n_el - 1) * pitch  # 38.1 mm
probe_geometry = np.stack(
    [np.linspace(-aperture / 2, aperture / 2, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
)
probe = Probe(
    probe_geometry=probe_geometry,
    center_frequency=5.208e6,
    sampling_frequency=20.832e6,
)

# Scan parameters — 3 plane-wave angles matching conversion script
# (consecutive angles from the 75-angle PICMUS set around 0 degrees)
n_tx = 3
angles = np.array([-0.4324324, 0.0, 0.4324324]) * np.pi / 180  # ~ ±0.43 deg
sound_speed = 1540.0
xlims = (-20e-3, 20e-3)
zlims = (5e-3, 40e-3)
wavelength = sound_speed / probe.center_frequency
grid_size_x = int((xlims[1] - xlims[0]) / (0.5 * wavelength)) + 1
grid_size_z = int((zlims[1] - zlims[0]) / (0.5 * wavelength)) + 1

t0_delays = compute_t0_delays_planewave(probe_geometry, angles, sound_speed)
tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

scan = Scan(
    n_tx=n_tx, n_el=n_el,
    center_frequency=probe.center_frequency, sampling_frequency=probe.sampling_frequency,
    probe_geometry=probe_geometry, t0_delays=t0_delays, tx_apodizations=tx_apodizations,
    element_width=pitch,
    focus_distances=np.ones(n_tx) * np.inf, polar_angles=angles,
    initial_times=np.zeros(n_tx), n_ax=1024,
    xlims=xlims, zlims=zlims, grid_size_x=grid_size_x, grid_size_z=grid_size_z,
    lens_sound_speed=1000, lens_thickness=1e-3, n_ch=1,
    selected_transmits="all", sound_speed=sound_speed,
    apply_lens_correction=False, attenuation_coef=0.0,
)

# Extent for imshow (in mm)
extent_mm = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]

# Lazily initialized pipeline and parameters
_pipeline = None
_parameters = None


def _get_pipeline(dynamic_range=(-50, 0)):
    """Lazily initialize and cache the ZEA pipeline and parameters."""
    global _pipeline, _parameters
    if _pipeline is None:
        _pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
        _parameters = _pipeline.prepare_parameters(probe, scan, dynamic_range=dynamic_range)
    return _pipeline, _parameters


def _single_rf_to_bmode(rf_frame, dynamic_range=(-50, 0)):
    """Convert a single RF frame (n_tx, n_ax, n_el) to B-mode image.

    Args:
        rf_frame: RF data with shape (n_tx, n_ax, n_el).
        dynamic_range: Tuple of (min_db, max_db) for log compression.

    Returns:
        B-mode image as uint8 array.
    """
    import torch
    pipeline, parameters = _get_pipeline(dynamic_range)
    if isinstance(rf_frame, torch.Tensor):
        rf_frame = rf_frame.detach().cpu().numpy()
    # Add channel dim: (n_tx, n_ax, n_el) -> (n_tx, n_ax, n_el, 1)
    rf_data = rf_frame[:, :, :, np.newaxis]
    inputs = {pipeline.key: rf_data}
    outputs = pipeline(**inputs, **parameters)
    image = outputs[pipeline.output_key]
    image = zea.display.to_8bit(image, dynamic_range=dynamic_range)
    return np.asarray(image)


def undo_normalization(data, image_range=(-1, 1), mu=255, data_min=None, data_max=None):
    """Undo ZeaDataset normalization: image_range -> mu-law expand -> min-max restore."""
    lo, hi = image_range
    data = (data - lo) / (hi - lo) * 2.0 - 1.0
    data = np.sign(data) * ((1 + mu) ** np.abs(data) - 1.0) / mu
    if data_min is not None and data_max is not None:
        data = (data + 1.0) / 2.0 * (data_max - data_min) + data_min
    return data


def rf_to_bmode(rf_data, dynamic_range=(-50, 0)):
    """Convert RF data batch to B-mode images.

    Args:
        rf_data: RF data with shape (B, C, H, W) where C=n_tx, H=n_ax, W=n_el.
        dynamic_range: Tuple of (min_db, max_db) for log compression.

    Returns:
        List of B-mode images (each a 2D uint8 array).
    """
    if rf_data.ndim == 4:
        frames = rf_data
    elif rf_data.ndim == 3:
        frames = rf_data[np.newaxis]
    else:
        raise ValueError(f"Expected 3D or 4D RF data, got shape {rf_data.shape}")

    bmode_images = []
    for i in range(len(frames)):
        frame = frames[i]
        bmode = _single_rf_to_bmode(frame, dynamic_range=dynamic_range)
        bmode_images.append(bmode)
    return bmode_images
