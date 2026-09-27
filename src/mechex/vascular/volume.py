"""CTA volume preparation.

Ref: Sec. 3.8, input volumes resampled to isotropic 1.0 mm spacing, windowed to the
arterial contrast range and intensity-normalised per scan.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from mechex.settings import VascularEncoderConfig


def resample_isotropic(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    target_spacing_mm: float = 1.0,
) -> np.ndarray:
    """Trilinearly resample a volume to isotropic spacing.

    The dtype is preserved as float32 because the downstream window is defined on
    Hounsfield units and integer rounding would shift the arterial range.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, received {volume.ndim} dimensions")
    if target_spacing_mm <= 0:
        raise ValueError("target spacing must be positive")
    current = np.asarray(volume, dtype=np.float32)
    ratios = [spacing[axis] / target_spacing_mm for axis in range(3)]
    target_shape = tuple(max(1, round(current.shape[axis] * ratios[axis])) for axis in range(3))
    tensor = torch.as_tensor(current).unsqueeze(0).unsqueeze(0)
    resampled = functional.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resampled.squeeze(0).squeeze(0).numpy()


def window_to_contrast_range(
    volume: np.ndarray,
    low_hu: float = 100.0,
    high_hu: float = 700.0,
) -> np.ndarray:
    """Clip to the arterial contrast window and rescale to the unit interval."""
    if high_hu <= low_hu:
        raise ValueError("the contrast window must have a positive width")
    clipped = np.clip(np.asarray(volume, dtype=np.float32), low_hu, high_hu)
    return (clipped - low_hu) / (high_hu - low_hu)


def normalise_per_scan(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean unit-variance normalisation over one scan."""
    values = np.asarray(volume, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    return (values - mean) / max(std, eps)


def fit_shape(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Centre-crop or centre-pad a volume to a fixed grid."""
    if volume.ndim != 3:
        raise ValueError("expected a 3D volume")
    out = np.zeros(shape, dtype=np.float32)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for axis in range(3):
        extent = min(shape[axis], volume.shape[axis])
        source_start = (volume.shape[axis] - extent) // 2
        target_start = (shape[axis] - extent) // 2
        source_slices.append(slice(source_start, source_start + extent))
        target_slices.append(slice(target_start, target_start + extent))
    out[tuple(target_slices)] = volume[tuple(source_slices)]
    return out


def prepare_phase(
    volume: np.ndarray, config: VascularEncoderConfig, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Full preparation path for one phase."""
    resampled = resample_isotropic(volume, spacing, config.isotropic_spacing_mm)
    windowed = window_to_contrast_range(resampled, *config.arterial_window_hu)
    fitted = fit_shape(windowed, config.volume_shape)
    return normalise_per_scan(fitted)


def prepare_pair(
    arterial: np.ndarray,
    delayed: np.ndarray,
    config: VascularEncoderConfig,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Stack the arterial and delayed phases into a two-channel input."""
    arterial_prepared = prepare_phase(arterial, config, spacing)
    delayed_prepared = prepare_phase(delayed, config, spacing)
    return np.stack([arterial_prepared, delayed_prepared], axis=0)


def to_tensor(prepared: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Wrap a prepared pair as a batched tensor."""
    return torch.as_tensor(prepared, dtype=torch.float32, device=device).unsqueeze(0)
