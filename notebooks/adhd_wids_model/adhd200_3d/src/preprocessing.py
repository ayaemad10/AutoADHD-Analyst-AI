"""
preprocessing.py
-----------------
Single-volume MRI preprocessing. Kept close to the original notebook's
transforms (they were reasonable) but:
  - every function raises a typed `PrepError` with a specific reason instead
    of crashing the whole pipeline on one bad file
  - `vol_to_3d` / `vol_to_slices` / `roi_features` are now pure functions with
    explicit args (no reliance on notebook-global CFG at import time), so they
    are testable and reusable from src/dataloader.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class PrepError(Exception):
    """Raised when a single subject's volume cannot be preprocessed.
    Callers should catch this per-subject and record it, never let it kill
    a whole batch/epoch."""
    pass


def load_vol(path: str) -> np.ndarray:
    import nibabel as nib
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise PrepError(f"Not 3D after squeeze: shape={vol.shape}")
    if not np.isfinite(vol).all():
        raise PrepError("Volume contains NaN/Inf")
    return vol


def clip_zscore(vol: np.ndarray, perc: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    nz = vol[vol > 0]
    if nz.size == 0:
        raise PrepError("Empty brain mask (no positive voxels)")
    lo, hi = np.percentile(nz, perc)
    vol = np.clip(vol, lo, hi)
    m, s = vol.mean(), vol.std()
    if s < 1e-6:
        raise PrepError("Zero variance after clipping")
    return (vol - m) / s


def minmax01(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-6:
        raise PrepError("Zero dynamic range")
    return (vol - vmin) / (vmax - vmin)


def resize_vol(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Resize a (X,Y,Z) volume to (D,H,W) target using per-axis bilinear resize.
    Returns array in (D,H,W) order (matches the original notebook's convention).

    BUG FIX vs. the original notebook: the original allocated
    `out_xy = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2]))` -- i.e. the
    SOURCE volume's native X/Y size -- and then tried to write a resized
    (H, W) slice into it. That only works by coincidence if the source volume
    already happens to have X==H and Y==W. Verified by smoke test: this
    crashes with a broadcast ValueError on any volume whose native resolution
    differs from `target`, which is the normal case for real MRI. Fixed here
    by allocating the intermediate array at the TARGET (H, W) size.
    """
    import tensorflow as tf
    D, H, W = target
    out_xy = np.zeros((H, W, vol.shape[2]), dtype=np.float32)
    for z in range(vol.shape[2]):
        sl = vol[:, :, z][np.newaxis, :, :, np.newaxis]
        out_xy[:, :, z] = tf.image.resize(sl, [H, W], method="bilinear").numpy()[0, :, :, 0]
    out = np.zeros((H, W, D), dtype=np.float32)
    for x in range(H):
        row = out_xy[x, :, :][np.newaxis, :, :, np.newaxis]
        out[x, :, :] = tf.image.resize(row, [W, D], method="bilinear").numpy()[0, :, :, 0]
    return np.transpose(out, (2, 0, 1))  # (D,H,W)


def vol_to_3d(path: str, target: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    """-> (D,H,W,1) float32 in [0,1]."""
    vol = load_vol(path)
    vol = clip_zscore(vol)
    vol = minmax01(vol)
    vol = resize_vol(vol, target)
    return vol[..., np.newaxis].astype(np.float32)


def vol_to_slices(path: str, n_slices: int = 32,
                   target: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
    """-> (S,H,W,1) float32, or None if the volume has fewer than n_slices
    non-empty slices (caller must handle None, this is NOT an error to raise
    and drop the whole subject silently -- log it)."""
    import tensorflow as tf
    vol = load_vol(path)
    vol = clip_zscore(vol)
    vol = minmax01(vol)
    d = vol.shape[2]
    nonempty = [z for z in range(d) if vol[:, :, z].mean() > 0.005]
    if len(nonempty) < n_slices:
        return None
    ci, half = len(nonempty) // 2, n_slices // 2
    w = nonempty[max(0, ci - half):max(0, ci - half) + n_slices]
    if len(w) < n_slices:
        w = nonempty[-n_slices:]
    H, W = target
    out = [tf.image.resize(vol[:, :, z][..., np.newaxis], [H, W]).numpy() for z in w]
    return np.stack(out, 0).astype(np.float32)


def roi_features(path: str) -> np.ndarray:
    """Compact volumetric/intensity summary statistics for the classical-ML
    baseline. Not a substitute for real ROI/atlas-based morphometry -- see
    README note on modality limits."""
    vol = load_vol(path)
    vol = clip_zscore(vol)
    vol = minmax01(vol)
    nz = vol[vol > 0.05]
    d = vol.shape[2]
    nonempty = [z for z in range(d) if vol[:, :, z].mean() > 0.005]

    feats = [nz.mean(), nz.std(), nz.min(), nz.max()]
    feats += np.percentile(nz, [5, 25, 50, 75, 95]).tolist()
    feats.append(float(len(nonempty) / d))

    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        z = int(d * pct / 100)
        sl = vol[:, :, z]
        sl_nz = sl[sl > 0.05]
        feats.append(float(sl_nz.mean()) if sl_nz.size else 0.0)
        feats.append(float(sl_nz.std()) if sl_nz.size else 0.0)

    return np.array(feats, dtype=np.float32)
