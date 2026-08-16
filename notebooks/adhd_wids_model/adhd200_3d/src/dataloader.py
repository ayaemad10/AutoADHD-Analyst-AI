"""
dataloader.py
-------------
Lazy / batched loading, per spec:

    MRI file paths -> subject-level split -> lazy loader -> preprocessing
    -> batch -> tf.data.Dataset -> prefetch -> GPU

Replaces the original notebook's `load_set()` (cell "6. Load All Data into
RAM"), which materialized every subject's full volume as a Python list
before training -- infeasible for the full ADHD-200 collection.

Design:
  - `tf.data.Dataset.from_generator` yields ONE subject at a time
  - preprocessing (preprocessing.vol_to_3d / vol_to_slices) runs lazily,
    inside the generator, not up front
  - failures are per-subject (logged + skipped), never crash the whole epoch
  - `.batch().prefetch(AUTOTUNE)` at the end; `.cache()` is opt-in and only
    recommended for `smoke_test` / `development` RUN_MODEs, never `full`
    (caching the full preprocessed dataset would defeat the memory-safety
    purpose of this module)
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocessing import PrepError, roi_features, vol_to_3d, vol_to_slices

logger = logging.getLogger("adhd200.dataloader")


def _subject_generator(files_df: pd.DataFrame, mode: str,
                        vol_target: Tuple[int, int, int],
                        slice_target: Tuple[int, int], n_slices: int
                        ) -> Iterator[Tuple[np.ndarray, int, str]]:
    """Yields (array, label, subject_id) one at a time. Never holds more than
    one subject's array in memory at once."""
    n_skipped = 0
    for _, row in files_df.iterrows():
        sid, lbl, path = row["subject_id"], int(row["label"]), row["filepath"]
        try:
            if mode == "3d":
                x = vol_to_3d(path, target=vol_target)
            elif mode == "slices":
                x = vol_to_slices(path, n_slices=n_slices, target=slice_target)
                if x is None:
                    n_skipped += 1
                    continue
            elif mode == "roi":
                x = roi_features(path)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except PrepError as e:
            n_skipped += 1
            logger.warning(f"[dataloader] skipping subject {sid} ({path}): {e}")
            continue
        yield x, lbl, sid
    if n_skipped:
        logger.info(f"[dataloader] {n_skipped} subjects skipped during this pass "
                     f"(mode={mode}) -- see warnings above for reasons.")


def build_tf_dataset(files_df: pd.DataFrame, mode: str = "3d", batch_size: int = 4,
                      vol_target: Tuple[int, int, int] = (64, 64, 64),
                      slice_target: Tuple[int, int] = (128, 128), n_slices: int = 32,
                      shuffle: bool = True, cache: bool = False,
                      shuffle_buffer: int = 64):
    """Build a lazy, batched, prefetching tf.data.Dataset over `files_df`.

    `cache=True` should only be used for RUN_MODE in {smoke_test, development}
    with a small subject count -- see module docstring.
    """
    import tensorflow as tf

    if mode == "3d":
        out_shape = (*vol_target, 1)
        out_sig = (tf.TensorSpec(shape=out_shape, dtype=tf.float32),
                   tf.TensorSpec(shape=(), dtype=tf.int32),
                   tf.TensorSpec(shape=(), dtype=tf.string))
    elif mode == "slices":
        out_shape = (n_slices, *slice_target, 1)
        out_sig = (tf.TensorSpec(shape=out_shape, dtype=tf.float32),
                   tf.TensorSpec(shape=(), dtype=tf.int32),
                   tf.TensorSpec(shape=(), dtype=tf.string))
    elif mode == "roi":
        out_sig = (tf.TensorSpec(shape=(None,), dtype=tf.float32),
                   tf.TensorSpec(shape=(), dtype=tf.int32),
                   tf.TensorSpec(shape=(), dtype=tf.string))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def gen():
        yield from _subject_generator(files_df, mode, vol_target, slice_target, n_slices)

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def apply_run_mode_limit(files_df: pd.DataFrame, run_mode: str,
                          smoke_n_subjects: int = 6, dev_n_subjects: int = 60,
                          seed: int = 42) -> pd.DataFrame:
    """Subject-level subsampling for smoke_test / development RUN_MODEs.
    `full` returns files_df unchanged. Sampling is stratified by label so a
    smoke test always has at least one subject of each class when possible."""
    if run_mode == "full":
        return files_df

    n = {"smoke_test": smoke_n_subjects, "development": dev_n_subjects}.get(run_mode)
    if n is None:
        raise ValueError(f"Unknown RUN_MODE: {run_mode}")

    subj = files_df.drop_duplicates("subject_id")
    if len(subj) <= n:
        return files_df

    per_class = max(1, n // 2)
    parts = []
    for lbl, grp in subj.groupby("label"):
        parts.append(grp.sample(n=min(per_class, len(grp)), random_state=seed))
    picked = pd.concat(parts)
    return files_df[files_df.subject_id.isin(set(picked.subject_id))].copy()
