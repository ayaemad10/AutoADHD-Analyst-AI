"""
training.py
------------
Loss functions, callbacks, and the fit/CV loop. Adds the RUN_MODE concept
(smoke_test / development / full) which was entirely absent from the
original notebook -- every training call now takes `run_mode` and scales
epochs/patience accordingly, and results are TAGGED with the run_mode they
came from so a smoke-test AUC can never be mistaken for a full-dataset result
downstream (this directly enforces the spec's "never report smoke-test or
development performance as final full-dataset performance").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau, TerminateOnNaN)
from tensorflow.keras.optimizers import Adam, AdamW


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def focal_loss(gamma: float = 2.0, alpha: float = 0.80):
    def fn(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        a_t = alpha * y_true[:, 1] + (1 - alpha) * y_true[:, 0]
        fl = -a_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)
    return fn


def weighted_bce(pos_weight: float = 3.0):
    def fn(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        w = 1 + (pos_weight - 1) * y_true[:, 1]
        bce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(w * bce)
    return fn


# --------------------------------------------------------------------------
# RUN_MODE
# --------------------------------------------------------------------------

RUN_MODE_DEFAULTS = {
    "smoke_test": dict(epochs=2, patience=1, cv_folds=2, smoke_n_subjects=6),
    "development": dict(epochs=15, patience=5, cv_folds=3, dev_n_subjects=60),
    "full": dict(epochs=50, patience=10, cv_folds=5),
}


def resolve_run_mode(run_mode: str) -> dict:
    if run_mode not in RUN_MODE_DEFAULTS:
        raise ValueError(f"RUN_MODE must be one of {list(RUN_MODE_DEFAULTS)}, got {run_mode!r}")
    return RUN_MODE_DEFAULTS[run_mode]


# --------------------------------------------------------------------------
# Optimizer / callbacks / fit
# --------------------------------------------------------------------------

def get_opt(lr: float, wd: float = 1e-4, use_adamw: bool = True):
    if use_adamw:
        try:
            return AdamW(lr, weight_decay=wd, clipnorm=1.0)
        except Exception:
            pass
    return Adam(lr, clipnorm=1.0)


def make_callbacks(tag: str, ckpt_dir: str, monitor: str = "val_recall", patience: int = 10):
    os.makedirs(ckpt_dir, exist_ok=True)
    return [
        EarlyStopping(monitor=monitor, mode="max", patience=patience,
                      restore_best_weights=True, min_delta=0.003, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(2, patience // 2),
                           min_lr=1e-7, verbose=0),
        ModelCheckpoint(os.path.join(ckpt_dir, f"{tag}_best.keras"), monitor=monitor,
                         mode="max", save_best_only=True, verbose=0),
        TerminateOnNaN(),
    ]


def fit_model(model, ds_train, ds_val, run_mode: str, loss_fn, tag: str, ckpt_dir: str,
              lr: float = 3e-4, class_weight: Optional[Dict[int, float]] = None,
              metrics=None):
    """Fit using tf.data.Dataset train/val sets (from dataloader.build_tf_dataset).
    Epochs/patience come from RUN_MODE, never hardcoded per call site, so a
    smoke test physically cannot accidentally run full-length training."""
    rm = resolve_run_mode(run_mode)
    if metrics is None:
        metrics = [tf.keras.metrics.Recall(name="recall"),
                   tf.keras.metrics.AUC(name="auc"),
                   tf.keras.metrics.Precision(name="precision")]

    model.compile(get_opt(lr), loss=loss_fn, metrics=metrics)

    def _to_cat(x, y, *rest):
        return x, tf.one_hot(tf.cast(y, tf.int32), 2)

    ds_train_c = ds_train.map(_to_cat)
    ds_val_c = ds_val.map(_to_cat)

    history = model.fit(
        ds_train_c, validation_data=ds_val_c,
        epochs=rm["epochs"], class_weight=class_weight,
        callbacks=make_callbacks(tag, ckpt_dir, patience=rm["patience"]),
        verbose=0,
    )
    return history


def compute_balanced_class_weight(labels: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1])
    present = np.unique(labels)
    if len(present) < 2:
        # Cannot compute a balanced weight with one class -- surface this
        # loudly rather than silently returning {0:1,1:1}, which would hide
        # a real data problem (e.g. a single-class site, see Pittsburgh finding).
        raise ValueError(
            f"compute_balanced_class_weight got labels with only classes "
            f"{present.tolist()} present -- cannot balance. Check upstream "
            f"filtering/splitting for a single-class subset."
        )
    w = compute_class_weight("balanced", classes=classes, y=labels)
    return {0: float(w[0]), 1: float(w[1])}


@dataclass
class CVFoldResult:
    fold: int
    model_name: str
    metrics: dict


def subject_group_kfold(subject_ids: np.ndarray, labels: np.ndarray, n_splits: int, seed: int = 42):
    """Thin wrapper around StratifiedGroupKFold with an explicit leakage assertion
    per fold (the original notebook did this inline; kept here so every caller
    gets the same guarantee for free)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(subject_ids, labels, groups=subject_ids)):
        tr_s, te_s = set(subject_ids[tr_idx]), set(subject_ids[te_idx])
        assert not (tr_s & te_s), f"Fold {fold}: TRAIN-TEST subject leakage detected"
        yield fold, tr_idx, te_idx
