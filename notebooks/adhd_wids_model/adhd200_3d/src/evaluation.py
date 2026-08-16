"""
evaluation.py
-------------
Metrics, threshold optimization, ensembling, calibration. The original
notebook's logic here (OOF-only threshold locking, test-blind ensemble
weighting) was methodologically SOUND -- kept faithfully, just decoupled
from notebook globals so it's reusable/testable.

Hard rule enforced by function signatures, not just convention: any function
that touches the test set (`evaluate_on_test`) takes an explicit,
already-locked `threshold: float` argument -- it CANNOT compute its own
threshold from the data it's given. If you want to sweep a threshold, use
`sweep_threshold()` on validation/OOF data only.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                              brier_score_loss, confusion_matrix, f1_score,
                              matthews_corrcoef, roc_auc_score)


def eval_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, tag: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = dict(
        tag=tag, threshold=float(threshold),
        accuracy=(tp + tn) / max(tp + tn + fp + fn, 1),
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        npv=tn / (tn + fn) if (tn + fn) else 0.0,
        f1=f1_score(y_true, y_pred, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    if len(set(y_true)) > 1:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
        result["pr_auc"] = average_precision_score(y_true, y_prob)
        result["brier"] = brier_score_loss(y_true, y_prob)
    else:
        result["roc_auc"] = result["pr_auc"] = result["brier"] = float("nan")
    return result


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "youden_j",
                     min_specificity: float = 0.0, label: str = "") -> Tuple[float, dict]:
    """Find the best threshold on VALIDATION/OOF data ONLY. Caller is
    responsible for never passing test-set y_true/y_prob here -- see
    module docstring for the enforced boundary at `evaluate_on_test`."""
    thresholds = np.linspace(0.01, 0.99, 197)
    best_t, best_score, best_metrics = 0.5, -np.inf, None

    for t in thresholds:
        m = eval_at(y_true, y_prob, t, tag=label)
        if objective == "f1":
            score = m["f1"]
        elif objective == "youden_j":
            score = m["recall"] + m["specificity"] - 1
        elif objective == "recall_at_specificity":
            score = m["recall"] if m["specificity"] >= min_specificity else -np.inf
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if score > best_score:
            best_score, best_t, best_metrics = score, t, m

    return float(best_t), best_metrics


def evaluate_on_test(y_true_test: np.ndarray, y_prob_test: np.ndarray,
                      locked_threshold: float, tag: str = "test") -> dict:
    """The ONLY function in this module allowed to touch test labels. Takes
    a threshold that must already have been locked from validation/OOF."""
    return eval_at(y_true_test, y_prob_test, locked_threshold, tag=tag)


def soft_vote_ensemble(oof_probs: Dict[str, np.ndarray]) -> np.ndarray:
    arrs = list(oof_probs.values())
    return np.mean(arrs, axis=0)


def optimize_ensemble_weights(oof_probs: Dict[str, np.ndarray], y_true: np.ndarray,
                               objective: str = "youden_j") -> Dict[str, float]:
    """Weights optimized on OOF predictions ONLY -- never on the test set."""
    from scipy.optimize import minimize

    names = list(oof_probs.keys())
    stacked = np.stack([oof_probs[n] for n in names], axis=0)  # (M, N)

    def neg_score(w):
        w = np.abs(w)
        w = w / (w.sum() + 1e-12)
        p = (w[:, None] * stacked).sum(axis=0)
        _, m = sweep_threshold(y_true, p, objective=objective)
        return -(m["recall"] + m["specificity"] - 1) if objective == "youden_j" else -m["f1"]

    x0 = np.ones(len(names)) / len(names)
    res = minimize(neg_score, x0, method="Nelder-Mead")
    w = np.abs(res.x)
    w = w / w.sum()
    return {n: float(wi) for n, wi in zip(names, w)}


def calibrate_platt(oof_probs: np.ndarray, oof_y: np.ndarray):
    """Fit Platt scaling (logistic regression on 1D score) using OOF data only.
    Returns a fitted sklearn estimator; apply with `.predict_proba(x)[:,1]`."""
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1.0)
    platt.fit(oof_probs.reshape(-1, 1), oof_y)
    return platt
