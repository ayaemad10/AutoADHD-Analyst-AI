"""
dataset_discovery.py
---------------------
Matches discovered NIfTI files to phenotypic labels and produces the
`usable` subject-level table, plus subject/site-aware splitting.

Fixes vs. the original notebook:
  - filename filter for the preprocessed volume ("normalized_resampled_128")
    is now CONFIGURABLE, not hardcoded -- because it was never verified
    against a real download of the Kaggle dataset in this sandbox. If the
    real filenames differ, `NIFTI_FILENAME_FILTER` in config.yaml is the one
    place to change it (previously this string was buried inline).
  - `_ok_nifti` is defined before use (the original notebook referenced it
    via `"_ok_nifti" in dir()` before its own definition, which silently made
    the `ok` variable meaningless dead code).
  - every filtering step now records a "why was this subject/file excluded"
    reason instead of just a boolean, so nothing is dropped silently.
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .phenotype_loader import _norm_id, load_phenotypic


def discover_nifti(root: str, filename_filter: Optional[str] = None) -> pd.DataFrame:
    """Walk `root` for subject NIfTI files without loading their contents.

    Expected layout: root/<site>/sub-<id>/<file>.nii[.gz]
    `filename_filter`: substring that a preprocessed filename must contain
    (e.g. "normalized_resampled_128"). Pass None to accept all .nii/.nii.gz
    files -- recommended for the FIRST real run against the Kaggle dataset,
    since the exact preprocessed filename convention has not been verified
    from this sandbox.
    """
    rows = []
    for site_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(site_dir):
            continue
        site = os.path.basename(site_dir)
        candidates = (glob.glob(os.path.join(site_dir, "sub-*", "*.nii")) +
                      glob.glob(os.path.join(site_dir, "sub-*", "*.nii.gz")))
        for p in candidates:
            fname = os.path.basename(p)
            if filename_filter and filename_filter not in fname:
                continue
            rows.append(dict(
                site=site,
                subject_id=_norm_id(os.path.basename(os.path.dirname(p))),
                filepath=p,
                filename=fname,
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        warnings.warn(
            f"[dataset_discovery] No NIfTI files matched under {root} "
            f"(filter={filename_filter!r}). If this is the first real run, "
            f"re-run with filename_filter=None and inspect actual filenames "
            f"before assuming a naming convention.",
            stacklevel=2,
        )
    return df


def _ok_nifti(path: str) -> bool:
    """Cheap header-only readability check -- does not load full voxel data."""
    try:
        import nibabel as nib
        img = nib.load(path)
        shape = img.shape
        return len(shape) >= 3 and all(s > 1 for s in shape[:3])
    except Exception:
        return False


def build_usable_table(pheno_df: pd.DataFrame, nii_df: pd.DataFrame,
                        check_readable: bool = True) -> tuple[pd.DataFrame, dict]:
    """Merge phenotype labels onto discovered NIfTI files and flag usability.

    Returns (usable_df, exclusion_report) -- exclusion_report explains exactly
    how many rows were dropped and why, at each stage.
    """
    report = {"n_nifti_files": len(nii_df), "n_phenotype_subjects": len(pheno_df)}

    if nii_df.empty:
        report["stop_reason"] = "no NIfTI files discovered"
        return nii_df, report

    meta = nii_df.merge(pheno_df, on=["subject_id", "site"], how="left")
    meta["has_label"] = meta["label"].notna()
    report["n_unmatched_to_phenotype"] = int((~meta["has_label"]).sum())

    if check_readable:
        meta["is_readable"] = meta["filepath"].apply(_ok_nifti)
    else:
        meta["is_readable"] = True
    report["n_unreadable"] = int((~meta["is_readable"]).sum())

    meta["is_dup"] = meta.get("is_dup", False)
    report["n_duplicate_flagged"] = int(meta["is_dup"].fillna(False).sum())

    meta["usable"] = meta["has_label"] & meta["is_readable"] & (~meta["is_dup"].fillna(False))
    usable = meta[meta["usable"]].copy()
    if not usable.empty:
        usable["label"] = usable["label"].astype(int)

    subj = usable.drop_duplicates("subject_id")
    report["n_usable_files"] = len(usable)
    report["n_usable_subjects"] = len(subj)
    report["n_control"] = int((subj["label"] == 0).sum()) if len(subj) else 0
    report["n_adhd"] = int((subj["label"] == 1).sum()) if len(subj) else 0
    report["n_sites"] = int(usable["site"].nunique()) if len(usable) else 0

    return meta, report


def site_bias_report(subjects: pd.DataFrame, bias_threshold_pct: float = 20.0) -> pd.DataFrame:
    ss = (subjects.groupby(["site", "label"]).size().unstack(fill_value=0)
          .rename(columns={0: "ctrl", 1: "adhd"}))
    ss["total"] = ss.sum(axis=1)
    ss["adhd_pct"] = np.where(ss["total"] > 0, ss.get("adhd", 0) / ss["total"] * 100, np.nan)
    mean_pct = ss["adhd_pct"].mean()
    ss["bias_flag"] = (ss["adhd_pct"] - mean_pct).abs() > bias_threshold_pct
    ss["single_class_site"] = (ss.get("ctrl", 0) == 0) | (ss.get("adhd", 0) == 0)
    return ss


def subject_level_split(usable: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Subject-level, stratified, leakage-verified split. Raises AssertionError
    (not a silent pass) if any subject appears in more than one split."""
    from sklearn.model_selection import train_test_split

    subjects = usable[["subject_id", "label", "site"]].drop_duplicates("subject_id")

    train_s, test_s = train_test_split(
        subjects, test_size=test_size, stratify=subjects["label"], random_state=seed)
    train_s, val_s = train_test_split(
        train_s, test_size=val_size, stratify=train_s["label"], random_state=seed)

    train_ids, val_ids, test_ids = set(train_s.subject_id), set(val_s.subject_id), set(test_s.subject_id)

    assert not (train_ids & val_ids), "TRAIN-VAL LEAKAGE"
    assert not (train_ids & test_ids), "TRAIN-TEST LEAKAGE"
    assert not (val_ids & test_ids), "VAL-TEST LEAKAGE"

    return (
        usable[usable.subject_id.isin(train_ids)].copy(),
        usable[usable.subject_id.isin(val_ids)].copy(),
        usable[usable.subject_id.isin(test_ids)].copy(),
        {"train_n": len(train_ids), "val_n": len(val_ids), "test_n": len(test_ids), "status": "PASS"},
    )
