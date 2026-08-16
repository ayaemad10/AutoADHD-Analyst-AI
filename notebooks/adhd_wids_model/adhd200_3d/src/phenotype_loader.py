"""
phenotype_loader.py
--------------------
Portable phenotypic loader for ADHD-200.

Fixes applied vs. the original ADHD200_MODEL_COMPETITION.ipynb (cell "2. Dataset Discovery"):

BUG 1 (silent data loss): the original id-column detector only recognized
    ("scandir id", "scandirid", "subject", "subid")
  Peking_1_TestRelease_phenotypic.csv uses a column literally named "ID", which
  does not match any of those strings (case-folded "id" != "scandirid" etc.),
  so that entire file (51 real subjects) was silently skipped -> `continue`
  with no warning. Verified against the actual uploaded CSV.

BUG 2 (silent feature loss): TABULAR_COLS included "Full4IQ", "VIQ", "PIQ",
  but every real ADHD-200 phenotypic file uses "Full4 IQ", "Verbal IQ",
  "Performance IQ" (with spaces). Because of this mismatch, IQ features were
  NEVER attached to any subject, even though the notebook's header claims an
  "MRI + Phenotypic Fusion" model uses them. Verified against real columns.

BUG 3 (data quality, not a code bug but must be reported, not silently
  dropped): Pittsburgh_phenotypic.csv contains DX==0 (control) for every row
  in the uploaded file -> zero ADHD-positive subjects at that site. The
  original site-bias check (cell "4. Site / Scanner Analysis") would catch
  this IF the subjects survive step 1-2, but bugs 1-2 could mask/compound it.
  This loader reports it explicitly instead of leaving it to be discovered
  downstream.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Real ADHD-200 DX coding (verified against uploaded CSVs):
#   0 = Typically Developing (control)
#   1 = ADHD-Combined
#   2 = ADHD-Hyperactive/Impulsive
#   3 = ADHD-Inattentive
# Binarized label: 0 -> control, {1,2,3} -> ADHD (matches original notebook's logic)

# Candidate ID column names actually observed across ADHD-200 phenotypic files,
# case-insensitive, whitespace-stripped. "id" is now included to catch the
# Peking_1_TestRelease schema.
ID_CANDIDATES = {"scandir id", "scandirid", "subject", "subid", "id"}

# Map of desired canonical feature name -> real column name variants seen in
# the actual files (verified, not guessed).
TABULAR_COL_MAP: Dict[str, List[str]] = {
    "age": ["Age"],
    "gender": ["Gender"],
    "handedness": ["Handedness"],
    "full4_iq": ["Full4 IQ", "Full4IQ"],
    "full2_iq": ["Full2 IQ", "Full2IQ"],
    "verbal_iq": ["Verbal IQ", "VIQ"],
    "performance_iq": ["Performance IQ", "PIQ"],
    "adhd_index": ["ADHD Index"],
    "inattentive": ["Inattentive"],
    "hyper_impulsive": ["Hyper/Impulsive"],
    "med_status": ["Med Status"],
}


def _norm_id(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "nan"):
        return None
    s = s.replace("sub-", "").replace("sub_", "")
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _find_id_col(columns: List[str]) -> Optional[str]:
    for c in columns:
        if c.strip().lower() in ID_CANDIDATES:
            return c
    return None


def _find_tabular_cols(columns: List[str]) -> Dict[str, str]:
    """Return {canonical_name: actual_column_name} for columns present in this file."""
    found = {}
    cols_stripped = {c.strip(): c for c in columns}
    for canon, variants in TABULAR_COL_MAP.items():
        for v in variants:
            if v in cols_stripped:
                found[canon] = cols_stripped[v]
                break
    return found


@dataclass
class PhenotypeLoadReport:
    files_found: List[str] = field(default_factory=list)
    files_loaded: List[str] = field(default_factory=list)
    files_skipped: Dict[str, str] = field(default_factory=dict)  # filename -> reason
    per_site_counts: Dict[str, dict] = field(default_factory=dict)
    total_subjects: int = 0
    duplicate_subjects: int = 0

    def to_dict(self):
        return {
            "files_found": self.files_found,
            "files_loaded": self.files_loaded,
            "files_skipped": self.files_skipped,
            "per_site_counts": self.per_site_counts,
            "total_subjects": self.total_subjects,
            "duplicate_subjects": self.duplicate_subjects,
        }


def load_phenotypic(root: str, verbose: bool = True) -> "tuple[pd.DataFrame, PhenotypeLoadReport]":
    """
    Load and normalize all *_phenotypic.csv files found directly under `root`.

    Returns (dataframe, report). The report explicitly lists any file that was
    skipped and WHY -- nothing is silently dropped.
    """
    report = PhenotypeLoadReport()
    frames = []

    csv_paths = sorted(glob.glob(os.path.join(root, "*_phenotypic.csv")))
    report.files_found = [os.path.basename(p) for p in csv_paths]

    for path in csv_paths:
        fname = os.path.basename(path)
        site = fname.replace("_TestRelease_phenotypic.csv", "").replace("_phenotypic.csv", "")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            report.files_skipped[fname] = f"read error: {e}"
            continue

        df.columns = [c.strip() for c in df.columns]

        id_col = _find_id_col(list(df.columns))
        if id_col is None:
            report.files_skipped[fname] = (
                f"no recognizable subject-ID column among {list(df.columns)[:5]}..."
            )
            continue
        if "DX" not in df.columns:
            report.files_skipped[fname] = "no 'DX' (diagnosis) column"
            continue

        tab_map = _find_tabular_cols(list(df.columns))
        keep_actual = [id_col, "DX"] + list(tab_map.values())
        sub = df[keep_actual].copy()
        sub.rename(columns={id_col: "subject_id_raw"}, inplace=True)
        sub.rename(columns={v: k for k, v in tab_map.items()}, inplace=True)

        sub["subject_id"] = sub["subject_id_raw"].apply(_norm_id)
        sub["site"] = site
        sub["source_file"] = fname
        sub["label_raw"] = pd.to_numeric(sub["DX"], errors="coerce")

        # Report per-site diagnostic composition immediately (no silent surprises later)
        dx_counts = sub["label_raw"].value_counts(dropna=False).to_dict()
        n_adhd = int(sub["label_raw"].isin([1, 2, 3]).sum())
        n_ctrl = int((sub["label_raw"] == 0).sum())
        report.per_site_counts[site] = {
            "n_subjects": len(sub),
            "n_control": n_ctrl,
            "n_adhd": n_adhd,
            "dx_value_counts": {str(k): int(v) for k, v in dx_counts.items()},
            "tabular_cols_matched": list(tab_map.keys()),
        }
        if n_adhd == 0 or n_ctrl == 0:
            warnings.warn(
                f"[phenotype_loader] Site '{site}' ({fname}) has only one class "
                f"present (control={n_ctrl}, adhd={n_adhd}). This site cannot "
                f"contribute to a stratified split on its own and will bias "
                f"pooled statistics.",
                stacklevel=2,
            )

        frames.append(sub)
        report.files_loaded.append(fname)

    if not frames:
        raise FileNotFoundError(f"No usable phenotypic CSVs found in {root}")

    pheno = pd.concat(frames, ignore_index=True, sort=False)
    pheno["is_dup"] = pheno.duplicated(subset=["subject_id", "site"], keep=False)
    report.duplicate_subjects = int(pheno["is_dup"].sum())
    pheno = pheno.drop_duplicates(subset=["subject_id", "site"], keep="first")
    report.total_subjects = len(pheno)

    pheno["label"] = np.where(
        pheno["label_raw"] == 0, 0,
        np.where(pheno["label_raw"].isin([1, 2, 3]), 1, np.nan),
    )

    if verbose:
        print(f"Loaded {len(report.files_loaded)}/{len(report.files_found)} phenotypic files "
              f"-> {report.total_subjects} unique subjects")
        if report.files_skipped:
            print("SKIPPED (not silent):")
            for f, reason in report.files_skipped.items():
                print(f"  - {f}: {reason}")

    return pheno, report


if __name__ == "__main__":
    import json

    ROOT = "/mnt/user-data/uploads"
    pheno, report = load_phenotypic(ROOT)

    print("\n=== Per-site composition ===")
    for site, stats in report.per_site_counts.items():
        print(f"{site:20s} n={stats['n_subjects']:4d}  ctrl={stats['n_control']:4d}  "
              f"adhd={stats['n_adhd']:4d}  tabular_matched={stats['tabular_cols_matched']}")

    print(f"\nTotal unique subjects (label-bearing): {report.total_subjects}")
    print(f"Duplicate (subject_id, site) rows found: {report.duplicate_subjects}")
    print(f"Missing label after DX parse: {int(pheno['label'].isna().sum())}")

    os.makedirs("/home/claude/reports", exist_ok=True)
    with open("/home/claude/reports/phenotype_load_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    pheno.to_csv("/home/claude/reports/pheno_merged.csv", index=False)
    print("\nSaved: reports/phenotype_load_report.json, reports/pheno_merged.csv")
