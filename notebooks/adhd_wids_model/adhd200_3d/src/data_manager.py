"""
data_manager.py
----------------
Universal Data Source Manager for the ADHD-200 pipeline.

Priority chain (first that exists wins), per spec:
    1. Kaggle mounted dataset   (/kaggle/input/...)
    2. KaggleHub download       (kagglehub.dataset_download(...))
    3. Environment variable     (ADHD200_DATA_DIR)
    4. Local Windows path       (D:\\adhd200-preprocessed)

Nothing downstream should hardcode a path. Everything uses DATA_DIR returned
by `resolve_data_dir()`.

NOTE ON WHAT THIS FILE CANNOT DO IN THIS SANDBOX:
This module was written and code-reviewed in an environment with no network
access to kaggle.com and no access to the user's local D:\\ drive or MRI
files. It has NOT been executed end-to-end against the real Kaggle dataset.
It has been exercised against the fallback/no-data-found path and against
the real phenotypic CSVs only. Run it for real in Kaggle/Colab/local and
report back what `resolve_data_dir()` + `discover_files()` actually find.
"""

from __future__ import annotations

import glob
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

KAGGLE_DATASET_SLUG = "purnimakumarrr/adhd200-preprocessed-anatomical-dataset"
KAGGLE_MOUNT_CANDIDATES = [
    "/kaggle/input/adhd200-preprocessed-anatomical-dataset",
    "/kaggle/input/adhd200-preprocessed",
]
ENV_VAR_NAME = "ADHD200_DATA_DIR"
WINDOWS_FALLBACK = r"D:\adhd200-preprocessed"


@dataclass
class DataSourceResult:
    data_dir: Optional[str] = None
    source: str = "none"  # kaggle_mount | kagglehub_download | env_var | local_windows | not_found
    environment: str = field(default_factory=lambda: detect_environment())
    detail: str = ""

    def to_dict(self):
        d = dict(self.__dict__)
        return d


def detect_environment() -> str:
    if os.path.isdir("/kaggle/input") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.isdir("/content"):
        return "colab"
    if platform.system() == "Windows":
        return "windows_local"
    if platform.system() == "Linux":
        return "linux"
    return platform.system().lower() or "unknown"


def _try_kaggle_mount() -> Optional[str]:
    for c in KAGGLE_MOUNT_CANDIDATES:
        if os.path.isdir(c):
            return c
    return None


def _try_kagglehub_download() -> Optional[str]:
    """Only attempted if kagglehub is importable AND network to kaggle.com works.
    Returns None (never raises) if unavailable -- caller falls through the chain."""
    try:
        import kagglehub
    except ImportError:
        return None
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        if path and os.path.isdir(path):
            return path
    except Exception:
        # Network unavailable, auth missing, quota, etc. -- fall through silently,
        # the caller will report this in `detail`.
        return None
    return None


def _try_env_var() -> Optional[str]:
    p = os.environ.get(ENV_VAR_NAME)
    if p and os.path.isdir(p):
        return p
    return None


def _try_local_windows() -> Optional[str]:
    if os.path.isdir(WINDOWS_FALLBACK):
        return WINDOWS_FALLBACK
    return None


def resolve_data_dir(verbose: bool = True) -> DataSourceResult:
    """Walk the priority chain and return the first hit, with full provenance."""
    attempts = []

    p = _try_kaggle_mount()
    attempts.append(("kaggle_mount", p))
    if p:
        res = DataSourceResult(data_dir=p, source="kaggle_mount",
                                detail="Found dataset already mounted under /kaggle/input")
        if verbose:
            print(f"[data_manager] DATA_DIR = {p}  (source=kaggle_mount)")
        return res

    p = _try_kagglehub_download()
    attempts.append(("kagglehub_download", p))
    if p:
        res = DataSourceResult(data_dir=p, source="kagglehub_download",
                                detail=f"Downloaded via kagglehub: {KAGGLE_DATASET_SLUG}")
        if verbose:
            print(f"[data_manager] DATA_DIR = {p}  (source=kagglehub_download)")
        return res

    p = _try_env_var()
    attempts.append(("env_var", p))
    if p:
        res = DataSourceResult(data_dir=p, source="env_var",
                                detail=f"Found via ${ENV_VAR_NAME}")
        if verbose:
            print(f"[data_manager] DATA_DIR = {p}  (source=env_var)")
        return res

    p = _try_local_windows()
    attempts.append(("local_windows", p))
    if p:
        res = DataSourceResult(data_dir=p, source="local_windows",
                                detail=f"Found at hardcoded fallback {WINDOWS_FALLBACK}")
        if verbose:
            print(f"[data_manager] DATA_DIR = {p}  (source=local_windows)")
        return res

    detail = ("No data source found. Tried: " +
              ", ".join(f"{name}={'FAIL' if v is None else v}" for name, v in attempts) +
              f". Set ${ENV_VAR_NAME} to your dataset root, or run inside Kaggle/Colab "
              f"with the dataset attached, or ensure {WINDOWS_FALLBACK} exists.")
    if verbose:
        print(f"[data_manager] NOT FOUND. {detail}")
    return DataSourceResult(data_dir=None, source="not_found", detail=detail)


# ---------------------------------------------------------------------------
# Discovery: inventory whatever DATA_DIR actually contains.
# Deliberately does NOT open/read MRI volumes -- just stats the filesystem.
# ---------------------------------------------------------------------------

RELEVANT_EXT = {".csv", ".xlsx", ".nii", ".gz", ".json", ".pkl", ".joblib", ".keras", ".h5"}


def discover_files(data_dir: str, max_files_to_list: int = 50) -> dict:
    """Inventory DATA_DIR without loading any MRI content. Safe to call on a
    directory with hundreds of thousands of files -- uses os.walk with counters,
    never materializes a full file list into memory beyond `max_files_to_list`
    samples per extension."""
    if not data_dir or not os.path.isdir(data_dir):
        return {"error": f"DATA_DIR does not exist: {data_dir}"}

    counts = {}
    total_size_bytes = 0
    total_files = 0
    samples = {}
    site_dirs = []

    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if entry.is_dir():
            site_dirs.append(entry.name)

    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            total_files += 1
            ext = "".join(Path(fname).suffixes[-2:]) if fname.endswith(".nii.gz") else Path(fname).suffix
            ext = ext.lower()
            counts[ext] = counts.get(ext, 0) + 1
            try:
                total_size_bytes += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
            if ext in RELEVANT_EXT and len(samples.get(ext, [])) < max_files_to_list:
                samples.setdefault(ext, []).append(os.path.join(root, fname))

    summary = {
        "data_dir": data_dir,
        "environment": detect_environment(),
        "site_subdirectories": site_dirs,
        "total_files": total_files,
        "total_size_gb": round(total_size_bytes / (1024 ** 3), 3),
        "extension_counts": counts,
        "n_nifti_files": counts.get(".nii", 0) + counts.get(".nii.gz", 0),
        "n_phenotypic_csv": len(glob.glob(os.path.join(data_dir, "*_phenotypic.csv"))),
        "sample_paths_by_ext": samples,
    }
    return summary


def save_summary(summary: dict, reports_dir: str = "./reports"):
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[data_manager] Saved reports/dataset_summary.json")


if __name__ == "__main__":
    result = resolve_data_dir()
    print(json.dumps(result.to_dict(), indent=2))

    if result.data_dir:
        summary = discover_files(result.data_dir)
        print(json.dumps(summary, indent=2)[:2000])
        save_summary(summary)
    else:
        print("\n[data_manager] No MRI data source available in THIS environment "
              "(expected -- this sandbox has no Kaggle network access and no "
              "D:\\ drive). This code is ready to run as-is in Kaggle/Colab/your PC.")
