from __future__ import annotations

import pandas as pd


def class_imbalance_report(df: pd.DataFrame, target_columns: list[str]) -> dict:
    """
    For each target column, reports class counts, percentages, and the
    imbalance ratio (majority / minority). A ratio > ~1.5 is worth
    class-weighting; > 3 is severe and worth resampling strategy too.

    Also reports the *combined* distribution across all target columns
    together (e.g. ADHD_Outcome x Sex_F), since the WiDS competition metric
    specifically weights the female-ADHD-positive intersection — that
    subgroup's size is exactly what makes this problem hard.
    """
    report: dict = {"per_target": {}, "combined": None}

    for col in target_columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(dropna=False, normalize=True) * 100
        majority = counts.max()
        minority = counts.min()
        report["per_target"][col] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): round(float(v), 2) for k, v in pct.items()},
            "imbalance_ratio": round(float(majority / minority), 2) if minority > 0 else None,
        }

    present_targets = [c for c in target_columns if c in df.columns]
    if len(present_targets) >= 2:
        combined_counts = df.groupby(present_targets, dropna=False).size()
        report["combined"] = {
            str(k): int(v) for k, v in combined_counts.items()
        }

    return report
