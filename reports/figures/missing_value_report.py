from __future__ import annotations

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: missing count, missing %, dtype."""
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [round(float(df[c].isna().mean()) * 100, 2) for c in df.columns],
        }
    ).sort_values("missing_pct", ascending=False).reset_index(drop=True)
    return report


def flag_high_missing_columns(report: pd.DataFrame, threshold_pct: float = 40.0) -> list[str]:
    """Columns above threshold are candidates to drop rather than impute —
    imputing >40% of a column tends to manufacture signal that isn't there."""
    return report.loc[report["missing_pct"] > threshold_pct, "column"].tolist()
