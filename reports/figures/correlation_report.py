from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_with_target(
    df: pd.DataFrame, feature_columns: list[str], target: pd.Series, top_n: int = 30
) -> pd.DataFrame:
    """
    Vectorized Pearson correlation of every feature column against a single
    (numeric or binary 0/1) target. Deliberately avoids `df.corr()` on the
    full frame — for the WiDS connectome (~19,901 columns) that would
    attempt to build an ~19,901 x 19,901 matrix (~1.6 billion cells), which
    is both slow and mostly meaningless (we only care about correlation
    *with the target*, not every feature against every other feature).
    """
    X = df[feature_columns].to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)

    # Mask rows where target is NaN
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    y_centered = y - y.mean()
    y_std = y_centered.std()

    col_means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_means, X)
    X_centered = X_filled - col_means

    numerator = X_centered.T @ y_centered
    denom = np.sqrt((X_centered ** 2).sum(axis=0)) * (y_std * np.sqrt(len(y)))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom != 0, numerator / denom, np.nan)

    result = pd.DataFrame({"feature": feature_columns, "corr_with_target": corr})
    result["abs_corr"] = result["corr_with_target"].abs()
    return result.sort_values("abs_corr", ascending=False).head(top_n).drop(columns="abs_corr")


def pairwise_correlation(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Full correlation matrix — only safe to call on a SMALL feature set
    (e.g. quantitative metadata), not on the connectome."""
    if len(feature_columns) > 200:
        raise ValueError(
            f"pairwise_correlation called with {len(feature_columns)} columns — "
            "this is meant for small feature sets only (<=200). Use "
            "correlation_with_target for high-dimensional data like the connectome."
        )
    return df[feature_columns].corr(numeric_only=True)


def top_correlated_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Flags highly-correlated feature PAIRS (multicollinearity candidates
    for dropping one of the pair)."""
    pairs = []
    cols = corr_matrix.columns
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 3)})
    return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "correlation"]
    )
