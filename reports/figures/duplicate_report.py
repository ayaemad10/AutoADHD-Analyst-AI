from __future__ import annotations

import pandas as pd


def duplicate_report(df: pd.DataFrame, id_column: str | None = None) -> dict:
    """
    Reports two distinct kinds of duplication, which matter differently:
    - full-row duplicates (identical across every column)
    - duplicate IDs (same participant/subject appearing more than once,
      which is more dangerous — it can silently leak a subject into both
      train and test if not handled before splitting)
    """
    full_dupe_mask = df.duplicated(keep=False)
    result = {
        "n_rows": int(len(df)),
        "n_full_duplicate_rows": int(full_dupe_mask.sum()),
        "full_duplicate_row_indices": df.index[full_dupe_mask].tolist()[:50],
    }

    if id_column and id_column in df.columns:
        id_counts = df[id_column].value_counts()
        dup_ids = id_counts[id_counts > 1]
        result["id_column"] = id_column
        result["n_duplicate_ids"] = int(len(dup_ids))
        result["duplicate_ids"] = dup_ids.head(50).to_dict()

    return result
