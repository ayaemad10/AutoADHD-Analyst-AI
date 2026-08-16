"""
inference.py
------------
`predict_subject(path_to_mri, model_dir)` -- portable inference that loads
everything it needs from `model_dir` (never a hardcoded training-machine
path). Expects the artifact layout written by training.py / the final
notebook:

    models/
        final_model.keras
        preprocessing_config.json   {"mode": "3d"|"slices", "vol_size":[...], ...}
        feature_config.json
        label_mapping.json          {"0": "control", "1": "adhd"}
        threshold.json              {"threshold": 0.43}
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def predict_subject(path_to_mri: str, model_dir: str) -> dict:
    """Run the full portable inference pipeline on a single subject's MRI file.

    Returns: {"prediction": str, "probability": float, "threshold": float,
              "model": str}
    Raises FileNotFoundError with a clear message if any required artifact
    is missing from model_dir -- never guesses a default silently.
    """
    import tensorflow as tf

    from .preprocessing import PrepError, roi_features, vol_to_3d, vol_to_slices

    required = ["final_model.keras", "preprocessing_config.json",
                "label_mapping.json", "threshold.json"]
    missing = [r for r in required if not os.path.exists(os.path.join(model_dir, r))]
    if missing:
        raise FileNotFoundError(
            f"model_dir={model_dir} is missing required artifact(s): {missing}. "
            f"predict_subject() will not guess defaults for a clinical-adjacent "
            f"prediction pipeline."
        )

    prep_cfg = _load_json(os.path.join(model_dir, "preprocessing_config.json"))
    label_map = _load_json(os.path.join(model_dir, "label_mapping.json"))
    threshold = _load_json(os.path.join(model_dir, "threshold.json"))["threshold"]
    model = tf.keras.models.load_model(os.path.join(model_dir, "final_model.keras"),
                                        compile=False)

    mode = prep_cfg["mode"]
    try:
        if mode == "3d":
            x = vol_to_3d(path_to_mri, target=tuple(prep_cfg["vol_size"]))
            x = x[np.newaxis, ...]
        elif mode == "slices":
            x = vol_to_slices(path_to_mri, n_slices=prep_cfg["num_slices"],
                               target=tuple(prep_cfg["image_size"]))
            if x is None:
                raise PrepError("Volume has too few non-empty slices for slice mode")
            x = x[np.newaxis, ...]
        else:
            raise ValueError(f"Unsupported preprocessing mode in config: {mode}")
    except PrepError as e:
        return {"prediction": None, "probability": None, "threshold": threshold,
                "model": model.name, "error": f"preprocessing failed: {e}"}

    prob = float(model.predict(x, verbose=0)[0, 1])
    pred_class = int(prob >= threshold)

    return {
        "prediction": label_map.get(str(pred_class), str(pred_class)),
        "probability": prob,
        "threshold": threshold,
        "model": model.name,
    }


def save_model_package(model, model_dir: str, mode: str, vol_size=None,
                        num_slices: Optional[int] = None, image_size=None,
                        label_mapping: dict = None, threshold: float = 0.5,
                        training_config: dict = None):
    """Write the full portable artifact set training.py should call after
    locking a final model + threshold."""
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "final_model.keras"))

    prep_cfg = {"mode": mode}
    if mode == "3d":
        prep_cfg["vol_size"] = list(vol_size)
    elif mode == "slices":
        prep_cfg["num_slices"] = num_slices
        prep_cfg["image_size"] = list(image_size)

    with open(os.path.join(model_dir, "preprocessing_config.json"), "w") as f:
        json.dump(prep_cfg, f, indent=2)
    with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping or {"0": "control", "1": "adhd"}, f, indent=2)
    with open(os.path.join(model_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    if training_config:
        with open(os.path.join(model_dir, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2)

    print(f"[inference] Saved portable model package to {model_dir}")
