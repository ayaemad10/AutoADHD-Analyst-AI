"""
models.py
---------
Deep model architectures (A-E) ported from the original notebook's cell
"9. Model Definitions", refactored into a registry so training.py and any
notebook can build a model by name without duplicating architecture code.

Kept intentionally close to the audited originals -- they were architecturally
reasonable for small-N 3D medical imaging (residual blocks, attention,
multi-scale, slice+attention). The main issue in the original was
duplication/coupling with notebook globals, not the architectures themselves.

IMPORTANT: none of these have been trained on real data from this sandbox
(no MRI access here). Shapes have been smoke-tested with random tensors only
to confirm the graphs build and a forward pass runs -- see
`tests/test_models_smoke.py`-style check at the bottom of this file.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras import layers as KL
from tensorflow.keras.regularizers import l2 as L2


def bn_relu(x):
    return KL.Activation("relu")(KL.BatchNormalization()(x))


def res_block3d(x, f, reg=1e-4):
    sc = x
    x = KL.Conv3D(f, 3, padding="same", kernel_regularizer=L2(reg))(x)
    x = bn_relu(x)
    x = KL.Conv3D(f, 3, padding="same", kernel_regularizer=L2(reg))(x)
    x = KL.BatchNormalization()(x)
    if sc.shape[-1] != f:
        sc = KL.Conv3D(f, 1, padding="same")(sc)
    x = KL.Add()([x, sc])
    return KL.Activation("relu")(x)


def channel_attention3d(x, reduction=8):
    ch = x.shape[-1]
    avg = KL.GlobalAveragePooling3D()(x)
    mx = KL.GlobalMaxPooling3D()(x)
    d1 = KL.Dense(max(ch // reduction, 4), activation="relu")
    d2 = KL.Dense(ch, activation="sigmoid")
    a = KL.Add()([d2(d1(avg)), d2(d1(mx))])
    a = KL.Reshape((1, 1, 1, ch))(a)
    return KL.Multiply()([x, a])


def spatial_attention3d(x):
    avg = KL.Lambda(lambda t: __import__("tensorflow").reduce_mean(t, axis=-1, keepdims=True))(x)
    mx = KL.Lambda(lambda t: __import__("tensorflow").reduce_max(t, axis=-1, keepdims=True))(x)
    cat = KL.Concatenate(axis=-1)([avg, mx])
    a = KL.Conv3D(1, 7, padding="same", activation="sigmoid")(cat)
    return KL.Multiply()([x, a])


def _head(x, name):
    x = KL.GlobalAveragePooling3D()(x)
    x = KL.Dense(64, activation="relu")(x)
    x = KL.Dropout(0.4)(x)
    out = KL.Dense(2, activation="softmax", dtype="float32", name=f"{name}_out")(x)
    return out


def build_model_A(vol_size: Tuple[int, int, int] = (64, 64, 64)) -> Model:
    """Strong 3D CNN (residual)."""
    inp = Input((*vol_size, 1), name="modelA_in")
    x = KL.Conv3D(16, 3, padding="same")(inp); x = bn_relu(x)
    x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 32); x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 64); x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 128)
    out = _head(x, "A")
    return Model(inp, out, name="ModelA_ResidualCNN3D")


def build_model_B(vol_size: Tuple[int, int, int] = (64, 64, 64)) -> Model:
    """3D ResNet-like, deeper stack."""
    inp = Input((*vol_size, 1), name="modelB_in")
    x = KL.Conv3D(16, 3, padding="same")(inp); x = bn_relu(x)
    x = KL.MaxPool3D(2)(x)
    for f in (32, 64, 128, 128):
        x = res_block3d(x, f)
        x = KL.MaxPool3D(2)(x) if f != 128 else x
    out = _head(x, "B")
    return Model(inp, out, name="ModelB_ResNet3D")


def build_model_C(vol_size: Tuple[int, int, int] = (64, 64, 64)) -> Model:
    """3D CNN + channel & spatial attention."""
    inp = Input((*vol_size, 1), name="modelC_in")
    x = KL.Conv3D(16, 3, padding="same")(inp); x = bn_relu(x)
    x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 32); x = channel_attention3d(x); x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 64); x = spatial_attention3d(x); x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 128)
    out = _head(x, "C")
    return Model(inp, out, name="ModelC_AttentionCNN3D")


def build_model_D(vol_size: Tuple[int, int, int] = (64, 64, 64)) -> Model:
    """Multi-scale 3D CNN: parallel branches at different receptive fields."""
    inp = Input((*vol_size, 1), name="modelD_in")
    b1 = KL.Conv3D(16, 3, padding="same")(inp); b1 = bn_relu(b1)
    b2 = KL.Conv3D(16, 5, padding="same")(inp); b2 = bn_relu(b2)
    b3 = KL.Conv3D(16, 7, padding="same")(inp); b3 = bn_relu(b3)
    x = KL.Concatenate()([b1, b2, b3])
    x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 64); x = KL.MaxPool3D(2)(x)
    x = res_block3d(x, 128)
    out = _head(x, "D")
    return Model(inp, out, name="ModelD_MultiScaleCNN3D")


def build_model_E(n_slices: int = 32, image_size: Tuple[int, int] = (128, 128)) -> Model:
    """Slice encoder (2D CNN per slice, TimeDistributed) + attention pooling
    across the slice axis."""
    import tensorflow as tf
    inp = Input((n_slices, *image_size, 1), name="modelE_in")
    td = KL.TimeDistributed(KL.Conv2D(16, 3, padding="same", activation="relu"))(inp)
    td = KL.TimeDistributed(KL.MaxPool2D(2))(td)
    td = KL.TimeDistributed(KL.Conv2D(32, 3, padding="same", activation="relu"))(td)
    td = KL.TimeDistributed(KL.GlobalAveragePooling2D())(td)  # (S, 32)

    attn_scores = KL.Dense(1)(td)                              # (S, 1)
    attn_weights = KL.Softmax(axis=1)(attn_scores)
    weighted = KL.Multiply()([td, attn_weights])
    pooled = KL.Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted)  # (32,)

    x = KL.Dense(64, activation="relu")(pooled)
    x = KL.Dropout(0.4)(x)
    out = KL.Dense(2, activation="softmax", dtype="float32", name="E_out")(x)
    return Model(inp, out, name="ModelE_SliceAttention")


MODEL_REGISTRY: Dict[str, Callable[..., Model]] = {
    "ModelA": build_model_A,
    "ModelB": build_model_B,
    "ModelC": build_model_C,
    "ModelD": build_model_D,
    "ModelE": build_model_E,
}

MODEL_MODE = {  # which dataloader mode each model consumes
    "ModelA": "3d", "ModelB": "3d", "ModelC": "3d", "ModelD": "3d", "ModelE": "slices",
}


def build_classical_pipeline(model_name: str = "random_forest", seed: int = 42):
    """Model F: classical ML on ROI features. Returns an sklearn-compatible
    (unfitted) estimator; caller fits a StandardScaler on train only, per the
    original notebook's leakage-safe convention."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8,
                                       class_weight="balanced", random_state=seed)
    if model_name == "svm":
        return SVC(probability=True, class_weight="balanced", random_state=seed)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, eval_metric="logloss",
                                  random_state=seed)
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, max_depth=4, class_weight="balanced",
                                   random_state=seed, verbosity=-1)
    raise ValueError(f"Unknown classical model_name: {model_name}")


if __name__ == "__main__":
    # CODE-MECHANICS SMOKE TEST ONLY: random tensors, not real MRI, purely to
    # confirm every architecture builds and does a forward pass without error.
    import numpy as np
    for name, builder in MODEL_REGISTRY.items():
        m = builder() if name != "ModelE" else builder(n_slices=8, image_size=(32, 32))
        if name == "ModelE":
            x = np.random.rand(2, 8, 32, 32, 1).astype("float32")
        else:
            m = builder(vol_size=(16, 16, 16))
            x = np.random.rand(2, 16, 16, 16, 1).astype("float32")
        y = m.predict(x, verbose=0)
        print(f"{name:8s} params={m.count_params():>9,d}  output_shape={y.shape}")
    print("MODEL SMOKE TEST PASSED (random tensors, code mechanics only, not a result)")
