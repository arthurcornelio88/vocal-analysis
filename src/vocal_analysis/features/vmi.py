"""Vocal Mechanism Index (VMI) - Continuous vocal mechanism index."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class VMILabel(str, Enum):
    """Categorical labels for VMI."""

    M1_HEAVY = "M1_HEAVY"
    M1_LIGHT = "M1_LIGHT"
    MIX_PASSAGGIO = "MIX_PASSAGGIO"
    M2_REINFORCED = "M2_REINFORCED"
    M2_LIGHT = "M2_LIGHT"


# VMI -> Label mapping
VMI_THRESHOLDS = {
    (0.0, 0.2): VMILabel.M1_HEAVY,
    (0.2, 0.4): VMILabel.M1_LIGHT,
    (0.4, 0.6): VMILabel.MIX_PASSAGGIO,
    (0.6, 0.8): VMILabel.M2_REINFORCED,
    (0.8, 1.0): VMILabel.M2_LIGHT,
}


@dataclass
class VMIWeights:
    """Weights for VMI computation."""

    alpha_ratio: float = 0.30
    cpps: float = 0.25
    h1_h2: float = 0.25
    spectral_tilt: float = 0.20

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha_ratio": self.alpha_ratio,
            "cpps": self.cpps,
            "h1_h2": self.h1_h2,
            "spectral_tilt": self.spectral_tilt,
        }


def normalize_features_global(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize features with global Z-score.

    Args:
        df: DataFrame with features.
        feature_cols: Columns to normalize.
        scaler: Pre-trained scaler (optional). If None, trains a new one.

    Returns:
        Tuple (normalized DataFrame, scaler used).
    """
    df_normalized = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
    else:
        df_normalized[feature_cols] = scaler.transform(df[feature_cols].fillna(0))

    return df_normalized, scaler


def compute_vmi_fixed(
    alpha_ratio: np.ndarray,
    cpps: np.ndarray,
    h1_h2: np.ndarray,
    spectral_tilt: np.ndarray,
    weights: VMIWeights | None = None,
) -> np.ndarray:
    """Compute VMI with fixed weights (baseline).

    Logic:
    - High Alpha Ratio -> M1 (low VMI)
    - High CPPS -> clean voice (does not directly distinguish M1/M2)
    - Low H1-H2 -> M1 (low VMI)
    - Negative Spectral Tilt (steep) -> M1 (low VMI)

    VMI is normalized to [0, 1] where:
    - 0.0 = Dense M1
    - 1.0 = Light M2

    Args:
        alpha_ratio: Alpha Ratio per frame (dB).
        cpps: CPPS per frame.
        h1_h2: H1-H2 per frame (dB).
        spectral_tilt: Spectral Tilt per frame.
        weights: Weights for each feature. Default: VMIWeights().

    Returns:
        Array with VMI per frame [0, 1].
    """
    if weights is None:
        weights = VMIWeights()

    # Ensure same size
    min_len = min(len(alpha_ratio), len(cpps), len(h1_h2), len(spectral_tilt))
    alpha_ratio = alpha_ratio[:min_len]
    cpps = cpps[:min_len]
    h1_h2 = h1_h2[:min_len]
    spectral_tilt = spectral_tilt[:min_len]

    # Replace NaN with median (robust to outliers)
    alpha_ratio = np.where(np.isnan(alpha_ratio), np.nanmedian(alpha_ratio), alpha_ratio)
    cpps = np.where(np.isnan(cpps), np.nanmedian(cpps), cpps)
    h1_h2 = np.where(np.isnan(h1_h2), np.nanmedian(h1_h2), h1_h2)
    spectral_tilt = np.where(np.isnan(spectral_tilt), np.nanmedian(spectral_tilt), spectral_tilt)

    # Normalize each feature to [0, 1] using min-max
    def minmax_norm(arr: np.ndarray) -> np.ndarray:
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        if arr_max - arr_min < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    alpha_norm = minmax_norm(alpha_ratio)
    cpps_norm = minmax_norm(cpps)
    h1_h2_norm = minmax_norm(h1_h2)
    tilt_norm = minmax_norm(spectral_tilt)

    # Invert features that correlate negatively with VMI
    # High Alpha Ratio -> M1 -> low VMI, so we invert
    alpha_contrib = (1 - alpha_norm) * weights.alpha_ratio

    # CPPS has no clear direction for VMI, use as is
    # (High CPPS = clean voice, present in both dense M1 AND reinforced M2)
    # Therefore, CPPS contributes less to M1/M2 discrimination
    cpps_contrib = cpps_norm * weights.cpps * 0.5 + 0.25 * weights.cpps  # Neutral contribution

    # High H1-H2 -> M2 -> high VMI
    h1_h2_contrib = h1_h2_norm * weights.h1_h2

    # Negative Spectral Tilt (steep) -> M1 -> low VMI
    # More positive/flat Tilt -> M2 -> high VMI
    tilt_contrib = tilt_norm * weights.spectral_tilt

    # Combine
    vmi = alpha_contrib + cpps_contrib + h1_h2_contrib + tilt_contrib

    # Renormalize to [0, 1]
    vmi = np.clip(vmi, 0, 1)

    # Final normalization to ensure full range
    vmi_min, vmi_max = np.nanmin(vmi), np.nanmax(vmi)
    if vmi_max - vmi_min > 1e-10:
        vmi = (vmi - vmi_min) / (vmi_max - vmi_min)

    return vmi


def train_vmi_regressor(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "vmi_target",
    test_size: float = 0.2,
    random_state: int = 42,
    use_temporal_smoothing: bool = True,
) -> tuple[xgb.XGBRegressor, StandardScaler, dict]:
    """Train XGBoost regressor to learn optimal VMI weights.

    Args:
        df: DataFrame with features and target (vmi_target: pseudo-labels or annotations).
        feature_cols: Spectral feature columns.
        target_col: Column with VMI target (0-1).
        test_size: Test set proportion.
        random_state: Seed for reproducibility.
        use_temporal_smoothing: If True, adds temporal regularization.

    Returns:
        Tuple (XGBoost model, scaler, metrics).
    """
    if feature_cols is None:
        feature_cols = ["alpha_ratio", "cpps_per_frame", "h1_h2", "spectral_tilt"]

    # Remove rows with NaN in target
    df_clean = df.dropna(subset=[target_col])

    # Global normalization
    df_normalized, scaler = normalize_features_global(df_clean, feature_cols)

    X = df_normalized[feature_cols]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        objective="reg:squarederror",
        random_state=random_state,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
    )

    model.fit(X_train, y_train)

    # Metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_rmse": np.sqrt(np.mean((y_train - y_pred_train) ** 2)),
        "test_rmse": np.sqrt(np.mean((y_test - y_pred_test) ** 2)),
        "train_mae": np.mean(np.abs(y_train - y_pred_train)),
        "test_mae": np.mean(np.abs(y_test - y_pred_test)),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_, strict=False)),
    }

    print("VMI Regressor Training Results:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
    print(f"  Test MAE:   {metrics['test_mae']:.4f}")
    print(f"  Feature Importance: {metrics['feature_importance']}")

    return model, scaler, metrics


def predict_vmi(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    scaler: StandardScaler,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Predict VMI using trained model.

    Args:
        df: DataFrame with features.
        model: Trained XGBoost model.
        scaler: Scaler used during training.
        feature_cols: Feature columns.

    Returns:
        Array with predicted VMI [0, 1].
    """
    if feature_cols is None:
        feature_cols = ["alpha_ratio", "cpps_per_frame", "h1_h2", "spectral_tilt"]

    df_normalized, _ = normalize_features_global(df, feature_cols, scaler)
    X = df_normalized[feature_cols]

    vmi = model.predict(X)

    # Ensure range [0, 1]
    vmi = np.clip(vmi, 0, 1)

    return vmi


def vmi_to_label(vmi: np.ndarray | float) -> np.ndarray | VMILabel:
    """Convert numeric VMI to categorical label.

    Args:
        vmi: VMI value(s) between 0 and 1.

    Returns:
        Categorical label(s).
    """
    if isinstance(vmi, int | float):
        for (low, high), label in VMI_THRESHOLDS.items():
            if low <= vmi < high:
                return label
        return VMILabel.M2_LIGHT  # Edge case: vmi == 1.0

    # Array
    labels = np.empty(len(vmi), dtype=object)
    for i, v in enumerate(vmi):
        for (low, high), label in VMI_THRESHOLDS.items():
            if low <= v < high:
                labels[i] = label.value
                break
        else:
            labels[i] = VMILabel.M2_LIGHT.value

    return labels


def create_pseudo_labels_from_gmm(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    f0_col: str = "f0",
) -> np.ndarray:
    """Create VMI pseudo-labels from existing GMM clusters.

    Strategy: assumes that the cluster with the highest mean F0 is M2,
    and converts to a continuous scale based on distance to the centroid.

    Args:
        df: DataFrame with clusters and f0.
        cluster_col: Column with cluster labels.
        f0_col: Column with F0 values.

    Returns:
        Array with VMI pseudo-labels [0, 1].
    """
    df = df.copy()

    # Identify which cluster is M1/M2 based on mean F0
    cluster_f0_mean = df.groupby(cluster_col)[f0_col].mean()
    m2_cluster = cluster_f0_mean.idxmax()
    m1_cluster = cluster_f0_mean.idxmin()

    # Compute normalized distance to each cluster centroid
    m1_centroid = df[df[cluster_col] == m1_cluster][f0_col].mean()
    m2_centroid = df[df[cluster_col] == m2_cluster][f0_col].mean()

    # VMI = relative distance between centroids
    # f0 close to m1_centroid -> low VMI
    # f0 close to m2_centroid -> high VMI
    f0_values = df[f0_col].values

    if m2_centroid - m1_centroid > 1e-10:
        vmi_pseudo = (f0_values - m1_centroid) / (m2_centroid - m1_centroid)
    else:
        vmi_pseudo = np.full_like(f0_values, 0.5)

    vmi_pseudo = np.clip(vmi_pseudo, 0, 1)

    return vmi_pseudo


def apply_temporal_smoothing(
    vmi: np.ndarray,
    window_size: int = 5,
    method: Literal["median", "mean", "exponential"] = "median",
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply temporal smoothing to VMI to avoid rapid oscillations.

    Args:
        vmi: VMI array per frame.
        window_size: Window size for mean/median.
        method: Smoothing method.
        alpha: Parameter for exponential smoothing.

    Returns:
        Smoothed VMI.
    """
    if method == "median":
        # Moving median (robust to outliers)
        vmi_smooth = np.zeros_like(vmi)
        half_window = window_size // 2
        for i in range(len(vmi)):
            start = max(0, i - half_window)
            end = min(len(vmi), i + half_window + 1)
            vmi_smooth[i] = np.nanmedian(vmi[start:end])
        return vmi_smooth

    elif method == "mean":
        # Moving average
        kernel = np.ones(window_size) / window_size
        return np.convolve(vmi, kernel, mode="same")

    elif method == "exponential":
        # Exponential moving average
        vmi_smooth = np.zeros_like(vmi)
        vmi_smooth[0] = vmi[0]
        for i in range(1, len(vmi)):
            vmi_smooth[i] = alpha * vmi[i] + (1 - alpha) * vmi_smooth[i - 1]
        return vmi_smooth

    return vmi
