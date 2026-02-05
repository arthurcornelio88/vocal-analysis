"""Vocal Mechanism Index (VMI) - Índice contínuo de mecanismo vocal."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class VMILabel(str, Enum):
    """Labels categóricas para VMI."""

    M1_DENSO = "M1_DENSO"
    M1_LIGEIRO = "M1_LIGEIRO"
    MIX_PASSAGGIO = "MIX_PASSAGGIO"
    M2_REFORCADO = "M2_REFORCADO"
    M2_LIGEIRO = "M2_LIGEIRO"


# Mapeamento VMI -> Label
VMI_THRESHOLDS = {
    (0.0, 0.2): VMILabel.M1_DENSO,
    (0.2, 0.4): VMILabel.M1_LIGEIRO,
    (0.4, 0.6): VMILabel.MIX_PASSAGGIO,
    (0.6, 0.8): VMILabel.M2_REFORCADO,
    (0.8, 1.0): VMILabel.M2_LIGEIRO,
}


@dataclass
class VMIWeights:
    """Pesos para cálculo do VMI."""

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
    """Normaliza features com Z-score global.

    Args:
        df: DataFrame com features.
        feature_cols: Colunas a normalizar.
        scaler: Scaler pré-treinado (opcional). Se None, treina novo.

    Returns:
        Tupla (DataFrame normalizado, scaler usado).
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
    """Calcula VMI com pesos fixos (baseline).

    Lógica:
    - Alpha Ratio alta → M1 (baixo VMI)
    - CPPS alto → voz limpa (não distingue M1/M2 diretamente)
    - H1-H2 baixo → M1 (baixo VMI)
    - Spectral Tilt negativo (steep) → M1 (baixo VMI)

    O VMI é normalizado para [0, 1] onde:
    - 0.0 = M1 Denso
    - 1.0 = M2 Ligeiro

    Args:
        alpha_ratio: Alpha Ratio por frame (dB).
        cpps: CPPS por frame.
        h1_h2: H1-H2 por frame (dB).
        spectral_tilt: Spectral Tilt por frame.
        weights: Pesos para cada feature. Default: VMIWeights().

    Returns:
        Array com VMI por frame [0, 1].
    """
    if weights is None:
        weights = VMIWeights()

    # Garantir mesmo tamanho
    min_len = min(len(alpha_ratio), len(cpps), len(h1_h2), len(spectral_tilt))
    alpha_ratio = alpha_ratio[:min_len]
    cpps = cpps[:min_len]
    h1_h2 = h1_h2[:min_len]
    spectral_tilt = spectral_tilt[:min_len]

    # Substituir NaN por mediana (robusto a outliers)
    alpha_ratio = np.where(np.isnan(alpha_ratio), np.nanmedian(alpha_ratio), alpha_ratio)
    cpps = np.where(np.isnan(cpps), np.nanmedian(cpps), cpps)
    h1_h2 = np.where(np.isnan(h1_h2), np.nanmedian(h1_h2), h1_h2)
    spectral_tilt = np.where(np.isnan(spectral_tilt), np.nanmedian(spectral_tilt), spectral_tilt)

    # Normalizar cada feature para [0, 1] usando min-max
    def minmax_norm(arr: np.ndarray) -> np.ndarray:
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        if arr_max - arr_min < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    alpha_norm = minmax_norm(alpha_ratio)
    cpps_norm = minmax_norm(cpps)
    h1_h2_norm = minmax_norm(h1_h2)
    tilt_norm = minmax_norm(spectral_tilt)

    # Inverter features que correlacionam negativamente com VMI
    # Alpha Ratio alta → M1 → baixo VMI, então invertemos
    alpha_contrib = (1 - alpha_norm) * weights.alpha_ratio

    # CPPS não tem direção clara para VMI, usar como está
    # (CPPS alto = voz limpa, presente em M1 denso E M2 reforçado)
    # Por isso, CPPS contribui menos para discriminação M1/M2
    cpps_contrib = cpps_norm * weights.cpps * 0.5 + 0.25 * weights.cpps  # Contribuição neutra

    # H1-H2 alto → M2 → alto VMI
    h1_h2_contrib = h1_h2_norm * weights.h1_h2

    # Spectral Tilt negativo (steep) → M1 → baixo VMI
    # Tilt mais positivo/plano → M2 → alto VMI
    tilt_contrib = tilt_norm * weights.spectral_tilt

    # Combinar
    vmi = alpha_contrib + cpps_contrib + h1_h2_contrib + tilt_contrib

    # Renormalizar para [0, 1]
    vmi = np.clip(vmi, 0, 1)

    # Normalização final para garantir range completo
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
    """Treina regressor XGBoost para aprender pesos ótimos do VMI.

    Args:
        df: DataFrame com features e target (vmi_target: pseudo-labels ou anotações).
        feature_cols: Colunas de features espectrais.
        target_col: Coluna com VMI target (0-1).
        test_size: Proporção do conjunto de teste.
        random_state: Seed para reprodutibilidade.
        use_temporal_smoothing: Se True, adiciona regularização temporal.

    Returns:
        Tupla (modelo XGBoost, scaler, métricas).
    """
    if feature_cols is None:
        feature_cols = ["alpha_ratio", "cpps_per_frame", "h1_h2", "spectral_tilt"]

    # Remover linhas com NaN no target
    df_clean = df.dropna(subset=[target_col])

    # Normalização global
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

    # Métricas
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
    """Prediz VMI usando modelo treinado.

    Args:
        df: DataFrame com features.
        model: Modelo XGBoost treinado.
        scaler: Scaler usado no treinamento.
        feature_cols: Colunas de features.

    Returns:
        Array com VMI predito [0, 1].
    """
    if feature_cols is None:
        feature_cols = ["alpha_ratio", "cpps_per_frame", "h1_h2", "spectral_tilt"]

    df_normalized, _ = normalize_features_global(df, feature_cols, scaler)
    X = df_normalized[feature_cols]

    vmi = model.predict(X)

    # Garantir range [0, 1]
    vmi = np.clip(vmi, 0, 1)

    return vmi


def vmi_to_label(vmi: np.ndarray | float) -> np.ndarray | VMILabel:
    """Converte VMI numérico para label categórica.

    Args:
        vmi: Valor(es) VMI entre 0 e 1.

    Returns:
        Label(s) categórica(s).
    """
    if isinstance(vmi, int | float):
        for (low, high), label in VMI_THRESHOLDS.items():
            if low <= vmi < high:
                return label
        return VMILabel.M2_LIGEIRO  # Edge case: vmi == 1.0

    # Array
    labels = np.empty(len(vmi), dtype=object)
    for i, v in enumerate(vmi):
        for (low, high), label in VMI_THRESHOLDS.items():
            if low <= v < high:
                labels[i] = label.value
                break
        else:
            labels[i] = VMILabel.M2_LIGEIRO.value

    return labels


def create_pseudo_labels_from_gmm(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    f0_col: str = "f0",
) -> np.ndarray:
    """Cria pseudo-labels VMI a partir de clusters GMM existentes.

    Estratégia: assume que o cluster com F0 médio mais alto é M2,
    e converte para escala contínua baseada na distância ao centróide.

    Args:
        df: DataFrame com clusters e f0.
        cluster_col: Coluna com labels de cluster.
        f0_col: Coluna com valores de F0.

    Returns:
        Array com pseudo-labels VMI [0, 1].
    """
    df = df.copy()

    # Identificar qual cluster é M1/M2 baseado em F0 médio
    cluster_f0_mean = df.groupby(cluster_col)[f0_col].mean()
    m2_cluster = cluster_f0_mean.idxmax()
    m1_cluster = cluster_f0_mean.idxmin()

    # Calcular distância normalizada ao centróide de cada cluster
    m1_centroid = df[df[cluster_col] == m1_cluster][f0_col].mean()
    m2_centroid = df[df[cluster_col] == m2_cluster][f0_col].mean()

    # VMI = distância relativa entre centroides
    # f0 próximo de m1_centroid → VMI baixo
    # f0 próximo de m2_centroid → VMI alto
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
    """Aplica suavização temporal ao VMI para evitar oscilações rápidas.

    Args:
        vmi: Array de VMI por frame.
        window_size: Tamanho da janela para média/mediana.
        method: Método de suavização.
        alpha: Parâmetro para exponential smoothing.

    Returns:
        VMI suavizado.
    """
    if method == "median":
        # Mediana móvel (robusto a outliers)
        vmi_smooth = np.zeros_like(vmi)
        half_window = window_size // 2
        for i in range(len(vmi)):
            start = max(0, i - half_window)
            end = min(len(vmi), i + half_window + 1)
            vmi_smooth[i] = np.nanmedian(vmi[start:end])
        return vmi_smooth

    elif method == "mean":
        # Média móvel
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
