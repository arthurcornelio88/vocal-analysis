"""Features de agilidade articulatória para análise do Choro."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def compute_f0_velocity(f0: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Calcula taxa de mudança de pitch (Hz/s).

    Args:
        f0: Array de frequências fundamentais.
        time: Array de tempo correspondente.

    Returns:
        Array com velocidade de mudança de f0.
    """
    if len(f0) < 2:
        return np.array([])

    # Velocidade: Δf0 / Δt
    f0_velocity = np.diff(f0) / np.diff(time)
    # Adicionar zero no início para manter tamanho
    f0_velocity = np.concatenate([[0], f0_velocity])

    return f0_velocity


def compute_f0_acceleration(f0: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Calcula aceleração de pitch (Hz/s²).

    Args:
        f0: Array de frequências fundamentais.
        time: Array de tempo correspondente.

    Returns:
        Array com aceleração de f0.
    """
    if len(f0) < 3:
        return np.array([])

    f0_velocity = compute_f0_velocity(f0, time)
    # Aceleração: Δvelocidade / Δt
    f0_accel = np.diff(f0_velocity) / np.diff(time[:-1])
    # Adicionar zeros no início para manter tamanho
    f0_accel = np.concatenate([[0, 0], f0_accel])

    return f0_accel


def compute_syllable_rate(
    energy: np.ndarray, time: np.ndarray, min_distance_s: float = 0.1
) -> float:
    """Estima taxa silábica (sílabas/segundo) via picos de energia.

    Args:
        energy: Array de energia RMS.
        time: Array de tempo correspondente.
        min_distance_s: Distância mínima entre picos em segundos (default 0.1s = 100ms).

    Returns:
        Taxa silábica estimada em sílabas/segundo.
    """
    if len(energy) < 2 or len(time) < 2:
        return 0.0

    # Converter distância mínima para índices
    time_step = np.mean(np.diff(time))
    min_distance_frames = int(min_distance_s / time_step)

    # Encontrar picos de energia
    peaks, _ = find_peaks(energy, distance=min_distance_frames)

    # Taxa = número de picos / duração total
    duration = time[-1] - time[0]
    if duration > 0:
        return len(peaks) / duration
    return 0.0


def compute_articulation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computa todas as features de agilidade articulatória.

    Args:
        df: DataFrame com colunas 'f0', 'time', 'energy'.

    Returns:
        DataFrame com features adicionadas: 'f0_velocity', 'f0_acceleration', 'syllable_rate'.
    """
    df = df.copy()

    # F0 velocity e acceleration
    df["f0_velocity"] = compute_f0_velocity(df["f0"].values, df["time"].values)
    df["f0_acceleration"] = compute_f0_acceleration(df["f0"].values, df["time"].values)

    # Taxa silábica global
    syllable_rate = compute_syllable_rate(df["energy"].values, df["time"].values)
    df["syllable_rate"] = syllable_rate

    return df


def get_articulation_stats(df: pd.DataFrame) -> dict:
    """Extrai estatísticas de agilidade articulatória.

    Args:
        df: DataFrame com features de articulação.

    Returns:
        Dicionário com estatísticas.
    """
    df_voiced = df[df["confidence"] > 0.8].copy()

    if "f0_velocity" not in df_voiced.columns:
        df_voiced = compute_articulation_features(df_voiced)

    stats = {
        "f0_velocity_mean": float(np.abs(df_voiced["f0_velocity"]).mean()),
        "f0_velocity_max": float(np.abs(df_voiced["f0_velocity"]).max()),
        "f0_acceleration_mean": float(np.abs(df_voiced["f0_acceleration"]).mean()),
        "syllable_rate": float(df_voiced["syllable_rate"].iloc[0]) if len(df_voiced) > 0 else 0.0,
    }

    return stats
