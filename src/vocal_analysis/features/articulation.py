"""Articulatory agility features for Choro analysis."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def compute_f0_velocity(f0: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute pitch change rate (Hz/s).

    Args:
        f0: Array of fundamental frequencies.
        time: Corresponding time array.

    Returns:
        Array with f0 change velocity.
    """
    if len(f0) < 2:
        return np.array([])

    dt = np.diff(time)
    f0_velocity = np.diff(f0) / dt
    # Zero out where there is a large gap between frames (unvoiced segment in between)
    f0_velocity[dt > 0.05] = 0.0
    f0_velocity = np.concatenate([[0], f0_velocity])

    return f0_velocity


def compute_f0_acceleration(f0: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute pitch acceleration (Hz/s^2).

    Args:
        f0: Array of fundamental frequencies.
        time: Corresponding time array.

    Returns:
        Array with f0 acceleration.
    """
    if len(f0) < 3:
        return np.array([])

    f0_velocity = compute_f0_velocity(f0, time)
    dt = np.diff(time)
    f0_accel = np.diff(f0_velocity) / dt
    # Zero out where there is a large gap between frames
    f0_accel[dt > 0.05] = 0.0
    # Limit extreme values (second derivative amplifies pitch noise)
    f0_accel = np.clip(f0_accel, -10000.0, 10000.0)
    f0_accel = np.concatenate([[0], f0_accel])

    return f0_accel


def compute_syllable_rate(
    energy: np.ndarray, time: np.ndarray, min_distance_s: float = 0.1
) -> float:
    """Estimate syllabic rate (syllables/second) via energy peaks.

    Args:
        energy: RMS energy array.
        time: Corresponding time array.
        min_distance_s: Minimum distance between peaks in seconds (default 0.1s = 100ms).

    Returns:
        Estimated syllabic rate in syllables/second.
    """
    if len(energy) < 2 or len(time) < 2:
        return 0.0

    # Convert minimum distance to indices
    time_step = np.mean(np.diff(time))
    min_distance_frames = int(min_distance_s / time_step)

    # Find energy peaks
    peaks, _ = find_peaks(energy, distance=min_distance_frames)

    # Rate = number of peaks / total duration
    duration = time[-1] - time[0]
    if duration > 0:
        return len(peaks) / duration
    return 0.0


def compute_articulation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all articulatory agility features.

    Args:
        df: DataFrame with columns 'f0', 'time', 'energy'.

    Returns:
        DataFrame with added features: 'f0_velocity', 'f0_acceleration', 'syllable_rate'.
    """
    df = df.copy()

    # Compute velocity and acceleration per song to avoid
    # artifacts at song boundaries (negative/zero delta-t)
    vel_parts = []
    accel_parts = []
    for _, group in df.groupby("song", sort=False):
        vel_parts.append(compute_f0_velocity(group["f0"].values, group["time"].values))
        accel_parts.append(compute_f0_acceleration(group["f0"].values, group["time"].values))
    df["f0_velocity"] = np.concatenate(vel_parts)
    df["f0_acceleration"] = np.concatenate(accel_parts)

    # Global syllabic rate
    syllable_rate = compute_syllable_rate(df["energy"].values, df["time"].values)
    df["syllable_rate"] = syllable_rate

    return df


def get_articulation_stats(df: pd.DataFrame) -> dict:
    """Extract articulatory agility statistics.

    Args:
        df: DataFrame with articulation features.

    Returns:
        Dictionary with statistics.
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
