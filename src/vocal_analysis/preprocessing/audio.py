"""Audio preprocessing functions."""

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path: str | Path,
    sr: int = 44100,
    mono: bool = True,
    normalize: bool = True,
    target_db: float = -3.0,
) -> tuple[np.ndarray, int]:
    """Load audio with default settings for vocal analysis.

    Args:
        audio_path: Path to the audio file.
        sr: Target sample rate (default 44.1kHz).
        mono: Convert to mono.
        normalize: Apply normalization (default True, per methodology).
        target_db: Target level in dB for normalization (default -3dBFS).

    Returns:
        Tuple with audio array and sample rate.
    """
    audio, sr_out = librosa.load(str(audio_path), sr=sr, mono=mono)

    # Normalization per methodology (-3 dBFS peak)
    if normalize:
        audio = normalize_audio(audio, target_db=target_db)

    return audio, sr_out


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to a target dB level.

    Args:
        audio: Audio array.
        target_db: Target level in dB (default -3dB).

    Returns:
        Normalized audio array.
    """
    # Compute the normalization factor
    current_max = np.max(np.abs(audio))
    if current_max > 0:
        # Convert dB to linear amplitude
        target_amplitude = 10 ** (target_db / 20)
        # Apply normalization
        audio = audio * (target_amplitude / current_max)
    return audio
