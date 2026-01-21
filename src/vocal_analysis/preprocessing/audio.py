"""Funções de pré-processamento de áudio."""

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
    """Carrega áudio com configurações padrão para análise vocal.

    Args:
        audio_path: Caminho para o arquivo de áudio.
        sr: Sample rate alvo (default 44.1kHz).
        mono: Converter para mono.
        normalize: Aplicar normalização (default True, conforme metodologia).
        target_db: Nível alvo em dB para normalização (default -3dBFS).

    Returns:
        Tuple com array de áudio e sample rate.
    """
    audio, sr_out = librosa.load(str(audio_path), sr=sr, mono=mono)

    # Normalização conforme metodologia (-3 dBFS de pico)
    if normalize:
        audio = normalize_audio(audio, target_db=target_db)

    return audio, sr_out


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normaliza áudio para um nível de dB alvo.

    Args:
        audio: Array de áudio.
        target_db: Nível alvo em dB (default -3dB).

    Returns:
        Array de áudio normalizado.
    """
    # Calcula o fator de normalização
    current_max = np.max(np.abs(audio))
    if current_max > 0:
        # Converte dB para amplitude linear
        target_amplitude = 10 ** (target_db / 20)
        # Aplica a normalização
        audio = audio * (target_amplitude / current_max)
    return audio
