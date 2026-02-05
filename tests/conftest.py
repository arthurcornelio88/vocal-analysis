"""Configurações e fixtures do pytest."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Adicionar src ao path para imports funcionarem
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def synthetic_audio() -> tuple[np.ndarray, int]:
    """Gera áudio sintético (senoide 440Hz, 1 segundo, 44.1kHz)."""
    sr = 44100
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def synthetic_f0() -> np.ndarray:
    """Array de F0 constante para testes."""
    return np.full(100, 440.0)


@pytest.fixture
def synthetic_f0_with_nan() -> np.ndarray:
    """Array de F0 com alguns valores NaN."""
    f0 = np.full(100, 440.0)
    f0[10:15] = np.nan  # Região sem pitch
    f0[50:55] = np.nan
    return f0


@pytest.fixture
def m1_spectral_features() -> dict[str, np.ndarray]:
    """Features espectrais típicas de M1 denso.

    M1 denso:
    - Alpha Ratio alta (mais energia em harmônicos superiores)
    - H1-H2 baixo (adução firme)
    - Spectral Tilt negativo/íngreme
    - CPPS alto (voz limpa)
    """
    n = 100
    rng = np.random.default_rng(42)
    return {
        "alpha_ratio": rng.uniform(3.0, 7.0, n),  # Alto (dB)
        "h1_h2": rng.uniform(0.0, 4.0, n),  # Baixo (dB)
        "spectral_tilt": rng.uniform(-0.04, -0.02, n),  # Íngreme
        "cpps_per_frame": rng.uniform(18.0, 22.0, n),  # Alto
    }


@pytest.fixture
def m2_spectral_features() -> dict[str, np.ndarray]:
    """Features espectrais típicas de M2 ligeiro.

    M2 ligeiro:
    - Alpha Ratio baixa (energia concentrada em fundamental)
    - H1-H2 alto (adução leve)
    - Spectral Tilt próximo de zero (plano)
    - CPPS moderado
    """
    n = 100
    rng = np.random.default_rng(123)  # Seed diferente de m1
    return {
        "alpha_ratio": rng.uniform(-7.0, -3.0, n),  # Baixo (dB)
        "h1_h2": rng.uniform(12.0, 18.0, n),  # Alto (dB)
        "spectral_tilt": rng.uniform(-0.015, -0.005, n),  # Suave
        "cpps_per_frame": rng.uniform(8.0, 12.0, n),  # Moderado
    }


@pytest.fixture
def noisy_spectral_features() -> dict[str, np.ndarray]:
    """Features espectrais com ruído e NaNs para testar robustez."""
    n = 100
    rng = np.random.default_rng(42)

    features = {
        "alpha_ratio": rng.uniform(-10, 10, n),
        "h1_h2": rng.uniform(0, 20, n),
        "spectral_tilt": rng.uniform(-0.05, 0, n),
        "cpps_per_frame": rng.uniform(0, 30, n),
    }

    # Adicionar NaNs aleatórios
    for key in features:
        nan_indices = rng.choice(n, size=10, replace=False)
        features[key][nan_indices] = np.nan

    return features
