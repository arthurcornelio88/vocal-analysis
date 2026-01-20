"""Pipeline híbrido de extração de features (Crepe + Praat)."""

from pathlib import Path
from typing import TypedDict

import numpy as np
import parselmouth
import torch
import torchcrepe

from vocal_analysis.preprocessing.audio import load_audio


class BioacousticFeatures(TypedDict):
    """Features bioacústicas extraídas."""

    f0: np.ndarray
    confidence: np.ndarray
    hnr: np.ndarray
    cpps_global: float
    time: np.ndarray


def extract_bioacoustic_features(
    audio_path: str | Path,
    hop_length: int = 882,
    fmin: float = 50.0,
    fmax: float = 800.0,
    device: str = "cpu",
    model: str = "tiny",
) -> BioacousticFeatures:
    """Pipeline Híbrido de extração de features.

    1. Crepe para f0 (SOTA em robustez de pitch).
    2. Parselmouth para métricas espectrais (Rigor acadêmico).

    Args:
        audio_path: Caminho para o arquivo de áudio.
        hop_length: Hop length em samples (default 441 = 10ms @ 44.1kHz).
        fmin: Frequência mínima para detecção de pitch.
        fmax: Frequência máxima para detecção de pitch.
        device: Dispositivo para inferência ('cpu' ou 'cuda').
        model: Modelo CREPE ('tiny', 'small', 'medium', 'large', 'full').

    Returns:
        Dicionário com features extraídas.
    """
    audio_path = Path(audio_path)
    audio, sr = load_audio(audio_path)

    # 1. Extração de f0 com CREPE (melhor que autocorrelação para canto)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

    f0, confidence = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model,
        decoder=torchcrepe.decode.viterbi,
        batch_size=2048,
        device=device,
        return_periodicity=True,
    )
    f0 = f0.squeeze().cpu().numpy()
    confidence = confidence.squeeze().cpu().numpy()

    # 2. Extração de Timbre com Parselmouth (Praat)
    sound = parselmouth.Sound(str(audio_path))

    # Harmonicity (HNR) - Proxy para "limpeza" da voz
    time_step = hop_length / sr
    harmonicity = sound.to_harmonicity(time_step=time_step)
    hnr_values = harmonicity.values[0]

    # Cepstral Peak Prominence (CPP) via Praat call
    # Usa a interface de scripting do Praat
    try:
        power_cepstrogram = parselmouth.praat.call(
            sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
        )
        cpps = parselmouth.praat.call(power_cepstrogram, "Get CPPS", False, 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0, "Exponential decay", "Robust slow")
    except Exception:
        # Fallback: usar HNR médio como proxy
        cpps = float(np.nanmean(hnr_values))

    # Ajustar tamanhos dos arrays
    min_len = min(len(f0), len(hnr_values))
    time = np.arange(min_len) * time_step

    return BioacousticFeatures(
        f0=f0[:min_len],
        confidence=confidence[:min_len],
        hnr=hnr_values[:min_len],
        cpps_global=cpps,
        time=time,
    )
