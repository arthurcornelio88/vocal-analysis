"""Features espectrais para análise de mecanismo vocal (VMI)."""

from pathlib import Path

import numpy as np
import parselmouth
from scipy import signal
from scipy.stats import linregress

from vocal_analysis.preprocessing.audio import load_audio


def compute_alpha_ratio(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 220,
    n_fft: int = 2048,
    low_band: tuple[float, float] = (50, 1000),
    high_band: tuple[float, float] = (1000, 5000),
) -> np.ndarray:
    """Calcula Alpha Ratio (razão de energia espectral) por frame.

    Alpha Ratio = energia em banda alta / energia em banda baixa (em dB).
    Valores altos indicam mais energia em harmônicos superiores (típico de M1).

    Args:
        audio: Sinal de áudio mono.
        sr: Sample rate.
        hop_length: Hop length em samples.
        n_fft: Tamanho da FFT.
        low_band: Banda baixa em Hz (default: 50-1000 Hz).
        high_band: Banda alta em Hz (default: 1000-5000 Hz).

    Returns:
        Array com Alpha Ratio por frame (em dB).
    """
    # Calcular STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power = np.abs(Zxx) ** 2

    # Índices das bandas de frequência
    low_idx = np.where((f >= low_band[0]) & (f < low_band[1]))[0]
    high_idx = np.where((f >= high_band[0]) & (f < high_band[1]))[0]

    # Energia em cada banda (soma ao longo das frequências)
    energy_low = np.sum(power[low_idx, :], axis=0)
    energy_high = np.sum(power[high_idx, :], axis=0)

    # Evitar divisão por zero e log de zero
    energy_low = np.maximum(energy_low, 1e-10)
    energy_high = np.maximum(energy_high, 1e-10)

    # Alpha Ratio em dB
    alpha_ratio = 10 * np.log10(energy_high / energy_low)

    return alpha_ratio


def compute_h1_h2(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    hop_length: int = 220,
    n_fft: int = 4096,
    harmonic_tolerance_hz: float = 50.0,
) -> np.ndarray:
    """Calcula H1-H2 (diferença de amplitude entre 1º e 2º harmônico) por frame.

    H1-H2 indica inclinação glotal:
    - Valores baixos (H1 ≈ H2): adução firme, típico de M1
    - Valores altos (H1 >> H2): adução leve, típico de M2/falsete

    Nota: H1-H2 é menos confiável quando F0 > 350Hz (H1 pode coincidir com F1).
    Use Spectral Tilt como complemento.

    Args:
        audio: Sinal de áudio mono.
        sr: Sample rate.
        f0: Array de F0 por frame (mesmo número de frames que output).
        hop_length: Hop length em samples.
        n_fft: Tamanho da FFT (maior = melhor resolução em frequência).
        harmonic_tolerance_hz: Tolerância para encontrar pico do harmônico.

    Returns:
        Array com H1-H2 por frame (em dB). NaN onde F0 é inválido.
    """
    # Calcular STFT com resolução maior
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)

    # Resolução de frequência
    freq_resolution = sr / n_fft

    n_frames = magnitude.shape[1]
    h1_h2 = np.full(n_frames, np.nan)

    # Ajustar tamanho do f0 se necessário
    f0_aligned = np.interp(
        np.linspace(0, 1, n_frames),
        np.linspace(0, 1, len(f0)),
        f0,
    )

    for i in range(n_frames):
        f0_frame = f0_aligned[i]

        # Pular frames sem pitch válido
        if np.isnan(f0_frame) or f0_frame <= 0:
            continue

        # Frequências dos harmônicos
        h1_freq = f0_frame
        h2_freq = 2 * f0_frame

        # Verificar se H2 está dentro do range de frequência
        if h2_freq >= sr / 2:
            continue

        # Encontrar índices dos picos
        h1_idx_center = int(h1_freq / freq_resolution)
        h2_idx_center = int(h2_freq / freq_resolution)
        tolerance_bins = int(harmonic_tolerance_hz / freq_resolution)

        # Buscar pico na região do harmônico
        h1_start = max(0, h1_idx_center - tolerance_bins)
        h1_end = min(len(f), h1_idx_center + tolerance_bins)
        h2_start = max(0, h2_idx_center - tolerance_bins)
        h2_end = min(len(f), h2_idx_center + tolerance_bins)

        if h1_end > h1_start and h2_end > h2_start:
            h1_amp = np.max(magnitude[h1_start:h1_end, i])
            h2_amp = np.max(magnitude[h2_start:h2_end, i])

            # Evitar log de zero
            if h1_amp > 1e-10 and h2_amp > 1e-10:
                h1_h2[i] = 20 * np.log10(h1_amp) - 20 * np.log10(h2_amp)

    return h1_h2


def compute_spectral_tilt(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 220,
    n_fft: int = 2048,
    fmin: float = 50.0,
    fmax: float = 5000.0,
) -> np.ndarray:
    """Calcula Spectral Tilt (inclinação espectral) por frame.

    Spectral Tilt é a inclinação da regressão linear no espectro de potência.
    Mais robusto que H1-H2 em registros agudos (F0 > 350Hz).

    - Valores negativos (steep): espectro decai rapidamente, típico de M1
    - Valores próximos de zero (flat): espectro mais plano, típico de M2

    Args:
        audio: Sinal de áudio mono.
        sr: Sample rate.
        hop_length: Hop length em samples.
        n_fft: Tamanho da FFT.
        fmin: Frequência mínima para regressão.
        fmax: Frequência máxima para regressão.

    Returns:
        Array com Spectral Tilt por frame (slope em dB/Hz).
    """
    # Calcular STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    # Filtrar frequências de interesse
    freq_mask = (f >= fmin) & (f <= fmax)
    f_filtered = f[freq_mask]
    log_f = np.log10(f_filtered + 1)  # Log de frequência para linearização

    n_frames = power_db.shape[1]
    spectral_tilt = np.zeros(n_frames)

    for i in range(n_frames):
        power_frame = power_db[freq_mask, i]

        # Regressão linear: amplitude (dB) vs log-frequência
        slope, _, _, _, _ = linregress(log_f, power_frame)
        spectral_tilt[i] = slope

    return spectral_tilt


def compute_cpps_per_frame(
    audio_path: str | Path,
    hop_length: int = 220,
    sr: int = 44100,
    fmin: float = 50.0,
    window_duration: float = 0.04,
    timeout: int | None = None,
) -> np.ndarray:
    """Calcula CPPS (Cepstral Peak Prominence Smoothed) por frame via Praat.

    CPPS indica periodicidade/limpeza vocal:
    - Valores altos: voz periódica, limpa
    - Valores baixos: ruído, aperiodicidade

    Args:
        audio_path: Caminho para o arquivo de áudio.
        hop_length: Hop length em samples.
        sr: Sample rate esperado.
        fmin: Frequência mínima para análise.
        window_duration: Duração da janela em segundos.
        timeout: Timeout em segundos (None = sem timeout).

    Returns:
        Array com CPPS por frame. Valores NaN onde extração falha.
    """
    audio_path = Path(audio_path)
    sound = parselmouth.Sound(str(audio_path))

    time_step = hop_length / sr
    duration = sound.duration
    n_frames = int(duration / time_step)

    cpps_values = np.full(n_frames, np.nan)

    try:
        # Criar PowerCepstrogram completo
        power_cepstrogram = parselmouth.praat.call(
            sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
        )

        # Extrair CPPS em cada time point
        for i in range(n_frames):
            t = i * time_step
            try:
                # Extrair CPPS para uma janela centrada em t
                cpps_at_t = parselmouth.praat.call(
                    power_cepstrogram,
                    "Get CPPS",
                    False,  # subtract_tilt_before_smoothing
                    0.02,  # time_averaging_window
                    0.0005,  # quefrency_averaging_window
                    60,  # peak_search_pitch_range_start
                    330,  # peak_search_pitch_range_end
                    0.05,  # tolerance
                    "Parabolic",  # interpolation_method
                    t,  # from_time (usar t específico)
                    t + window_duration,  # to_time
                    "Exponential decay",
                    "Robust slow",
                )
                cpps_values[i] = cpps_at_t
            except Exception:
                # Frame específico falhou, manter NaN
                pass

    except Exception as e:
        print(f"  ⚠ CPPS per-frame falhou: {e}", flush=True)

    return cpps_values


def compute_f0_f1_distance(
    f0: np.ndarray,
    f1: np.ndarray,
) -> np.ndarray:
    """Calcula distância F0-F1 em semitons por frame.

    Indica estratégia de ressonância (vowel tuning), não mecanismo.
    Útil como feature auxiliar para análise de técnica vocal.

    Args:
        f0: Array de F0 por frame.
        f1: Array de F1 por frame.

    Returns:
        Array com distância F0-F1 em semitons. NaN onde dados são inválidos.
    """
    # Garantir mesmo tamanho
    min_len = min(len(f0), len(f1))
    f0 = f0[:min_len]
    f1 = f1[:min_len]

    # Máscaras para valores válidos
    valid_mask = (f0 > 0) & (f1 > 0) & ~np.isnan(f0) & ~np.isnan(f1)

    distance = np.full(min_len, np.nan)

    # Distância em semitons: 12 * log2(f1/f0)
    distance[valid_mask] = 12 * np.log2(f1[valid_mask] / f0[valid_mask])

    return distance


def extract_spectral_features(
    audio_path: str | Path,
    f0: np.ndarray,
    f1: np.ndarray | None = None,
    hop_length: int = 220,
    sr: int = 44100,
    skip_cpps_per_frame: bool = False,
) -> dict:
    """Extrai todas as features espectrais para VMI.

    Função de conveniência que agrupa todas as extrações espectrais.

    Args:
        audio_path: Caminho para o arquivo de áudio.
        f0: Array de F0 por frame (necessário para H1-H2).
        f1: Array de F1 por frame (opcional, para F0-F1 distance).
        hop_length: Hop length em samples.
        sr: Sample rate.
        skip_cpps_per_frame: Se True, pula CPPS per-frame (economiza tempo).

    Returns:
        Dicionário com:
        - alpha_ratio: np.ndarray
        - h1_h2: np.ndarray
        - spectral_tilt: np.ndarray
        - cpps_per_frame: np.ndarray (se não pulado)
        - f0_f1_distance: np.ndarray (se f1 fornecido)
    """
    audio_path = Path(audio_path)
    audio, sr_loaded = load_audio(audio_path)

    if sr_loaded != sr:
        print(f"  ⚠ Sample rate mismatch: esperado {sr}, obtido {sr_loaded}", flush=True)
        sr = sr_loaded

    result = {}

    # Alpha Ratio
    result["alpha_ratio"] = compute_alpha_ratio(audio, sr, hop_length)

    # H1-H2
    result["h1_h2"] = compute_h1_h2(audio, sr, f0, hop_length)

    # Spectral Tilt
    result["spectral_tilt"] = compute_spectral_tilt(audio, sr, hop_length)

    # CPPS per-frame (opcional - pode ser lento)
    if not skip_cpps_per_frame:
        result["cpps_per_frame"] = compute_cpps_per_frame(audio_path, hop_length, sr)
    else:
        result["cpps_per_frame"] = None

    # F0-F1 distance (se F1 disponível)
    if f1 is not None:
        result["f0_f1_distance"] = compute_f0_f1_distance(f0, f1)
    else:
        result["f0_f1_distance"] = None

    return result
