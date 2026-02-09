"""Spectral features for vocal mechanism analysis (VMI)."""

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
    """Compute Alpha Ratio (spectral energy ratio) per frame.

    Alpha Ratio = high band energy / low band energy (in dB).
    High values indicate more energy in upper harmonics (typical of M1).

    Args:
        audio: Mono audio signal.
        sr: Sample rate.
        hop_length: Hop length in samples.
        n_fft: FFT size.
        low_band: Low band in Hz (default: 50-1000 Hz).
        high_band: High band in Hz (default: 1000-5000 Hz).

    Returns:
        Array with Alpha Ratio per frame (in dB).
    """
    # Compute STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power = np.abs(Zxx) ** 2

    # Frequency band indices
    low_idx = np.where((f >= low_band[0]) & (f < low_band[1]))[0]
    high_idx = np.where((f >= high_band[0]) & (f < high_band[1]))[0]

    # Energy in each band (sum across frequencies)
    energy_low = np.sum(power[low_idx, :], axis=0)
    energy_high = np.sum(power[high_idx, :], axis=0)

    # Avoid division by zero and log of zero
    energy_low = np.maximum(energy_low, 1e-10)
    energy_high = np.maximum(energy_high, 1e-10)

    # Alpha Ratio in dB
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
    """Compute H1-H2 (amplitude difference between 1st and 2nd harmonic) per frame.

    H1-H2 indicates glottal slope:
    - Low values (H1 ≈ H2): firm adduction, typical of M1
    - High values (H1 >> H2): light adduction, typical of M2/falsetto

    Note: H1-H2 is less reliable when F0 > 350Hz (H1 may coincide with F1).
    Use Spectral Tilt as a complement.

    Args:
        audio: Mono audio signal.
        sr: Sample rate.
        f0: F0 array per frame (same number of frames as output).
        hop_length: Hop length in samples.
        n_fft: FFT size (larger = better frequency resolution).
        harmonic_tolerance_hz: Tolerance for finding the harmonic peak.

    Returns:
        Array with H1-H2 per frame (in dB). NaN where F0 is invalid.
    """
    # Compute STFT with higher resolution
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)

    # Frequency resolution
    freq_resolution = sr / n_fft

    n_frames = magnitude.shape[1]
    h1_h2 = np.full(n_frames, np.nan)

    # Adjust f0 size if necessary
    f0_aligned = np.interp(
        np.linspace(0, 1, n_frames),
        np.linspace(0, 1, len(f0)),
        f0,
    )

    for i in range(n_frames):
        f0_frame = f0_aligned[i]

        # Skip frames without valid pitch
        if np.isnan(f0_frame) or f0_frame <= 0:
            continue

        # Harmonic frequencies
        h1_freq = f0_frame
        h2_freq = 2 * f0_frame

        # Check if H2 is within the frequency range
        if h2_freq >= sr / 2:
            continue

        # Find peak indices
        h1_idx_center = int(h1_freq / freq_resolution)
        h2_idx_center = int(h2_freq / freq_resolution)
        tolerance_bins = int(harmonic_tolerance_hz / freq_resolution)

        # Search for peak in the harmonic region
        h1_start = max(0, h1_idx_center - tolerance_bins)
        h1_end = min(len(f), h1_idx_center + tolerance_bins)
        h2_start = max(0, h2_idx_center - tolerance_bins)
        h2_end = min(len(f), h2_idx_center + tolerance_bins)

        if h1_end > h1_start and h2_end > h2_start:
            h1_amp = np.max(magnitude[h1_start:h1_end, i])
            h2_amp = np.max(magnitude[h2_start:h2_end, i])

            # Avoid log of zero
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
    """Compute Spectral Tilt per frame.

    Spectral Tilt is the slope of the linear regression on the power spectrum.
    More robust than H1-H2 in high registers (F0 > 350Hz).

    - Negative values (steep): spectrum decays rapidly, typical of M1
    - Values close to zero (flat): flatter spectrum, typical of M2

    Args:
        audio: Mono audio signal.
        sr: Sample rate.
        hop_length: Hop length in samples.
        n_fft: FFT size.
        fmin: Minimum frequency for regression.
        fmax: Maximum frequency for regression.

    Returns:
        Array with Spectral Tilt per frame (slope in dB/Hz).
    """
    # Compute STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)

    # Filter frequencies of interest
    freq_mask = (f >= fmin) & (f <= fmax)
    f_filtered = f[freq_mask]
    log_f = np.log10(f_filtered + 1)  # Log frequency for linearization

    n_frames = power_db.shape[1]
    spectral_tilt = np.zeros(n_frames)

    for i in range(n_frames):
        power_frame = power_db[freq_mask, i]

        # Linear regression: amplitude (dB) vs log-frequency
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
    """Compute CPPS (Cepstral Peak Prominence Smoothed) per frame via Praat.

    CPPS indicates vocal periodicity/clarity:
    - High values: periodic, clean voice
    - Low values: noise, aperiodicity

    Args:
        audio_path: Path to the audio file.
        hop_length: Hop length in samples.
        sr: Expected sample rate.
        fmin: Minimum frequency for analysis.
        window_duration: Window duration in seconds.
        timeout: Timeout in seconds (None = no timeout).

    Returns:
        Array with CPPS per frame. NaN values where extraction fails.
    """
    audio_path = Path(audio_path)
    sound = parselmouth.Sound(str(audio_path))

    time_step = hop_length / sr
    duration = sound.duration
    n_frames = int(duration / time_step)

    cpps_values = np.full(n_frames, np.nan)

    try:
        # Create complete PowerCepstrogram
        power_cepstrogram = parselmouth.praat.call(
            sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
        )

        # Extract CPPS at each time point
        for i in range(n_frames):
            t = i * time_step
            try:
                # Extract CPPS for a window centered at t
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
                    t,  # from_time (use specific t)
                    t + window_duration,  # to_time
                    "Exponential decay",
                    "Robust slow",
                )
                cpps_values[i] = cpps_at_t
            except Exception:
                # Specific frame failed, keep NaN
                pass

    except Exception as e:
        print(f"  ⚠ CPPS per-frame failed: {e}", flush=True)

    return cpps_values


def compute_f0_f1_distance(
    f0: np.ndarray,
    f1: np.ndarray,
) -> np.ndarray:
    """Compute F0-F1 distance in semitones per frame.

    Indicates resonance strategy (vowel tuning), not mechanism.
    Useful as an auxiliary feature for vocal technique analysis.

    Args:
        f0: F0 array per frame.
        f1: F1 array per frame.

    Returns:
        Array with F0-F1 distance in semitones. NaN where data is invalid.
    """
    # Ensure same size
    min_len = min(len(f0), len(f1))
    f0 = f0[:min_len]
    f1 = f1[:min_len]

    # Masks for valid values
    valid_mask = (f0 > 0) & (f1 > 0) & ~np.isnan(f0) & ~np.isnan(f1)

    distance = np.full(min_len, np.nan)

    # Distance in semitones: 12 * log2(f1/f0)
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
    """Extract all spectral features for VMI.

    Convenience function that groups all spectral extractions.

    Args:
        audio_path: Path to the audio file.
        f0: F0 array per frame (required for H1-H2).
        f1: F1 array per frame (optional, for F0-F1 distance).
        hop_length: Hop length in samples.
        sr: Sample rate.
        skip_cpps_per_frame: If True, skip per-frame CPPS (saves time).

    Returns:
        Dictionary with:
        - alpha_ratio: np.ndarray
        - h1_h2: np.ndarray
        - spectral_tilt: np.ndarray
        - cpps_per_frame: np.ndarray (if not skipped)
        - f0_f1_distance: np.ndarray (if f1 provided)
    """
    audio_path = Path(audio_path)
    audio, sr_loaded = load_audio(audio_path)

    if sr_loaded != sr:
        print(f"  ⚠ Sample rate mismatch: expected {sr}, got {sr_loaded}", flush=True)
        sr = sr_loaded

    result = {}

    # Alpha Ratio
    result["alpha_ratio"] = compute_alpha_ratio(audio, sr, hop_length)

    # H1-H2
    result["h1_h2"] = compute_h1_h2(audio, sr, f0, hop_length)

    # Spectral Tilt
    result["spectral_tilt"] = compute_spectral_tilt(audio, sr, hop_length)

    # CPPS per-frame (optional - can be slow)
    if not skip_cpps_per_frame:
        result["cpps_per_frame"] = compute_cpps_per_frame(audio_path, hop_length, sr)
    else:
        result["cpps_per_frame"] = None

    # F0-F1 distance (if F1 available)
    if f1 is not None:
        result["f0_f1_distance"] = compute_f0_f1_distance(f0, f1)
    else:
        result["f0_f1_distance"] = None

    return result
