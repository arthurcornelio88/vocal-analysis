"""Hybrid feature extraction pipeline (Crepe + Praat)."""

from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
import parselmouth
import torch
import torchcrepe

from vocal_analysis.preprocessing.audio import load_audio


class BioacousticFeatures(TypedDict):
    """Extracted bioacoustic features."""

    f0: np.ndarray
    confidence: np.ndarray
    hnr: np.ndarray
    cpps_global: float
    jitter: float
    shimmer: float
    energy: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    f4: np.ndarray
    time: np.ndarray


class ExtendedFeatures(TypedDict):
    """Extended features including spectral features for VMI."""

    # Basic features
    f0: np.ndarray
    confidence: np.ndarray
    hnr: np.ndarray
    cpps_global: float
    jitter: float
    shimmer: float
    energy: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    f4: np.ndarray
    time: np.ndarray
    # Spectral features (VMI)
    alpha_ratio: np.ndarray
    h1_h2: np.ndarray
    spectral_tilt: np.ndarray
    cpps_per_frame: np.ndarray | None
    f0_f1_distance: np.ndarray | None


def extract_bioacoustic_features(
    audio_path: str | Path,
    hop_length: int = 220,  # <--- CHANGE 1: Reduce from 441 to 220 (5ms) to capture fast notes
    fmin: float = 50.0,
    fmax: float = 800.0,
    device: str = "cpu",
    model: str = "full",  # <--- CHANGE 2: Ensure 'full' model for maximum accuracy
    skip_formants: bool = False,
    skip_jitter_shimmer: bool = False,
    use_praat_f0: bool = False,
    skip_cpps: bool = False,
    cpps_timeout: int | None = None,
    batch_size: int = 2048,
) -> BioacousticFeatures:
    """Hybrid feature extraction pipeline.

    1. Crepe for f0 (SOTA in pitch robustness).
    2. Parselmouth for spectral metrics (academic rigor).

    Args:
        audio_path: Path to the audio file.
        hop_length: Hop length in samples (default 441 = 10ms @ 44.1kHz, per methodology).
        fmin: Minimum frequency for pitch detection.
        fmax: Maximum frequency for pitch detection.
        device: Device for inference ('cpu' or 'cuda').
        model: CREPE model ('tiny', 'small', 'medium', 'large', 'full').
        skip_formants: If True, skip F1-F4 formant extraction (saves ~30% of time).
        skip_jitter_shimmer: If True, skip Jitter/Shimmer (saves ~20% of time).
        use_praat_f0: If True, use Praat instead of CREPE (much faster, less accurate).
        skip_cpps: If True, skip CPPS entirely (returns None).
        cpps_timeout: Timeout in seconds for CPPS (None = no timeout). Use only if CPPS hangs.
        batch_size: Batch size for CREPE (default 2048 for GPU, use 512 for macOS CPU).

    Returns:
        Dictionary with extracted features.
    """
    audio_path = Path(audio_path)
    audio, sr = load_audio(audio_path)

    # Load Parselmouth Sound (used by several methods)
    sound = parselmouth.Sound(str(audio_path))
    time_step = hop_length / sr

    # 1. f0 extraction
    if use_praat_f0:
        # Use Praat autocorrelation (much faster than CREPE, but less robust)
        pitch = sound.to_pitch(time_step=time_step, pitch_floor=fmin, pitch_ceiling=fmax)

        # Extract f0 arrays directly from the Pitch object values
        # Praat returns 2D matrices with time and values
        f0_values = pitch.selected_array["frequency"]

        # Create confidence based on voiced/unvoiced
        # If f0 > 0, we consider it voiced with high confidence
        confidence_values = np.where(f0_values > 0, 0.9, 0.0)

        f0 = f0_values
        confidence = confidence_values

    else:
        # Use CREPE (SOTA in robustness, but slow)
        # CREPE (Kim et al., 2018) is chosen over traditional autocorrelation methods
        # (such as Praat's "To Pitch (cc)") due to its superior robustness with signals containing:
        #   - Intense vibrato (common in Choro)
        #   - Background noise (historical recordings)
        #   - Fast ornamentations (glissandi, portamenti)
        #
        # CREPE internally uses its own optimized windowing (approximately 25ms)
        # which is not user-configurable. This CNN architectural choice was validated
        # in MIR (Music Information Retrieval) benchmarks and outperforms autocorrelation-based
        # methods for pitch detection in complex musical signals.
        #
        # Reference: Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018).
        # "Crepe: A convolutional representation for pitch estimation." ICASSP 2018.
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

        f0, confidence = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            model=model,
            decoder=torchcrepe.decode.weighted_argmax,  # <--- CRUCIAL: Change from .viterbi to .weighted_argmax
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
        )

        # Note: weighted_argmax may generate more "noise" or false jumps,
        # but it won't "swallow" the real high-pitched note.

        # Manual post-processing filtering (Optional, but recommended when using argmax)
        torchcrepe.filter.median(f0, 3)  # Light median filter to remove point noise

        f0 = f0.squeeze().cpu().numpy()
        confidence = confidence.squeeze().cpu().numpy()

        # Clip F0 to valid range - remove spurious detections outside fmin/fmax
        invalid_mask = (f0 < fmin) | (f0 > fmax)
        f0[invalid_mask] = 0
        confidence[invalid_mask] = 0

    # 2. Timbre extraction with Parselmouth (Praat)
    # Harmonicity (HNR) - Proxy for voice "clarity"
    harmonicity = sound.to_harmonicity(time_step=time_step)
    hnr_values = harmonicity.values[0]

    # Cepstral Peak Prominence (CPP) via Praat call
    # Uses the Praat scripting interface
    if skip_cpps:
        cpps = None
    else:
        try:
            if cpps_timeout:
                # Extraction with timeout (for macOS/cases that hang)
                import threading

                result = {"cpps": None, "error": None, "timeout": False}

                def extract_cpps_target():
                    try:
                        power_cepstrogram = parselmouth.praat.call(
                            sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
                        )
                        result["cpps"] = parselmouth.praat.call(
                            power_cepstrogram,
                            "Get CPPS",
                            False,
                            0.02,
                            0.0005,
                            60,
                            330,
                            0.05,
                            "Parabolic",
                            0.001,
                            0,
                            "Exponential decay",
                            "Robust slow",
                        )
                    except Exception as e:
                        result["error"] = str(e)

                thread = threading.Thread(target=extract_cpps_target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=cpps_timeout)

                if thread.is_alive():
                    result["timeout"] = True
                    cpps = None
                    print(f"  ⚠ CPPS timeout ({cpps_timeout}s) - returning None", flush=True)
                elif result["error"]:
                    cpps = None
                    print(f"  ⚠ CPPS error: {result['error']} - returning None", flush=True)
                else:
                    cpps = result["cpps"]
            else:
                # Direct extraction without timeout
                power_cepstrogram = parselmouth.praat.call(
                    sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
                )
                cpps = parselmouth.praat.call(
                    power_cepstrogram,
                    "Get CPPS",
                    False,
                    0.02,
                    0.0005,
                    60,
                    330,
                    0.05,
                    "Parabolic",
                    0.001,
                    0,
                    "Exponential decay",
                    "Robust slow",
                )
        except Exception as e:
            # Extraction error: return explicit None
            print(f"  ⚠ CPPS failed: {e} - returning None", flush=True)
            cpps = None

    # 3. Jitter and Shimmer extraction (glottal instability)
    # Jitter (ppq5): Period Perturbation Quotient - period instability
    # Shimmer (apq11): Amplitude Perturbation Quotient - amplitude instability
    if skip_jitter_shimmer:
        jitter_ppq5 = np.nan
        shimmer_apq11 = np.nan
    else:
        try:
            point_process = parselmouth.praat.call(
                sound, "To PointProcess (periodic, cc)", fmin, fmax
            )
            jitter_ppq5 = parselmouth.praat.call(
                point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
            )
            shimmer_apq11 = parselmouth.praat.call(
                [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )
        except Exception:
            # Fallback: default values for signals with unstable pitch
            jitter_ppq5 = np.nan
            shimmer_apq11 = np.nan

    # 4. Spectral Energy extraction (RMS)
    # Frame length of 25ms per typical windowing (1102 samples @ 44.1kHz)
    energy = librosa.feature.rms(y=audio, frame_length=int(0.025 * sr), hop_length=hop_length)[0]

    # 5. Formant F1-F4 extraction via LPC (Burg method)
    # Formants indicate vocal tract resonances
    if skip_formants:
        # If disabled, return empty arrays
        f1_values = np.full_like(hnr_values, np.nan)
        f2_values = np.full_like(hnr_values, np.nan)
        f3_values = np.full_like(hnr_values, np.nan)
        f4_values = np.full_like(hnr_values, np.nan)
    else:
        try:
            formants = sound.to_formant_burg(
                time_step=time_step, max_number_of_formants=5, maximum_formant=5500
            )
            # Extract temporal arrays from formants
            f1_values = np.array(
                [formants.get_value_at_time(1, t) for t in np.arange(0, sound.duration, time_step)]
            )
            f2_values = np.array(
                [formants.get_value_at_time(2, t) for t in np.arange(0, sound.duration, time_step)]
            )
            f3_values = np.array(
                [formants.get_value_at_time(3, t) for t in np.arange(0, sound.duration, time_step)]
            )
            f4_values = np.array(
                [formants.get_value_at_time(4, t) for t in np.arange(0, sound.duration, time_step)]
            )
        except Exception:
            # Fallback: empty arrays if extraction fails
            f1_values = np.full_like(hnr_values, np.nan)
            f2_values = np.full_like(hnr_values, np.nan)
            f3_values = np.full_like(hnr_values, np.nan)
            f4_values = np.full_like(hnr_values, np.nan)

    # Adjust array sizes for temporal synchronization
    min_len = min(len(f0), len(hnr_values), len(energy), len(f1_values))
    time = np.arange(min_len) * time_step

    return BioacousticFeatures(
        f0=f0[:min_len],
        confidence=confidence[:min_len],
        hnr=hnr_values[:min_len],
        cpps_global=cpps,
        jitter=jitter_ppq5,
        shimmer=shimmer_apq11,
        energy=energy[:min_len],
        f1=f1_values[:min_len],
        f2=f2_values[:min_len],
        f3=f3_values[:min_len],
        f4=f4_values[:min_len],
        time=time,
    )


def extract_extended_features(
    audio_path: str | Path,
    hop_length: int = 220,
    fmin: float = 50.0,
    fmax: float = 800.0,
    device: str = "cpu",
    model: str = "full",
    skip_formants: bool = False,
    skip_jitter_shimmer: bool = False,
    use_praat_f0: bool = False,
    skip_cpps: bool = False,
    cpps_timeout: int | None = None,
    batch_size: int = 2048,
    skip_cpps_per_frame: bool = False,
) -> ExtendedFeatures:
    """Extract basic + spectral features for VMI.

    Combines standard bioacoustic feature extraction with the new
    spectral features required for VMI computation.

    Args:
        audio_path: Path to the audio file.
        hop_length: Hop length in samples.
        fmin: Minimum frequency for pitch detection.
        fmax: Maximum frequency for pitch detection.
        device: Device for inference ('cpu' or 'cuda').
        model: CREPE model.
        skip_formants: If True, skip formant extraction.
        skip_jitter_shimmer: If True, skip Jitter/Shimmer.
        use_praat_f0: If True, use Praat instead of CREPE.
        skip_cpps: If True, skip global CPPS.
        cpps_timeout: Timeout in seconds for global CPPS.
        batch_size: Batch size for CREPE.
        skip_cpps_per_frame: If True, skip per-frame CPPS (saves time).

    Returns:
        Dictionary with all features (basic + spectral).
    """
    from vocal_analysis.features.spectral import extract_spectral_features

    audio_path = Path(audio_path)

    # 1. Extract basic features
    basic_features = extract_bioacoustic_features(
        audio_path=audio_path,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        device=device,
        model=model,
        skip_formants=skip_formants,
        skip_jitter_shimmer=skip_jitter_shimmer,
        use_praat_f0=use_praat_f0,
        skip_cpps=skip_cpps,
        cpps_timeout=cpps_timeout,
        batch_size=batch_size,
    )

    # 2. Extract spectral features
    audio, sr = load_audio(audio_path)
    spectral_features = extract_spectral_features(
        audio_path=audio_path,
        f0=basic_features["f0"],
        f1=basic_features["f1"] if not skip_formants else None,
        hop_length=hop_length,
        sr=sr,
        skip_cpps_per_frame=skip_cpps_per_frame,
    )

    # 3. Synchronize sizes
    min_len = min(
        len(basic_features["f0"]),
        len(spectral_features["alpha_ratio"]),
        len(spectral_features["h1_h2"]),
        len(spectral_features["spectral_tilt"]),
    )

    # CPPS per-frame may have a different size
    cpps_per_frame = spectral_features["cpps_per_frame"]
    if cpps_per_frame is not None:
        cpps_per_frame = cpps_per_frame[:min_len] if len(cpps_per_frame) >= min_len else None

    f0_f1_distance = spectral_features["f0_f1_distance"]
    if f0_f1_distance is not None:
        f0_f1_distance = f0_f1_distance[:min_len]

    return ExtendedFeatures(
        # Basic features
        f0=basic_features["f0"][:min_len],
        confidence=basic_features["confidence"][:min_len],
        hnr=basic_features["hnr"][:min_len],
        cpps_global=basic_features["cpps_global"],
        jitter=basic_features["jitter"],
        shimmer=basic_features["shimmer"],
        energy=basic_features["energy"][:min_len],
        f1=basic_features["f1"][:min_len],
        f2=basic_features["f2"][:min_len],
        f3=basic_features["f3"][:min_len],
        f4=basic_features["f4"][:min_len],
        time=basic_features["time"][:min_len],
        # Spectral features
        alpha_ratio=spectral_features["alpha_ratio"][:min_len],
        h1_h2=spectral_features["h1_h2"][:min_len],
        spectral_tilt=spectral_features["spectral_tilt"][:min_len],
        cpps_per_frame=cpps_per_frame,
        f0_f1_distance=f0_f1_distance,
    )
