"""Pipeline híbrido de extração de features (Crepe + Praat)."""

from pathlib import Path
from typing import TypedDict

import librosa
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
    jitter: float
    shimmer: float
    energy: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    f4: np.ndarray
    time: np.ndarray


def extract_bioacoustic_features(
    audio_path: str | Path,
    hop_length: int = 220, # <--- MUDANÇA 1: Reduzir de 441 para 220 (5ms) para captar notas rápidas
    fmin: float = 50.0,
    fmax: float = 800.0,
    device: str = "cpu",
    model: str = "full", # <--- MUDANÇA 2: Garantir modelo 'full' para precisão máxima
    skip_formants: bool = False,
    skip_jitter_shimmer: bool = False,
    use_praat_f0: bool = False,
    skip_cpps: bool = False,
    cpps_timeout: int | None = None,
    batch_size: int = 2048,
) -> BioacousticFeatures:
    """Pipeline Híbrido de extração de features.

    1. Crepe para f0 (SOTA em robustez de pitch).
    2. Parselmouth para métricas espectrais (Rigor acadêmico).

    Args:
        audio_path: Caminho para o arquivo de áudio.
        hop_length: Hop length em samples (default 441 = 10ms @ 44.1kHz, conforme metodologia).
        fmin: Frequência mínima para detecção de pitch.
        fmax: Frequência máxima para detecção de pitch.
        device: Dispositivo para inferência ('cpu' ou 'cuda').
        model: Modelo CREPE ('tiny', 'small', 'medium', 'large', 'full').
        skip_formants: Se True, pula extração de formantes F1-F4 (economiza ~30% do tempo).
        skip_jitter_shimmer: Se True, pula Jitter/Shimmer (economiza ~20% do tempo).
        use_praat_f0: Se True, usa Praat ao invés de CREPE (muito mais rápido, menos preciso).
        skip_cpps: Se True, pula CPPS completamente (retorna None).
        cpps_timeout: Timeout em segundos para CPPS (None = sem timeout). Use apenas se CPPS travar.
        batch_size: Batch size para CREPE (default 2048 para GPU, use 512 para macOS CPU).

    Returns:
        Dicionário com features extraídas.
    """
    audio_path = Path(audio_path)
    audio, sr = load_audio(audio_path)

    # Carregar Parselmouth Sound (usado por vários métodos)
    sound = parselmouth.Sound(str(audio_path))
    time_step = hop_length / sr

    # 1. Extração de f0
    if use_praat_f0:
        # Usar Praat autocorrelation (muito mais rápido que CREPE, mas menos robusto)
        pitch = sound.to_pitch(time_step=time_step, pitch_floor=fmin, pitch_ceiling=fmax)

        # Extrair arrays de f0 diretamente dos valores do Pitch object
        # Praat retorna matrizes 2D com tempo e valores
        f0_values = pitch.selected_array["frequency"]

        # Criar confidence baseada em voiced/unvoiced
        # Se f0 > 0, consideramos voiced com confidence alta
        confidence_values = np.where(f0_values > 0, 0.9, 0.0)

        f0 = f0_values
        confidence = confidence_values

    else:
        # Usar CREPE (SOTA em robustez, mas lento)
        # CREPE (Kim et al., 2018) é escolhido sobre métodos tradicionais de autocorrelação
        # (como Praat's "To Pitch (cc)") por sua robustez superior em sinais com:
        #   - Vibrato intenso (comum no Choro)
        #   - Ruído de fundo (gravações históricas)
        #   - Ornamentações rápidas (glissandi, portamenti)
        #
        # O CREPE utiliza internamente janelamento próprio otimizado (aproximadamente 25ms)
        # que não é configurável pelo usuário. Essa escolha arquitetural da CNN foi validada
        # em benchmarks do MIR (Music Information Retrieval) e supera métodos baseados em
        # autocorrelação na detecção de pitch em sinais musicais complexos.
        #
        # Referência: Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018).
        # "Crepe: A convolutional representation for pitch estimation." ICASSP 2018.
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

        f0, confidence = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            model=model,
            decoder=torchcrepe.decode.weighted_argmax, # <--- CRUCIAL: Mude de .viterbi para .weighted_argmax
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
        )

        # Nota: weighted_argmax pode gerar mais "ruído" ou saltos falsos,
        # mas não vai "comer" a nota aguda real.
        
        # Filtragem pós-processamento manual (Opcional, mas recomendada se usar argmax)
        torchcrepe.filter.median(f0, 3)  # Filtro mediano leve para tirar ruído pontual

        f0 = f0.squeeze().cpu().numpy()
        confidence = confidence.squeeze().cpu().numpy()

    # 2. Extração de Timbre com Parselmouth (Praat)
    # Harmonicity (HNR) - Proxy para "limpeza" da voz
    harmonicity = sound.to_harmonicity(time_step=time_step)
    hnr_values = harmonicity.values[0]

    # Cepstral Peak Prominence (CPP) via Praat call
    # Usa a interface de scripting do Praat
    if skip_cpps:
        cpps = None
    else:
        try:
            if cpps_timeout:
                # Extração com timeout (para macOS/casos que travam)
                import threading
                result = {"cpps": None, "error": None, "timeout": False}

                def extract_cpps_target():
                    try:
                        power_cepstrogram = parselmouth.praat.call(
                            sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
                        )
                        result["cpps"] = parselmouth.praat.call(
                            power_cepstrogram, "Get CPPS", False, 0.02, 0.0005,
                            60, 330, 0.05, "Parabolic", 0.001, 0,
                            "Exponential decay", "Robust slow"
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
                    print(f"  ⚠ CPPS timeout ({cpps_timeout}s) - retornando None", flush=True)
                elif result["error"]:
                    cpps = None
                    print(f"  ⚠ CPPS erro: {result['error']} - retornando None", flush=True)
                else:
                    cpps = result["cpps"]
            else:
                # Extração direta sem timeout
                power_cepstrogram = parselmouth.praat.call(
                    sound, "To PowerCepstrogram", fmin, time_step, 5000.0, 50.0
                )
                cpps = parselmouth.praat.call(
                    power_cepstrogram, "Get CPPS", False, 0.02, 0.0005,
                    60, 330, 0.05, "Parabolic", 0.001, 0,
                    "Exponential decay", "Robust slow"
                )
        except Exception as e:
            # Erro na extração: retornar None explícito
            print(f"  ⚠ CPPS falhou: {e} - retornando None", flush=True)
            cpps = None

    # 3. Extração de Jitter e Shimmer (instabilidade glótica)
    # Jitter (ppq5): Period Perturbation Quotient - instabilidade de período
    # Shimmer (apq11): Amplitude Perturbation Quotient - instabilidade de amplitude
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
            # Fallback: valores padrão para sinais com pitch instável
            jitter_ppq5 = np.nan
            shimmer_apq11 = np.nan

    # 4. Extração de Energia Espectral (RMS)
    # Frame length de 25ms conforme janelamento típico (1102 samples @ 44.1kHz)
    energy = librosa.feature.rms(y=audio, frame_length=int(0.025 * sr), hop_length=hop_length)[0]

    # 5. Extração de Formantes F1-F4 via LPC (método de Burg)
    # Formantes indicam ressonâncias do trato vocal
    if skip_formants:
        # Se desativado, retorna arrays vazios
        f1_values = np.full_like(hnr_values, np.nan)
        f2_values = np.full_like(hnr_values, np.nan)
        f3_values = np.full_like(hnr_values, np.nan)
        f4_values = np.full_like(hnr_values, np.nan)
    else:
        try:
            formants = sound.to_formant_burg(
                time_step=time_step, max_number_of_formants=5, maximum_formant=5500
            )
            # Extrair arrays temporais dos formantes
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
            # Fallback: arrays vazios se extração falhar
            f1_values = np.full_like(hnr_values, np.nan)
            f2_values = np.full_like(hnr_values, np.nan)
            f3_values = np.full_like(hnr_values, np.nan)
            f4_values = np.full_like(hnr_values, np.nan)

    # Ajustar tamanhos dos arrays para sincronização temporal
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
