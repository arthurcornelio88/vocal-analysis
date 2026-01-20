"""Testes para o módulo de preprocessing."""

import numpy as np
import pytest

from vocal_analysis.preprocessing.audio import load_audio, normalize_audio


class TestNormalizeAudio:
    """Testes para a função normalize_audio."""

    def test_normalize_to_target_db(self) -> None:
        """Deve normalizar o áudio para o nível de dB alvo."""
        audio = np.array([0.5, -0.5, 0.25, -0.25])
        normalized = normalize_audio(audio, target_db=-3.0)

        # -3dB = 10^(-3/20) ≈ 0.708
        expected_max = 10 ** (-3.0 / 20)
        assert np.isclose(np.max(np.abs(normalized)), expected_max, rtol=1e-5)

    def test_normalize_preserves_relative_amplitudes(self) -> None:
        """Deve preservar as amplitudes relativas entre samples."""
        audio = np.array([1.0, 0.5, 0.25])
        normalized = normalize_audio(audio, target_db=-6.0)

        # Razões devem ser mantidas
        assert np.isclose(normalized[1] / normalized[0], 0.5)
        assert np.isclose(normalized[2] / normalized[0], 0.25)

    def test_normalize_silent_audio(self) -> None:
        """Não deve alterar áudio silencioso (evita divisão por zero)."""
        audio = np.zeros(100)
        normalized = normalize_audio(audio, target_db=-3.0)

        assert np.allclose(normalized, audio)

    def test_normalize_default_target(self) -> None:
        """Deve usar -3dB como padrão."""
        audio = np.array([1.0, -1.0])
        normalized = normalize_audio(audio)

        expected_max = 10 ** (-3.0 / 20)
        assert np.isclose(np.max(np.abs(normalized)), expected_max, rtol=1e-5)


class TestLoadAudio:
    """Testes para a função load_audio."""

    def test_load_audio_returns_tuple(self, tmp_path) -> None:
        """Deve retornar uma tupla (audio, sample_rate)."""
        # Criar arquivo WAV temporário
        import soundfile as sf

        audio_data = np.random.randn(44100).astype(np.float32)
        audio_file = tmp_path / "test.wav"
        sf.write(audio_file, audio_data, 44100)

        audio, sr = load_audio(audio_file)

        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int)
        assert sr == 44100

    def test_load_audio_resamples(self, tmp_path) -> None:
        """Deve resamplear para o sample rate especificado."""
        import soundfile as sf

        # Criar arquivo com sr diferente
        audio_data = np.random.randn(22050).astype(np.float32)
        audio_file = tmp_path / "test_22k.wav"
        sf.write(audio_file, audio_data, 22050)

        audio, sr = load_audio(audio_file, sr=44100)

        assert sr == 44100
        # Áudio deve ter aproximadamente o dobro de samples após resample
        assert len(audio) == pytest.approx(44100, rel=0.1)
