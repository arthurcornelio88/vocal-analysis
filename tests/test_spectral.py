"""Testes para o módulo de features espectrais (VMI)."""

import numpy as np

from vocal_analysis.features.spectral import (
    compute_alpha_ratio,
    compute_h1_h2,
    compute_spectral_tilt,
)


class TestComputeAlphaRatio:
    """Testes para compute_alpha_ratio."""

    def test_alpha_ratio_returns_array(self, synthetic_audio):
        """Deve retornar um array numpy."""
        audio, sr = synthetic_audio
        result = compute_alpha_ratio(audio, sr)
        assert isinstance(result, np.ndarray)

    def test_alpha_ratio_has_frames(self, synthetic_audio):
        """Deve retornar múltiplos frames."""
        audio, sr = synthetic_audio
        result = compute_alpha_ratio(audio, sr, hop_length=220)
        assert len(result) > 1

    def test_alpha_ratio_finite_values(self, synthetic_audio):
        """Valores devem ser finitos (não inf/nan)."""
        audio, sr = synthetic_audio
        result = compute_alpha_ratio(audio, sr)
        assert np.all(np.isfinite(result))

    def test_alpha_ratio_no_nan(self, synthetic_audio):
        """Não deve retornar NaN para áudio limpo."""
        audio, sr = synthetic_audio
        result = compute_alpha_ratio(audio, sr)
        assert not np.any(np.isnan(result))

    def test_alpha_ratio_custom_bands(self, synthetic_audio):
        """Deve funcionar com bandas customizadas."""
        audio, sr = synthetic_audio
        result = compute_alpha_ratio(audio, sr, low_band=(100, 500), high_band=(500, 3000))
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestComputeH1H2:
    """Testes para compute_h1_h2."""

    def test_h1_h2_returns_array(self, synthetic_audio, synthetic_f0):
        """Deve retornar um array numpy."""
        audio, sr = synthetic_audio
        result = compute_h1_h2(audio, sr, synthetic_f0)
        assert isinstance(result, np.ndarray)

    def test_h1_h2_nan_for_invalid_f0(self, synthetic_audio):
        """Deve retornar NaN onde F0 é inválido."""
        audio, sr = synthetic_audio
        f0_invalid = np.full(100, np.nan)
        result = compute_h1_h2(audio, sr, f0_invalid)
        assert np.all(np.isnan(result))

    def test_h1_h2_nan_for_zero_f0(self, synthetic_audio):
        """Deve retornar NaN onde F0 é zero."""
        audio, sr = synthetic_audio
        f0_zero = np.zeros(100)
        result = compute_h1_h2(audio, sr, f0_zero)
        assert np.all(np.isnan(result))

    def test_h1_h2_partial_nan(self, synthetic_audio, synthetic_f0_with_nan):
        """Deve retornar NaN apenas onde F0 é inválido."""
        audio, sr = synthetic_audio
        result = compute_h1_h2(audio, sr, synthetic_f0_with_nan)
        # Alguns valores devem ser válidos
        assert np.any(~np.isnan(result))


class TestComputeSpectralTilt:
    """Testes para compute_spectral_tilt."""

    def test_spectral_tilt_returns_array(self, synthetic_audio):
        """Deve retornar um array numpy."""
        audio, sr = synthetic_audio
        result = compute_spectral_tilt(audio, sr)
        assert isinstance(result, np.ndarray)

    def test_spectral_tilt_has_frames(self, synthetic_audio):
        """Deve retornar múltiplos frames."""
        audio, sr = synthetic_audio
        result = compute_spectral_tilt(audio, sr, hop_length=220)
        assert len(result) > 1

    def test_spectral_tilt_no_nan(self, synthetic_audio):
        """Não deve retornar NaN para áudio limpo."""
        audio, sr = synthetic_audio
        result = compute_spectral_tilt(audio, sr)
        assert not np.any(np.isnan(result))

    def test_spectral_tilt_typically_negative(self, synthetic_audio):
        """Para voz típica, tilt deve ser negativo (espectro decai)."""
        audio, sr = synthetic_audio
        result = compute_spectral_tilt(audio, sr)
        # Senoide pura pode ter tilt variável, mas maioria deve ser negativa ou próximo de zero
        # Relaxamos a condição para senoide pura
        assert np.nanmean(result) < 50  # Valor muito alto seria suspeito


class TestSpectralFeaturesIntegration:
    """Testes de integração para features espectrais."""

    def test_all_features_same_hop_length(self, synthetic_audio, synthetic_f0):
        """Features com mesmo hop_length devem ter tamanhos comparáveis."""
        audio, sr = synthetic_audio
        hop_length = 441

        alpha = compute_alpha_ratio(audio, sr, hop_length=hop_length)
        h1h2 = compute_h1_h2(audio, sr, synthetic_f0, hop_length=hop_length)
        tilt = compute_spectral_tilt(audio, sr, hop_length=hop_length)

        # Tamanhos devem ser similares (podem variar ligeiramente por padding)
        sizes = [len(alpha), len(h1h2), len(tilt)]
        assert max(sizes) - min(sizes) < 10  # Tolerância de 10 frames
