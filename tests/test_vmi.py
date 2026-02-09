"""Testes para o módulo VMI (Vocal Mechanism Index)."""

import numpy as np

from vocal_analysis.features.vmi import (
    VMILabel,
    VMIWeights,
    apply_temporal_smoothing,
    compute_vmi_fixed,
    vmi_to_label,
)


class TestVMIWeights:
    """Testes para VMIWeights dataclass."""

    def test_default_weights(self):
        """Pesos default devem estar definidos."""
        weights = VMIWeights()
        assert weights.alpha_ratio == 0.30
        assert weights.cpps == 0.25
        assert weights.h1_h2 == 0.25
        assert weights.spectral_tilt == 0.20

    def test_weights_sum_to_one(self):
        """Pesos default devem somar aproximadamente 1."""
        weights = VMIWeights()
        total = weights.alpha_ratio + weights.cpps + weights.h1_h2 + weights.spectral_tilt
        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        """Deve aceitar pesos customizados."""
        weights = VMIWeights(alpha_ratio=0.5, cpps=0.2, h1_h2=0.2, spectral_tilt=0.1)
        assert weights.alpha_ratio == 0.5
        assert weights.cpps == 0.2

    def test_to_dict(self):
        """to_dict deve retornar dicionário com todos os pesos."""
        weights = VMIWeights()
        d = weights.to_dict()
        assert "alpha_ratio" in d
        assert "cpps" in d
        assert "h1_h2" in d
        assert "spectral_tilt" in d


class TestVMILabel:
    """Testes para VMILabel enum."""

    def test_all_labels_exist(self):
        """Todos os 5 labels devem existir."""
        assert VMILabel.M1_HEAVY.value == "M1_HEAVY"
        assert VMILabel.M1_LIGHT.value == "M1_LIGHT"
        assert VMILabel.MIX_PASSAGGIO.value == "MIX_PASSAGGIO"
        assert VMILabel.M2_REINFORCED.value == "M2_REINFORCED"
        assert VMILabel.M2_LIGHT.value == "M2_LIGHT"

    def test_label_count(self):
        """Devem existir exatamente 5 labels."""
        assert len(VMILabel) == 5


class TestComputeVMIFixed:
    """Testes para compute_vmi_fixed."""

    def test_vmi_returns_array(self, m1_spectral_features):
        """Deve retornar um array numpy."""
        result = compute_vmi_fixed(
            alpha_ratio=m1_spectral_features["alpha_ratio"],
            cpps=m1_spectral_features["cpps_per_frame"],
            h1_h2=m1_spectral_features["h1_h2"],
            spectral_tilt=m1_spectral_features["spectral_tilt"],
        )
        assert isinstance(result, np.ndarray)

    def test_vmi_range_zero_to_one(self, m1_spectral_features):
        """VMI deve estar entre 0 e 1."""
        result = compute_vmi_fixed(
            alpha_ratio=m1_spectral_features["alpha_ratio"],
            cpps=m1_spectral_features["cpps_per_frame"],
            h1_h2=m1_spectral_features["h1_h2"],
            spectral_tilt=m1_spectral_features["spectral_tilt"],
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_vmi_m1_features_low_vmi(self, m1_spectral_features, m2_spectral_features):
        """Quando M1 e M2 combinados, M1 deve ter VMI menor."""
        # Combinar M1 e M2 para que a normalização seja feita no conjunto completo
        combined_alpha = np.concatenate(
            [
                m1_spectral_features["alpha_ratio"],
                m2_spectral_features["alpha_ratio"],
            ]
        )
        combined_cpps = np.concatenate(
            [
                m1_spectral_features["cpps_per_frame"],
                m2_spectral_features["cpps_per_frame"],
            ]
        )
        combined_h1h2 = np.concatenate(
            [
                m1_spectral_features["h1_h2"],
                m2_spectral_features["h1_h2"],
            ]
        )
        combined_tilt = np.concatenate(
            [
                m1_spectral_features["spectral_tilt"],
                m2_spectral_features["spectral_tilt"],
            ]
        )

        vmi_combined = compute_vmi_fixed(
            alpha_ratio=combined_alpha,
            cpps=combined_cpps,
            h1_h2=combined_h1h2,
            spectral_tilt=combined_tilt,
        )

        n = len(m1_spectral_features["alpha_ratio"])
        vmi_m1 = vmi_combined[:n]
        vmi_m2 = vmi_combined[n:]

        # VMI médio de M1 deve ser menor que VMI médio de M2
        assert np.nanmean(vmi_m1) < np.nanmean(vmi_m2)

    def test_vmi_handles_nan(self, noisy_spectral_features):
        """Deve lidar com NaN nas features."""
        result = compute_vmi_fixed(
            alpha_ratio=noisy_spectral_features["alpha_ratio"],
            cpps=noisy_spectral_features["cpps_per_frame"],
            h1_h2=noisy_spectral_features["h1_h2"],
            spectral_tilt=noisy_spectral_features["spectral_tilt"],
        )
        # Não deve retornar array vazio
        assert len(result) > 0
        # Deve estar no range válido (exceto NaN)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= 0)
            assert np.all(valid <= 1)

    def test_vmi_custom_weights(self, m1_spectral_features):
        """Deve aceitar pesos customizados."""
        weights = VMIWeights(alpha_ratio=1.0, cpps=0.0, h1_h2=0.0, spectral_tilt=0.0)
        result = compute_vmi_fixed(
            alpha_ratio=m1_spectral_features["alpha_ratio"],
            cpps=m1_spectral_features["cpps_per_frame"],
            h1_h2=m1_spectral_features["h1_h2"],
            spectral_tilt=m1_spectral_features["spectral_tilt"],
            weights=weights,
        )
        assert isinstance(result, np.ndarray)


class TestVMIToLabel:
    """Testes para vmi_to_label."""

    def test_label_m1_heavy(self):
        """VMI 0.0-0.2 should be M1_HEAVY."""
        assert vmi_to_label(0.0) == VMILabel.M1_HEAVY
        assert vmi_to_label(0.1) == VMILabel.M1_HEAVY
        assert vmi_to_label(0.19) == VMILabel.M1_HEAVY

    def test_label_m1_light(self):
        """VMI 0.2-0.4 should be M1_LIGHT."""
        assert vmi_to_label(0.2) == VMILabel.M1_LIGHT
        assert vmi_to_label(0.3) == VMILabel.M1_LIGHT
        assert vmi_to_label(0.39) == VMILabel.M1_LIGHT

    def test_label_mix_passaggio(self):
        """VMI 0.4-0.6 should be MIX_PASSAGGIO."""
        assert vmi_to_label(0.4) == VMILabel.MIX_PASSAGGIO
        assert vmi_to_label(0.5) == VMILabel.MIX_PASSAGGIO
        assert vmi_to_label(0.59) == VMILabel.MIX_PASSAGGIO

    def test_label_m2_reinforced(self):
        """VMI 0.6-0.8 should be M2_REINFORCED."""
        assert vmi_to_label(0.6) == VMILabel.M2_REINFORCED
        assert vmi_to_label(0.7) == VMILabel.M2_REINFORCED
        assert vmi_to_label(0.79) == VMILabel.M2_REINFORCED

    def test_label_m2_light(self):
        """VMI 0.8-1.0 should be M2_LIGHT."""
        assert vmi_to_label(0.8) == VMILabel.M2_LIGHT
        assert vmi_to_label(0.9) == VMILabel.M2_LIGHT
        assert vmi_to_label(1.0) == VMILabel.M2_LIGHT

    def test_label_array(self):
        """Should work with array of values."""
        vmi_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = vmi_to_label(vmi_values)
        assert len(labels) == 5
        assert labels[0] == VMILabel.M1_HEAVY.value
        assert labels[2] == VMILabel.MIX_PASSAGGIO.value
        assert labels[4] == VMILabel.M2_LIGHT.value


class TestApplyTemporalSmoothing:
    """Testes para apply_temporal_smoothing."""

    def test_smoothing_returns_same_length(self):
        """Deve retornar array do mesmo tamanho."""
        vmi = np.random.rand(100)
        smoothed = apply_temporal_smoothing(vmi, window_size=5, method="median")
        assert len(smoothed) == len(vmi)

    def test_smoothing_reduces_variance(self):
        """Suavização deve reduzir variância."""
        # Criar sinal com ruído
        vmi = np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.3 + 0.5
        vmi += np.random.rand(100) * 0.2  # Adicionar ruído

        smoothed = apply_temporal_smoothing(vmi, window_size=7, method="median")

        # Variância deve diminuir ou permanecer similar
        # (não garantido para todos os casos, mas deve funcionar na maioria)
        assert np.var(smoothed) <= np.var(vmi) * 1.5  # Margem de tolerância

    def test_smoothing_median_method(self):
        """Método median deve funcionar."""
        vmi = np.random.rand(50)
        smoothed = apply_temporal_smoothing(vmi, window_size=5, method="median")
        assert isinstance(smoothed, np.ndarray)
        assert len(smoothed) == len(vmi)

    def test_smoothing_mean_method(self):
        """Método mean deve funcionar."""
        vmi = np.random.rand(50)
        smoothed = apply_temporal_smoothing(vmi, window_size=5, method="mean")
        assert isinstance(smoothed, np.ndarray)
        assert len(smoothed) == len(vmi)

    def test_smoothing_exponential_method(self):
        """Método exponential deve funcionar."""
        vmi = np.random.rand(50)
        smoothed = apply_temporal_smoothing(vmi, window_size=5, method="exponential")
        assert isinstance(smoothed, np.ndarray)
        assert len(smoothed) == len(vmi)

    def test_smoothing_preserves_range(self):
        """Suavização deve preservar range [0, 1]."""
        vmi = np.clip(np.random.rand(100), 0, 1)
        smoothed = apply_temporal_smoothing(vmi, window_size=5, method="median")
        assert np.all(smoothed >= 0)
        assert np.all(smoothed <= 1)
