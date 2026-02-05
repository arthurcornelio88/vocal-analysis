"""Testes para o módulo de features."""

import numpy as np

from vocal_analysis.features.extraction import BioacousticFeatures, ExtendedFeatures


class TestBioacousticFeatures:
    """Testes para a estrutura BioacousticFeatures."""

    def test_bioacoustic_features_structure(self) -> None:
        """Deve aceitar os campos esperados."""
        features = BioacousticFeatures(
            f0=np.array([100.0, 150.0, 200.0]),
            confidence=np.array([0.9, 0.8, 0.7]),
            hnr=np.array([10.0, 12.0, 11.0]),
            cpps_global=5.5,
            time=np.array([0.0, 0.01, 0.02]),
        )

        assert "f0" in features
        assert "confidence" in features
        assert "hnr" in features
        assert "cpps_global" in features
        assert "time" in features

    def test_bioacoustic_features_types(self) -> None:
        """Deve ter os tipos corretos."""
        features = BioacousticFeatures(
            f0=np.array([100.0]),
            confidence=np.array([0.9]),
            hnr=np.array([10.0]),
            cpps_global=5.5,
            time=np.array([0.0]),
        )

        assert isinstance(features["f0"], np.ndarray)
        assert isinstance(features["confidence"], np.ndarray)
        assert isinstance(features["hnr"], np.ndarray)
        assert isinstance(features["cpps_global"], float)
        assert isinstance(features["time"], np.ndarray)


class TestExtendedFeatures:
    """Testes para a estrutura ExtendedFeatures (inclui features VMI)."""

    def test_extended_features_has_spectral_keys(self) -> None:
        """ExtendedFeatures deve incluir campos espectrais para VMI."""
        features = ExtendedFeatures(
            f0=np.array([100.0]),
            confidence=np.array([0.9]),
            hnr=np.array([10.0]),
            cpps_global=5.5,
            jitter=0.5,
            shimmer=1.0,
            energy=np.array([0.1]),
            f1=np.array([500.0]),
            f2=np.array([1500.0]),
            f3=np.array([2500.0]),
            f4=np.array([3500.0]),
            time=np.array([0.0]),
            alpha_ratio=np.array([5.0]),
            h1_h2=np.array([3.0]),
            spectral_tilt=np.array([-0.02]),
            cpps_per_frame=np.array([20.0]),
        )

        # Campos espectrais (VMI)
        assert "alpha_ratio" in features
        assert "h1_h2" in features
        assert "spectral_tilt" in features
        assert "cpps_per_frame" in features

    def test_extended_features_inherits_bioacoustic(self) -> None:
        """ExtendedFeatures deve conter todos os campos base."""
        features = ExtendedFeatures(
            f0=np.array([100.0]),
            confidence=np.array([0.9]),
            hnr=np.array([10.0]),
            cpps_global=5.5,
            jitter=0.5,
            shimmer=1.0,
            energy=np.array([0.1]),
            f1=np.array([500.0]),
            f2=np.array([1500.0]),
            f3=np.array([2500.0]),
            f4=np.array([3500.0]),
            time=np.array([0.0]),
            alpha_ratio=np.array([5.0]),
            h1_h2=np.array([3.0]),
            spectral_tilt=np.array([-0.02]),
            cpps_per_frame=None,
        )

        # Campos base
        assert "f0" in features
        assert "confidence" in features
        assert "hnr" in features
        assert "cpps_global" in features
        assert "jitter" in features
        assert "shimmer" in features
        assert "energy" in features
        assert "time" in features

        # Formantes
        assert "f1" in features
        assert "f2" in features
        assert "f3" in features
        assert "f4" in features

    def test_extended_features_cpps_per_frame_optional(self) -> None:
        """cpps_per_frame pode ser None."""
        features = ExtendedFeatures(
            f0=np.array([100.0]),
            confidence=np.array([0.9]),
            hnr=np.array([10.0]),
            cpps_global=5.5,
            jitter=0.5,
            shimmer=1.0,
            energy=np.array([0.1]),
            f1=np.array([500.0]),
            f2=np.array([1500.0]),
            f3=np.array([2500.0]),
            f4=np.array([3500.0]),
            time=np.array([0.0]),
            alpha_ratio=np.array([5.0]),
            h1_h2=np.array([3.0]),
            spectral_tilt=np.array([-0.02]),
            cpps_per_frame=None,
        )

        assert features["cpps_per_frame"] is None
