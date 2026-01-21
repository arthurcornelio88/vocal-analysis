"""Testes para o módulo de features."""

import numpy as np

from vocal_analysis.features.extraction import BioacousticFeatures


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
