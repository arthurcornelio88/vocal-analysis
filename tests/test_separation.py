"""Testes para o módulo de source separation."""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from vocal_analysis.preprocessing.separation import (
    VocalSeparator,
    separate_vocals,
    separate_vocals_safe,
    HTDEMUCS_SAMPLE_RATE,
)


class TestVocalSeparator:
    """Testes para a classe VocalSeparator."""

    def test_init_default_device(self) -> None:
        """Deve inicializar com CPU por padrão."""
        separator = VocalSeparator()
        assert separator.device == torch.device("cpu")

    def test_init_custom_device(self) -> None:
        """Deve aceitar dispositivo customizado."""
        separator = VocalSeparator(device="cpu")
        assert separator.device == torch.device("cpu")

    def test_lazy_loading(self) -> None:
        """Modelo só deve carregar quando acessado."""
        separator = VocalSeparator()
        assert separator._model is None
        assert separator._sample_rate is None

    def test_init_segment_seconds(self) -> None:
        """Deve aceitar segment_seconds customizado."""
        separator = VocalSeparator(segment_seconds=5.0)
        assert separator.segment_seconds == 5.0

    def test_init_overlap(self) -> None:
        """Deve aceitar overlap customizado."""
        separator = VocalSeparator(overlap=0.2)
        assert separator.overlap == 0.2


class TestSeparateVocals:
    """Testes para função separate_vocals."""

    def test_cache_loads_existing(self, tmp_path: Path) -> None:
        """Deve carregar do cache se existir."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Criar cache fake
        vocals = np.random.randn(44100).astype(np.float32)
        np.save(cache_dir / "test_vocals.npy", vocals)

        # Criar arquivo de áudio fake (não será lido se cache existe)
        audio_path = tmp_path / "test.mp3"
        audio_path.touch()

        result, sr = separate_vocals(
            audio_path,
            cache_dir=cache_dir,
        )

        assert np.array_equal(result, vocals)
        assert sr == HTDEMUCS_SAMPLE_RATE

    def test_cache_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """Deve criar diretório de cache se não existir."""
        cache_dir = tmp_path / "new_cache_dir"

        # Criar cache fake no local esperado após criação
        cache_dir.mkdir()
        vocals = np.random.randn(44100).astype(np.float32)
        np.save(cache_dir / "test_vocals.npy", vocals)

        audio_path = tmp_path / "test.mp3"
        audio_path.touch()

        result, sr = separate_vocals(
            audio_path,
            cache_dir=cache_dir,
        )

        assert result is not None


class TestSeparateVocalsSafe:
    """Testes para função separate_vocals_safe."""

    def test_returns_tuple_with_success_flag(self, tmp_path: Path) -> None:
        """Deve retornar tupla com flag de sucesso."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Criar cache fake
        vocals = np.random.randn(44100).astype(np.float32)
        np.save(cache_dir / "test_vocals.npy", vocals)

        audio_path = tmp_path / "test.mp3"
        audio_path.touch()

        result, sr, success = separate_vocals_safe(
            audio_path,
            cache_dir=cache_dir,
        )

        assert success is True
        assert result is not None
        assert sr == HTDEMUCS_SAMPLE_RATE

    @patch("vocal_analysis.preprocessing.separation.separate_vocals")
    def test_returns_none_on_failure(
        self, mock_separate: MagicMock, tmp_path: Path
    ) -> None:
        """Deve retornar None e False se separação falhar."""
        mock_separate.side_effect = Exception("Test error")

        audio_path = tmp_path / "test.mp3"
        audio_path.touch()

        result, sr, success = separate_vocals_safe(audio_path)

        assert success is False
        assert result is None

    @patch("vocal_analysis.preprocessing.separation.separate_vocals")
    def test_fallback_to_cpu_on_oom(
        self, mock_separate: MagicMock, tmp_path: Path
    ) -> None:
        """Deve tentar CPU se GPU ficar sem memória."""
        # Primeira chamada (GPU) falha com OOM, segunda (CPU) funciona
        vocals = np.random.randn(44100).astype(np.float32)
        mock_separate.side_effect = [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            (vocals, 44100),
        ]

        audio_path = tmp_path / "test.mp3"
        audio_path.touch()

        result, sr, success = separate_vocals_safe(audio_path, device="cuda")

        assert success is True
        assert result is not None
        # Verifica que foi chamado duas vezes (GPU falhou, CPU funcionou)
        assert mock_separate.call_count == 2


class TestHTDemucsConstants:
    """Testes para constantes do módulo."""

    def test_sample_rate_constant(self) -> None:
        """Sample rate deve ser 44100 Hz."""
        assert HTDEMUCS_SAMPLE_RATE == 44100
