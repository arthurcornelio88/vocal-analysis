"""Source separation para isolar voz usando HTDemucs via torchaudio."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torchaudio

if TYPE_CHECKING:
    pass

# Tipo para stems disponíveis no HTDemucs
StemName = Literal["drums", "bass", "other", "vocals"]

# Sample rate do modelo HTDemucs
HTDEMUCS_SAMPLE_RATE = 44100


class VocalSeparator:
    """Separador de voz usando HTDemucs via torchaudio.

    O modelo HTDemucs é carregado sob demanda (lazy loading) para evitar
    overhead na inicialização quando a separação não é necessária.

    Attributes:
        device: Dispositivo para inferência ('cpu' ou 'cuda').
        segment_seconds: Tamanho dos chunks em segundos para processamento.
        overlap: Sobreposição entre chunks (0.0-0.5).
    """

    def __init__(
        self,
        device: str = "cpu",
        segment_seconds: float = 10.0,
        overlap: float = 0.1,
    ) -> None:
        """Inicializa o separador.

        Args:
            device: 'cpu' ou 'cuda' para GPU.
            segment_seconds: Tamanho dos chunks em segundos (para controle de memória).
            overlap: Sobreposição entre chunks (0.0-0.5).
        """
        self.device = torch.device(device)
        self.segment_seconds = segment_seconds
        self.overlap = overlap

        # Lazy loading do modelo
        self._model: torch.nn.Module | None = None
        self._sample_rate: int | None = None

    @property
    def model(self) -> torch.nn.Module:
        """Carrega modelo sob demanda (lazy loading)."""
        if self._model is None:
            print("  Carregando modelo HTDemucs (primeira execução baixa ~1GB)...")
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            self._model = bundle.get_model().to(self.device)
            self._model.eval()
            self._sample_rate = bundle.sample_rate  # 44100
            print(f"  Modelo carregado (sample rate: {self._sample_rate} Hz)")
        return self._model

    @property
    def sample_rate(self) -> int:
        """Sample rate do modelo (44100 Hz)."""
        _ = self.model  # Garante que modelo foi carregado
        return self._sample_rate  # type: ignore[return-value]

    @property
    def sources(self) -> list[str]:
        """Lista de stems disponíveis no modelo."""
        return list(self.model.sources)  # type: ignore[attr-defined]

    def separate(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int = HTDEMUCS_SAMPLE_RATE,
    ) -> dict[StemName, np.ndarray]:
        """Separa áudio em stems (drums, bass, other, vocals).

        Args:
            audio: Array mono ou stereo (samples,) ou (2, samples).
            sr: Sample rate do áudio de entrada.

        Returns:
            Dict com stems separados como numpy arrays mono (samples,).
        """
        # Converter para tensor se necessário
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))

        # Garantir formato stereo (2, samples) para o modelo
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)  # Mono -> Stereo
        elif audio.dim() == 2 and audio.shape[0] != 2:
            # Shape (samples, channels) -> (channels, samples)
            audio = audio.T

        # Resample se necessário
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Adicionar dimensão batch: (1, 2, samples)
        audio = audio.unsqueeze(0).to(self.device)

        # Separar com chunking para evitar OOM
        with torch.no_grad():
            sources = self._separate_chunked(audio)

        # Converter para dict de numpy arrays mono
        stem_names = self.sources
        result: dict[StemName, np.ndarray] = {}
        for i, name in enumerate(stem_names):
            stem = sources[0, i]  # Remove batch dim: (2, samples)
            stem_mono = stem.mean(dim=0).cpu().numpy()  # Stereo -> Mono
            result[name] = stem_mono  # type: ignore[literal-required]

        return result

    def extract_vocals(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int = HTDEMUCS_SAMPLE_RATE,
    ) -> np.ndarray:
        """Extrai apenas a pista de voz.

        Args:
            audio: Array mono ou stereo.
            sr: Sample rate.

        Returns:
            Array mono com voz isolada (samples,).
        """
        stems = self.separate(audio, sr)
        return stems["vocals"]

    def _separate_chunked(self, audio: torch.Tensor) -> torch.Tensor:
        """Processa áudio em chunks para evitar estouro de memória.

        Implementação baseada no tutorial do torchaudio para HTDemucs.

        Args:
            audio: Tensor (batch, channels, samples).

        Returns:
            Tensor (batch, sources, channels, samples).
        """
        batch, channels, length = audio.shape
        segment_length = int(self.segment_seconds * self.sample_rate)
        overlap_length = int(segment_length * self.overlap)
        stride = segment_length - overlap_length

        # Se o áudio é menor que um segmento, processa direto
        if length <= segment_length:
            return self.model(audio)  # type: ignore[no-any-return]

        # Processar em chunks com overlap
        num_sources = len(self.sources)
        output = torch.zeros(batch, num_sources, channels, length, device=self.device)
        weight = torch.zeros(length, device=self.device)

        # Janela de fade para suavizar transições
        fade_length = overlap_length
        fade_in = torch.linspace(0, 1, fade_length, device=self.device)
        fade_out = torch.linspace(1, 0, fade_length, device=self.device)

        start = 0
        while start < length:
            end = min(start + segment_length, length)
            chunk = audio[:, :, start:end]

            # Pad se necessário
            if chunk.shape[2] < segment_length:
                pad_length = segment_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_length))

            # Processar chunk
            chunk_output = self.model(chunk)

            # Remover padding se aplicado
            if end - start < segment_length:
                chunk_output = chunk_output[:, :, :, : end - start]

            # Aplicar fade e acumular
            chunk_length = chunk_output.shape[3]
            chunk_weight = torch.ones(chunk_length, device=self.device)

            # Fade in no início (exceto primeiro chunk)
            if start > 0:
                chunk_output[:, :, :, :fade_length] *= fade_in
                chunk_weight[:fade_length] *= fade_in

            # Fade out no final (exceto último chunk)
            if end < length:
                chunk_output[:, :, :, -fade_length:] *= fade_out
                chunk_weight[-fade_length:] *= fade_out

            output[:, :, :, start:end] += chunk_output
            weight[start:end] += chunk_weight

            start += stride

        # Normalizar pelo peso acumulado
        output /= weight.view(1, 1, 1, -1).clamp(min=1e-8)

        return output


def _get_file_hash(file_path: Path) -> str:
    """Calcula hash MD5 do arquivo para invalidação de cache."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Ler apenas os primeiros 64KB para hash rápido
        hasher.update(f.read(65536))
    return hasher.hexdigest()[:12]


def separate_vocals(
    audio_path: str | Path,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, int]:
    """Função de conveniência para separar voz de um arquivo.

    Args:
        audio_path: Caminho do arquivo de áudio.
        device: 'cpu' ou 'cuda'.
        cache_dir: Diretório para cachear resultados (opcional).

    Returns:
        Tuple (vocals_array, sample_rate).
    """
    import librosa

    audio_path = Path(audio_path)

    # Verificar cache
    if cache_dir:
        cache_file = cache_dir / f"{audio_path.stem}_vocals.npy"
        if cache_file.exists():
            print(f"  Cache encontrado: {cache_file.name}")
            vocals = np.load(cache_file)
            return vocals, HTDEMUCS_SAMPLE_RATE

    # Carregar áudio usando librosa (mais robusto que torchaudio.load)
    # Mantém stereo se disponível, resamplea para 44.1kHz
    waveform, sr = librosa.load(str(audio_path), sr=HTDEMUCS_SAMPLE_RATE, mono=False)

    # librosa retorna (samples,) para mono ou (channels, samples) para stereo
    # Garantir formato consistente
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]  # (1, samples)

    # Separar
    separator = VocalSeparator(device=device)
    vocals = separator.extract_vocals(waveform, sr)

    # Salvar cache
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vocals)
        print(f"  Cache salvo: {cache_file.name}")

    return vocals, HTDEMUCS_SAMPLE_RATE


def separate_vocals_safe(
    audio_path: Path,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple[np.ndarray | None, int, bool]:
    """Versão segura com fallback automático.

    Args:
        audio_path: Caminho do arquivo de áudio.
        device: 'cpu' ou 'cuda'.
        cache_dir: Diretório para cachear resultados.

    Returns:
        Tuple (vocals_or_none, sample_rate, success_flag).
    """
    try:
        vocals, sr = separate_vocals(audio_path, device, cache_dir)
        return vocals, sr, True
    except torch.cuda.OutOfMemoryError:
        print("  GPU sem memória. Tentando CPU...")
        try:
            vocals, sr = separate_vocals(audio_path, "cpu", cache_dir)
            return vocals, sr, True
        except Exception as e:
            print(f"  Falha na separação (CPU): {e}")
            return None, HTDEMUCS_SAMPLE_RATE, False
    except Exception as e:
        print(f"  Falha na separação: {e}")
        return None, HTDEMUCS_SAMPLE_RATE, False
