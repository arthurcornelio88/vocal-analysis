"""Source separation to isolate vocals using HTDemucs via torchaudio."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torchaudio

if TYPE_CHECKING:
    pass

# Type for available stems in HTDemucs
StemName = Literal["drums", "bass", "other", "vocals"]

# HTDemucs model sample rate
HTDEMUCS_SAMPLE_RATE = 44100


class VocalSeparator:
    """Vocal separator using HTDemucs via torchaudio.

    The HTDemucs model is loaded on demand (lazy loading) to avoid
    overhead during initialization when separation is not needed.

    Attributes:
        device: Device for inference ('cpu' or 'cuda').
        segment_seconds: Chunk size in seconds for processing.
        overlap: Overlap between chunks (0.0-0.5).
    """

    def __init__(
        self,
        device: str = "cpu",
        segment_seconds: float = 10.0,
        overlap: float = 0.1,
    ) -> None:
        """Initialize the separator.

        Args:
            device: 'cpu' or 'cuda' for GPU.
            segment_seconds: Chunk size in seconds (for memory control).
            overlap: Overlap between chunks (0.0-0.5).
        """
        self.device = torch.device(device)
        self.segment_seconds = segment_seconds
        self.overlap = overlap

        # Lazy loading of the model
        self._model: torch.nn.Module | None = None
        self._sample_rate: int | None = None

    @property
    def model(self) -> torch.nn.Module:
        """Load model on demand (lazy loading)."""
        if self._model is None:
            print("  Loading HTDemucs model (first run downloads ~1GB)...")
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            self._model = bundle.get_model().to(self.device)
            self._model.eval()
            self._sample_rate = bundle.sample_rate  # 44100
            print(f"  Model loaded (sample rate: {self._sample_rate} Hz)")
        return self._model

    @property
    def sample_rate(self) -> int:
        """Model sample rate (44100 Hz)."""
        _ = self.model  # Ensure model is loaded
        return self._sample_rate  # type: ignore[return-value]

    @property
    def sources(self) -> list[str]:
        """List of available stems in the model."""
        return list(self.model.sources)  # type: ignore[attr-defined]

    def separate(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int = HTDEMUCS_SAMPLE_RATE,
    ) -> dict[StemName, np.ndarray]:
        """Separate audio into stems (drums, bass, other, vocals).

        Args:
            audio: Mono or stereo array (samples,) or (2, samples).
            sr: Input audio sample rate.

        Returns:
            Dict with separated stems as mono numpy arrays (samples,).
        """
        # Convert to tensor if necessary
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))

        # Ensure stereo format (2, samples) for the model
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)  # Mono -> Stereo
        elif audio.dim() == 2 and audio.shape[0] != 2:
            # Shape (samples, channels) -> (channels, samples)
            audio = audio.T

        # Resample if necessary
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Add batch dimension: (1, 2, samples)
        audio = audio.unsqueeze(0).to(self.device)

        # Separate with chunking to avoid OOM
        with torch.no_grad():
            sources = self._separate_chunked(audio)

        # Convert to dict of mono numpy arrays
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
        """Extract only the vocal track.

        Args:
            audio: Mono or stereo array.
            sr: Sample rate.

        Returns:
            Mono array with isolated vocals (samples,).
        """
        stems = self.separate(audio, sr)
        return stems["vocals"]

    def _separate_chunked(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio in chunks to avoid memory overflow.

        Implementation based on the torchaudio tutorial for HTDemucs.

        Args:
            audio: Tensor (batch, channels, samples).

        Returns:
            Tensor (batch, sources, channels, samples).
        """
        batch, channels, length = audio.shape
        segment_length = int(self.segment_seconds * self.sample_rate)
        overlap_length = int(segment_length * self.overlap)
        stride = segment_length - overlap_length

        # If audio is shorter than one segment, process directly
        if length <= segment_length:
            return self.model(audio)  # type: ignore[no-any-return]

        # Process in chunks with overlap
        num_sources = len(self.sources)
        output = torch.zeros(batch, num_sources, channels, length, device=self.device)
        weight = torch.zeros(length, device=self.device)

        # Fade window to smooth transitions
        fade_length = overlap_length
        fade_in = torch.linspace(0, 1, fade_length, device=self.device)
        fade_out = torch.linspace(1, 0, fade_length, device=self.device)

        start = 0
        while start < length:
            end = min(start + segment_length, length)
            chunk = audio[:, :, start:end]

            # Pad if necessary
            if chunk.shape[2] < segment_length:
                pad_length = segment_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_length))

            # Process chunk
            chunk_output = self.model(chunk)

            # Remove padding if applied
            if end - start < segment_length:
                chunk_output = chunk_output[:, :, :, : end - start]

            # Apply fade and accumulate
            chunk_length = chunk_output.shape[3]
            chunk_weight = torch.ones(chunk_length, device=self.device)

            # Fade in at the beginning (except first chunk)
            if start > 0:
                chunk_output[:, :, :, :fade_length] *= fade_in
                chunk_weight[:fade_length] *= fade_in

            # Fade out at the end (except last chunk)
            if end < length:
                chunk_output[:, :, :, -fade_length:] *= fade_out
                chunk_weight[-fade_length:] *= fade_out

            output[:, :, :, start:end] += chunk_output
            weight[start:end] += chunk_weight

            start += stride

        # Normalize by accumulated weight
        output /= weight.view(1, 1, 1, -1).clamp(min=1e-8)

        return output


def _get_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of the file for cache invalidation."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read only the first 64KB for fast hashing
        hasher.update(f.read(65536))
    return hasher.hexdigest()[:12]


def separate_vocals(
    audio_path: str | Path,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, int]:
    """Convenience function to separate vocals from a file.

    Args:
        audio_path: Path to the audio file.
        device: 'cpu' or 'cuda'.
        cache_dir: Directory to cache results (optional).

    Returns:
        Tuple (vocals_array, sample_rate).
    """
    import librosa

    audio_path = Path(audio_path)

    # Check cache
    if cache_dir:
        cache_file = cache_dir / f"{audio_path.stem}_vocals.npy"
        if cache_file.exists():
            print(f"  Cache found: {cache_file.name}")
            vocals = np.load(cache_file)
            return vocals, HTDEMUCS_SAMPLE_RATE

    # Load audio using librosa (more robust than torchaudio.load)
    # Keeps stereo if available, resamples to 44.1kHz
    waveform, sr = librosa.load(str(audio_path), sr=HTDEMUCS_SAMPLE_RATE, mono=False)

    # librosa returns (samples,) for mono or (channels, samples) for stereo
    # Ensure consistent format
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]  # (1, samples)

    # Separate
    separator = VocalSeparator(device=device)
    vocals = separator.extract_vocals(waveform, sr)

    # Save cache
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vocals)
        print(f"  Cache saved: {cache_file.name}")

    return vocals, HTDEMUCS_SAMPLE_RATE


def separate_vocals_safe(
    audio_path: Path,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple[np.ndarray | None, int, bool]:
    """Safe version with automatic fallback.

    Args:
        audio_path: Path to the audio file.
        device: 'cpu' or 'cuda'.
        cache_dir: Directory to cache results.

    Returns:
        Tuple (vocals_or_none, sample_rate, success_flag).
    """
    try:
        vocals, sr = separate_vocals(audio_path, device, cache_dir)
        return vocals, sr, True
    except torch.cuda.OutOfMemoryError:
        print("  GPU out of memory. Trying CPU...")
        try:
            vocals, sr = separate_vocals(audio_path, "cpu", cache_dir)
            return vocals, sr, True
        except Exception as e:
            print(f"  Separation failed (CPU): {e}")
            return None, HTDEMUCS_SAMPLE_RATE, False
    except Exception as e:
        print(f"  Separation failed: {e}")
        return None, HTDEMUCS_SAMPLE_RATE, False
