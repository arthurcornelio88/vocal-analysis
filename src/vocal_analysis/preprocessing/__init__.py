"""Audio preprocessing module."""

from vocal_analysis.preprocessing.audio import load_audio, normalize_audio
from vocal_analysis.preprocessing.separation import (
    VocalSeparator,
    separate_vocals,
    separate_vocals_safe,
)

__all__ = [
    "normalize_audio",
    "load_audio",
    "VocalSeparator",
    "separate_vocals",
    "separate_vocals_safe",
]
