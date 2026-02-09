"""Regenerate separation validation plot using cached data.

Usage:
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot

    # Specific song
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot \
        --song "Apanhei-te Cavaquinho"

    # All songs with cache
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all
"""

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from vocal_analysis.features.extraction import extract_bioacoustic_features
from vocal_analysis.preprocessing.separation import HTDEMUCS_SAMPLE_RATE
from vocal_analysis.visualization.plots import plot_separation_validation


def _normalize_song_name(name: str) -> str:
    """Normalize song name for matching with files."""
    return re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")


def _parse_excerpt_interval(interval_str: str) -> tuple[float, float] | None:
    """Parse interval in 'MMSS-MMSS' format to seconds."""
    match = re.match(r"(\d{4})-(\d{4})", interval_str.strip("\"'"))
    if not match:
        return None

    def mmss_to_seconds(mmss: str) -> float:
        minutes = int(mmss[:2])
        seconds = int(mmss[2:])
        return minutes * 60 + seconds

    start = mmss_to_seconds(match.group(1))
    end = mmss_to_seconds(match.group(2))
    return start, end


def _get_excerpt_from_env(song_stem: str, project_root: Path) -> tuple[float, float] | None:
    """Look up excerpt interval from .env for a song."""
    song_key = _normalize_song_name(song_stem).upper()
    env_path = project_root / ".env"

    if not env_path.exists():
        return None

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.upper().startswith("EXCERPT_"):
                env_song = key.upper().replace("EXCERPT_", "")
                if (
                    song_key == env_song
                    or song_key.startswith(env_song)
                    or env_song.startswith(song_key)
                ):
                    return _parse_excerpt_interval(value.split("#")[0].strip())

    return None


def regenerate_plot(
    song_name: str,
    project_root: Path,
    use_praat_f0: bool = True,
) -> Path | None:
    """Regenerate validation plot for a song using cache.

    Args:
        song_name: Song name (e.g.: "Apanhei-te Cavaquinho")
        project_root: Project root
        use_praat_f0: If True, use Praat (fast). If False, use CREPE (slow, accurate).

    Returns:
        Path of the generated plot or None if failed.
    """
    song_stem = _normalize_song_name(song_name)

    # Find files
    audio_files = list((project_root / "data" / "raw").glob("*.mp3"))
    audio_path = None
    for f in audio_files:
        if _normalize_song_name(f.stem) == song_stem:
            audio_path = f
            break

    if not audio_path:
        print(f"❌ Audio not found for: {song_name}")
        return None

    cache_file = project_root / "data" / "cache" / "separated" / f"{song_stem}_vocals.npy"
    if not cache_file.exists():
        print(f"❌ Cache not found: {cache_file}")
        return None

    output_path = project_root / "outputs" / "plots" / f"{song_stem}_separation_validation.png"

    # Look up excerpt from .env
    excerpt = _get_excerpt_from_env(song_stem, project_root)
    start_time, end_time = excerpt if excerpt else (None, None)

    if excerpt:
        print(f"📊 {song_name} (excerpt: {start_time:.0f}s-{end_time:.0f}s)")
    else:
        print(f"📊 {song_name} (full audio)")

    # Extract features from original
    print("  Extracting features from original audio...")
    orig_features = extract_bioacoustic_features(
        audio_path,
        skip_formants=True,
        skip_jitter_shimmer=True,
        skip_cpps=True,
        use_praat_f0=use_praat_f0,
    )

    # Load separated vocals from cache
    print("  Loading separated vocals from cache...")
    vocals = np.load(cache_file)

    # Create temporary WAV for extraction
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, vocals, HTDEMUCS_SAMPLE_RATE)
        temp_path = Path(f.name)

    try:
        print("  Extracting features from separated vocals...")
        sep_features = extract_bioacoustic_features(
            temp_path,
            skip_formants=True,
            skip_jitter_shimmer=True,
            skip_cpps=True,
            use_praat_f0=use_praat_f0,
        )

        print("  Generating plot...")
        plot_separation_validation(
            time_original=orig_features["time"],
            f0_original=orig_features["f0"],
            confidence_original=orig_features["confidence"],
            time_separated=sep_features["time"],
            f0_separated=sep_features["f0"],
            confidence_separated=sep_features["confidence"],
            title=f"Separation Validation - {song_stem}",
            save_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"  ✅ Plot salvo: {output_path.name}")
        return output_path

    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate separation validation plots using cached data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--song",
        type=str,
        help="Song name (e.g.: 'Apanhei-te Cavaquinho'). Lists songs if omitted.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate plots for all songs with available cache",
    )
    parser.add_argument(
        "--use-crepe",
        action="store_true",
        help="Use CREPE instead of Praat for f0 (slower, more accurate)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / "data" / "cache" / "separated"

    if not cache_dir.exists():
        print("❌ Cache directory not found. Run first:")
        print("   uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals")
        return

    cached_files = list(cache_dir.glob("*_vocals.npy"))
    if not cached_files:
        print("❌ No separation cache found.")
        return

    if args.all:
        print(f"Regenerating plots for {len(cached_files)} songs...\n")
        for cache_file in cached_files:
            song_stem = cache_file.stem.replace("_vocals", "")
            regenerate_plot(song_stem, project_root, use_praat_f0=not args.use_crepe)
            print()
    elif args.song:
        regenerate_plot(args.song, project_root, use_praat_f0=not args.use_crepe)
    else:
        print("Songs with available cache:")
        for cache_file in cached_files:
            song_stem = cache_file.stem.replace("_vocals", "")
            print(f"  - {song_stem}")
        print("\nUsage:")
        print("  uv run python -m vocal_analysis.scripts.regenerate_validation_plot --song 'nome'")
        print("  uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all")


if __name__ == "__main__":
    main()
