"""Script to process Ademilde Fonseca recordings."""

import argparse
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vocal_analysis.features.extraction import (
    extract_bioacoustic_features,
    extract_extended_features,
)
from vocal_analysis.utils.pitch import hz_range_to_notes, hz_to_note


def _parse_excerpt_interval(interval_str: str) -> tuple[float, float] | None:
    """Parse interval in 'MMSS-MMSS' format to seconds.

    Args:
        interval_str: String in "0022-0103" format (from second 22 to 1:03).

    Returns:
        Tuple (start_seconds, end_seconds) or None if invalid.
    """
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


def _get_excerpt_from_env(song_stem: str) -> tuple[float, float] | None:
    """Look up excerpt interval from .env for a given song.

    Args:
        song_stem: Song name (file stem, e.g. "delicado", "apanheite_cavaquinho").

    Returns:
        Tuple (start, end) in seconds, or None if not found.
    """
    # Normalize name for lookup (e.g. "delicado" -> "DELICADO")
    song_key = song_stem.upper().replace("-", "_")

    # Try loading from .env (project root = 4 levels up from this file)
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if not env_path.exists():
        return None

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # Check if this is an EXCERPT_* variable matching the song
            if key.upper().startswith("EXCERPT_"):
                env_song = key.upper().replace("EXCERPT_", "")
                # Exact or partial match (e.g. "DELICADO" matches "DELICADO")
                if (
                    song_key == env_song
                    or song_key.startswith(env_song)
                    or env_song.startswith(song_key)
                ):
                    return _parse_excerpt_interval(value)

    return None


class ProcessingConfig:
    """Configuration to control which features to extract (debug mode)."""

    def __init__(
        self,
        skip_formants: bool = False,
        skip_plots: bool = False,
        skip_jitter_shimmer: bool = False,
        limit_files: int | None = None,
        use_praat_f0: bool = False,
        skip_cpps: bool = False,
        cpps_timeout: int | None = None,
        batch_size: int = 2048,
        crepe_model: str = "full",
        device: str = "cpu",
        separate_vocals: bool = False,
        separation_device: str | None = None,
        use_separation_cache: bool = True,
        validate_separation: bool = False,
        extract_spectral: bool = False,
        skip_cpps_per_frame: bool = True,
    ):
        self.skip_formants = skip_formants
        self.skip_plots = skip_plots
        self.skip_jitter_shimmer = skip_jitter_shimmer
        self.limit_files = limit_files
        self.use_praat_f0 = use_praat_f0
        self.skip_cpps = skip_cpps
        self.cpps_timeout = cpps_timeout
        self.batch_size = batch_size
        self.crepe_model = crepe_model
        self.device = device
        self.separate_vocals = separate_vocals
        self.separation_device = separation_device or device
        self.use_separation_cache = use_separation_cache
        self.validate_separation = validate_separation
        self.extract_spectral = extract_spectral
        self.skip_cpps_per_frame = skip_cpps_per_frame


def _generate_validation_plot(
    original_audio_path: Path,
    separated_features: dict,
    config: ProcessingConfig,
    output_dir: Path,
) -> None:
    """Generate comparison plot to validate vocal separation.

    Reads excerpt interval from .env (if available) to plot only
    the relevant segment instead of the full audio.

    Args:
        original_audio_path: Path to the original audio file.
        separated_features: Features extracted from separated vocals.
        config: Processing configuration.
        output_dir: Output directory.
    """
    from vocal_analysis.features.extraction import extract_bioacoustic_features
    from vocal_analysis.visualization.plots import plot_separation_validation

    # Look up excerpt interval from .env
    excerpt_interval = _get_excerpt_from_env(original_audio_path.stem)
    start_time, end_time = excerpt_interval if excerpt_interval else (None, None)

    if excerpt_interval:
        print(f"  Generating validation plot (excerpt: {start_time:.0f}s-{end_time:.0f}s)...")
    else:
        print("  Generating validation plot (full audio)...")

    # Extract features from original audio (for comparison)
    original_features = extract_bioacoustic_features(
        original_audio_path,
        skip_formants=True,  # Only need f0 for validation
        skip_jitter_shimmer=True,
        use_praat_f0=config.use_praat_f0,
        skip_cpps=True,
        batch_size=config.batch_size,
        model=config.crepe_model,
        device=config.device,
    )

    # Generate plot
    plot_path = output_dir / "plots" / f"{original_audio_path.stem}_separation_validation.png"
    plot_separation_validation(
        time_original=original_features["time"],
        f0_original=original_features["f0"],
        confidence_original=original_features["confidence"],
        time_separated=separated_features["time"],
        f0_separated=separated_features["f0"],
        confidence_separated=separated_features["confidence"],
        title=f"Separation Validation - {original_audio_path.stem}",
        save_path=plot_path,
        start_time=start_time,
        end_time=end_time,
    )
    print(f"  Validation plot saved: {plot_path.name}")


def process_audio_files(
    data_dir: Path, output_dir: Path, config: ProcessingConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """Process all audio files and extract features.

    Args:
        data_dir: Directory containing audio files.
        output_dir: Directory to save outputs.

    Returns:
        Tuple with features DataFrame and processing metadata.
    """
    if config is None:
        config = ProcessingConfig()

    audio_files = list(data_dir.glob("*.mp3"))

    # Limit number of files if specified (useful for debug)
    if config.limit_files:
        audio_files = audio_files[: config.limit_files]

    print(f"Found {len(audio_files)} audio files")
    if config.skip_formants:
        print("DEBUG: Formants DISABLED")
    if config.skip_plots:
        print("DEBUG: Plots DISABLED")
    if config.skip_jitter_shimmer:
        print("DEBUG: Jitter/Shimmer DISABLED")
    if config.use_praat_f0:
        print("DEBUG: Using Praat f0 (fast) instead of CREPE")
    else:
        print(f"Using CREPE model '{config.crepe_model}' for f0 extraction")
        if config.device == "cuda":
            print("GPU ENABLED (cuda) - accelerated processing!")
        else:
            print("CPU mode (slow processing - use GPU if available)")
    if config.skip_cpps:
        print("DEBUG: CPPS DISABLED (avoids hang on macOS)")
    if config.separate_vocals:
        print(f"SOURCE SEPARATION ENABLED (HTDemucs on {config.separation_device})")
        if config.validate_separation:
            print("Visual validation enabled (comparison plots)")
    if config.extract_spectral:
        print("SPECTRAL FEATURES ENABLED (Alpha Ratio, H1-H2, Spectral Tilt)")

    all_features = []
    songs_metadata = []

    # Cache directory for separation
    cache_dir = None
    if config.separate_vocals and config.use_separation_cache:
        cache_dir = output_dir.parent / "data" / "cache" / "separated"

    for audio_path in audio_files:
        print(f"\nProcessing: {audio_path.name}")

        # Source separation if enabled
        audio_path_for_features = audio_path
        temp_wav_path = None
        vocals_array = None

        if config.separate_vocals:
            from vocal_analysis.preprocessing.separation import separate_vocals_safe

            print("  Applying source separation (HTDemucs)...")
            vocals, sr, success = separate_vocals_safe(
                audio_path,
                device=config.separation_device,
                cache_dir=cache_dir,
            )

            if success and vocals is not None:
                # Create temporary WAV file (Praat needs a file)
                fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
                sf.write(temp_wav_path, vocals, sr)
                audio_path_for_features = Path(temp_wav_path)
                vocals_array = vocals
                print("  Vocals separated successfully")
            else:
                print("  Source separation failed, using original audio")

        try:
            if config.extract_spectral:
                # Use extended extraction with spectral features (VMI)
                features = extract_extended_features(
                    audio_path_for_features,
                    skip_formants=config.skip_formants,
                    skip_jitter_shimmer=config.skip_jitter_shimmer,
                    use_praat_f0=config.use_praat_f0,
                    skip_cpps=config.skip_cpps,
                    cpps_timeout=config.cpps_timeout,
                    batch_size=config.batch_size,
                    model=config.crepe_model,
                    device=config.device,
                    skip_cpps_per_frame=config.skip_cpps_per_frame,
                )
            else:
                features = extract_bioacoustic_features(
                    audio_path_for_features,
                    skip_formants=config.skip_formants,
                    skip_jitter_shimmer=config.skip_jitter_shimmer,
                    use_praat_f0=config.use_praat_f0,
                    skip_cpps=config.skip_cpps,
                    cpps_timeout=config.cpps_timeout,
                    batch_size=config.batch_size,
                    model=config.crepe_model,
                    device=config.device,
                )

            # Generate validation plot if enabled
            if config.validate_separation and config.separate_vocals and vocals_array is not None:
                _generate_validation_plot(audio_path, features, config, output_dir)

            # Create DataFrame for this song
            df_data = {
                "time": features["time"],
                "f0": features["f0"],
                "confidence": features["confidence"],
                "hnr": features["hnr"],
                "energy": features["energy"],
            }

            # Add formants only if not disabled
            if not config.skip_formants:
                df_data.update(
                    {
                        "f1": features["f1"],
                        "f2": features["f2"],
                        "f3": features["f3"],
                        "f4": features["f4"],
                    }
                )

            # Add spectral features if enabled
            if config.extract_spectral:
                df_data.update(
                    {
                        "alpha_ratio": features["alpha_ratio"],
                        "h1_h2": features["h1_h2"],
                        "spectral_tilt": features["spectral_tilt"],
                    }
                )
                # CPPS per-frame is optional (slow)
                if not config.skip_cpps_per_frame and features.get("cpps_per_frame") is not None:
                    df_data["cpps_per_frame"] = features["cpps_per_frame"]

            df = pd.DataFrame(df_data)
            df["song"] = audio_path.stem
            df["cpps_global"] = features["cpps_global"]

            if not config.skip_jitter_shimmer:
                df["jitter"] = features["jitter"]
                df["shimmer"] = features["shimmer"]

            # Filter frames with low confidence or silence
            # confidence > 0.85: CREPE periodicity threshold
            # hnr > 0: removes silence/noise (Praat returns -200 dB for unvoiced frames)
            df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)].copy()

            all_features.append(df_voiced)

            # Generate f0 plot (unless disabled)
            plot_path = None
            if not config.skip_plots:
                # Import only when needed to avoid matplotlib initialization overhead
                from vocal_analysis.visualization.plots import plot_f0_contour

                plot_path = output_dir / "plots" / f"{audio_path.stem}_f0.png"
                plot_f0_contour(
                    features["time"],
                    features["f0"],
                    features["confidence"],
                    title=f"f0 Contour - {audio_path.stem}",
                    save_path=plot_path,
                )

            # Song metadata (convert numpy types to Python native for JSON serialization)
            song_meta = {
                "song": audio_path.stem,
                "file": audio_path.name,
                "total_frames": int(len(df)),
                "voiced_frames": int(len(df_voiced)),
                "f0_mean_hz": float(round(df_voiced["f0"].mean(), 1)),
                "f0_mean_note": hz_to_note(df_voiced["f0"].mean()),
                "f0_min_hz": float(round(df_voiced["f0"].min(), 1)),
                "f0_max_hz": float(round(df_voiced["f0"].max(), 1)),
                "f0_range_notes": hz_range_to_notes(df_voiced["f0"].min(), df_voiced["f0"].max()),
                "f0_std_hz": float(round(df_voiced["f0"].std(), 1)),
                "hnr_mean_db": float(round(df_voiced["hnr"].mean(), 1)),
                "cpps_global": (
                    float(round(features["cpps_global"], 2))
                    if features["cpps_global"] is not None
                    else None
                ),
                "energy_mean": float(round(df_voiced["energy"].mean(), 4)),
            }

            # Add jitter/shimmer if not disabled
            if not config.skip_jitter_shimmer:
                song_meta["jitter_ppq5"] = (
                    float(round(features["jitter"], 4))
                    if not np.isnan(features["jitter"])
                    else None
                )
                song_meta["shimmer_apq11"] = (
                    float(round(features["shimmer"], 4))
                    if not np.isnan(features["shimmer"])
                    else None
                )

            # Add plot path if generated
            if plot_path:
                song_meta["plot_path"] = str(plot_path.relative_to(output_dir.parent))
            songs_metadata.append(song_meta)

            # Print summary
            print(f"  f0: {song_meta['f0_mean_hz']} Hz ({song_meta['f0_mean_note']})")
            print(f"  Range: {song_meta['f0_range_notes']}")
            print(f"  HNR: {song_meta['hnr_mean_db']} dB | CPPS: {song_meta['cpps_global']}")

            if not config.skip_jitter_shimmer:
                jitter_str = (
                    f"{song_meta['jitter_ppq5']}" if song_meta.get("jitter_ppq5") else "N/A"
                )
                shimmer_str = (
                    f"{song_meta['shimmer_apq11']}" if song_meta.get("shimmer_apq11") else "N/A"
                )
                print(f"  Jitter: {jitter_str} | Shimmer: {shimmer_str}")

        except Exception as e:
            print(f"  ERROR: {e}")
            songs_metadata.append(
                {
                    "song": audio_path.stem,
                    "file": audio_path.name,
                    "error": str(e),
                }
            )
        finally:
            # Clean up temporary file if created
            if temp_wav_path and Path(temp_wav_path).exists():
                Path(temp_wav_path).unlink()

    metadata = {
        "processed_at": datetime.now().isoformat(),
        "artist": "Ademilde Fonseca",
        "n_songs": len(audio_files),
        "n_success": len(all_features),
        "songs": songs_metadata,
    }

    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)

        # Add global stats to metadata (convert numpy types to Python native)
        df_voiced = df_all[(df_all["confidence"] > 0.85) & (df_all["hnr"] > 0)]
        metadata["global"] = {
            "total_voiced_frames": int(len(df_voiced)),
            "f0_mean_hz": float(round(df_voiced["f0"].mean(), 1)),
            "f0_mean_note": hz_to_note(df_voiced["f0"].mean()),
            "f0_min_hz": float(round(df_voiced["f0"].min(), 1)),
            "f0_max_hz": float(round(df_voiced["f0"].max(), 1)),
            "f0_range_notes": hz_range_to_notes(df_voiced["f0"].min(), df_voiced["f0"].max()),
            "f0_std_hz": float(round(df_voiced["f0"].std(), 1)),
            "hnr_mean_db": float(round(df_voiced["hnr"].mean(), 1)),
        }

        return df_all, metadata

    return pd.DataFrame(), metadata


def save_outputs(
    df: pd.DataFrame,
    metadata: dict,
    project_root: Path,
) -> None:
    """Save CSV, JSON and processing log.

    Args:
        df: Features DataFrame.
        metadata: Metadata dictionary.
        project_root: Project root directory.
    """
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # CSV with features
    csv_path = processed_dir / "ademilde_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # JSON with metadata
    json_path = processed_dir / "ademilde_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")

    # Markdown log
    log_path = processed_dir / "processing_log.md"
    _write_log_markdown(metadata, log_path)
    print(f"Log saved: {log_path}")


def _write_log_markdown(metadata: dict, path: Path) -> None:
    """Generate markdown processing log."""
    lines = [
        f"# Processing Log - {metadata['artist']}",
        "",
        f"**Date:** {metadata['processed_at']}",
        f"**Songs processed:** {metadata['n_success']}/{metadata['n_songs']}",
        "",
    ]

    if "global" in metadata:
        g = metadata["global"]
        lines.extend(
            [
                "## Global Summary",
                "",
                "| Metric | Value | Note |",
                "|--------|-------|------|",
                f"| Mean f0 | {g['f0_mean_hz']} Hz | {g['f0_mean_note']} |",
                f"| Min f0 | {g['f0_min_hz']} Hz | – |",
                f"| Max f0 | {g['f0_max_hz']} Hz | – |",
                f"| Range | – | {g['f0_range_notes']} |",
                f"| f0 Std Dev | {g['f0_std_hz']} Hz | – |",
                f"| Mean HNR | {g['hnr_mean_db']} dB | – |",
                f"| Total Frames | {g['total_voiced_frames']} | – |",
                "",
            ]
        )

    lines.extend(
        [
            "## Per Song",
            "",
        ]
    )

    for song in metadata["songs"]:
        if "error" in song:
            lines.append(f"### {song['song']}")
            lines.append(f"Error: {song['error']}")
        else:
            lines.extend(
                [
                    f"### {song['song']}",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Mean f0 | {song['f0_mean_hz']} Hz ({song['f0_mean_note']}) |",
                    f"| Range | {song['f0_range_notes']} |",
                    f"| HNR | {song['hnr_mean_db']} dB |",
                    f"| CPPS | {song['cpps_global']} |",
                    f"| Frames | {song['voiced_frames']}/{song['total_frames']} |",
                    "",
                ]
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Ademilde Fonseca audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Full processing
  python -m vocal_analysis.preprocessing.process_ademilde

  # Quick debug (no formants, plots, jitter/shimmer)
  python -m vocal_analysis.preprocessing.process_ademilde --skip-formants --skip-plots --skip-jitter-shimmer

  # Process only 1 file
  python -m vocal_analysis.preprocessing.process_ademilde --limit 1

  # Ultra-fast mode (f0 + HNR only)
  python -m vocal_analysis.preprocessing.process_ademilde --fast
        """,
    )

    parser.add_argument(
        "--skip-formants",
        action="store_true",
        help="Skip formant extraction (F1-F4) - saves ~30%% of time",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip f0 plot generation - saves I/O"
    )
    parser.add_argument(
        "--skip-jitter-shimmer",
        action="store_true",
        help="Skip Jitter/Shimmer (Praat Point Process) - saves ~20%% of time",
    )
    parser.add_argument("--limit", type=int, metavar="N", help="Process only the first N files")
    parser.add_argument(
        "--use-praat-f0",
        action="store_true",
        help="Use Praat (autocorrelation) for f0 instead of CREPE - MUCH faster but less accurate",
    )
    parser.add_argument(
        "--skip-cpps",
        action="store_true",
        help="Skip CPPS entirely (returns None)",
    )
    parser.add_argument(
        "--cpps-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Timeout in seconds for CPPS (None = no timeout). Use only if CPPS hangs on your system",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        metavar="SIZE",
        help="Batch size for CREPE (default: 2048 for GPU, use 512 for memory-limited macOS CPU)",
    )
    parser.add_argument(
        "--crepe-model",
        type=str,
        default="full",
        choices=["tiny", "small", "full"],
        help="CREPE model for f0 extraction (default: full). Use 'small' to save memory on macOS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for CREPE (default: cpu). Use 'cuda' for GPU (Google Colab, Windows with NVIDIA)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: enables --skip-formants, --skip-plots, --skip-jitter-shimmer, --skip-cpps, --use-praat-f0",
    )

    # Source separation arguments (enabled by default)
    parser.add_argument(
        "--no-separate-vocals",
        action="store_true",
        help="Disable source separation (HTDemucs). By default, vocal separation "
        "is enabled to improve pitch detection in complex arrangements.",
    )
    parser.add_argument(
        "--separation-device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for source separation (default: same as --device)",
    )
    parser.add_argument(
        "--no-separation-cache",
        action="store_true",
        help="Disable separated audio cache (forces reprocessing)",
    )
    parser.add_argument(
        "--validate-separation",
        action="store_true",
        help="Generate comparison plot original vs separated vocals (Hz + notes) for visual validation",
    )

    # VMI / Spectral features arguments
    parser.add_argument(
        "--extract-spectral",
        action="store_true",
        help="Extract spectral features (Alpha Ratio, H1-H2, Spectral Tilt) for VMI analysis. "
        "Required for the VMI pipeline in run_analysis.py.",
    )
    parser.add_argument(
        "--cpps-per-frame",
        action="store_true",
        help="Extract CPPS per-frame (slow). Requires --extract-spectral.",
    )

    args = parser.parse_args()

    # Fast mode enables all optimizations
    if args.fast:
        args.skip_formants = True
        args.skip_plots = True
        args.skip_jitter_shimmer = True
        args.use_praat_f0 = True
        args.skip_cpps = True

    config = ProcessingConfig(
        skip_formants=args.skip_formants,
        skip_plots=args.skip_plots,
        skip_jitter_shimmer=args.skip_jitter_shimmer,
        limit_files=args.limit,
        use_praat_f0=args.use_praat_f0,
        skip_cpps=args.skip_cpps,
        cpps_timeout=args.cpps_timeout,
        batch_size=args.batch_size,
        crepe_model=args.crepe_model,
        device=args.device,
        separate_vocals=not args.no_separate_vocals,
        separation_device=args.separation_device,
        use_separation_cache=not args.no_separation_cache,
        validate_separation=args.validate_separation,
        extract_spectral=args.extract_spectral,
        skip_cpps_per_frame=not args.cpps_per_frame,
    )

    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"

    # Ensure directories exist
    if not config.skip_plots:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    df, metadata = process_audio_files(data_dir, output_dir, config)

    if not df.empty:
        save_outputs(df, metadata, project_root)

        # Print final summary
        g = metadata["global"]
        print("\n" + "=" * 50)
        print(f"SUMMARY - {metadata['artist']}")
        print("=" * 50)
        print(f"Mean f0: {g['f0_mean_hz']} Hz ({g['f0_mean_note']})")
        print(f"Range: {g['f0_range_notes']}")
        print(f"HNR: {g['hnr_mean_db']} dB")


if __name__ == "__main__":
    main()
