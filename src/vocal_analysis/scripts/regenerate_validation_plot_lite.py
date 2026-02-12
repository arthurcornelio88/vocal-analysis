"""Regenerate separation validation plots from CSV + original MP3 (no HTDemucs needed).

Reads separated vocal f0/confidence from xgb_predictions.csv,
extracts original mix f0 via Praat (fast), and regenerates the plots.

Usage:
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot_lite
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot_lite --song delicado
"""

import argparse
from pathlib import Path

import pandas as pd

from vocal_analysis.features.extraction import extract_bioacoustic_features
from vocal_analysis.visualization.plots import plot_separation_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate validation plots (lite)")
    parser.add_argument("--song", type=str, help="Song stem (e.g. 'delicado'). All if omitted.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    csv_path = project_root / "outputs" / "xgb_predictions.csv"
    raw_dir = project_root / "data" / "raw"
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    songs = df["song"].unique()

    if args.song:
        songs = [s for s in songs if args.song.lower() in s.lower()]
        if not songs:
            print(f"No song matching '{args.song}' in CSV.")
            return

    for song in songs:
        print(f"\n--- {song} ---")

        # Find original MP3
        mp3_path = raw_dir / f"{song}.mp3"
        if not mp3_path.exists():
            print(f"  MP3 not found: {mp3_path}")
            continue

        # Extract f0 from original mix via Praat (fast)
        print("  Extracting f0 from original mix (Praat)...")
        orig_features = extract_bioacoustic_features(
            mp3_path,
            skip_formants=True,
            skip_jitter_shimmer=True,
            skip_cpps=True,
            use_praat_f0=True,
        )

        # Get separated vocal data from CSV
        song_df = df[df["song"] == song].sort_values("time")
        time_sep = song_df["time"].values
        f0_sep = song_df["f0"].values
        conf_sep = song_df["confidence"].values

        # Generate plot
        output_path = plot_dir / f"{song}_separation_validation.png"
        print("  Generating plot...")
        plot_separation_validation(
            time_original=orig_features["time"],
            f0_original=orig_features["f0"],
            confidence_original=orig_features["confidence"],
            time_separated=time_sep,
            f0_separated=f0_sep,
            confidence_separated=conf_sep,
            title=f"Separation Validation - {song}",
            save_path=output_path,
        )
        print(f"  Saved: {output_path.name}")


if __name__ == "__main__":
    main()
