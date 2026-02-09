"""Script to run the full exploratory analysis."""

import json
import os
import re
from pathlib import Path

# --- Ensure .env is loaded ---
try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback if dotenv is not installed
    def load_dotenv():
        pass


import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from vocal_analysis.analysis.exploratory import (
    analyze_mechanism_regions,
    analyze_mechanism_vmi,
    cluster_mechanisms,
    generate_report,
    generate_vmi_report,
)
from vocal_analysis.analysis.llm_report import generate_narrative_report
from vocal_analysis.features.articulation import (
    compute_articulation_features,
)
from vocal_analysis.features.spectral import extract_spectral_features
from vocal_analysis.modeling.classifier import train_mechanism_classifier
from vocal_analysis.visualization.plots import (
    plot_vmi_scatter,
    plot_xgb_mechanism_excerpt,
    plot_xgb_mechanism_timeline,
)


def parse_time_string(time_str: str) -> float:
    """Convert 'MMSS' string or raw seconds to float seconds."""
    if not isinstance(time_str, str):
        return float(time_str)

    clean_str = time_str.replace('"', "").replace("'", "").strip()

    # MMSS format (4 digits)
    if len(clean_str) == 4 and clean_str.isdigit():
        minutes = int(clean_str[:2])
        seconds = int(clean_str[2:])
        return minutes * 60 + seconds

    # Otherwise assume raw seconds
    try:
        return float(clean_str)
    except ValueError:
        return 0.0


def get_manual_excerpt_from_env(song_name: str) -> tuple[float, float] | None:
    """Look up manual excerpt interval from environment variables with name normalization."""
    # Sanitize: "Apanhei-te Cavaquinho" -> "APANHEITE_CAVAQUINHO"
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", song_name).upper()
    safe_name = re.sub(r"_+", "_", safe_name)  # Remove duplicate underscores

    env_key = f"EXCERPT_{safe_name}"
    env_value = os.environ.get(env_key)

    if env_value:
        try:
            start_str, end_str = env_value.split("-")
            start_time = parse_time_string(start_str.strip())
            end_time = parse_time_string(end_str.strip())
            return start_time, end_time
        except Exception as e:
            print(f"  Warning: Error reading ENV {env_key}='{env_value}': {e}")

    return None


def save_audio_excerpt(
    song_name: str, start_time: float, end_time: float, project_root: Path, output_dir: Path
) -> None:
    """Crop and save the audio excerpt corresponding to the selected interval."""
    raw_dir = project_root / "data" / "raw"
    audio_files = list(raw_dir.glob(f"*{song_name}*.mp3")) + list(
        raw_dir.glob(f"*{song_name}*.wav")
    )

    if not audio_files:
        return

    audio_path = audio_files[0]

    try:
        # Load only the desired segment
        y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time - start_time)
        out_path = output_dir / f"excerpt_{song_name}.wav"
        sf.write(out_path, y, sr)
        print(f"    Audio excerpt saved: {out_path.name}")
    except Exception as e:
        print(f"    Warning: Error saving audio excerpt: {e}")


def compute_spectral_features_for_df(
    df: pd.DataFrame,
    project_root: Path,
    hop_length: int = 220,
    sr: int = 44100,
    skip_cpps_per_frame: bool = True,
) -> pd.DataFrame:
    """Compute spectral features for each song in the DataFrame.

    Args:
        df: DataFrame with data (must have 'song' column).
        project_root: Project root to locate audio files.
        hop_length: Hop length used in the original extraction.
        sr: Sample rate.
        skip_cpps_per_frame: If True, skip CPPS per-frame (faster).

    Returns:
        DataFrame with spectral features added.
    """
    raw_dir = project_root / "data" / "raw"
    separated_dir = project_root / "data" / "separated"

    # Initialize columns
    df["alpha_ratio"] = np.nan
    df["h1_h2"] = np.nan
    df["spectral_tilt"] = np.nan
    if not skip_cpps_per_frame:
        df["cpps_per_frame"] = np.nan

    for song_name in df["song"].unique():
        print(f"  Computing spectral features: {song_name}...")

        # Find audio file (prefer separated if available)
        audio_path = None
        for pattern in [f"*{song_name}*_vocals.wav", f"*{song_name}*.wav", f"*{song_name}*.mp3"]:
            candidates = list(separated_dir.glob(pattern)) + list(raw_dir.glob(pattern))
            if candidates:
                audio_path = candidates[0]
                break

        if audio_path is None:
            print(f"    Audio not found for {song_name}")
            continue

        # Mask for this song
        song_mask = df["song"] == song_name
        song_df = df[song_mask]

        try:
            # Extract spectral features
            spectral = extract_spectral_features(
                audio_path=audio_path,
                f0=song_df["f0"].values,
                f1=song_df["f1"].values if "f1" in song_df.columns else None,
                hop_length=hop_length,
                sr=sr,
                skip_cpps_per_frame=skip_cpps_per_frame,
            )

            # Align sizes
            n_frames = len(song_df)
            for col in ["alpha_ratio", "h1_h2", "spectral_tilt"]:
                values = spectral[col]
                if len(values) >= n_frames:
                    df.loc[song_mask, col] = values[:n_frames]
                else:
                    # Pad with NaN if needed
                    padded = np.full(n_frames, np.nan)
                    padded[: len(values)] = values
                    df.loc[song_mask, col] = padded

            if not skip_cpps_per_frame and spectral["cpps_per_frame"] is not None:
                values = spectral["cpps_per_frame"]
                if len(values) >= n_frames:
                    df.loc[song_mask, "cpps_per_frame"] = values[:n_frames]

        except Exception as e:
            print(f"    Error extracting spectral features: {e}")

    return df


def main() -> None:
    """Run exploratory analysis on processed data."""
    # Load ENV from .env file if it exists
    load_dotenv()

    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "processed" / "ademilde_features.csv"
    metadata_path = project_root / "data" / "processed" / "ademilde_metadata.json"
    output_dir = project_root / "outputs"

    # Flag to use VMI (can be ENV or argument)
    use_vmi = os.environ.get("USE_VMI", "true").lower() == "true"

    # Report language (en or pt-BR)
    report_lang = os.environ.get("REPORT_LANG", "en")

    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)

    metadata = None
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Artist: {metadata.get('artist', 'Unknown')}")

    print(f"  Total Frames: {len(df)}")

    # Compute articulatory agility features
    print("\nComputing articulatory agility features...")
    df = compute_articulation_features(df)

    # Threshold analysis
    print("\nAnalyzing by threshold (400 Hz / G4)...")
    stats = analyze_mechanism_regions(df, threshold_hz=400.0, output_dir=output_dir / "plots")

    # Clustering
    print("\nRunning GMM clustering...")
    plots_dir = output_dir / "plots"
    df_clustered = cluster_mechanisms(df, n_clusters=2, method="gmm", output_dir=plots_dir)

    # VMI Analysis (if enabled)
    vmi_stats = None
    if use_vmi:
        print("\n" + "=" * 50)
        print("VMI ANALYSIS (Vocal Mechanism Index)")
        print("=" * 50)

        # Check if spectral features are already in CSV
        spectral_cols = ["alpha_ratio", "h1_h2", "spectral_tilt"]
        has_spectral = all(col in df_clustered.columns for col in spectral_cols)

        if not has_spectral:
            print("\nComputing spectral features (Alpha Ratio, H1-H2, Spectral Tilt)...")
            df_clustered = compute_spectral_features_for_df(
                df_clustered,
                project_root,
                hop_length=220,
                skip_cpps_per_frame=True,  # CPPS per-frame is slow
            )

        # Check if we have valid features
        valid_spectral = df_clustered[spectral_cols].notna().any().all()

        if valid_spectral:
            print("\nComputing VMI...")
            try:
                df_vmi, vmi_stats = analyze_mechanism_vmi(
                    df_clustered,
                    smoothing_method="median",
                    smoothing_window=5,
                    output_dir=plots_dir,
                )

                # Copy VMI to df_clustered
                df_clustered["vmi"] = df_vmi["vmi"]
                df_clustered["vmi_label"] = df_vmi["vmi_label"]

                print("\n  VMI Distribution:")
                for label, s in vmi_stats.items():
                    print(f"    {label}: {s['count']} frames ({s['percentage']:.1f}%)")

                # VMI scatter plot
                try:
                    vmi_scatter_path = plots_dir / "vmi_scatter.png"
                    plot_vmi_scatter(
                        df_clustered,
                        x_col="f0",
                        y_col="alpha_ratio",
                        color_col="vmi",
                        save_path=vmi_scatter_path,
                    )
                    print(f"\n  VMI plot saved: {vmi_scatter_path.name}")
                except Exception as e:
                    print(f"  Error generating VMI plot: {e}")

                # Generate VMI report
                artist_name = metadata.get("artist", "Unknown") if metadata else "Unknown"
                vmi_report_path = output_dir / "vmi_analysis.md"
                generate_vmi_report(
                    df_clustered,
                    vmi_stats,
                    vmi_report_path,
                    artist_name=artist_name,
                    lang=report_lang,
                )
                print(f"  VMI report saved: {vmi_report_path.name}")

            except Exception as e:
                print(f"  Error in VMI analysis: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("  Spectral features not available. Skipping VMI analysis.")
            print("  (Check that audio files are in data/raw or data/separated)")

        print("=" * 50 + "\n")

    # XGBoost
    print("\nTraining XGBoost with GMM pseudo-labels...")
    base_cols = ["f0", "hnr", "energy", "f0_velocity", "f0_acceleration"]
    optional_cols = ["f1", "f2", "f3", "f4"]
    feature_cols = base_cols + [c for c in optional_cols if c in df_clustered.columns]

    df_train = df_clustered[feature_cols].copy()
    df_train["mechanism_label"] = df_clustered["mechanism"].map({"M1": 0, "M2": 1})

    xgb_report = None
    try:
        model, xgb_report = train_mechanism_classifier(
            df_train, feature_cols=feature_cols, target_col="mechanism_label"
        )

        # Predict and assign to DataFrame
        predictions = model.predict(df_clustered[feature_cols])
        df_clustered["xgb_mechanism"] = predictions
        df_clustered["xgb_mechanism"] = df_clustered["xgb_mechanism"].map({0: "M1", 1: "M2"})

        # Save predictions
        pred_path = output_dir / "xgb_predictions.csv"
        df_clustered.to_csv(pred_path, index=False)
        print(f"  Predictions saved: {pred_path}")

        # Timeline plot
        timeline_path = plots_dir / "xgb_mechanism_timeline.png"
        plot_xgb_mechanism_timeline(df_clustered, save_path=timeline_path)
        print("  Timeline plot generated.")

        # Excerpts
        print("\nGenerating excerpts per song...")

        for song_name in df_clustered["song"].unique():
            song_df = df_clustered[df_clustered["song"] == song_name].sort_values("time")

            if song_df.empty:
                continue

            t_min_song = song_df["time"].min()
            t_max_song = song_df["time"].max()

            # 1. Try to get interval from environment variable
            manual_excerpt = get_manual_excerpt_from_env(song_name)

            if manual_excerpt:
                best_start, best_end = manual_excerpt
                print(f"  > {song_name}: ENV FOUND ({best_start:.1f}s - {best_end:.1f}s)")

                if best_start > t_max_song:
                    print(
                        f"    Warning: Start {best_start}s > Song duration ({t_max_song:.1f}s). Skipping."
                    )
                    continue

                # Count frames for logging
                best_count = len(
                    song_df[(song_df["time"] >= best_start) & (song_df["time"] < best_end)]
                )

            else:
                # 2. Automatic (density-based)
                print(f"  > {song_name}: Automatic (density)...")
                best_start = t_min_song
                best_count = 0

                if t_max_song - 5 <= t_min_song:
                    search_range = [t_min_song]
                else:
                    search_range = np.arange(t_min_song, t_max_song - 5, 0.5)

                for t in search_range:
                    count = len(song_df[(song_df["time"] >= t) & (song_df["time"] < t + 5)])
                    if count > best_count:
                        best_count = count
                        best_start = t
                best_end = best_start + 5.0

            # Generate plot
            excerpt_path = plots_dir / f"excerpt_{song_name}.png"
            plot_xgb_mechanism_excerpt(
                df_clustered,
                song=song_name,
                start_time=best_start,
                end_time=best_end,
                save_path=excerpt_path,
            )
            print(f"    Plot saved: {best_start:.1f}s - {best_end:.1f}s")

            # Save audio excerpt
            save_audio_excerpt(song_name, best_start, best_end, project_root, output_dir)

    except Exception as e:
        print(f"  Critical error in XGBoost/Plots: {e}")
        import traceback

        traceback.print_exc()

    # Generate basic report
    artist_name = metadata.get("artist", "Unknown") if metadata else "Unknown"
    report_path = output_dir / "analysis_report.md"
    generate_report(
        df,
        stats,
        report_path,
        artist_name=artist_name,
        xgb_report=xgb_report,
        xgb_feature_cols=feature_cols,
        lang=report_lang,
    )

    # Generate LLM report
    if os.environ.get("GEMINI_API_KEY"):
        llm_report_path = output_dir / "llm_report.md"
        print("\nGenerating narrative report with Gemini...")
        plot_paths = list((output_dir / "plots").glob("*.png"))

        try:
            generate_narrative_report(
                stats, metadata, llm_report_path, plot_paths=plot_paths, lang=report_lang
            )
            print("  LLM report generated successfully!")
        except Exception as e:
            print(f"  Error generating LLM report: {e}")
    else:
        print("\n(Tip: Set GEMINI_API_KEY to generate an AI-powered report)")

    print("\nDone!")


if __name__ == "__main__":
    main()
