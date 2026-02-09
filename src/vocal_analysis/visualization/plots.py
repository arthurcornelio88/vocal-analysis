"""Academic-style plots for publication."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_mechanism_clusters(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of Pitch vs HNR to visualize M1/M2 clusters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(
        data=df,
        x="f0",
        y="hnr",
        hue="mechanism",
        alpha=0.6,
        palette="viridis",
        ax=ax,
    )

    ax.set_title("Spectral Distribution: Mechanism 1 vs Mechanism 2")
    ax.set_xlabel("Fundamental Frequency (Hz)")
    ax.set_ylabel("Harmonic-to-Noise Ratio (dB)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_xgb_mechanism_timeline(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> sns.FacetGrid:
    """Temporal f0 contour colored by XGBoost prediction, split by song.

    Uses FacetGrid to avoid overlapping different songs on the same axis.
    """
    sns.set_theme(style="whitegrid")

    palette = {"M1": "steelblue", "M2": "coral"}

    g = sns.FacetGrid(
        df, row="song", hue="xgb_mechanism", palette=palette, aspect=4, height=2.5, sharex=False
    )

    g.map(plt.scatter, "time", "f0", s=1.5, alpha=0.7)
    g.add_legend(title="Mechanism")

    g.set_axis_labels("Time (s)", "f0 (Hz)")
    g.set_titles(row_template="{row_name}")

    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(g.figure)

    return g


def plot_xgb_mechanism_excerpt(
    df: pd.DataFrame,
    song: str,
    start_time: float,
    end_time: float,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Song excerpt with f0 colored by XGBoost prediction and note labels on Y axis."""
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    subset = df[(df["song"] == song) & (df["time"] >= start_time) & (df["time"] <= end_time)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.set_theme(style="whitegrid")

    colors = {"M1": "steelblue", "M2": "coral"}

    if not subset.empty:
        for mech in ["M1", "M2"]:
            mech_data = subset[subset["xgb_mechanism"] == mech]
            if not mech_data.empty:
                ax.scatter(
                    mech_data["time"],
                    mech_data["f0"],
                    c=colors[mech],
                    s=15,
                    alpha=0.8,
                    label=mech,
                    zorder=2,
                )

    # Y-axis configuration (note labels)
    f0_min = subset["f0"].min() if not subset.empty else 100
    f0_max = subset["f0"].max() if not subset.empty else 800

    f0_min = max(f0_min * 0.9, 50)
    f0_max = f0_max * 1.1
    ax.set_ylim(f0_min, f0_max)

    # Fixed X-axis to requested window
    ax.set_xlim(start_time, end_time)

    # Generate note ticks
    midi_min = int(np.floor(hz_to_midi(f0_min)))
    midi_max = int(np.ceil(hz_to_midi(f0_max)))
    note_ticks_hz = []
    note_ticks_labels = []
    for midi in range(midi_min, midi_max + 1):
        hz = float(midi_to_hz(midi))
        if f0_min <= hz <= f0_max:
            note_ticks_hz.append(hz)
            note_ticks_labels.append(hz_to_note(hz))

    ax2 = ax.twinx()
    ax2.set_ylim(f0_min, f0_max)
    ax2.set_yticks(note_ticks_hz)
    ax2.set_yticklabels(note_ticks_labels, fontsize=9)
    ax2.set_ylabel("Note")

    for hz in note_ticks_hz:
        ax.axhline(hz, color="gray", linewidth=0.3, alpha=0.5, zorder=1)

    ax.set_title(f"{song} — {start_time:.1f}s to {end_time:.1f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")

    if not subset.empty:
        ax.legend(loc="upper left")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_f0_contour(
    time: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray | None = None,
    title: str = "f0 Contour",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot f0 contour over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.set_theme(style="whitegrid")

    if confidence is not None:
        scatter = ax.scatter(time, f0, c=confidence, cmap="viridis", s=2, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Confidence")
    else:
        ax.plot(time, f0, linewidth=0.8, color="steelblue")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_separation_validation(
    time_original: np.ndarray,
    f0_original: np.ndarray,
    confidence_original: np.ndarray,
    time_separated: np.ndarray,
    f0_separated: np.ndarray,
    confidence_separated: np.ndarray,
    title: str = "Validation: Original vs Separated Vocals",
    save_path: str | Path | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> plt.Figure:
    """Comparison plot showing f0 before/after source separation.

    Allows visual validation that the separation is capturing the voice
    and not other instruments (cavaquinho, flute, etc.).

    Args:
        time_original: Time array from original audio.
        f0_original: f0 array from original audio.
        confidence_original: CREPE confidence array from original.
        time_separated: Time array from separated vocals.
        f0_separated: f0 array from separated vocals.
        confidence_separated: CREPE confidence array from separated vocals.
        title: Plot title.
        save_path: Path to save the plot (optional).
        start_time: Excerpt start time (seconds). If None, uses full audio.
        end_time: Excerpt end time (seconds). If None, uses full audio.

    Returns:
        Matplotlib figure.
    """
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    # Filter by time interval if specified
    if start_time is not None and end_time is not None:
        mask_time_orig = (time_original >= start_time) & (time_original <= end_time)
        mask_time_sep = (time_separated >= start_time) & (time_separated <= end_time)

        time_original = time_original[mask_time_orig]
        f0_original = f0_original[mask_time_orig]
        confidence_original = confidence_original[mask_time_orig]

        time_separated = time_separated[mask_time_sep]
        f0_separated = f0_separated[mask_time_sep]
        confidence_separated = confidence_separated[mask_time_sep]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    sns.set_theme(style="whitegrid")

    conf_threshold = 0.5

    # Compute f0 range for both plots
    f0_all = np.concatenate(
        [
            f0_original[confidence_original > conf_threshold],
            f0_separated[confidence_separated > conf_threshold],
        ]
    )
    if len(f0_all) > 0:
        f0_min = max(f0_all.min() * 0.9, 50)
        f0_max = f0_all.max() * 1.1
    else:
        f0_min, f0_max = 100, 800

    # Generate note ticks (natural notes only for readability)
    midi_min = int(np.floor(hz_to_midi(f0_min)))
    midi_max = int(np.ceil(hz_to_midi(f0_max)))
    note_ticks_hz = []
    note_ticks_labels = []
    for midi in range(midi_min, midi_max + 1):
        hz = float(midi_to_hz(midi))
        note = hz_to_note(hz)
        # Show only natural notes (no # or b) to keep axis clean
        if f0_min <= hz <= f0_max and "#" not in note and "b" not in note:
            note_ticks_hz.append(hz)
            note_ticks_labels.append(note)

    # Plot 1: Original audio (mix)
    ax1 = axes[0]
    mask1 = confidence_original > conf_threshold
    scatter1 = ax1.scatter(
        time_original[mask1],
        f0_original[mask1],
        c=confidence_original[mask1],
        cmap="viridis",
        s=3,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax1.set_ylim(f0_min, f0_max)
    ax1.set_ylabel("f0 (Hz)")
    ax1.set_title("Original (full mix)")

    # Right Y-axis with notes (original)
    ax1_notes = ax1.twinx()
    ax1_notes.set_ylim(f0_min, f0_max)
    ax1_notes.set_yticks(note_ticks_hz)
    ax1_notes.set_yticklabels(note_ticks_labels, fontsize=8)
    ax1_notes.set_ylabel("Note")

    for hz in note_ticks_hz:
        ax1.axhline(hz, color="gray", linewidth=0.3, alpha=0.4)

    # Plot 2: Separated vocals
    ax2 = axes[1]
    mask2 = confidence_separated > conf_threshold
    scatter2 = ax2.scatter(
        time_separated[mask2],
        f0_separated[mask2],
        c=confidence_separated[mask2],
        cmap="viridis",
        s=3,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax2.set_ylim(f0_min, f0_max)
    ax2.set_ylabel("f0 (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Separated Vocals (HTDemucs)")

    # Right Y-axis with notes (separated)
    ax2_notes = ax2.twinx()
    ax2_notes.set_ylim(f0_min, f0_max)
    ax2_notes.set_yticks(note_ticks_hz)
    ax2_notes.set_yticklabels(note_ticks_labels, fontsize=8)
    ax2_notes.set_ylabel("Note")

    for hz in note_ticks_hz:
        ax2.axhline(hz, color="gray", linewidth=0.3, alpha=0.4)

    # Shared colorbar
    cbar = fig.colorbar(
        scatter2, ax=axes, location="right", label="CREPE Confidence", shrink=0.8, pad=0.02
    )

    fig.suptitle(title, fontsize=12, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_vmi_scatter(
    df: pd.DataFrame,
    x_col: str = "f0",
    y_col: str = "alpha_ratio",
    color_col: str = "vmi",
    title: str = "F0 vs Alpha Ratio (VMI)",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
    show_note_axis: bool = True,
) -> plt.Figure:
    """Scatter plot F0 vs Alpha Ratio colored by VMI.

    Main plot for VMI visualization, enabling identification of the
    "turning point" where laryngeal configuration changes.

    Args:
        df: DataFrame with data (must have columns f0, alpha_ratio, vmi).
        x_col: Column for X axis (default: f0).
        y_col: Column for Y axis (default: alpha_ratio).
        color_col: Column for coloring (default: vmi).
        title: Plot title.
        save_path: Path to save (optional).
        figsize: Figure size.
        show_note_axis: If True, shows secondary axis with musical notes.

    Returns:
        Matplotlib figure.
    """
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    # Filter valid data
    df_valid = df.dropna(subset=[x_col, y_col, color_col])

    if df_valid.empty:
        ax.text(
            0.5,
            0.5,
            "No valid data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        return fig

    # Scatter plot with diverging colormap
    scatter = ax.scatter(
        df_valid[x_col],
        df_valid[y_col],
        c=df_valid[color_col],
        cmap="RdBu_r",  # Blue (M1) -> White (mix) -> Red (M2)
        s=5,
        alpha=0.6,
        vmin=0,
        vmax=1,
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("VMI (0=Dense M1, 1=Light M2)", fontsize=10)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["Dense M1", "Light M1", "Mix", "Reinf. M2", "Light M2", ""])

    # Labels
    ax.set_xlabel("F0 (Hz)", fontsize=11)
    ax.set_ylabel("Alpha Ratio (dB)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Secondary axis with musical notes (if enabled and x is f0)
    if show_note_axis and x_col == "f0":
        f0_min, f0_max = df_valid[x_col].min(), df_valid[x_col].max()
        f0_min = max(f0_min * 0.95, 50)
        f0_max = f0_max * 1.05

        ax.set_xlim(f0_min, f0_max)

        midi_min = int(np.floor(hz_to_midi(f0_min)))
        midi_max = int(np.ceil(hz_to_midi(f0_max)))

        note_ticks_hz = []
        note_ticks_labels = []
        for midi in range(midi_min, midi_max + 1, 2):  # Every 2 semitones to avoid clutter
            hz = float(midi_to_hz(midi))
            if f0_min <= hz <= f0_max:
                note_ticks_hz.append(hz)
                note_ticks_labels.append(hz_to_note(hz))

        ax2 = ax.twiny()
        ax2.set_xlim(f0_min, f0_max)
        ax2.set_xticks(note_ticks_hz)
        ax2.set_xticklabels(note_ticks_labels, fontsize=8, rotation=45)
        ax2.set_xlabel("Musical Note", fontsize=10)

    # Reference lines for VMI zones
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_vmi_timeline(
    df: pd.DataFrame,
    time_col: str = "time",
    f0_col: str = "f0",
    vmi_col: str = "vmi",
    song_col: str | None = "song",
    title: str = "F0 Contour with VMI",
    save_path: str | Path | None = None,
) -> plt.Figure | sns.FacetGrid:
    """Timeline of F0 colored by VMI, split by song.

    Args:
        df: DataFrame with data.
        time_col: Time column.
        f0_col: F0 column.
        vmi_col: VMI column.
        song_col: Song column (if None, no splitting).
        title: Plot title.
        save_path: Path to save (optional).

    Returns:
        Matplotlib figure or seaborn FacetGrid.
    """
    sns.set_theme(style="whitegrid")

    df_valid = df.dropna(subset=[time_col, f0_col, vmi_col])

    if song_col and song_col in df_valid.columns:
        # FacetGrid split by song
        g = sns.FacetGrid(
            df_valid,
            row=song_col,
            aspect=4,
            height=2.5,
            sharex=False,
            sharey=False,
        )

        def scatter_vmi(x, y, c, **kwargs):
            plt.scatter(x, y, c=c, cmap="RdBu_r", s=3, alpha=0.7, vmin=0, vmax=1)

        g.map(scatter_vmi, time_col, f0_col, vmi_col)
        g.set_axis_labels("Time (s)", "F0 (Hz)")
        g.set_titles(row_template="{row_name}")
        g.figure.suptitle(title, y=1.02, fontsize=12, fontweight="bold")

        if save_path:
            g.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(g.figure)

        return g

    else:
        fig, ax = plt.subplots(figsize=(14, 5))

        scatter = ax.scatter(
            df_valid[time_col],
            df_valid[f0_col],
            c=df_valid[vmi_col],
            cmap="RdBu_r",
            s=3,
            alpha=0.7,
            vmin=0,
            vmax=1,
        )

        plt.colorbar(scatter, ax=ax, label="VMI")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("F0 (Hz)")
        ax.set_title(title)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        return fig
