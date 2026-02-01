"""Plots com estética acadêmica para artigo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_mechanism_clusters(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot de Pitch vs HNR para visualizar clusters M1/M2.

    Args:
        df: DataFrame com colunas 'f0', 'hnr' e 'mechanism'.
        save_path: Caminho para salvar o plot (opcional).

    Returns:
        Figura matplotlib.
    """
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

    ax.set_title("Distribuição Espectral: Mecanismo 1 vs Mecanismo 2")
    ax.set_xlabel("Frequência Fundamental (Hz)")
    ax.set_ylabel("Harmonic-to-Noise Ratio (dB)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_xgb_mechanism_timeline(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Contorno temporal de f0 colorido pela predição do XGBoost.

    Args:
        df: DataFrame com colunas 'time', 'f0', 'xgb_mechanism'.
        save_path: Caminho para salvar o plot (opcional).

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    colors = {"M1": "steelblue", "M2": "coral"}
    for mech in ["M1", "M2"]:
        subset = df[df["xgb_mechanism"] == mech]
        ax.scatter(subset["time"], subset["f0"], c=colors[mech], s=1.5, alpha=0.7, label=mech)

    ax.set_title("Predição XGBoost: M1 vs M2 ao longo do tempo")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.legend(loc="upper right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_xgb_mechanism_excerpt(
    df: pd.DataFrame,
    song: str,
    start_time: float,
    end_time: float,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Trecho de um música com f0 colorido por predição XGBoost e notas no eixo Y.

    Args:
        df: DataFrame com colunas 'time', 'f0', 'xgb_mechanism', 'song'.
        song: Nome da música a plotar.
        start_time: Tempo de início do trecho (s).
        end_time: Tempo de fim do trecho (s).
        save_path: Caminho para salvar o plot (opcional).

    Returns:
        Figura matplotlib.
    """
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    subset = df[(df["song"] == song) & (df["time"] >= start_time) & (df["time"] <= end_time)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.set_theme(style="whitegrid")

    colors = {"M1": "steelblue", "M2": "coral"}
    for mech in ["M1", "M2"]:
        mech_data = subset[subset["xgb_mechanism"] == mech]
        ax.scatter(
            mech_data["time"], mech_data["f0"], c=colors[mech], s=8, alpha=0.8, label=mech, zorder=2
        )

    # Eixo Y secundário com notas musicais
    f0_min = subset["f0"].min() if len(subset) > 0 else 100
    f0_max = subset["f0"].max() if len(subset) > 0 else 800
    # Expandir range para margem visual
    f0_min = max(f0_min * 0.85, 50)
    f0_max = f0_max * 1.15
    ax.set_ylim(f0_min, f0_max)

    # Gerar ticks de notas dentro do range visível
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
    ax2.set_ylabel("Nota")

    # Linhas horizontais sutis nas notas
    for hz in note_ticks_hz:
        ax.axhline(hz, color="gray", linewidth=0.3, alpha=0.5, zorder=1)

    ax.set_title(f"{song} — {start_time:.1f}s a {end_time:.1f}s")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("f0 (Hz)")
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
    title: str = "Contorno de f0",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot do contorno de f0 ao longo do tempo.

    Args:
        time: Array de tempo em segundos.
        f0: Array de frequência fundamental.
        confidence: Array de confiança da estimativa (opcional).
        title: Título do gráfico.
        save_path: Caminho para salvar o plot (opcional).

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.set_theme(style="whitegrid")

    if confidence is not None:
        scatter = ax.scatter(time, f0, c=confidence, cmap="viridis", s=2, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Confiança")
    else:
        ax.plot(time, f0, linewidth=0.8, color="steelblue")

    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("f0 (Hz)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
