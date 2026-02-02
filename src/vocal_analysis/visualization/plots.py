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
    """Scatter plot de Pitch vs HNR para visualizar clusters M1/M2."""
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
) -> sns.FacetGrid:
    """Contorno temporal de f0 colorido pela predição, separado por música.
    
    CORREÇÃO: Usa FacetGrid para não sobrepor músicas diferentes no mesmo eixo.
    """
    sns.set_theme(style="whitegrid")
    
    # Define cores fixas para garantir consistência
    palette = {"M1": "steelblue", "M2": "coral"}
    
    # Cria um grid com uma linha por música
    g = sns.FacetGrid(df, row="song", hue="xgb_mechanism", palette=palette, 
                      aspect=4, height=2.5, sharex=False)
    
    g.map(plt.scatter, "time", "f0", s=1.5, alpha=0.7)
    g.add_legend(title="Mecanismo")
    
    # Ajustes de títulos e eixos
    g.set_axis_labels("Tempo (s)", "f0 (Hz)")
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
    """Trecho de um música com f0 colorido por predição XGBoost e notas no eixo Y."""
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    subset = df[(df["song"] == song) & (df["time"] >= start_time) & (df["time"] <= end_time)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.set_theme(style="whitegrid")

    colors = {"M1": "steelblue", "M2": "coral"}
    
    # Plotar dados (se vazio, cria plot vazio para não quebrar)
    if not subset.empty:
        for mech in ["M1", "M2"]:
            mech_data = subset[subset["xgb_mechanism"] == mech]
            if not mech_data.empty:
                ax.scatter(
                    mech_data["time"], mech_data["f0"], 
                    c=colors[mech], s=15, alpha=0.8, label=mech, zorder=2
                )
    
    # Configuração do Eixo Y (Notas)
    f0_min = subset["f0"].min() if not subset.empty else 100
    f0_max = subset["f0"].max() if not subset.empty else 800
    
    # Margem de segurança
    f0_min = max(f0_min * 0.9, 50)
    f0_max = f0_max * 1.1
    ax.set_ylim(f0_min, f0_max)
    
    # Eixo X fixo na janela pedida (importante para conferência visual)
    ax.set_xlim(start_time, end_time)

    # Gerar ticks de notas
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

    for hz in note_ticks_hz:
        ax.axhline(hz, color="gray", linewidth=0.3, alpha=0.5, zorder=1)

    ax.set_title(f"{song} — {start_time:.1f}s a {end_time:.1f}s")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("f0 (Hz)")
    
    # Só adiciona legenda se houve dados plotados
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
    title: str = "Contorno de f0",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot do contorno de f0 ao longo do tempo."""
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