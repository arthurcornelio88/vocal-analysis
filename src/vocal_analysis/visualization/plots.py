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
