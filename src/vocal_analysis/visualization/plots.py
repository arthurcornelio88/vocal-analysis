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
    g = sns.FacetGrid(
        df, row="song", hue="xgb_mechanism", palette=palette, aspect=4, height=2.5, sharex=False
    )

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
                    mech_data["time"],
                    mech_data["f0"],
                    c=colors[mech],
                    s=15,
                    alpha=0.8,
                    label=mech,
                    zorder=2,
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


def plot_separation_validation(
    time_original: np.ndarray,
    f0_original: np.ndarray,
    confidence_original: np.ndarray,
    time_separated: np.ndarray,
    f0_separated: np.ndarray,
    confidence_separated: np.ndarray,
    title: str = "Validacao: Original vs Voz Separada",
    save_path: str | Path | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> plt.Figure:
    """Plot comparativo mostrando f0 antes/depois da separacao.

    Permite validar visualmente se a separacao esta captando a voz
    e nao outros instrumentos (cavaquinho, flauta, etc).

    Args:
        time_original: Array de tempo do audio original.
        f0_original: Array de f0 do audio original.
        confidence_original: Array de confianca CREPE do original.
        time_separated: Array de tempo da voz separada.
        f0_separated: Array de f0 da voz separada.
        confidence_separated: Array de confianca CREPE da voz separada.
        title: Titulo do grafico.
        save_path: Caminho para salvar o plot (opcional).
        start_time: Tempo inicial do excerpt (segundos). Se None, usa todo o audio.
        end_time: Tempo final do excerpt (segundos). Se None, usa todo o audio.

    Returns:
        Figura matplotlib.
    """
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    # Filtrar por intervalo de tempo se especificado
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

    # Filtrar frames com confianca razoavel para visualizacao
    conf_threshold = 0.5

    # Calcular range de f0 para ambos os plots
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

    # Gerar ticks de notas (apenas notas naturais para legibilidade)
    midi_min = int(np.floor(hz_to_midi(f0_min)))
    midi_max = int(np.ceil(hz_to_midi(f0_max)))
    note_ticks_hz = []
    note_ticks_labels = []
    for midi in range(midi_min, midi_max + 1):
        hz = float(midi_to_hz(midi))
        note = hz_to_note(hz)
        # Mostrar apenas notas naturais (sem # ou b) para não poluir o eixo
        if f0_min <= hz <= f0_max and "#" not in note and "b" not in note:
            note_ticks_hz.append(hz)
            note_ticks_labels.append(note)

    # Plot 1: Audio Original (mix)
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
    ax1.set_title("Original (mix completo)")

    # Eixo Y direito com notas (original)
    ax1_notes = ax1.twinx()
    ax1_notes.set_ylim(f0_min, f0_max)
    ax1_notes.set_yticks(note_ticks_hz)
    ax1_notes.set_yticklabels(note_ticks_labels, fontsize=8)
    ax1_notes.set_ylabel("Nota")

    # Grid nas notas
    for hz in note_ticks_hz:
        ax1.axhline(hz, color="gray", linewidth=0.3, alpha=0.4)

    # Plot 2: Voz Separada
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
    ax2.set_xlabel("Tempo (s)")
    ax2.set_title("Voz Separada (HTDemucs)")

    # Eixo Y direito com notas (separado)
    ax2_notes = ax2.twinx()
    ax2_notes.set_ylim(f0_min, f0_max)
    ax2_notes.set_yticks(note_ticks_hz)
    ax2_notes.set_yticklabels(note_ticks_labels, fontsize=8)
    ax2_notes.set_ylabel("Nota")

    # Grid nas notas
    for hz in note_ticks_hz:
        ax2.axhline(hz, color="gray", linewidth=0.3, alpha=0.4)

    # Colorbar compartilhada (location='right' evita sobreposição com eixo de notas)
    cbar = fig.colorbar(
        scatter2, ax=axes, location="right", label="Confiança CREPE", shrink=0.8, pad=0.02
    )

    # Titulo geral
    fig.suptitle(title, fontsize=12, fontweight="bold")
    # Nota: tight_layout() removido pois constrained_layout=True já cuida do layout

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
    """Scatter plot F0 vs Alpha Ratio colorido pelo VMI.

    Este é o plot principal para visualização do VMI, permitindo
    identificar o "turning point" onde a configuração laríngea muda.

    Args:
        df: DataFrame com dados (deve ter colunas f0, alpha_ratio, vmi).
        x_col: Coluna para eixo X (default: f0).
        y_col: Coluna para eixo Y (default: alpha_ratio).
        color_col: Coluna para coloração (default: vmi).
        title: Título do gráfico.
        save_path: Caminho para salvar (opcional).
        figsize: Tamanho da figura.
        show_note_axis: Se True, mostra eixo secundário com notas musicais.

    Returns:
        Figura matplotlib.
    """
    from vocal_analysis.utils.pitch import hz_to_midi, hz_to_note, midi_to_hz

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    # Filtrar dados válidos
    df_valid = df.dropna(subset=[x_col, y_col, color_col])

    if df_valid.empty:
        ax.text(
            0.5,
            0.5,
            "Sem dados válidos para plotar",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        return fig

    # Scatter plot com colormap divergente
    scatter = ax.scatter(
        df_valid[x_col],
        df_valid[y_col],
        c=df_valid[color_col],
        cmap="RdBu_r",  # Azul (M1) -> Branco (mix) -> Vermelho (M2)
        s=5,
        alpha=0.6,
        vmin=0,
        vmax=1,
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("VMI (0=M1 Denso, 1=M2 Ligeiro)", fontsize=10)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["M1 Denso", "M1 Lig.", "Mix", "M2 Ref.", "M2 Lig.", ""])

    # Labels
    ax.set_xlabel("F0 (Hz)", fontsize=11)
    ax.set_ylabel("Alpha Ratio (dB)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Eixo secundário com notas musicais (se habilitado e x é f0)
    if show_note_axis and x_col == "f0":
        f0_min, f0_max = df_valid[x_col].min(), df_valid[x_col].max()
        f0_min = max(f0_min * 0.95, 50)
        f0_max = f0_max * 1.05

        ax.set_xlim(f0_min, f0_max)

        midi_min = int(np.floor(hz_to_midi(f0_min)))
        midi_max = int(np.ceil(hz_to_midi(f0_max)))

        note_ticks_hz = []
        note_ticks_labels = []
        for midi in range(midi_min, midi_max + 1, 2):  # A cada 2 semitons para não poluir
            hz = float(midi_to_hz(midi))
            if f0_min <= hz <= f0_max:
                note_ticks_hz.append(hz)
                note_ticks_labels.append(hz_to_note(hz))

        ax2 = ax.twiny()
        ax2.set_xlim(f0_min, f0_max)
        ax2.set_xticks(note_ticks_hz)
        ax2.set_xticklabels(note_ticks_labels, fontsize=8, rotation=45)
        ax2.set_xlabel("Nota Musical", fontsize=10)

    # Linhas de referência para zonas VMI
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
    title: str = "Contorno F0 com VMI",
    save_path: str | Path | None = None,
) -> plt.Figure | sns.FacetGrid:
    """Timeline de F0 colorido pelo VMI, separado por música.

    Args:
        df: DataFrame com dados.
        time_col: Coluna de tempo.
        f0_col: Coluna de F0.
        vmi_col: Coluna de VMI.
        song_col: Coluna de música (se None, não separa).
        title: Título do gráfico.
        save_path: Caminho para salvar (opcional).

    Returns:
        Figura matplotlib ou FacetGrid seaborn.
    """
    sns.set_theme(style="whitegrid")

    df_valid = df.dropna(subset=[time_col, f0_col, vmi_col])

    if song_col and song_col in df_valid.columns:
        # FacetGrid separado por música
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
        g.set_axis_labels("Tempo (s)", "F0 (Hz)")
        g.set_titles(row_template="{row_name}")
        g.figure.suptitle(title, y=1.02, fontsize=12, fontweight="bold")

        if save_path:
            g.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(g.figure)

        return g

    else:
        # Plot único
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
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("F0 (Hz)")
        ax.set_title(title)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        return fig
