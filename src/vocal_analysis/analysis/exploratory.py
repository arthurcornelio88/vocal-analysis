"""Análise exploratória de mecanismos laríngeos M1/M2."""

from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from vocal_analysis.features.vmi import (
    VMILabel,
    VMIWeights,
    apply_temporal_smoothing,
    compute_vmi_fixed,
    vmi_to_label,
)
from vocal_analysis.utils.pitch import hz_range_to_notes, hz_to_note


class MechanismStats(TypedDict):
    """Estatísticas por mecanismo."""

    count: int
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    hnr_mean: float
    note_mean: str
    note_range: str


def analyze_mechanism_regions(
    df: pd.DataFrame,
    threshold_hz: float = 400.0,
    output_dir: Path | None = None,
) -> dict[str, MechanismStats]:
    """Analisa regiões de pitch separando por limiar M1/M2.

    Args:
        df: DataFrame com colunas 'f0', 'hnr', 'confidence'.
        threshold_hz: Limiar de separação M1/M2 em Hz (default 400 Hz ~ G4).
        output_dir: Diretório para salvar plots.

    Returns:
        Dicionário com estatísticas por mecanismo.
    """
    # Filtrar voiced frames: confidence > 0.85 + HNR > 0 (remove silêncio e faux positifs)
    df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)].copy()

    # Classificar por threshold
    df_voiced["mechanism"] = np.where(df_voiced["f0"] < threshold_hz, "M1", "M2")

    stats = {}
    for mech in ["M1", "M2"]:
        subset = df_voiced[df_voiced["mechanism"] == mech]
        if len(subset) > 0:
            stats[mech] = MechanismStats(
                count=len(subset),
                f0_mean=float(subset["f0"].mean()),
                f0_std=float(subset["f0"].std()),
                f0_min=float(subset["f0"].min()),
                f0_max=float(subset["f0"].max()),
                hnr_mean=float(subset["hnr"].mean()),
                note_mean=hz_to_note(subset["f0"].mean()),
                note_range=hz_range_to_notes(subset["f0"].min(), subset["f0"].max()),
            )

    if output_dir:
        _plot_mechanism_analysis(df_voiced, threshold_hz, output_dir)

    return stats


def cluster_mechanisms(
    df: pd.DataFrame,
    n_clusters: int = 2,
    method: str = "gmm",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Clusteriza frames em mecanismos usando f0 e HNR.

    Args:
        df: DataFrame com colunas 'f0', 'hnr'.
        n_clusters: Número de clusters.
        method: Método de clustering ('kmeans' ou 'gmm').
        output_dir: Diretório para salvar plots.

    Returns:
        DataFrame com coluna 'cluster' adicionada.
    """
    df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)].copy()

    # Normalizar features (RobustScaler usa mediana/IQR, mais robusto a outliers)
    features = df_voiced[["f0", "hnr"]].values
    scaler = RobustScaler()
    features_norm = scaler.fit_transform(features)

    # Clustering
    if method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        df_voiced["cluster"] = model.fit_predict(features_norm)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_voiced["cluster"] = model.fit_predict(features_norm)

    # Ordenar clusters por f0 médio (cluster 0 = mais grave = M1)
    cluster_means = df_voiced.groupby("cluster")["f0"].mean()
    cluster_order = cluster_means.sort_values().index.tolist()
    df_voiced["mechanism"] = df_voiced["cluster"].map(
        {cluster_order[i]: f"M{i + 1}" for i in range(len(cluster_order))}
    )

    if output_dir:
        _plot_clusters(df_voiced, output_dir)

    return df_voiced


class VMIStats(TypedDict):
    """Estatísticas por categoria VMI."""

    label: str
    count: int
    percentage: float
    vmi_mean: float
    vmi_std: float
    f0_mean: float
    f0_std: float
    alpha_ratio_mean: float
    h1_h2_mean: float


def analyze_mechanism_vmi(
    df: pd.DataFrame,
    weights: VMIWeights | None = None,
    smoothing_method: Literal["median", "mean", "exponential", "none"] = "median",
    smoothing_window: int = 5,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, VMIStats]]:
    """Analisa mecanismos usando VMI (Vocal Mechanism Index) agnóstico.

    Substitui o threshold fixo de G4 por análise baseada em features espectrais.

    Args:
        df: DataFrame com colunas espectrais: 'alpha_ratio', 'h1_h2', 'spectral_tilt',
            e opcionalmente 'cpps_per_frame'. Deve ter também 'f0', 'confidence'.
        weights: Pesos para cálculo do VMI. Default: pesos padrão.
        smoothing_method: Método de suavização temporal ('median', 'mean', 'exponential', 'none').
        smoothing_window: Tamanho da janela de suavização.
        output_dir: Diretório para salvar plots.

    Returns:
        Tupla (DataFrame com VMI, estatísticas por categoria VMI).
    """
    # Filtrar voiced frames
    df_voiced = df[(df["confidence"] > 0.85)].copy()

    # Verificar se temos as features necessárias
    required_cols = ["alpha_ratio", "h1_h2", "spectral_tilt"]
    missing_cols = [c for c in required_cols if c not in df_voiced.columns]
    if missing_cols:
        raise ValueError(f"Colunas faltando para VMI: {missing_cols}")

    # Usar CPPS per-frame se disponível, senão usar CPPS global ou placeholder
    if "cpps_per_frame" in df_voiced.columns:
        cpps = df_voiced["cpps_per_frame"].values
    elif "cpps_global" in df_voiced.columns:
        # Usar CPPS global para todos os frames
        cpps = np.full(len(df_voiced), df_voiced["cpps_global"].iloc[0])
    else:
        # Placeholder neutro
        cpps = np.full(len(df_voiced), 0.5)

    # Calcular VMI
    vmi_raw = compute_vmi_fixed(
        alpha_ratio=df_voiced["alpha_ratio"].values,
        cpps=cpps,
        h1_h2=df_voiced["h1_h2"].values,
        spectral_tilt=df_voiced["spectral_tilt"].values,
        weights=weights,
    )

    # Aplicar suavização temporal
    if smoothing_method != "none":
        vmi_smoothed = apply_temporal_smoothing(
            vmi_raw,
            window_size=smoothing_window,
            method=smoothing_method,
        )
    else:
        vmi_smoothed = vmi_raw

    # Adicionar ao DataFrame
    df_voiced["vmi"] = vmi_smoothed
    df_voiced["vmi_label"] = vmi_to_label(vmi_smoothed)

    # Calcular estatísticas por categoria
    stats = {}
    total_frames = len(df_voiced)

    for label in VMILabel:
        subset = df_voiced[df_voiced["vmi_label"] == label.value]
        if len(subset) > 0:
            stats[label.value] = VMIStats(
                label=label.value,
                count=len(subset),
                percentage=(len(subset) / total_frames) * 100,
                vmi_mean=float(subset["vmi"].mean()),
                vmi_std=float(subset["vmi"].std()),
                f0_mean=float(subset["f0"].mean()),
                f0_std=float(subset["f0"].std()),
                alpha_ratio_mean=float(subset["alpha_ratio"].mean()),
                h1_h2_mean=float(subset["h1_h2"].mean()),
            )

    if output_dir:
        _plot_vmi_analysis(df_voiced, output_dir)

    return df_voiced, stats


def _plot_vmi_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Gera plots de análise VMI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # 1. Scatter F0 vs Alpha Ratio colorido por VMI
    ax = axes[0, 0]
    scatter = ax.scatter(
        df["f0"],
        df["alpha_ratio"],
        c=df["vmi"],
        cmap="RdBu_r",
        alpha=0.5,
        s=5,
    )
    plt.colorbar(scatter, ax=ax, label="VMI")
    ax.set_xlabel("F0 (Hz)")
    ax.set_ylabel("Alpha Ratio (dB)")
    ax.set_title("F0 vs Alpha Ratio (cor = VMI)")

    # 2. Histograma de VMI
    ax = axes[0, 1]
    ax.hist(df["vmi"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        ax.axvline(threshold, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("VMI")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição do VMI")

    # 3. Timeline VMI
    ax = axes[1, 0]
    if "time" in df.columns:
        scatter = ax.scatter(
            df["time"],
            df["f0"],
            c=df["vmi"],
            cmap="RdBu_r",
            s=3,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label="VMI")
        ax.set_xlabel("Tempo (s)")
    else:
        scatter = ax.scatter(
            range(len(df)),
            df["f0"],
            c=df["vmi"],
            cmap="RdBu_r",
            s=3,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label="VMI")
        ax.set_xlabel("Frame")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("Contorno F0 (cor = VMI)")

    # 4. Boxplot de VMI por categoria
    ax = axes[1, 1]
    order = [label.value for label in VMILabel]
    present_labels = [lbl for lbl in order if lbl in df["vmi_label"].values]
    if present_labels:
        sns.boxplot(data=df, x="vmi_label", y="f0", order=present_labels, ax=ax)
        ax.set_xlabel("VMI Label")
        ax.set_ylabel("F0 (Hz)")
        ax.set_title("F0 por Categoria VMI")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(output_dir / "vmi_analysis.png", dpi=150)
    plt.close(fig)


def _plot_mechanism_analysis(df: pd.DataFrame, threshold_hz: float, output_dir: Path) -> None:
    """Gera plots de análise por mecanismo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # 1. Histograma de f0 com threshold
    ax = axes[0, 0]
    for mech, color in [("M1", "steelblue"), ("M2", "coral")]:
        subset = df[df["mechanism"] == mech]
        ax.hist(subset["f0"], bins=50, alpha=0.6, label=mech, color=color)
    ax.axvline(threshold_hz, color="red", linestyle="--", label=f"Threshold ({threshold_hz} Hz)")
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de f0 por Mecanismo")
    ax.legend()

    # 2. Scatter f0 vs HNR
    ax = axes[0, 1]
    sns.scatterplot(data=df, x="f0", y="hnr", hue="mechanism", alpha=0.5, ax=ax)
    ax.axvline(threshold_hz, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel("HNR (dB)")
    ax.set_title("f0 vs HNR por Mecanismo")

    # 3. Boxplot de HNR por mecanismo
    ax = axes[1, 0]
    sns.boxplot(data=df, x="mechanism", y="hnr", ax=ax)
    ax.set_xlabel("Mecanismo")
    ax.set_ylabel("HNR (dB)")
    ax.set_title("HNR por Mecanismo")

    # 4. Timeline colorido por mecanismo
    ax = axes[1, 1]
    colors = {"M1": "steelblue", "M2": "coral"}
    for mech in ["M1", "M2"]:
        subset = df[df["mechanism"] == mech]
        ax.scatter(subset["time"], subset["f0"], c=colors[mech], s=1, alpha=0.6, label=mech)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.set_title("Contorno de f0 por Mecanismo")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "mechanism_analysis.png", dpi=150)
    plt.close(fig)


def _plot_clusters(df: pd.DataFrame, output_dir: Path) -> None:
    """Plota resultado do clustering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(data=df, x="f0", y="hnr", hue="mechanism", alpha=0.6, ax=ax)
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel("HNR (dB)")
    ax.set_title("Clusters de Mecanismo (GMM)")

    fig.savefig(output_dir / "mechanism_clusters.png", dpi=150)
    plt.close(fig)


def generate_report(
    df: pd.DataFrame,
    stats: dict[str, MechanismStats],
    output_path: Path,
    artist_name: str = "Ademilde Fonseca",
    xgb_report: str | None = None,
    xgb_feature_cols: list[str] | None = None,
) -> None:
    """Gera relatório markdown com análise completa.

    Args:
        df: DataFrame com dados.
        stats: Estatísticas por mecanismo.
        output_path: Caminho do arquivo .md.
        artist_name: Nome do artista.
        xgb_report: Classification report do XGBoost (string), se disponível.
        xgb_feature_cols: Lista de features usadas no XGBoost.
    """
    df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)]

    f0_global_mean = df_voiced["f0"].mean()
    f0_global_min = df_voiced["f0"].min()
    f0_global_max = df_voiced["f0"].max()
    hnr_global_mean = df_voiced["hnr"].mean()

    report = f"""# Análise Bioacústica - {artist_name}

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | {f0_global_mean:.1f} Hz | {hz_to_note(f0_global_mean)} |
| **f0 mínimo** | {f0_global_min:.1f} Hz | {hz_to_note(f0_global_min)} |
| **f0 máximo** | {f0_global_max:.1f} Hz | {hz_to_note(f0_global_max)} |
| **Extensão** | {hz_range_to_notes(f0_global_min, f0_global_max)} | ~{np.log2(f0_global_max / f0_global_min):.1f} oitavas |
| **HNR médio** | {hnr_global_mean:.1f} dB | – |
| **Total frames** | {len(df_voiced)} | – |

## Análise por Mecanismo

"""

    for mech, s in stats.items():
        pct = (s["count"] / len(df_voiced)) * 100
        report += f"""### {mech} ({"Peito/Chest" if mech == "M1" else "Cabeça/Head"})

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | {s["count"]} ({pct:.1f}%) | – |
| **f0 médio** | {s["f0_mean"]:.1f} Hz | {s["note_mean"]} |
| **f0 desvio** | {s["f0_std"]:.1f} Hz | – |
| **Extensão** | {s["note_range"]} | – |
| **HNR médio** | {s["hnr_mean"]:.1f} dB | – |

"""

    # Songs breakdown
    if "song" in df.columns:
        report += "## Por Música\n\n"
        for song in df["song"].unique():
            song_df = df_voiced[df_voiced["song"] == song]
            report += f"""### {song}

- f0 médio: {song_df["f0"].mean():.1f} Hz ({hz_to_note(song_df["f0"].mean())})
- Extensão: {hz_range_to_notes(song_df["f0"].min(), song_df["f0"].max())}
- HNR médio: {song_df["hnr"].mean():.1f} dB

"""

    if xgb_report:
        features_str = ", ".join(f"`{c}`" for c in (xgb_feature_cols or []))
        report += f"""## Classificação XGBoost (Pseudo-Labels GMM)

Features utilizadas: {features_str}
Labels de treinamento: clusters do GMM (não-supervisionado)
Split: 80% treino / 20% teste

```
{xgb_report}```

"""

    report += """## Interpretação

### Padrão Bimodal

O contorno de f0 mostra alternância clara entre duas regiões:
- **Região grave (M1)**: Mecanismo 1 / voz de peito
- **Região aguda (M2)**: Mecanismo 2 / voz de cabeça

### Implicações para Classificação "Fach"

A análise sugere que a classificação tradicional de "tipo vocal" não captura
a realidade fisiológica dos mecanismos laríngeos. A cantora utiliza ambos
os mecanismos de forma fluida, contradizendo categorizações rígidas.

### Limitações

- Gravações históricas com baixa qualidade (HNR reduzido)
- Threshold M1/M2 baseado em heurística (400 Hz)
- CPPS comprometido pelo ruído de fundo

## Próximos Passos

1. Validar threshold com clustering não supervisionado
2. Analisar transições entre mecanismos (quebras de registro)
3. Comparar com cantoras contemporâneas (gravações de alta qualidade)
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def generate_vmi_report(
    df: pd.DataFrame,
    vmi_stats: dict[str, VMIStats],
    output_path: Path,
    artist_name: str = "Artista",
) -> None:
    """Gera relatório markdown com análise VMI.

    Args:
        df: DataFrame com dados e VMI calculado.
        vmi_stats: Estatísticas por categoria VMI.
        output_path: Caminho do arquivo .md.
        artist_name: Nome do artista.
    """
    df_voiced = df[df["confidence"] > 0.85]

    f0_global_mean = df_voiced["f0"].mean()
    f0_global_min = df_voiced["f0"].min()
    f0_global_max = df_voiced["f0"].max()
    vmi_global_mean = df_voiced["vmi"].mean()

    report = f"""# Análise VMI (Vocal Mechanism Index) - {artist_name}

## Metodologia

Este relatório utiliza o **VMI (Vocal Mechanism Index)**, uma métrica contínua que
substitui o threshold arbitrário de G4 (400 Hz) por análise baseada em features espectrais:

- **Alpha Ratio**: Razão de energia 0-1kHz vs 1-5kHz
- **H1-H2**: Diferença entre 1º e 2º harmônico (inclinação glotal)
- **Spectral Tilt**: Inclinação do espectro de potência
- **CPPS**: Proeminência do pico cepstral (periodicidade)

O VMI varia de **0.0 (M1 Denso)** a **1.0 (M2 Ligeiro)**, permitindo identificação
gradual do mecanismo vocal sem depender de frequências fixas.

---

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **F0 médio** | {f0_global_mean:.1f} Hz | {hz_to_note(f0_global_mean)} |
| **F0 mínimo** | {f0_global_min:.1f} Hz | {hz_to_note(f0_global_min)} |
| **F0 máximo** | {f0_global_max:.1f} Hz | {hz_to_note(f0_global_max)} |
| **Extensão** | {hz_range_to_notes(f0_global_min, f0_global_max)} | ~{np.log2(f0_global_max / f0_global_min):.1f} oitavas |
| **VMI médio** | {vmi_global_mean:.3f} | – |
| **Total frames** | {len(df_voiced)} | – |

---

## Análise por Categoria VMI

| Categoria | Frames | % | VMI médio | F0 médio | Alpha Ratio | H1-H2 |
|-----------|--------|---|-----------|----------|-------------|-------|
"""

    for label_name, s in vmi_stats.items():
        report += f"| **{label_name}** | {s['count']} | {s['percentage']:.1f}% | {s['vmi_mean']:.3f} | {s['f0_mean']:.1f} Hz | {s['alpha_ratio_mean']:.1f} dB | {s['h1_h2_mean']:.1f} dB |\n"

    report += """
### Interpretação das Categorias

- **M1_DENSO (VMI 0.0-0.2)**: Mecanismo pesado, adução firme, voz de peito plena
- **M1_LIGEIRO (VMI 0.2-0.4)**: M1 de borda fina, comum em tenores/registro médio
- **MIX_PASSAGGIO (VMI 0.4-0.6)**: Zona de passagem, instabilidade acústica, voz mista
- **M2_REFORCADO (VMI 0.6-0.8)**: M2 com adução glótica, ressonância frontal
- **M2_LIGEIRO (VMI 0.8-1.0)**: Mecanismo leve, falsete, piano M2

---

## Análise por Música

"""

    if "song" in df.columns:
        for song in df["song"].unique():
            song_df = df_voiced[df_voiced["song"] == song]
            if len(song_df) > 0:
                report += f"""### {song}

- F0 médio: {song_df["f0"].mean():.1f} Hz ({hz_to_note(song_df["f0"].mean())})
- VMI médio: {song_df["vmi"].mean():.3f}
- Distribuição: {(song_df["vmi_label"].value_counts() / len(song_df) * 100).to_dict()}

"""

    report += """---

## Vantagens do VMI

1. **Agnóstico à tessitura**: Não depende de frequências fixas como G4
2. **Contínuo**: Captura gradações entre mecanismos (passaggio)
3. **Multi-dimensional**: Combina múltiplas features espectrais
4. **Interpretável**: Cada feature tem significado fisiológico claro

## Limitações

1. **Pesos fixos**: Versão atual usa pesos default, não treinados
2. **CPPS global**: Ideal seria CPPS per-frame (mais lento)
3. **H1-H2 instável**: Pode ser menos preciso para F0 > 350Hz

## Próximos Passos

1. Treinar pesos VMI via XGBoost com pseudo-labels GMM
2. Validar com anotações manuais em trechos conhecidos
3. Adicionar regularização temporal para estabilidade
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
