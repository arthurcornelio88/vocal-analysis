"""Análise exploratória de mecanismos laríngeos M1/M2."""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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
    # Filtrar voiced frames: confidence > 0.8 + HNR > -10 (remove silêncio)
    df_voiced = df[(df["confidence"] > 0.8) & (df["hnr"] > -10)].copy()

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
    df_voiced = df[(df["confidence"] > 0.8) & (df["hnr"] > -10)].copy()

    # Normalizar features
    features = df_voiced[["f0", "hnr"]].copy()
    features_norm = (features - features.mean()) / features.std()

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
    df_voiced = df[(df["confidence"] > 0.8) & (df["hnr"] > -10)]

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
