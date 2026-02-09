"""Exploratory analysis of laryngeal mechanisms M1/M2."""

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
    """Statistics per mechanism."""

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
    """Analyze pitch regions by separating with M1/M2 threshold.

    Args:
        df: DataFrame with columns 'f0', 'hnr', 'confidence'.
        threshold_hz: M1/M2 separation threshold in Hz (default 400 Hz ~ G4).
        output_dir: Directory to save plots.

    Returns:
        Dictionary with statistics per mechanism.
    """
    # Filter voiced frames: confidence > 0.85 + HNR > 0 (removes silence and false positives)
    df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)].copy()

    # Classify by threshold
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
    """Cluster frames into mechanisms using f0 and HNR.

    Args:
        df: DataFrame with columns 'f0', 'hnr'.
        n_clusters: Number of clusters.
        method: Clustering method ('kmeans' or 'gmm').
        output_dir: Directory to save plots.

    Returns:
        DataFrame with 'cluster' column added.
    """
    df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)].copy()

    # Normalize features (RobustScaler uses median/IQR, more robust to outliers)
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

    # Sort clusters by mean f0 (cluster 0 = lowest = M1)
    cluster_means = df_voiced.groupby("cluster")["f0"].mean()
    cluster_order = cluster_means.sort_values().index.tolist()
    df_voiced["mechanism"] = df_voiced["cluster"].map(
        {cluster_order[i]: f"M{i + 1}" for i in range(len(cluster_order))}
    )

    if output_dir:
        _plot_clusters(df_voiced, output_dir)

    return df_voiced


class VMIStats(TypedDict):
    """Statistics per VMI category."""

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
    """Analyze mechanisms using tessitura-agnostic VMI (Vocal Mechanism Index).

    Replaces the fixed G4 threshold with spectral feature-based analysis.

    Args:
        df: DataFrame with spectral columns: 'alpha_ratio', 'h1_h2', 'spectral_tilt',
            and optionally 'cpps_per_frame'. Must also have 'f0', 'confidence'.
        weights: Weights for VMI computation. Default: standard weights.
        smoothing_method: Temporal smoothing method ('median', 'mean', 'exponential', 'none').
        smoothing_window: Smoothing window size.
        output_dir: Directory to save plots.

    Returns:
        Tuple (DataFrame with VMI, statistics per VMI category).
    """
    # Filter voiced frames
    df_voiced = df[(df["confidence"] > 0.85)].copy()

    # Check required features
    required_cols = ["alpha_ratio", "h1_h2", "spectral_tilt"]
    missing_cols = [c for c in required_cols if c not in df_voiced.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for VMI: {missing_cols}")

    # Use CPPS per-frame if available, otherwise use global CPPS or placeholder
    if "cpps_per_frame" in df_voiced.columns:
        cpps = df_voiced["cpps_per_frame"].values
    elif "cpps_global" in df_voiced.columns:
        cpps = np.full(len(df_voiced), df_voiced["cpps_global"].iloc[0])
    else:
        # Neutral placeholder
        cpps = np.full(len(df_voiced), 0.5)

    # Compute VMI
    vmi_raw = compute_vmi_fixed(
        alpha_ratio=df_voiced["alpha_ratio"].values,
        cpps=cpps,
        h1_h2=df_voiced["h1_h2"].values,
        spectral_tilt=df_voiced["spectral_tilt"].values,
        weights=weights,
    )

    # Apply temporal smoothing
    if smoothing_method != "none":
        vmi_smoothed = apply_temporal_smoothing(
            vmi_raw,
            window_size=smoothing_window,
            method=smoothing_method,
        )
    else:
        vmi_smoothed = vmi_raw

    # Add to DataFrame
    df_voiced["vmi"] = vmi_smoothed
    df_voiced["vmi_label"] = vmi_to_label(vmi_smoothed)

    # Compute statistics per category
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
    """Generate VMI analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # 1. Scatter F0 vs Alpha Ratio colored by VMI
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
    ax.set_title("F0 vs Alpha Ratio (color = VMI)")

    # 2. VMI histogram
    ax = axes[0, 1]
    ax.hist(df["vmi"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        ax.axvline(threshold, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("VMI")
    ax.set_ylabel("Frequency")
    ax.set_title("VMI Distribution")

    # 3. VMI timeline
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
        ax.set_xlabel("Time (s)")
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
    ax.set_title("F0 Contour (color = VMI)")

    # 4. Boxplot of VMI by category
    ax = axes[1, 1]
    order = [label.value for label in VMILabel]
    present_labels = [lbl for lbl in order if lbl in df["vmi_label"].values]
    if present_labels:
        sns.boxplot(data=df, x="vmi_label", y="f0", order=present_labels, ax=ax)
        ax.set_xlabel("VMI Label")
        ax.set_ylabel("F0 (Hz)")
        ax.set_title("F0 by VMI Category")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(output_dir / "vmi_analysis.png", dpi=150)
    plt.close(fig)


def _plot_mechanism_analysis(df: pd.DataFrame, threshold_hz: float, output_dir: Path) -> None:
    """Generate mechanism analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # 1. f0 histogram with threshold
    ax = axes[0, 0]
    for mech, color in [("M1", "steelblue"), ("M2", "coral")]:
        subset = df[df["mechanism"] == mech]
        ax.hist(subset["f0"], bins=50, alpha=0.6, label=mech, color=color)
    ax.axvline(threshold_hz, color="red", linestyle="--", label=f"Threshold ({threshold_hz} Hz)")
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("f0 Distribution by Mechanism")
    ax.legend()

    # 2. Scatter f0 vs HNR
    ax = axes[0, 1]
    sns.scatterplot(data=df, x="f0", y="hnr", hue="mechanism", alpha=0.5, ax=ax)
    ax.axvline(threshold_hz, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel("HNR (dB)")
    ax.set_title("f0 vs HNR by Mechanism")

    # 3. HNR boxplot by mechanism
    ax = axes[1, 0]
    sns.boxplot(data=df, x="mechanism", y="hnr", ax=ax)
    ax.set_xlabel("Mechanism")
    ax.set_ylabel("HNR (dB)")
    ax.set_title("HNR by Mechanism")

    # 4. Timeline colored by mechanism
    ax = axes[1, 1]
    colors = {"M1": "steelblue", "M2": "coral"}
    for mech in ["M1", "M2"]:
        subset = df[df["mechanism"] == mech]
        ax.scatter(subset["time"], subset["f0"], c=colors[mech], s=1, alpha=0.6, label=mech)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.set_title("f0 Contour by Mechanism")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "mechanism_analysis.png", dpi=150)
    plt.close(fig)


def _plot_clusters(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot clustering results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(data=df, x="f0", y="hnr", hue="mechanism", alpha=0.6, ax=ax)
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel("HNR (dB)")
    ax.set_title("Mechanism Clusters (GMM)")

    fig.savefig(output_dir / "mechanism_clusters.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation — dual-language templates (EN / PT-BR)
# ---------------------------------------------------------------------------


def generate_report(
    df: pd.DataFrame,
    stats: dict[str, MechanismStats],
    output_path: Path,
    artist_name: str = "Ademilde Fonseca",
    xgb_report: str | None = None,
    xgb_feature_cols: list[str] | None = None,
    lang: str = "en",
) -> None:
    """Generate markdown report with full analysis.

    Args:
        df: DataFrame with data.
        stats: Statistics per mechanism.
        output_path: Path for the .md file.
        artist_name: Artist name.
        xgb_report: XGBoost classification report (string), if available.
        xgb_feature_cols: List of features used in XGBoost.
        lang: Report language ('en' or 'pt-BR').
    """
    if lang == "pt-BR":
        report = _generate_report_pt(df, stats, artist_name, xgb_report, xgb_feature_cols)
    else:
        report = _generate_report_en(df, stats, artist_name, xgb_report, xgb_feature_cols)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def _generate_report_en(
    df: pd.DataFrame,
    stats: dict[str, MechanismStats],
    artist_name: str,
    xgb_report: str | None,
    xgb_feature_cols: list[str] | None,
) -> str:
    """Generate English analysis report."""
    df_voiced = df[(df["confidence"] > 0.85) & (df["hnr"] > 0)]

    f0_global_mean = df_voiced["f0"].mean()
    f0_global_min = df_voiced["f0"].min()
    f0_global_max = df_voiced["f0"].max()
    hnr_global_mean = df_voiced["hnr"].mean()

    report = f"""# Bioacoustic Analysis - {artist_name}

## Global Summary

| Metric | Value | Note |
|--------|-------|------|
| **Mean f0** | {f0_global_mean:.1f} Hz | {hz_to_note(f0_global_mean)} |
| **Min f0** | {f0_global_min:.1f} Hz | {hz_to_note(f0_global_min)} |
| **Max f0** | {f0_global_max:.1f} Hz | {hz_to_note(f0_global_max)} |
| **Range** | {hz_range_to_notes(f0_global_min, f0_global_max)} | ~{np.log2(f0_global_max / f0_global_min):.1f} octaves |
| **Mean HNR** | {hnr_global_mean:.1f} dB | – |
| **Total frames** | {len(df_voiced)} | – |

## Mechanism Analysis

"""

    for mech, s in stats.items():
        pct = (s["count"] / len(df_voiced)) * 100
        report += f"""### {mech} ({"Chest" if mech == "M1" else "Head"})

| Metric | Value | Note |
|--------|-------|------|
| **Frames** | {s["count"]} ({pct:.1f}%) | – |
| **Mean f0** | {s["f0_mean"]:.1f} Hz | {s["note_mean"]} |
| **f0 Std Dev** | {s["f0_std"]:.1f} Hz | – |
| **Range** | {s["note_range"]} | – |
| **Mean HNR** | {s["hnr_mean"]:.1f} dB | – |

"""

    if "song" in df.columns:
        report += "## Per Song\n\n"
        for song in df["song"].unique():
            song_df = df_voiced[df_voiced["song"] == song]
            report += f"""### {song}

- Mean f0: {song_df["f0"].mean():.1f} Hz ({hz_to_note(song_df["f0"].mean())})
- Range: {hz_range_to_notes(song_df["f0"].min(), song_df["f0"].max())}
- Mean HNR: {song_df["hnr"].mean():.1f} dB

"""

    if xgb_report:
        features_str = ", ".join(f"`{c}`" for c in (xgb_feature_cols or []))
        report += f"""## XGBoost Classification (GMM Pseudo-Labels)

Features used: {features_str}
Training labels: GMM clusters (unsupervised)
Split: 80% train / 20% test

```
{xgb_report}```

"""

    report += """## Interpretation

### Bimodal Pattern

The f0 contour shows clear alternation between two regions:
- **Low region (M1)**: Mechanism 1 / chest voice
- **High region (M2)**: Mechanism 2 / head voice

### Implications for "Fach" Classification

The analysis suggests that traditional "voice type" classification does not capture
the physiological reality of laryngeal mechanisms. The singer uses both mechanisms
fluidly, contradicting rigid categorizations.

### Limitations

- Historical recordings with low quality (reduced HNR)
- M1/M2 classification via GMM (sensitive to training data)
- CPPS compromised by background noise

## Next Steps

1. Analyze transitions between mechanisms (register breaks)
2. Compare with contemporary singers (high-quality recordings)
3. Validate VMI with manual annotations from a speech-language pathologist
"""
    return report


def _generate_report_pt(
    df: pd.DataFrame,
    stats: dict[str, MechanismStats],
    artist_name: str,
    xgb_report: str | None,
    xgb_feature_cols: list[str] | None,
) -> str:
    """Generate Portuguese analysis report."""
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
- Classificação M1/M2 via GMM (sensível a dados de treino)
- CPPS comprometido pelo ruído de fundo

## Próximos Passos

1. Analisar transições entre mecanismos (quebras de registro)
2. Comparar com cantoras contemporâneas (gravações de alta qualidade)
3. Validar VMI com anotações manuais de fonoaudiólogo
"""
    return report


def generate_vmi_report(
    df: pd.DataFrame,
    vmi_stats: dict[str, VMIStats],
    output_path: Path,
    artist_name: str = "Artist",
    lang: str = "en",
) -> None:
    """Generate markdown report with VMI analysis.

    Args:
        df: DataFrame with data and computed VMI.
        vmi_stats: Statistics per VMI category.
        output_path: Path for the .md file.
        artist_name: Artist name.
        lang: Report language ('en' or 'pt-BR').
    """
    if lang == "pt-BR":
        report = _generate_vmi_report_pt(df, vmi_stats, artist_name)
    else:
        report = _generate_vmi_report_en(df, vmi_stats, artist_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def _generate_vmi_report_en(
    df: pd.DataFrame,
    vmi_stats: dict[str, VMIStats],
    artist_name: str,
) -> str:
    """Generate English VMI report."""
    df_voiced = df[df["confidence"] > 0.85]

    f0_global_mean = df_voiced["f0"].mean()
    f0_global_min = df_voiced["f0"].min()
    f0_global_max = df_voiced["f0"].max()
    vmi_global_mean = df_voiced["vmi"].mean()

    report = f"""# VMI Analysis (Vocal Mechanism Index) - {artist_name}

## Methodology

This report uses the **VMI (Vocal Mechanism Index)**, a continuous metric that
replaces the arbitrary G4 threshold (400 Hz) with spectral feature-based analysis:

- **Alpha Ratio**: Energy ratio 0-1kHz vs 1-5kHz
- **H1-H2**: Difference between 1st and 2nd harmonic (glottal slope)
- **Spectral Tilt**: Power spectrum slope
- **CPPS**: Cepstral peak prominence (periodicity)

VMI ranges from **0.0 (Dense M1)** to **1.0 (Light M2)**, enabling gradual
identification of vocal mechanism without relying on fixed frequencies.

---

## Global Summary

| Metric | Value | Note |
|--------|-------|------|
| **Mean F0** | {f0_global_mean:.1f} Hz | {hz_to_note(f0_global_mean)} |
| **Min F0** | {f0_global_min:.1f} Hz | {hz_to_note(f0_global_min)} |
| **Max F0** | {f0_global_max:.1f} Hz | {hz_to_note(f0_global_max)} |
| **Range** | {hz_range_to_notes(f0_global_min, f0_global_max)} | ~{np.log2(f0_global_max / f0_global_min):.1f} octaves |
| **Mean VMI** | {vmi_global_mean:.3f} | – |
| **Total frames** | {len(df_voiced)} | – |

---

## VMI Category Analysis

| Category | Frames | % | Mean VMI | Mean F0 | Alpha Ratio | H1-H2 |
|----------|--------|---|----------|---------|-------------|-------|
"""

    for label_name, s in vmi_stats.items():
        report += f"| **{label_name}** | {s['count']} | {s['percentage']:.1f}% | {s['vmi_mean']:.3f} | {s['f0_mean']:.1f} Hz | {s['alpha_ratio_mean']:.1f} dB | {s['h1_h2_mean']:.1f} dB |\n"

    report += """
### Category Interpretation

- **M1_HEAVY (VMI 0.0-0.2)**: Heavy mechanism, firm adduction, full chest voice
- **M1_LIGHT (VMI 0.2-0.4)**: Thin-edge M1, common in tenors/middle register
- **MIX_PASSAGGIO (VMI 0.4-0.6)**: Passaggio zone, acoustic instability, mixed voice
- **M2_REINFORCED (VMI 0.6-0.8)**: M2 with glottic adduction, frontal resonance
- **M2_LIGHT (VMI 0.8-1.0)**: Light mechanism, falsetto, piano M2

---

## Per Song Analysis

"""

    if "song" in df.columns:
        for song in df["song"].unique():
            song_df = df_voiced[df_voiced["song"] == song]
            if len(song_df) > 0:
                report += f"""### {song}

- Mean F0: {song_df["f0"].mean():.1f} Hz ({hz_to_note(song_df["f0"].mean())})
- Mean VMI: {song_df["vmi"].mean():.3f}
- Distribution: {(song_df["vmi_label"].value_counts() / len(song_df) * 100).to_dict()}

"""

    report += """---

## VMI Advantages

1. **Tessitura-agnostic**: Does not depend on fixed frequencies like G4
2. **Continuous**: Captures gradations between mechanisms (passaggio)
3. **Multi-dimensional**: Combines multiple spectral features
4. **Interpretable**: Each feature has clear physiological meaning

## Limitations

1. **Fixed weights**: Current version uses default weights, not trained
2. **Global CPPS**: Ideally would use CPPS per-frame (slower)
3. **Unstable H1-H2**: May be less accurate for F0 > 350Hz

## Next Steps

1. Train VMI weights via XGBoost with GMM pseudo-labels
2. Validate with manual annotations on known passages
3. Add temporal regularization for stability
"""
    return report


def _generate_vmi_report_pt(
    df: pd.DataFrame,
    vmi_stats: dict[str, VMIStats],
    artist_name: str,
) -> str:
    """Generate Portuguese VMI report."""
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

- **M1_HEAVY (VMI 0.0-0.2)**: Mecanismo pesado, adução firme, voz de peito plena
- **M1_LIGHT (VMI 0.2-0.4)**: M1 de borda fina, comum em tenores/registro médio
- **MIX_PASSAGGIO (VMI 0.4-0.6)**: Zona de passagem, instabilidade acústica, voz mista
- **M2_REINFORCED (VMI 0.6-0.8)**: M2 com adução glótica, ressonância frontal
- **M2_LIGHT (VMI 0.8-1.0)**: Mecanismo leve, falsete, piano M2

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
    return report
