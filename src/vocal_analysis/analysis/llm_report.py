"""Narrative report generation using Gemini."""

import json
import os
from pathlib import Path

import google.generativeai as genai
from PIL import Image

# ---------------------------------------------------------------------------
# Dual-language system prompts
# ---------------------------------------------------------------------------

_TECHNICAL_TERMS = """
Technical terms:
- f0: fundamental frequency, correlates with pitch perception
- HNR: harmonic-to-noise ratio, indicates voice "clarity" (higher = cleaner voice)
- CPPS: Cepstral Peak Prominence Smoothed, proxy for vocal quality
- Jitter (ppq5): period perturbation, cycle-to-cycle frequency variation (%)
- Shimmer (apq11): amplitude perturbation, cycle-to-cycle amplitude variation (%)
- Formants (F1-F4): vocal tract resonances, related to timbral quality and vowel
- M1/M2: laryngeal mechanisms (chest register vs head register)
- VMI (Vocal Mechanism Index): continuous 0-1 metric replacing fixed frequency threshold
  - 0.0-0.2 = Dense M1 (full chest voice)
  - 0.2-0.4 = Light M1 (thin-edge chest voice)
  - 0.4-0.6 = Mix/Passaggio (transition zone)
  - 0.6-0.8 = Reinforced M2 (supported head voice)
  - 0.8-1.0 = Light M2 (light head voice)
- Alpha Ratio: energy ratio 0-1kHz vs 1-5kHz (more negative = darker/M1)
- H1-H2: difference between 1st and 2nd harmonic (glottal slope)
- Spectral Tilt: power spectrum slope
"""

SYSTEM_PROMPT_EN = f"""You are an expert in bioacoustics and vocal physiology, writing for an academic paper.

Project context:
- Analysis of the voice of Ademilde Fonseca, a Brazilian Choro singer
- Objective: critique the "Fach" vocal classification system using physiological analysis
- Laryngeal mechanisms: M1 (chest voice) and M2 (head voice)
- Extracted features: f0 (pitch), HNR (harmonic-to-noise ratio), CPPS, Jitter, Shimmer, Formants (F1-F4)

{_TECHNICAL_TERMS}

Write in academic but accessible English. Use musical notation (C4, G5) alongside Hz when relevant. Incorporate analysis of instability features (jitter/shimmer) and formants when available."""

SYSTEM_PROMPT_PT = """Você é um especialista em bioacústica e fisiologia vocal, escrevendo para um artigo acadêmico.

Contexto do projeto:
- Análise da voz de Ademilde Fonseca, cantora brasileira de Choro
- Objetivo: criticar o sistema de classificação vocal "Fach" usando análise fisiológica
- Mecanismos laríngeos: M1 (voz de peito) e M2 (voz de cabeça)
- Features extraídas: f0 (pitch), HNR (harmonic-to-noise ratio), CPPS, Jitter, Shimmer, Formantes (F1-F4)

Termos técnicos importantes:
- f0: frequência fundamental, correlaciona com percepção de altura (pitch)
- HNR: razão harmônico-ruído, indica "limpeza" da voz (valores altos = voz mais limpa)
- CPPS: Cepstral Peak Prominence Smoothed, proxy para qualidade vocal
- Jitter (ppq5): instabilidade de período, variação ciclo-a-ciclo na frequência (%)
- Shimmer (apq11): instabilidade de amplitude, variação ciclo-a-ciclo na amplitude (%)
- Formantes (F1-F4): ressonâncias do trato vocal, relacionadas à qualidade timbral e vogal
- M1/M2: mecanismos laríngeos (registro de peito vs cabeça)
- VMI (Vocal Mechanism Index): métrica contínua 0-1 que substitui threshold fixo de frequência
  - 0.0-0.2 = M1 Denso (voz de peito plena)
  - 0.2-0.4 = M1 Ligeiro (voz de peito leve)
  - 0.4-0.6 = Mix/Passaggio (zona de transição)
  - 0.6-0.8 = M2 Reforçado (voz de cabeça com corpo)
  - 0.8-1.0 = M2 Ligeiro (voz de cabeça leve)
- Alpha Ratio: razão de energia 0-1kHz vs 1-5kHz (valores mais negativos = mais grave/M1)
- H1-H2: diferença entre 1º e 2º harmônico (indica inclinação glotal)
- Spectral Tilt: inclinação do espectro de potência

Escreva em português brasileiro acadêmico, mas acessível. Use notação musical (C4, G5) ao lado de Hz quando relevante. Incorpore análise das features de instabilidade (jitter/shimmer) e formantes quando disponíveis."""

# ---------------------------------------------------------------------------
# Dual-language task prompts
# ---------------------------------------------------------------------------

_TASK_PROMPT_EN = """
## Task

Write an academic analysis (~500 words) with the following sections:

1. **Vocal Characterization**: Describe the singer's vocal profile based on the data
2. **Mechanism Analysis**: Interpret the M1/M2 distribution and what it reveals
3. **VMI Analysis**: If available, interpret the Vocal Mechanism Index distribution and how it captures nuances that binary M1/M2 classification misses (transition zones, mixed voice, etc.)
4. **Implications for the Fach System**: How this data challenges traditional classification
5. **Limitations**: Briefly mention limitations (historical recordings, etc.)

Use concrete data (musical notes, percentages) to support each point.
Maintain an academic but accessible tone. Avoid unnecessary jargon."""

_TASK_PROMPT_PT = """
## Tarefa

Escreva uma análise acadêmica (~500 palavras) com as seguintes seções:

1. **Caracterização Vocal**: Descreva o perfil vocal da cantora baseado nos dados
2. **Análise de Mecanismos**: Interprete a distribuição M1/M2 e o que isso revela
3. **Análise VMI**: Se disponível, interprete a distribuição do Vocal Mechanism Index e como ele captura nuances que a classificação binária M1/M2 não captura (zonas de transição, mix voice, etc)
4. **Implicações para o Sistema Fach**: Como esses dados desafiam a classificação tradicional
5. **Limitações**: Mencione brevemente as limitações (gravações históricas, etc)

Use os dados concretos (notas musicais, percentuais) para embasar cada ponto.
Mantenha tom acadêmico mas acessível. Evite jargão desnecessário."""

# ---------------------------------------------------------------------------
# Dual-language plot instructions
# ---------------------------------------------------------------------------

_PLOT_INSTRUCTIONS_EN = """
## Visualizations

Also analyze the attached plots.

**IMPORTANT RULES for referencing plots:**
1. When mentioning a plot, ALWAYS include a markdown link: [plot_name](plots/plot_name.png)
2. Correct example: "The pitch contour in [brasileirinho_f0](plots/brasileirinho_f0.png) shows..."
3. Correct example: "The cluster analysis in [mechanism_clusters](plots/mechanism_clusters.png) reveals..."
4. NEVER use "Figure 1", "Figure 2", etc.
5. NEVER mention a plot without including the markdown link

Plot types:
- *_f0.png: f0 contours per song - observe ornamentation patterns, vibrato, intervallic leaps
- mechanism_analysis.png: 4 subplots with M1/M2 analysis (histogram, scatter, boxplot, temporal)
- mechanism_clusters.png: GMM clustering of mechanisms
- vmi_analysis.png: 4 subplots with VMI analysis (F0 vs Alpha Ratio, VMI distribution, temporal contour, boxplot by category)
- vmi_scatter.png: Scatter plot F0 vs Alpha Ratio colored by VMI

Integrate visual observations with numerical data in your analysis."""

_PLOT_INSTRUCTIONS_PT = """
## Visualizações

Analise também os gráficos anexados.

**REGRAS IMPORTANTES para referenciar gráficos:**
1. Ao mencionar um gráfico, SEMPRE inclua um link markdown no formato: [nome_do_grafico](plots/nome_do_grafico.png)
2. Exemplo correto: "O contorno de pitch em [brasileirinho_f0](plots/brasileirinho_f0.png) mostra..."
3. Exemplo correto: "A análise de clusters em [mechanism_clusters](plots/mechanism_clusters.png) revela..."
4. NUNCA use "Figura 1", "Figura 2", etc.
5. NUNCA mencione um gráfico sem incluir o link markdown

Tipos de gráficos:
- *_f0.png: Contornos de f0 por música - observe padrões de ornamentação, vibrato, saltos intervalares
- mechanism_analysis.png: 4 subplots com análise M1/M2 (histograma, scatter, boxplot, temporal)
- mechanism_clusters.png: Clustering GMM dos mecanismos
- vmi_analysis.png: 4 subplots com análise VMI (F0 vs Alpha Ratio, distribuição VMI, contorno temporal, boxplot por categoria)
- vmi_scatter.png: Scatter plot F0 vs Alpha Ratio colorido por VMI

Integre observações visuais com os dados numéricos na sua análise."""

# ---------------------------------------------------------------------------
# Dual-language figure descriptions
# ---------------------------------------------------------------------------

_FIGURE_DESCS_EN = {
    "_f0": "f0 Contour - {}",
    "mechanism_analysis": "M1/M2 mechanism analysis (histogram, scatter, boxplot, temporal)",
    "mechanism_clusters": "GMM clustering of laryngeal mechanisms",
    "vmi_analysis": "VMI analysis (F0 vs Alpha Ratio, distribution, temporal contour)",
    "vmi_scatter": "Scatter F0 vs Alpha Ratio colored by VMI",
}

_FIGURE_DESCS_PT = {
    "_f0": "Contorno de f0 - {}",
    "mechanism_analysis": "Análise de mecanismos M1/M2 (histograma, scatter, boxplot, temporal)",
    "mechanism_clusters": "Clustering GMM dos mecanismos laríngeos",
    "vmi_analysis": "Análise VMI (F0 vs Alpha Ratio, distribuição, contorno temporal)",
    "vmi_scatter": "Scatter F0 vs Alpha Ratio colorido por VMI",
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_narrative_report(
    stats: dict,
    metadata: dict,
    output_path: Path,
    plot_paths: list[Path] | None = None,
    api_key: str | None = None,
    lang: str = "en",
) -> str:
    """Generate narrative report using Gemini with image support.

    Args:
        stats: Statistics per mechanism (M1/M2).
        metadata: Processing metadata.
        output_path: Path to save the report.
        plot_paths: List of paths to PNG plots (optional).
        api_key: Gemini API key (or uses GEMINI_API_KEY env var).
        lang: Report language ('en' or 'pt-BR').

    Returns:
        Generated report text.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured. Use: export GEMINI_API_KEY=your_key")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Select language-specific prompts
    system_prompt = SYSTEM_PROMPT_PT if lang == "pt-BR" else SYSTEM_PROMPT_EN
    task_prompt = _TASK_PROMPT_PT if lang == "pt-BR" else _TASK_PROMPT_EN
    plot_instructions = _PLOT_INSTRUCTIONS_PT if lang == "pt-BR" else _PLOT_INSTRUCTIONS_EN
    figure_descs = _FIGURE_DESCS_PT if lang == "pt-BR" else _FIGURE_DESCS_EN

    # Build prompt with data
    data_summary = _format_data_for_prompt(stats, metadata, lang=lang)

    data_header = "## Dados da Análise" if lang == "pt-BR" else "## Analysis Data"
    prompt = f"""{system_prompt}

{data_header}

{data_summary}

{task_prompt}"""

    # Prepare multimodal content if plots exist
    content = []

    if plot_paths:
        plot_list = "\n".join([f"- {p.stem}" for p in plot_paths if p.exists()])

        available_label = "Gráficos disponíveis:" if lang == "pt-BR" else "Available plots:"
        prompt += f"""
{plot_instructions}

{available_label}
{plot_list}"""

        content.append(prompt)

        # Load and attach images with caption
        chart_label = "Gráfico" if lang == "pt-BR" else "Chart"
        for plot_path in plot_paths:
            if plot_path.exists():
                content.append(f"[{chart_label}: {plot_path.stem}]")
                img = Image.open(plot_path)
                content.append(img)
                print(f"  Attaching plot: {plot_path.name}")
    else:
        content = prompt

    response = model.generate_content(content)
    report_text = response.text

    # Save report
    _write_report_file(output_path, report_text, stats, metadata, plot_paths, figure_descs, lang)

    return report_text


def _write_report_file(
    output_path: Path,
    report_text: str,
    stats: dict,
    metadata: dict,
    plot_paths: list[Path] | None,
    figure_descs: dict,
    lang: str,
) -> None:
    """Write the final report file with header, text, figures, and raw data."""
    # Language-specific labels
    if lang == "pt-BR":
        title_prefix = "Análise Bioacústica"
        default_artist = "Cantora"
        ai_note = "*Relatório gerado com auxílio de IA (Gemini 2.0 Flash)*"
        multimodal_note = "*Análise multimodal com {} visualizações*"
        figures_header = "## Figuras"
        raw_data_header = "## Dados Brutos"
    else:
        title_prefix = "Bioacoustic Analysis"
        default_artist = "Singer"
        ai_note = "*Report generated with AI assistance (Gemini 2.0 Flash)*"
        multimodal_note = "*Multimodal analysis with {} visualizations*"
        figures_header = "## Figures"
        raw_data_header = "## Raw Data"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {title_prefix} - {metadata.get('artist', default_artist)}\n\n")
        f.write(f"{ai_note}\n\n")

        if plot_paths:
            f.write(multimodal_note.format(len([p for p in plot_paths if p.exists()])) + "\n\n")

        f.write("---\n\n")
        f.write(report_text)

        # Figure gallery at the end
        if plot_paths:
            f.write("\n\n---\n\n")
            f.write(f"{figures_header}\n\n")
            for plot_path in sorted(plot_paths):
                if plot_path.exists():
                    name = plot_path.stem
                    desc = _get_figure_description(name, figure_descs)
                    f.write(f"### {name}\n\n")
                    f.write(f"![{desc}](plots/{name}.png)\n\n")
                    f.write(f"*{desc}*\n\n")

        f.write("\n\n---\n\n")
        f.write(f"{raw_data_header}\n\n")
        f.write("```json\n")
        f.write(
            json.dumps(
                {"stats": stats, "global": metadata.get("global", {})}, indent=2, ensure_ascii=False
            )
        )
        f.write("\n```\n")


def _get_figure_description(name: str, figure_descs: dict) -> str:
    """Get localized figure description based on filename."""
    if "_f0" in name:
        return figure_descs["_f0"].format(name.replace("_f0", ""))
    for key, desc in figure_descs.items():
        if key in name and key != "_f0":
            return desc
    return name


def _format_data_for_prompt(stats: dict, metadata: dict, lang: str = "en") -> str:
    """Format data for the LLM prompt."""
    from pathlib import Path

    import pandas as pd

    lines = []

    # Language-specific labels
    if lang == "pt-BR":
        lbl = {
            "global_stats": "### Estatísticas Globais",
            "artist": "Artista",
            "unknown": "Desconhecida",
            "mean_f0": "f0 médio",
            "vocal_range": "Extensão vocal",
            "f0_std": "Desvio padrão f0",
            "mean_hnr": "HNR médio",
            "mean_cpps": "CPPS médio",
            "mean_jitter": "Jitter médio (ppq5)",
            "mean_shimmer": "Shimmer médio (apq11)",
            "formants_header": "\n### Formantes (F1-F4) - Médias Globais",
            "vmi_header": "\n### VMI (Vocal Mechanism Index)",
            "mean_vmi": "VMI médio",
            "spectral_header": "\n### Features Espectrais - Médias Globais",
            "total_frames": "Total de frames analisados",
            "by_mechanism": "### Por Mecanismo Laríngeo",
            "chest": "Peito/Chest (M1)",
            "head": "Cabeça/Head (M2)",
            "proportion": "Proporção",
            "range": "Extensão",
            "by_song": "### Por Música",
            "csv_error": "Nota: Erro ao carregar dados do CSV",
        }
    else:
        lbl = {
            "global_stats": "### Global Statistics",
            "artist": "Artist",
            "unknown": "Unknown",
            "mean_f0": "Mean f0",
            "vocal_range": "Vocal range",
            "f0_std": "f0 standard deviation",
            "mean_hnr": "Mean HNR",
            "mean_cpps": "Mean CPPS",
            "mean_jitter": "Mean Jitter (ppq5)",
            "mean_shimmer": "Mean Shimmer (apq11)",
            "formants_header": "\n### Formants (F1-F4) - Global Means",
            "vmi_header": "\n### VMI (Vocal Mechanism Index)",
            "mean_vmi": "Mean VMI",
            "spectral_header": "\n### Spectral Features - Global Means",
            "total_frames": "Total analyzed frames",
            "by_mechanism": "### By Laryngeal Mechanism",
            "chest": "Chest (M1)",
            "head": "Head (M2)",
            "proportion": "Proportion",
            "range": "Range",
            "by_song": "### Per Song",
            "csv_error": "Note: Error loading CSV data",
        }

    # Global stats
    if "global" in metadata:
        g = metadata["global"]

        # Compute global means for jitter, shimmer
        jitter_values = [
            s.get("jitter_ppq5")
            for s in metadata.get("songs", [])
            if "jitter_ppq5" in s and "error" not in s
        ]
        jitter_mean = sum(jitter_values) / len(jitter_values) if jitter_values else None

        shimmer_values = [
            s.get("shimmer_apq11")
            for s in metadata.get("songs", [])
            if "shimmer_apq11" in s and "error" not in s
        ]
        shimmer_mean = sum(shimmer_values) / len(shimmer_values) if shimmer_values else None

        cpps_values = [
            s.get("cpps_global")
            for s in metadata.get("songs", [])
            if "cpps_global" in s and "error" not in s
        ]
        cpps_mean = sum(cpps_values) / len(cpps_values) if cpps_values else None

        lines.extend(
            [
                lbl["global_stats"],
                f"- {lbl['artist']}: {metadata.get('artist', lbl['unknown'])}",
                f"- {lbl['mean_f0']}: {g['f0_mean_hz']} Hz ({g['f0_mean_note']})",
                f"- {lbl['vocal_range']}: {g['f0_range_notes']} ({g['f0_min_hz']} - {g['f0_max_hz']} Hz)",
                f"- {lbl['f0_std']}: {g['f0_std_hz']} Hz",
                f"- {lbl['mean_hnr']}: {g['hnr_mean_db']} dB",
            ]
        )

        if cpps_mean is not None:
            lines.append(f"- {lbl['mean_cpps']}: {cpps_mean:.2f}")
        if jitter_mean is not None:
            lines.append(f"- {lbl['mean_jitter']}: {jitter_mean:.3%}")
        if shimmer_mean is not None:
            lines.append(f"- {lbl['mean_shimmer']}: {shimmer_mean:.3%}")

        # Try loading CSV for formant and VMI statistics
        csv_path = Path("data/processed/ademilde_features.csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df_voiced = df[(df["confidence"] > 0.8) & (df["hnr"] > -10)]

                formant_cols = ["f1", "f2", "f3", "f4"]
                available_formants = [col for col in formant_cols if col in df_voiced.columns]

                if available_formants:
                    lines.append(lbl["formants_header"])
                    for col in available_formants:
                        mean_val = df_voiced[col].mean()
                        lines.append(f"- {col.upper()}: {mean_val:.1f} Hz")

                if "vmi" in df_voiced.columns and "vmi_label" in df_voiced.columns:
                    lines.append(lbl["vmi_header"])
                    lines.append(f"- {lbl['mean_vmi']}: {df_voiced['vmi'].mean():.3f}")

                    vmi_counts = df_voiced["vmi_label"].value_counts()
                    total = len(df_voiced)
                    for label in [
                        "M1_HEAVY",
                        "M1_LIGHT",
                        "MIX_PASSAGGIO",
                        "M2_REINFORCED",
                        "M2_LIGHT",
                    ]:
                        if label in vmi_counts.index:
                            count = vmi_counts[label]
                            pct = count / total * 100
                            lines.append(f"- {label}: {count} frames ({pct:.1f}%)")

                spectral_cols = ["alpha_ratio", "h1_h2", "spectral_tilt"]
                available_spectral = [col for col in spectral_cols if col in df_voiced.columns]
                if available_spectral:
                    lines.append(lbl["spectral_header"])
                    for col in available_spectral:
                        mean_val = df_voiced[col].mean()
                        lines.append(f"- {col}: {mean_val:.2f} dB")
            except Exception as e:
                lines.append(f"\n*{lbl['csv_error']}: {e}*")

        lines.extend(
            [
                f"- {lbl['total_frames']}: {g['total_voiced_frames']}",
                "",
            ]
        )

    # Per mechanism
    lines.append(lbl["by_mechanism"])
    total_frames = sum(s.get("count", 0) for s in stats.values())

    for mech, s in stats.items():
        pct = (s["count"] / total_frames * 100) if total_frames > 0 else 0
        mech_name = lbl["chest"] if mech == "M1" else lbl["head"]
        lines.extend(
            [
                f"\n**{mech_name}:**",
                f"- {lbl['proportion']}: {s['count']} frames ({pct:.1f}%)",
                f"- {lbl['mean_f0']}: {s['f0_mean']:.1f} Hz ({s['note_mean']})",
                f"- {lbl['range']}: {s['note_range']}",
                f"- {lbl['mean_hnr']}: {s['hnr_mean']:.1f} dB",
            ]
        )

    # Per song
    if "songs" in metadata:
        lines.extend(["", lbl["by_song"]])
        for song in metadata["songs"]:
            if "error" not in song:
                features = [
                    f"f0={song['f0_mean_hz']} Hz ({song['f0_mean_note']})",
                    f"range={song['f0_range_notes']}",
                    f"HNR={song.get('hnr_mean_db', '?')} dB",
                ]

                if "cpps_global" in song:
                    features.append(f"CPPS={song['cpps_global']:.2f}")
                if "jitter_ppq5" in song:
                    features.append(f"Jitter={song['jitter_ppq5']:.3%}")
                if "shimmer_apq11" in song:
                    features.append(f"Shimmer={song['shimmer_apq11']:.3%}")

                try:
                    csv_path = Path("data/processed/ademilde_features.csv")
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        song_df = df[df["song"] == song["song"]]
                        song_df_voiced = song_df[
                            (song_df["confidence"] > 0.8) & (song_df["hnr"] > -10)
                        ]

                        formant_cols = ["f1", "f2", "f3", "f4"]
                        for col in formant_cols:
                            if col in song_df_voiced.columns:
                                mean_val = song_df_voiced[col].mean()
                                features.append(f"{col.upper()}={mean_val:.1f} Hz")
                except Exception:
                    pass

                lines.append(f"- **{song['song']}**: {', '.join(features)}")

    return "\n".join(lines)


def main() -> None:
    """CLI to generate narrative report."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate narrative report with Gemini")
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/ademilde_metadata.json")
    )
    parser.add_argument("--stats", type=Path, default=None, help="JSON with M1/M2 stats (optional)")
    parser.add_argument("--output", type=Path, default=Path("outputs/llm_report.md"))
    parser.add_argument("--plots-dir", type=Path, default=None, help="Directory with PNG plots")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=["en", "pt-BR"],
        help="Report language (default: REPORT_LANG env var or 'en')",
    )
    args = parser.parse_args()

    lang = args.lang or os.environ.get("REPORT_LANG", "en")

    # Load metadata
    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)

    # Stats placeholder if not provided
    stats = {}
    if args.stats and args.stats.exists():
        with open(args.stats, encoding="utf-8") as f:
            stats = json.load(f)
    else:
        # Use global data as proxy
        g = metadata.get("global", {})
        stats = {
            "M1": {
                "count": int(g.get("total_voiced_frames", 0) * 0.4),
                "f0_mean": g.get("f0_mean_hz", 300) * 0.8,
                "note_mean": "D4",
                "note_range": "D3 – G4",
                "hnr_mean": g.get("hnr_mean_db", 5),
            },
            "M2": {
                "count": int(g.get("total_voiced_frames", 0) * 0.6),
                "f0_mean": g.get("f0_mean_hz", 300) * 1.4,
                "note_mean": "A4",
                "note_range": "G4 – G5",
                "hnr_mean": g.get("hnr_mean_db", 5) * 0.8,
            },
        }

    # Collect plots if directory provided
    plot_paths = None
    if args.plots_dir and args.plots_dir.exists():
        plot_paths = list(args.plots_dir.glob("*.png"))
        print(f"Found {len(plot_paths)} plots")

    print("Generating report with Gemini...")
    report = generate_narrative_report(
        stats, metadata, args.output, plot_paths=plot_paths, lang=lang
    )
    print(f"Report saved to: {args.output}")
    print("\n--- Preview ---\n")
    print(report[:500] + "...")


if __name__ == "__main__":
    main()
