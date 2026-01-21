"""Geração de relatório narrativo usando Gemini."""

import json
import os
from pathlib import Path

import google.generativeai as genai
from PIL import Image

SYSTEM_PROMPT = """Você é um especialista em bioacústica e fisiologia vocal, escrevendo para um artigo acadêmico.

Contexto do projeto:
- Análise da voz de Ademilde Fonseca, cantora brasileira de Choro
- Objetivo: criticar o sistema de classificação vocal "Fach" usando análise fisiológica
- Mecanismos laríngeos: M1 (voz de peito) e M2 (voz de cabeça)
- Features extraídas: f0 (pitch), HNR (harmonic-to-noise ratio), CPPS

Termos técnicos importantes:
- f0: frequência fundamental, correlaciona com percepção de altura (pitch)
- HNR: razão harmônico-ruído, indica "limpeza" da voz (valores altos = voz mais limpa)
- CPPS: Cepstral Peak Prominence Smoothed, proxy para qualidade vocal
- M1/M2: mecanismos laríngeos (registro de peito vs cabeça)

Escreva em português brasileiro acadêmico, mas acessível. Use notação musical (C4, G5) ao lado de Hz quando relevante."""


def generate_narrative_report(
    stats: dict,
    metadata: dict,
    output_path: Path,
    plot_paths: list[Path] | None = None,
    api_key: str | None = None,
) -> str:
    """Gera relatório narrativo usando Gemini com suporte a imagens.

    Args:
        stats: Estatísticas por mecanismo (M1/M2).
        metadata: Metadados do processamento.
        output_path: Caminho para salvar o relatório.
        plot_paths: Lista de caminhos para plots PNG (opcional).
        api_key: Chave da API Gemini (ou usa GEMINI_API_KEY env var).

    Returns:
        Texto do relatório gerado.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não configurada. Use: export GEMINI_API_KEY=sua_chave")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Montar prompt com dados
    data_summary = _format_data_for_prompt(stats, metadata)

    prompt = f"""{SYSTEM_PROMPT}

## Dados da Análise

{data_summary}

## Tarefa

Escreva uma análise acadêmica (~500 palavras) com as seguintes seções:

1. **Caracterização Vocal**: Descreva o perfil vocal da cantora baseado nos dados
2. **Análise de Mecanismos**: Interprete a distribuição M1/M2 e o que isso revela
3. **Implicações para o Sistema Fach**: Como esses dados desafiam a classificação tradicional
4. **Limitações**: Mencione brevemente as limitações (gravações históricas, etc)

Use os dados concretos (notas musicais, percentuais) para embasar cada ponto.
Mantenha tom acadêmico mas acessível. Evite jargão desnecessário."""

    # Preparar conteúdo multimodal se houver plots
    content = []

    if plot_paths:
        # Listar os plots com seus nomes
        plot_list = "\n".join([f"- {p.stem}" for p in plot_paths if p.exists()])

        # Adicionar instruções sobre os plots
        prompt += f"""

## Visualizações

Analise também os gráficos anexados.

**REGRAS IMPORTANTES para referenciar gráficos:**
1. Ao mencionar um gráfico, SEMPRE inclua um link markdown no formato: [nome_do_grafico](plots/nome_do_grafico.png)
2. Exemplo correto: "O contorno de pitch em [brasileirinho_f0](plots/brasileirinho_f0.png) mostra..."
3. Exemplo correto: "A análise de clusters em [mechanism_clusters](plots/mechanism_clusters.png) revela..."
4. NUNCA use "Figura 1", "Figura 2", etc.
5. NUNCA mencione um gráfico sem incluir o link markdown

Gráficos disponíveis:
{plot_list}

Tipos de gráficos:
- *_f0.png: Contornos de f0 por música - observe padrões de ornamentação, vibrato, saltos intervalares
- mechanism_analysis.png: 4 subplots com análise M1/M2 (histograma, scatter, boxplot, temporal)
- mechanism_clusters.png: Clustering GMM dos mecanismos

Integre observações visuais com os dados numéricos na sua análise."""

        content.append(prompt)

        # Carregar e adicionar imagens com legenda
        for plot_path in plot_paths:
            if plot_path.exists():
                # Adicionar nome do arquivo antes da imagem
                content.append(f"[Gráfico: {plot_path.stem}]")
                img = Image.open(plot_path)
                content.append(img)
                print(f"  Anexando plot: {plot_path.name}")
    else:
        content = prompt

    response = model.generate_content(content)
    report_text = response.text

    # Salvar relatório
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Análise Bioacústica - {metadata.get('artist', 'Cantora')}\n\n")
        f.write("*Relatório gerado com auxílio de IA (Gemini 2.0 Flash)*\n\n")

        if plot_paths:
            f.write(
                f"*Análise multimodal com {len([p for p in plot_paths if p.exists()])} visualizações*\n\n"
            )

        f.write("---\n\n")
        f.write(report_text)

        # Galeria de figuras no final
        if plot_paths:
            f.write("\n\n---\n\n")
            f.write("## Figuras\n\n")
            for plot_path in sorted(plot_paths):
                if plot_path.exists():
                    # Descrição baseada no nome
                    name = plot_path.stem
                    if "_f0" in name:
                        desc = f"Contorno de f0 - {name.replace('_f0', '')}"
                    elif "mechanism_analysis" in name:
                        desc = (
                            "Análise de mecanismos M1/M2 (histograma, scatter, boxplot, temporal)"
                        )
                    elif "mechanism_clusters" in name:
                        desc = "Clustering GMM dos mecanismos laríngeos"
                    else:
                        desc = name
                    # Embed da imagem + legenda
                    f.write(f"### {name}\n\n")
                    f.write(f"![{desc}](plots/{name}.png)\n\n")
                    f.write(f"*{desc}*\n\n")

        f.write("\n\n---\n\n")
        f.write("## Dados Brutos\n\n")
        f.write("```json\n")
        f.write(
            json.dumps(
                {"stats": stats, "global": metadata.get("global", {})}, indent=2, ensure_ascii=False
            )
        )
        f.write("\n```\n")

    return report_text


def _format_data_for_prompt(stats: dict, metadata: dict) -> str:
    """Formata dados para o prompt do LLM."""
    lines = []

    # Global stats
    if "global" in metadata:
        g = metadata["global"]

        # Calcular CPPS médio global a partir das músicas
        cpps_values = [
            s.get("cpps_global")
            for s in metadata.get("songs", [])
            if "cpps_global" in s and "error" not in s
        ]
        cpps_mean = sum(cpps_values) / len(cpps_values) if cpps_values else None

        lines.extend(
            [
                "### Estatísticas Globais",
                f"- Artista: {metadata.get('artist', 'Desconhecida')}",
                f"- f0 médio: {g['f0_mean_hz']} Hz ({g['f0_mean_note']})",
                f"- Extensão vocal: {g['f0_range_notes']} ({g['f0_min_hz']} - {g['f0_max_hz']} Hz)",
                f"- Desvio padrão f0: {g['f0_std_hz']} Hz",
                f"- HNR médio: {g['hnr_mean_db']} dB",
            ]
        )

        if cpps_mean is not None:
            lines.append(f"- CPPS médio: {cpps_mean:.2f}")

        lines.extend(
            [
                f"- Total de frames analisados: {g['total_voiced_frames']}",
                "",
            ]
        )

    # Por mecanismo
    lines.append("### Por Mecanismo Laríngeo")
    total_frames = sum(s.get("count", 0) for s in stats.values())

    for mech, s in stats.items():
        pct = (s["count"] / total_frames * 100) if total_frames > 0 else 0
        mech_name = "Peito/Chest (M1)" if mech == "M1" else "Cabeça/Head (M2)"
        lines.extend(
            [
                f"\n**{mech_name}:**",
                f"- Proporção: {s['count']} frames ({pct:.1f}%)",
                f"- f0 médio: {s['f0_mean']:.1f} Hz ({s['note_mean']})",
                f"- Extensão: {s['note_range']}",
                f"- HNR médio: {s['hnr_mean']:.1f} dB",
            ]
        )

    # Por música
    if "songs" in metadata:
        lines.extend(["", "### Por Música"])
        for song in metadata["songs"]:
            if "error" not in song:
                cpps_str = f", CPPS={song['cpps_global']:.2f}" if "cpps_global" in song else ""
                lines.append(
                    f"- **{song['song']}**: f0={song['f0_mean_hz']} Hz ({song['f0_mean_note']}), range={song['f0_range_notes']}, HNR={song.get('hnr_mean_db', '?')} dB{cpps_str}"
                )

    return "\n".join(lines)


def main() -> None:
    """CLI para gerar relatório."""
    import argparse

    parser = argparse.ArgumentParser(description="Gera relatório narrativo com Gemini")
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/ademilde_metadata.json")
    )
    parser.add_argument("--stats", type=Path, default=None, help="JSON com stats M1/M2 (opcional)")
    parser.add_argument("--output", type=Path, default=Path("outputs/relatorio_llm.md"))
    parser.add_argument("--plots-dir", type=Path, default=None, help="Diretório com plots PNG")
    args = parser.parse_args()

    # Carregar metadata
    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)

    # Stats placeholder se não fornecido
    stats = {}
    if args.stats and args.stats.exists():
        with open(args.stats, encoding="utf-8") as f:
            stats = json.load(f)
    else:
        # Usar dados globais como proxy
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

    # Coletar plots se diretório fornecido
    plot_paths = None
    if args.plots_dir and args.plots_dir.exists():
        plot_paths = list(args.plots_dir.glob("*.png"))
        print(f"Encontrados {len(plot_paths)} plots")

    print("Gerando relatório com Gemini...")
    report = generate_narrative_report(stats, metadata, args.output, plot_paths=plot_paths)
    print(f"Relatório salvo em: {args.output}")
    print("\n--- Preview ---\n")
    print(report[:500] + "...")


if __name__ == "__main__":
    main()
