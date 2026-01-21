"""Script para rodar análise exploratória completa."""

import json
import os
from pathlib import Path

import pandas as pd

from vocal_analysis.analysis.exploratory import (
    analyze_mechanism_regions,
    cluster_mechanisms,
    generate_report,
)
from vocal_analysis.analysis.llm_report import generate_narrative_report
from vocal_analysis.features.articulation import (
    compute_articulation_features,
    get_articulation_stats,
)
from vocal_analysis.modeling.classifier import train_mechanism_classifier


def main() -> None:
    """Executa análise exploratória dos dados processados."""
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "processed" / "ademilde_features.csv"
    metadata_path = project_root / "data" / "processed" / "ademilde_metadata.json"
    output_dir = project_root / "outputs"

    if not data_path.exists():
        print(f"Arquivo não encontrado: {data_path}")
        print("Execute primeiro: uv run python -m vocal_analysis.preprocessing.process_ademilde")
        return

    # Carregar dados
    print("Carregando dados...")
    df = pd.read_csv(data_path)

    metadata = None
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Artista: {metadata.get('artist', 'Desconhecido')}")
        print(f"  Músicas: {metadata.get('n_success', '?')}")

    print(f"  Frames: {len(df)}")

    # Computar features de agilidade articulatória
    print("\nComputando features de agilidade articulatória...")
    df = compute_articulation_features(df)
    articulation_stats = get_articulation_stats(df)
    print(f"  f0 velocity médio: {articulation_stats['f0_velocity_mean']:.1f} Hz/s")
    print(f"  Taxa silábica: {articulation_stats['syllable_rate']:.2f} sílabas/s")

    # Análise por threshold
    print("\nAnalisando por threshold (400 Hz / G4)...")
    stats = analyze_mechanism_regions(df, threshold_hz=400.0, output_dir=output_dir / "plots")

    for mech, s in stats.items():
        print(f"\n  {mech} ({'Peito' if mech == 'M1' else 'Cabeça'}):")
        print(f"    Frames: {s['count']}")
        print(f"    f0: {s['f0_mean']:.1f} Hz ({s['note_mean']})")
        print(f"    Range: {s['note_range']}")
        print(f"    HNR: {s['hnr_mean']:.1f} dB")

    # Clustering
    print("\nExecutando clustering GMM...")
    plots_dir = output_dir / "plots"
    df_clustered = cluster_mechanisms(df, n_clusters=2, method="gmm", output_dir=plots_dir)

    # XGBoost: treinar com labels do GMM como pseudo-labels
    print("\nTreinando XGBoost com pseudo-labels do GMM...")
    df_train = df_clustered[["f0", "hnr", "energy"]].copy()
    df_train["mechanism_label"] = df_clustered["mechanism"].map({"M1": 0, "M2": 1})

    try:
        train_mechanism_classifier(df_train, target_col="mechanism_label")
        print("  Modelo XGBoost treinado com sucesso!")
    except Exception as e:
        print(f"  Erro ao treinar XGBoost: {e}")

    # Gerar relatório básico
    artist_name = metadata.get("artist", "Desconhecido") if metadata else "Desconhecido"
    report_path = output_dir / "analise_ademilde.md"
    print(f"\nGerando relatório básico: {report_path}")
    generate_report(df, stats, report_path, artist_name=artist_name)

    # Gerar relatório com LLM se API key disponível
    if os.environ.get("GEMINI_API_KEY"):
        llm_report_path = output_dir / "relatorio_llm.md"
        print(f"Gerando relatório narrativo com Gemini: {llm_report_path}")

        # Coletar plots para análise multimodal
        plot_paths = list((output_dir / "plots").glob("*.png"))
        print(f"  Anexando {len(plot_paths)} plots para análise multimodal")

        try:
            generate_narrative_report(stats, metadata, llm_report_path, plot_paths=plot_paths)
            print("  Relatório LLM gerado com sucesso!")
        except Exception as e:
            print(f"  Erro ao gerar relatório LLM: {e}")
    else:
        print("\nDica: Configure GEMINI_API_KEY para gerar relatório narrativo com IA")
        print("  export GEMINI_API_KEY=sua_chave")

    print("\nConcluído!")
    print(f"  - Plots: {output_dir / 'plots'}")
    print(f"  - Relatório: {report_path}")


if __name__ == "__main__":
    main()
