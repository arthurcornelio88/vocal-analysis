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
from vocal_analysis.visualization.plots import (
    plot_xgb_mechanism_excerpt,
    plot_xgb_mechanism_timeline,
)


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

    # XGBoost: treinar com labels do GMM como pseudo-labels e predizer todos os dados
    print("\nTreinando XGBoost com pseudo-labels do GMM...")
    base_cols = ["f0", "hnr", "energy", "f0_velocity", "f0_acceleration"]
    optional_cols = ["f1", "f2", "f3", "f4"]
    feature_cols = base_cols + [c for c in optional_cols if c in df_clustered.columns]
    print(f"  Features do modelo: {feature_cols}")
    df_train = df_clustered[feature_cols].copy()
    df_train["mechanism_label"] = df_clustered["mechanism"].map({"M1": 0, "M2": 1})

    xgb_report = None
    try:
        model, xgb_report = train_mechanism_classifier(
            df_train, feature_cols=feature_cols, target_col="mechanism_label"
        )
        # Predizer sobre todos os dados voiced (não apenas o split de teste)
        df_clustered["xgb_mechanism"] = model.predict(df_clustered[feature_cols]).tolist()
        df_clustered["xgb_mechanism"] = df_clustered["xgb_mechanism"].map({0: "M1", 1: "M2"})

        # Salvar predições
        pred_path = output_dir / "xgb_predictions.csv"
        df_clustered.to_csv(pred_path, index=False)
        print(f"  Predições salvas: {pred_path}")

        # Plot temporal da predição (todas as músicas)
        timeline_path = plots_dir / "xgb_mechanism_timeline.png"
        plot_xgb_mechanism_timeline(df_clustered, save_path=timeline_path)
        print("  Plot temporal gerado: xgb_mechanism_timeline.png")

        # Plots de excerpt por música — janela mais densa (5s, nota a nota, para eval humano)
        print("  Gerando excerpts por música...")
        import numpy as np

        for song_name in df_clustered["song"].unique():
            song_df = df_clustered[df_clustered["song"] == song_name].sort_values("time")
            t_min_song = song_df["time"].min()
            t_max_song = song_df["time"].max()
            # Encontrar janela de 5s com maior densidade de frames
            best_start = t_min_song
            best_count = 0
            for t in np.arange(t_min_song, t_max_song - 5, 0.5):
                count = len(song_df[(song_df["time"] >= t) & (song_df["time"] < t + 5)])
                if count > best_count:
                    best_count = count
                    best_start = t
            excerpt_path = plots_dir / f"excerpt_{song_name}.png"
            plot_xgb_mechanism_excerpt(
                df_clustered,
                song=song_name,
                start_time=best_start,
                end_time=best_start + 5.0,
                save_path=excerpt_path,
            )
            print(
                f"    {song_name}: {best_start:.1f}s – {best_start + 5:.1f}s ({best_count} frames)"
            )
    except Exception as e:
        print(f"  Erro ao treinar XGBoost: {e}")

    # Gerar relatório básico
    artist_name = metadata.get("artist", "Desconhecido") if metadata else "Desconhecido"
    report_path = output_dir / "analise_ademilde.md"
    print(f"\nGerando relatório básico: {report_path}")
    generate_report(
        df,
        stats,
        report_path,
        artist_name=artist_name,
        xgb_report=xgb_report,
        xgb_feature_cols=feature_cols,
    )

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
