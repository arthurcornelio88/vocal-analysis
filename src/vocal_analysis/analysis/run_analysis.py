"""Script para rodar análise exploratória completa."""

import json
import os
import re
from pathlib import Path

# --- GARANTIR CARREGAMENTO DE ENV ---
try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback caso não tenha dotenv instalado, define função dummy
    def load_dotenv(): pass

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

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


def parse_time_string(time_str: str) -> float:
    """Converte string 'MMSS' ou segundos puros para float segundos."""
    if not isinstance(time_str, str):
        return float(time_str)
        
    clean_str = time_str.replace('"', '').replace("'", "").strip()
    
    # Se for formato MMSS (4 dígitos)
    if len(clean_str) == 4 and clean_str.isdigit():
        minutes = int(clean_str[:2])
        seconds = int(clean_str[2:])
        return minutes * 60 + seconds
    
    # Caso contrário assume segundos
    try:
        return float(clean_str)
    except ValueError:
        return 0.0


def get_manual_excerpt_from_env(song_name: str) -> tuple[float, float] | None:
    """Busca intervalo manual em variáveis de ambiente com normalização de nome."""
    # Sanitiza: "Apanhei-te Cavaquinho" -> "APANHEITE_CAVAQUINHO" ou "APANHEI_TE_CAVAQUINHO"
    # Remove caracteres especiais e joga para uppercase
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", song_name).upper()
    safe_name = re.sub(r"_+", "_", safe_name) # Remove underscores duplicados
    
    env_key = f"EXCERPT_{safe_name}"
    env_value = os.environ.get(env_key)
    
    # Debug: ajuda a entender o que o script está procurando
    # print(f"  [DEBUG] Procurando ENV: {env_key} (Valor: {env_value})")

    if env_value:
        try:
            start_str, end_str = env_value.split("-")
            start_time = parse_time_string(start_str.strip())
            end_time = parse_time_string(end_str.strip())
            return start_time, end_time
        except Exception as e:
            print(f"  ⚠️ Aviso: Erro ao ler ENV {env_key}='{env_value}': {e}")
            
    return None


def save_audio_excerpt(
    song_name: str,
    start_time: float,
    end_time: float,
    project_root: Path,
    output_dir: Path
) -> None:
    """Recorta e salva o áudio correspondente ao excerpt."""
    raw_dir = project_root / "data" / "raw"
    audio_files = list(raw_dir.glob(f"*{song_name}*.mp3")) + list(raw_dir.glob(f"*{song_name}*.wav"))
    
    if not audio_files:
        return

    audio_path = audio_files[0]
    
    try:
        # Carrega apenas o trecho desejado
        y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time - start_time)
        out_path = output_dir / f"excerpt_{song_name}.wav"
        sf.write(out_path, y, sr)
        print(f"    🎵 Áudio salvo: {out_path.name}")
    except Exception as e:
        print(f"    ⚠️ Erro ao salvar áudio: {e}")


def main() -> None:
    """Executa análise exploratória dos dados processados."""
    # Carregar ENV do arquivo .env se existir
    load_dotenv()

    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "processed" / "ademilde_features.csv"
    metadata_path = project_root / "data" / "processed" / "ademilde_metadata.json"
    output_dir = project_root / "outputs"

    if not data_path.exists():
        print(f"Arquivo não encontrado: {data_path}")
        return

    # Carregar dados
    print("Carregando dados...")
    df = pd.read_csv(data_path)

    metadata = None
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Artista: {metadata.get('artist', 'Desconhecido')}")

    print(f"  Total de Frames: {len(df)}")

    # Computar features de agilidade articulatória
    print("\nComputando features de agilidade articulatória...")
    df = compute_articulation_features(df)
    
    # Análise por threshold
    print("\nAnalisando por threshold (400 Hz / G4)...")
    stats = analyze_mechanism_regions(df, threshold_hz=400.0, output_dir=output_dir / "plots")

    # Clustering
    print("\nExecutando clustering GMM...")
    plots_dir = output_dir / "plots"
    df_clustered = cluster_mechanisms(df, n_clusters=2, method="gmm", output_dir=plots_dir)

    # XGBoost
    print("\nTreinando XGBoost com pseudo-labels do GMM...")
    base_cols = ["f0", "hnr", "energy", "f0_velocity", "f0_acceleration"]
    optional_cols = ["f1", "f2", "f3", "f4"]
    feature_cols = base_cols + [c for c in optional_cols if c in df_clustered.columns]
    
    df_train = df_clustered[feature_cols].copy()
    df_train["mechanism_label"] = df_clustered["mechanism"].map({"M1": 0, "M2": 1})

    xgb_report = None
    try:
        model, xgb_report = train_mechanism_classifier(
            df_train, feature_cols=feature_cols, target_col="mechanism_label"
        )
        
        # --- CORREÇÃO DO ERRO ---
        # 1. Predizer (retorna numpy array)
        predictions = model.predict(df_clustered[feature_cols])
        # 2. Atribuir ao DataFrame (vira Series)
        df_clustered["xgb_mechanism"] = predictions
        # 3. Mapear (agora funciona porque é Series)
        df_clustered["xgb_mechanism"] = df_clustered["xgb_mechanism"].map({0: "M1", 1: "M2"})
        # ------------------------

        # Salvar predições
        pred_path = output_dir / "xgb_predictions.csv"
        df_clustered.to_csv(pred_path, index=False)
        print(f"  Predições salvas: {pred_path}")

        # Plot temporal
        timeline_path = plots_dir / "xgb_mechanism_timeline.png"
        plot_xgb_mechanism_timeline(df_clustered, save_path=timeline_path)
        print("  Plot temporal gerado.")

        # Excerpts
        print("\nGerando excerpts por música...")
        
        for song_name in df_clustered["song"].unique():
            song_df = df_clustered[df_clustered["song"] == song_name].sort_values("time")
            
            if song_df.empty:
                continue
                
            t_min_song = song_df["time"].min()
            t_max_song = song_df["time"].max()
            
            # 1. Tentar pegar intervalo da Variável de Ambiente
            manual_excerpt = get_manual_excerpt_from_env(song_name)
            
            if manual_excerpt:
                best_start, best_end = manual_excerpt
                print(f"  > {song_name}: ENV ENCONTRADO ({best_start:.1f}s - {best_end:.1f}s)")
                
                if best_start > t_max_song:
                    print(f"    ⚠️ Aviso: Início {best_start}s > Duração da música ({t_max_song:.1f}s). Ignorando.")
                    continue
                
                # Ajusta contagem apenas para log
                best_count = len(song_df[(song_df["time"] >= best_start) & (song_df["time"] < best_end)])
            
            else:
                # 2. Automático (densidade)
                print(f"  > {song_name}: Automático (densidade)...")
                best_start = t_min_song
                best_count = 0
                
                if t_max_song - 5 <= t_min_song:
                    search_range = [t_min_song]
                else:
                    search_range = np.arange(t_min_song, t_max_song - 5, 0.5)

                for t in search_range:
                    count = len(song_df[(song_df["time"] >= t) & (song_df["time"] < t + 5)])
                    if count > best_count:
                        best_count = count
                        best_start = t
                best_end = best_start + 5.0
            
            # Gerar Plot
            excerpt_path = plots_dir / f"excerpt_{song_name}.png"
            plot_xgb_mechanism_excerpt(
                df_clustered,
                song=song_name,
                start_time=best_start,
                end_time=best_end,
                save_path=excerpt_path,
            )
            print(f"    Plot salvo: {best_start:.1f}s – {best_end:.1f}s")
            
            # Salvar Áudio
            save_audio_excerpt(song_name, best_start, best_end, project_root, output_dir)
            
    except Exception as e:
        print(f"  Erro crítico no XGBoost/Plots: {e}")
        import traceback
        traceback.print_exc()

    # Gerar relatório básico
    artist_name = metadata.get("artist", "Desconhecido") if metadata else "Desconhecido"
    report_path = output_dir / "analise_ademilde.md"
    generate_report(
        df,
        stats,
        report_path,
        artist_name=artist_name,
        xgb_report=xgb_report,
        xgb_feature_cols=feature_cols,
    )

    # Gerar relatório LLM
    if os.environ.get("GEMINI_API_KEY"):
        llm_report_path = output_dir / "relatorio_llm.md"
        print(f"\nGerando relatório narrativo com Gemini...")
        plot_paths = list((output_dir / "plots").glob("*.png"))
        
        try:
            generate_narrative_report(stats, metadata, llm_report_path, plot_paths=plot_paths)
            print("  Relatório LLM gerado com sucesso!")
        except Exception as e:
            print(f"  Erro ao gerar relatório LLM: {e}")
    else:
        print("\n(Dica: Defina GEMINI_API_KEY para gerar o relatório com IA)")

    print("\nConcluído!")

if __name__ == "__main__":
    main()