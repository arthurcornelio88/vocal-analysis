"""Script para processar gravações de Ademilde Fonseca."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vocal_analysis.features.extraction import extract_bioacoustic_features
from vocal_analysis.utils.pitch import hz_range_to_notes, hz_to_note


class ProcessingConfig:
    """Configuração para controlar quais features extrair (debug mode)."""

    def __init__(
        self,
        skip_formants: bool = False,
        skip_plots: bool = False,
        skip_jitter_shimmer: bool = False,
        limit_files: int | None = None,
        use_praat_f0: bool = False,
        skip_cpps: bool = False,
        crepe_model: str = "full",
        device: str = "cpu",
    ):
        self.skip_formants = skip_formants
        self.skip_plots = skip_plots
        self.skip_jitter_shimmer = skip_jitter_shimmer
        self.limit_files = limit_files
        self.use_praat_f0 = use_praat_f0
        self.skip_cpps = skip_cpps
        self.crepe_model = crepe_model
        self.device = device


def process_audio_files(
    data_dir: Path, output_dir: Path, config: ProcessingConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """Processa todos os arquivos de áudio e extrai features.

    Args:
        data_dir: Diretório com arquivos de áudio.
        output_dir: Diretório para salvar outputs.

    Returns:
        Tuple com DataFrame de features e metadados do processamento.
    """
    if config is None:
        config = ProcessingConfig()

    audio_files = list(data_dir.glob("*.mp3"))

    # Limitar número de arquivos se especificado (útil para debug)
    if config.limit_files:
        audio_files = audio_files[: config.limit_files]

    print(f"Encontrados {len(audio_files)} arquivos de áudio")
    if config.skip_formants:
        print("⚡ DEBUG: Formants DESATIVADOS")
    if config.skip_plots:
        print("⚡ DEBUG: Plots DESATIVADOS")
    if config.skip_jitter_shimmer:
        print("⚡ DEBUG: Jitter/Shimmer DESATIVADOS")
    if config.use_praat_f0:
        print("⚡ DEBUG: Usando Praat f0 (rápido) ao invés de CREPE")
    else:
        print(f"🎵 Usando CREPE modelo '{config.crepe_model}' para extração de f0")
        if config.device == "cuda":
            print(f"🚀 GPU HABILITADA (cuda) - processamento acelerado!")
        else:
            print(f"💻 CPU (processamento lento - use GPU se disponível)")
    if config.skip_cpps:
        print("⚡ DEBUG: CPPS DESATIVADO (evita travamento em macOS)")

    all_features = []
    songs_metadata = []

    for audio_path in audio_files:
        print(f"\nProcessando: {audio_path.name}")

        try:
            features = extract_bioacoustic_features(
                audio_path,
                skip_formants=config.skip_formants,
                skip_jitter_shimmer=config.skip_jitter_shimmer,
                use_praat_f0=config.use_praat_f0,
                skip_cpps=config.skip_cpps,
                model=config.crepe_model,
                device=config.device,
            )

            # Criar DataFrame para esta música
            df_data = {
                "time": features["time"],
                "f0": features["f0"],
                "confidence": features["confidence"],
                "hnr": features["hnr"],
                "energy": features["energy"],
            }

            # Adicionar formants apenas se não foram desativados
            if not config.skip_formants:
                df_data.update(
                    {
                        "f1": features["f1"],
                        "f2": features["f2"],
                        "f3": features["f3"],
                        "f4": features["f4"],
                    }
                )

            df = pd.DataFrame(df_data)
            df["song"] = audio_path.stem
            df["cpps_global"] = features["cpps_global"]

            if not config.skip_jitter_shimmer:
                df["jitter"] = features["jitter"]
                df["shimmer"] = features["shimmer"]

            # Filtrar frames com baixa confiança ou silêncio
            # confidence > 0.8: conforme metodologia (CREPE periodicity)
            # hnr > -10: remove silêncio/ruído (Praat retorna -200 dB em frames não-voiced)
            df_voiced = df[(df["confidence"] > 0.8) & (df["hnr"] > -10)].copy()

            all_features.append(df_voiced)

            # Gerar plot de f0 (apenas se não desativado)
            plot_path = None
            if not config.skip_plots:
                # Import only when needed to avoid matplotlib initialization overhead
                from vocal_analysis.visualization.plots import plot_f0_contour

                plot_path = output_dir / "plots" / f"{audio_path.stem}_f0.png"
                plot_f0_contour(
                    features["time"],
                    features["f0"],
                    features["confidence"],
                    title=f"Contorno de f0 - {audio_path.stem}",
                    save_path=plot_path,
                )

            # Metadata da música (convert numpy types to Python native types for JSON serialization)
            song_meta = {
                "song": audio_path.stem,
                "file": audio_path.name,
                "total_frames": int(len(df)),
                "voiced_frames": int(len(df_voiced)),
                "f0_mean_hz": float(round(df_voiced["f0"].mean(), 1)),
                "f0_mean_note": hz_to_note(df_voiced["f0"].mean()),
                "f0_min_hz": float(round(df_voiced["f0"].min(), 1)),
                "f0_max_hz": float(round(df_voiced["f0"].max(), 1)),
                "f0_range_notes": hz_range_to_notes(df_voiced["f0"].min(), df_voiced["f0"].max()),
                "f0_std_hz": float(round(df_voiced["f0"].std(), 1)),
                "hnr_mean_db": float(round(df_voiced["hnr"].mean(), 1)),
                "cpps_global": float(round(features["cpps_global"], 2)),
                "energy_mean": float(round(df_voiced["energy"].mean(), 4)),
            }

            # Adicionar jitter/shimmer se não foram desativados
            if not config.skip_jitter_shimmer:
                song_meta["jitter_ppq5"] = (
                    float(round(features["jitter"], 4))
                    if not np.isnan(features["jitter"])
                    else None
                )
                song_meta["shimmer_apq11"] = (
                    float(round(features["shimmer"], 4))
                    if not np.isnan(features["shimmer"])
                    else None
                )

            # Adicionar path do plot se foi gerado
            if plot_path:
                song_meta["plot_path"] = str(plot_path.relative_to(output_dir.parent))
            songs_metadata.append(song_meta)

            # Print resumo
            print(f"  f0: {song_meta['f0_mean_hz']} Hz ({song_meta['f0_mean_note']})")
            print(f"  Range: {song_meta['f0_range_notes']}")
            print(f"  HNR: {song_meta['hnr_mean_db']} dB | CPPS: {song_meta['cpps_global']}")

            if not config.skip_jitter_shimmer:
                jitter_str = (
                    f"{song_meta['jitter_ppq5']}" if song_meta.get("jitter_ppq5") else "N/A"
                )
                shimmer_str = (
                    f"{song_meta['shimmer_apq11']}" if song_meta.get("shimmer_apq11") else "N/A"
                )
                print(f"  Jitter: {jitter_str} | Shimmer: {shimmer_str}")

        except Exception as e:
            print(f"  ERRO: {e}")
            songs_metadata.append(
                {
                    "song": audio_path.stem,
                    "file": audio_path.name,
                    "error": str(e),
                }
            )
            continue

    metadata = {
        "processed_at": datetime.now().isoformat(),
        "artist": "Ademilde Fonseca",
        "n_songs": len(audio_files),
        "n_success": len(all_features),
        "songs": songs_metadata,
    }

    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)

        # Adicionar stats globais ao metadata (convert numpy types to Python native types)
        df_voiced = df_all[(df_all["confidence"] > 0.8) & (df_all["hnr"] > -10)]
        metadata["global"] = {
            "total_voiced_frames": int(len(df_voiced)),
            "f0_mean_hz": float(round(df_voiced["f0"].mean(), 1)),
            "f0_mean_note": hz_to_note(df_voiced["f0"].mean()),
            "f0_min_hz": float(round(df_voiced["f0"].min(), 1)),
            "f0_max_hz": float(round(df_voiced["f0"].max(), 1)),
            "f0_range_notes": hz_range_to_notes(df_voiced["f0"].min(), df_voiced["f0"].max()),
            "f0_std_hz": float(round(df_voiced["f0"].std(), 1)),
            "hnr_mean_db": float(round(df_voiced["hnr"].mean(), 1)),
        }

        return df_all, metadata

    return pd.DataFrame(), metadata


def save_outputs(
    df: pd.DataFrame,
    metadata: dict,
    project_root: Path,
) -> None:
    """Salva CSV, JSON e log do processamento.

    Args:
        df: DataFrame com features.
        metadata: Dicionário de metadados.
        project_root: Raiz do projeto.
    """
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # CSV com features
    csv_path = processed_dir / "ademilde_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV salvo: {csv_path}")

    # JSON com metadados
    json_path = processed_dir / "ademilde_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"JSON salvo: {json_path}")

    # Log markdown
    log_path = processed_dir / "processing_log.md"
    _write_log_markdown(metadata, log_path)
    print(f"Log salvo: {log_path}")


def _write_log_markdown(metadata: dict, path: Path) -> None:
    """Gera log em markdown."""
    lines = [
        f"# Log de Processamento - {metadata['artist']}",
        "",
        f"**Data:** {metadata['processed_at']}",
        f"**Músicas processadas:** {metadata['n_success']}/{metadata['n_songs']}",
        "",
    ]

    if "global" in metadata:
        g = metadata["global"]
        lines.extend(
            [
                "## Resumo Global",
                "",
                "| Métrica | Valor | Nota |",
                "|---------|-------|------|",
                f"| f0 médio | {g['f0_mean_hz']} Hz | {g['f0_mean_note']} |",
                f"| f0 mínimo | {g['f0_min_hz']} Hz | – |",
                f"| f0 máximo | {g['f0_max_hz']} Hz | – |",
                f"| Extensão | – | {g['f0_range_notes']} |",
                f"| f0 desvio | {g['f0_std_hz']} Hz | – |",
                f"| HNR médio | {g['hnr_mean_db']} dB | – |",
                f"| Total frames | {g['total_voiced_frames']} | – |",
                "",
            ]
        )

    lines.extend(
        [
            "## Por Música",
            "",
        ]
    )

    for song in metadata["songs"]:
        if "error" in song:
            lines.append(f"### {song['song']} ❌")
            lines.append(f"Erro: {song['error']}")
        else:
            lines.extend(
                [
                    f"### {song['song']}",
                    "",
                    "| Métrica | Valor |",
                    "|---------|-------|",
                    f"| f0 médio | {song['f0_mean_hz']} Hz ({song['f0_mean_note']}) |",
                    f"| Extensão | {song['f0_range_notes']} |",
                    f"| HNR | {song['hnr_mean_db']} dB |",
                    f"| CPPS | {song['cpps_global']} |",
                    f"| Frames | {song['voiced_frames']}/{song['total_frames']} |",
                    "",
                ]
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Ponto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Processar arquivos de áudio de Ademilde Fonseca",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Processamento completo
  python -m vocal_analysis.preprocessing.process_ademilde

  # Debug rápido (sem formants, plots, jitter/shimmer)
  python -m vocal_analysis.preprocessing.process_ademilde --skip-formants --skip-plots --skip-jitter-shimmer

  # Processar apenas 1 arquivo
  python -m vocal_analysis.preprocessing.process_ademilde --limit 1

  # Modo ultra-rápido (apenas f0 + HNR)
  python -m vocal_analysis.preprocessing.process_ademilde --fast
        """,
    )

    parser.add_argument(
        "--skip-formants",
        action="store_true",
        help="Pular extração de formantes (F1-F4) - economiza ~30%% do tempo",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Não gerar plots de f0 - economiza I/O"
    )
    parser.add_argument(
        "--skip-jitter-shimmer",
        action="store_true",
        help="Pular Jitter/Shimmer (Praat Point Process) - economiza ~20%% do tempo",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N", help="Processar apenas os primeiros N arquivos"
    )
    parser.add_argument(
        "--use-praat-f0",
        action="store_true",
        help="Usar Praat (autocorrelation) para f0 ao invés de CREPE - MUITO mais rápido mas menos preciso",
    )
    parser.add_argument(
        "--skip-cpps",
        action="store_true",
        help="Pular CPPS (recomendado para macOS com arquivos longos que podem travar)",
    )
    parser.add_argument(
        "--crepe-model",
        type=str,
        default="full",
        choices=["tiny", "small", "full"],
        help="Modelo CREPE para extração de f0 (default: full). Use 'small' para economizar memória no macOS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo para CREPE (default: cpu). Use 'cuda' para GPU (Google Colab, Windows com NVIDIA)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Modo rápido: ativa --skip-formants, --skip-plots, --skip-jitter-shimmer, --skip-cpps, --use-praat-f0",
    )

    args = parser.parse_args()

    # Modo fast ativa todas as otimizações
    if args.fast:
        args.skip_formants = True
        args.skip_plots = True
        args.skip_jitter_shimmer = True
        args.use_praat_f0 = True
        args.skip_cpps = True

    config = ProcessingConfig(
        skip_formants=args.skip_formants,
        skip_plots=args.skip_plots,
        skip_jitter_shimmer=args.skip_jitter_shimmer,
        limit_files=args.limit,
        use_praat_f0=args.use_praat_f0,
        skip_cpps=args.skip_cpps,
        crepe_model=args.crepe_model,
        device=args.device,
    )

    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"

    # Garantir que diretórios existem
    if not config.skip_plots:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    df, metadata = process_audio_files(data_dir, output_dir, config)

    if not df.empty:
        save_outputs(df, metadata, project_root)

        # Print resumo final
        g = metadata["global"]
        print("\n" + "=" * 50)
        print(f"RESUMO - {metadata['artist']}")
        print("=" * 50)
        print(f"f0 médio: {g['f0_mean_hz']} Hz ({g['f0_mean_note']})")
        print(f"Extensão: {g['f0_range_notes']}")
        print(f"HNR: {g['hnr_mean_db']} dB")


if __name__ == "__main__":
    main()
