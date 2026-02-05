"""Script para processar gravações de Ademilde Fonseca."""

import argparse
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vocal_analysis.features.extraction import (
    extract_bioacoustic_features,
    extract_extended_features,
)
from vocal_analysis.utils.pitch import hz_range_to_notes, hz_to_note


def _parse_excerpt_interval(interval_str: str) -> tuple[float, float] | None:
    """Parseia intervalo no formato 'MMSS-MMSS' para segundos.

    Args:
        interval_str: String no formato "0022-0103" (do segundo 22 ao 1:03).

    Returns:
        Tuple (start_seconds, end_seconds) ou None se inválido.
    """
    match = re.match(r"(\d{4})-(\d{4})", interval_str.strip("\"'"))
    if not match:
        return None

    def mmss_to_seconds(mmss: str) -> float:
        minutes = int(mmss[:2])
        seconds = int(mmss[2:])
        return minutes * 60 + seconds

    start = mmss_to_seconds(match.group(1))
    end = mmss_to_seconds(match.group(2))
    return start, end


def _get_excerpt_from_env(song_stem: str) -> tuple[float, float] | None:
    """Busca intervalo de excerpt do .env para uma música.

    Args:
        song_stem: Nome da música (stem do arquivo, ex: "delicado", "apanheite_cavaquinho").

    Returns:
        Tuple (start, end) em segundos ou None se não encontrado.
    """
    # Normalizar nome para busca (ex: "delicado" -> "DELICADO", "apanheite_cavaquinho" -> "APANHEITE_CAVAQUINHO")
    song_key = song_stem.upper().replace("-", "_")

    # Tentar carregar do .env (project root = 4 levels up from this file)
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if not env_path.exists():
        return None

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # Verificar se é uma variável EXCERPT_* que corresponde à música
            if key.upper().startswith("EXCERPT_"):
                env_song = key.upper().replace("EXCERPT_", "")
                # Match exato ou parcial (ex: "DELICADO" matches "DELICADO")
                if (
                    song_key == env_song
                    or song_key.startswith(env_song)
                    or env_song.startswith(song_key)
                ):
                    return _parse_excerpt_interval(value)

    return None


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
        cpps_timeout: int | None = None,
        batch_size: int = 2048,
        crepe_model: str = "full",
        device: str = "cpu",
        separate_vocals: bool = False,
        separation_device: str | None = None,
        use_separation_cache: bool = True,
        validate_separation: bool = False,
        extract_spectral: bool = False,
        skip_cpps_per_frame: bool = True,
    ):
        self.skip_formants = skip_formants
        self.skip_plots = skip_plots
        self.skip_jitter_shimmer = skip_jitter_shimmer
        self.limit_files = limit_files
        self.use_praat_f0 = use_praat_f0
        self.skip_cpps = skip_cpps
        self.cpps_timeout = cpps_timeout
        self.batch_size = batch_size
        self.crepe_model = crepe_model
        self.device = device
        self.separate_vocals = separate_vocals
        self.separation_device = separation_device or device
        self.use_separation_cache = use_separation_cache
        self.validate_separation = validate_separation
        self.extract_spectral = extract_spectral
        self.skip_cpps_per_frame = skip_cpps_per_frame


def _generate_validation_plot(
    original_audio_path: Path,
    separated_features: dict,
    config: ProcessingConfig,
    output_dir: Path,
) -> None:
    """Gera plot comparativo para validar separação de voz.

    Lê intervalo de excerpt do .env (se disponível) para plotar apenas
    o trecho relevante ao invés do áudio completo.

    Args:
        original_audio_path: Caminho do áudio original.
        separated_features: Features extraídas da voz separada.
        config: Configuração de processamento.
        output_dir: Diretório de saída.
    """
    from vocal_analysis.features.extraction import extract_bioacoustic_features
    from vocal_analysis.visualization.plots import plot_separation_validation

    # Buscar intervalo de excerpt do .env
    excerpt_interval = _get_excerpt_from_env(original_audio_path.stem)
    start_time, end_time = excerpt_interval if excerpt_interval else (None, None)

    if excerpt_interval:
        print(f"  Gerando plot de validação (excerpt: {start_time:.0f}s-{end_time:.0f}s)...")
    else:
        print("  Gerando plot de validação (áudio completo)...")

    # Extrair features do áudio original (para comparação)
    original_features = extract_bioacoustic_features(
        original_audio_path,
        skip_formants=True,  # Só precisamos de f0 para validação
        skip_jitter_shimmer=True,
        use_praat_f0=config.use_praat_f0,
        skip_cpps=True,
        batch_size=config.batch_size,
        model=config.crepe_model,
        device=config.device,
    )

    # Gerar plot
    plot_path = output_dir / "plots" / f"{original_audio_path.stem}_separation_validation.png"
    plot_separation_validation(
        time_original=original_features["time"],
        f0_original=original_features["f0"],
        confidence_original=original_features["confidence"],
        time_separated=separated_features["time"],
        f0_separated=separated_features["f0"],
        confidence_separated=separated_features["confidence"],
        title=f"Validação Separação - {original_audio_path.stem}",
        save_path=plot_path,
        start_time=start_time,
        end_time=end_time,
    )
    print(f"  ✓ Plot de validação salvo: {plot_path.name}")


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
            print("🚀 GPU HABILITADA (cuda) - processamento acelerado!")
        else:
            print("💻 CPU (processamento lento - use GPU se disponível)")
    if config.skip_cpps:
        print("⚡ DEBUG: CPPS DESATIVADO (evita travamento em macOS)")
    if config.separate_vocals:
        print(f"🎤 SOURCE SEPARATION HABILITADA (HTDemucs no {config.separation_device})")
        if config.validate_separation:
            print("📊 Validação visual habilitada (plots comparativos)")
    if config.extract_spectral:
        print("📈 FEATURES ESPECTRAIS HABILITADAS (Alpha Ratio, H1-H2, Spectral Tilt)")

    all_features = []
    songs_metadata = []

    # Diretório de cache para separação
    cache_dir = None
    if config.separate_vocals and config.use_separation_cache:
        cache_dir = output_dir.parent / "data" / "cache" / "separated"

    for audio_path in audio_files:
        print(f"\nProcessando: {audio_path.name}")

        # Source separation se habilitado
        audio_path_for_features = audio_path
        temp_wav_path = None
        vocals_array = None

        if config.separate_vocals:
            from vocal_analysis.preprocessing.separation import separate_vocals_safe

            print("  Aplicando source separation (HTDemucs)...")
            vocals, sr, success = separate_vocals_safe(
                audio_path,
                device=config.separation_device,
                cache_dir=cache_dir,
            )

            if success and vocals is not None:
                # Criar arquivo WAV temporário (Praat precisa de arquivo)
                fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
                sf.write(temp_wav_path, vocals, sr)
                audio_path_for_features = Path(temp_wav_path)
                vocals_array = vocals
                print("  ✓ Voz separada com sucesso")
            else:
                print("  ⚠ Source separation falhou, usando audio original")

        try:
            if config.extract_spectral:
                # Usar extração estendida com features espectrais (VMI)
                features = extract_extended_features(
                    audio_path_for_features,
                    skip_formants=config.skip_formants,
                    skip_jitter_shimmer=config.skip_jitter_shimmer,
                    use_praat_f0=config.use_praat_f0,
                    skip_cpps=config.skip_cpps,
                    cpps_timeout=config.cpps_timeout,
                    batch_size=config.batch_size,
                    model=config.crepe_model,
                    device=config.device,
                    skip_cpps_per_frame=config.skip_cpps_per_frame,
                )
            else:
                features = extract_bioacoustic_features(
                    audio_path_for_features,
                    skip_formants=config.skip_formants,
                    skip_jitter_shimmer=config.skip_jitter_shimmer,
                    use_praat_f0=config.use_praat_f0,
                    skip_cpps=config.skip_cpps,
                    cpps_timeout=config.cpps_timeout,
                    batch_size=config.batch_size,
                    model=config.crepe_model,
                    device=config.device,
                )

            # Gerar plot de validação se habilitado
            if config.validate_separation and config.separate_vocals and vocals_array is not None:
                _generate_validation_plot(audio_path, features, config, output_dir)

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

            # Adicionar features espectrais se habilitado
            if config.extract_spectral:
                df_data.update(
                    {
                        "alpha_ratio": features["alpha_ratio"],
                        "h1_h2": features["h1_h2"],
                        "spectral_tilt": features["spectral_tilt"],
                    }
                )
                # CPPS per-frame é opcional (lento)
                if not config.skip_cpps_per_frame and features.get("cpps_per_frame") is not None:
                    df_data["cpps_per_frame"] = features["cpps_per_frame"]

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
                "cpps_global": (
                    float(round(features["cpps_global"], 2))
                    if features["cpps_global"] is not None
                    else None
                ),
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
        finally:
            # Limpar arquivo temporário se criado
            if temp_wav_path and Path(temp_wav_path).exists():
                Path(temp_wav_path).unlink()

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
        help="Pular CPPS completamente (retorna None)",
    )
    parser.add_argument(
        "--cpps-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Timeout em segundos para CPPS (None = sem timeout). Use apenas se CPPS travar no seu sistema",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        metavar="SIZE",
        help="Batch size para CREPE (default: 2048 para GPU, use 512 para macOS CPU limitado)",
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

    # Source separation arguments (habilitado por padrão)
    parser.add_argument(
        "--no-separate-vocals",
        action="store_true",
        help="Desabilitar source separation (HTDemucs). Por padrão, a separação de voz "
        "é habilitada para melhorar detecção de pitch em arranjos complexos.",
    )
    parser.add_argument(
        "--separation-device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo para source separation (default: mesmo que --device)",
    )
    parser.add_argument(
        "--no-separation-cache",
        action="store_true",
        help="Desabilitar cache de áudio separado (força reprocessamento)",
    )
    parser.add_argument(
        "--validate-separation",
        action="store_true",
        help="Gerar plot comparativo original vs voz separada (Hz + notas) para validação visual",
    )

    # VMI / Spectral features arguments
    parser.add_argument(
        "--extract-spectral",
        action="store_true",
        help="Extrair features espectrais (Alpha Ratio, H1-H2, Spectral Tilt) para análise VMI. "
        "Necessário para usar o pipeline VMI em run_analysis.py.",
    )
    parser.add_argument(
        "--cpps-per-frame",
        action="store_true",
        help="Extrair CPPS per-frame (lento). Requer --extract-spectral.",
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
        cpps_timeout=args.cpps_timeout,
        batch_size=args.batch_size,
        crepe_model=args.crepe_model,
        device=args.device,
        separate_vocals=not args.no_separate_vocals,
        separation_device=args.separation_device,
        use_separation_cache=not args.no_separation_cache,
        validate_separation=args.validate_separation,
        extract_spectral=args.extract_spectral,
        skip_cpps_per_frame=not args.cpps_per_frame,
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
