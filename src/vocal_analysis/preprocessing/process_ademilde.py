"""Script para processar gravações de Ademilde Fonseca."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from vocal_analysis.features.extraction import extract_bioacoustic_features
from vocal_analysis.utils.pitch import hz_to_note, hz_range_to_notes
from vocal_analysis.visualization.plots import plot_f0_contour


def process_audio_files(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Processa todos os arquivos de áudio e extrai features.

    Args:
        data_dir: Diretório com arquivos de áudio.
        output_dir: Diretório para salvar outputs.

    Returns:
        Tuple com DataFrame de features e metadados do processamento.
    """
    audio_files = list(data_dir.glob("*.mp3"))
    print(f"Encontrados {len(audio_files)} arquivos de áudio")

    all_features = []
    songs_metadata = []

    for audio_path in audio_files:
        print(f"\nProcessando: {audio_path.name}")

        try:
            features = extract_bioacoustic_features(audio_path)

            # Criar DataFrame para esta música
            df = pd.DataFrame({
                "time": features["time"],
                "f0": features["f0"],
                "confidence": features["confidence"],
                "hnr": features["hnr"],
            })
            df["song"] = audio_path.stem
            df["cpps_global"] = features["cpps_global"]

            # Filtrar frames com baixa confiança (ruído/silêncio)
            df_voiced = df[df["confidence"] > 0.5].copy()

            all_features.append(df_voiced)

            # Gerar plot de f0
            plot_path = output_dir / "plots" / f"{audio_path.stem}_f0.png"
            plot_f0_contour(
                features["time"],
                features["f0"],
                features["confidence"],
                title=f"Contorno de f0 - {audio_path.stem}",
                save_path=plot_path,
            )

            # Metadata da música
            song_meta = {
                "song": audio_path.stem,
                "file": audio_path.name,
                "total_frames": len(df),
                "voiced_frames": len(df_voiced),
                "f0_mean_hz": round(df_voiced["f0"].mean(), 1),
                "f0_mean_note": hz_to_note(df_voiced["f0"].mean()),
                "f0_min_hz": round(df_voiced["f0"].min(), 1),
                "f0_max_hz": round(df_voiced["f0"].max(), 1),
                "f0_range_notes": hz_range_to_notes(df_voiced["f0"].min(), df_voiced["f0"].max()),
                "f0_std_hz": round(df_voiced["f0"].std(), 1),
                "hnr_mean_db": round(df_voiced["hnr"].mean(), 1),
                "cpps_global": round(features["cpps_global"], 2),
                "plot_path": str(plot_path.relative_to(output_dir.parent)),
            }
            songs_metadata.append(song_meta)

            # Print resumo
            print(f"  f0: {song_meta['f0_mean_hz']} Hz ({song_meta['f0_mean_note']})")
            print(f"  Range: {song_meta['f0_range_notes']}")
            print(f"  HNR: {song_meta['hnr_mean_db']} dB | CPPS: {song_meta['cpps_global']}")

        except Exception as e:
            print(f"  ERRO: {e}")
            songs_metadata.append({
                "song": audio_path.stem,
                "file": audio_path.name,
                "error": str(e),
            })
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

        # Adicionar stats globais ao metadata
        df_voiced = df_all[df_all["confidence"] > 0.5]
        metadata["global"] = {
            "total_voiced_frames": len(df_voiced),
            "f0_mean_hz": round(df_voiced["f0"].mean(), 1),
            "f0_mean_note": hz_to_note(df_voiced["f0"].mean()),
            "f0_min_hz": round(df_voiced["f0"].min(), 1),
            "f0_max_hz": round(df_voiced["f0"].max(), 1),
            "f0_range_notes": hz_range_to_notes(df_voiced["f0"].min(), df_voiced["f0"].max()),
            "f0_std_hz": round(df_voiced["f0"].std(), 1),
            "hnr_mean_db": round(df_voiced["hnr"].mean(), 1),
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
        lines.extend([
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
        ])

    lines.extend([
        "## Por Música",
        "",
    ])

    for song in metadata["songs"]:
        if "error" in song:
            lines.append(f"### {song['song']} ❌")
            lines.append(f"Erro: {song['error']}")
        else:
            lines.extend([
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
            ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Ponto de entrada principal."""
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"

    # Garantir que diretórios existem
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    df, metadata = process_audio_files(data_dir, output_dir)

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
