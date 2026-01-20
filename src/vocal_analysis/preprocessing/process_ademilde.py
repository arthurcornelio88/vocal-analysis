"""Script para processar gravações de Ademilde Fonseca."""

from pathlib import Path

import pandas as pd

from vocal_analysis.features.extraction import extract_bioacoustic_features
from vocal_analysis.visualization.plots import plot_f0_contour


def process_audio_files(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Processa todos os arquivos de áudio e extrai features.

    Args:
        data_dir: Diretório com arquivos de áudio.
        output_dir: Diretório para salvar outputs.

    Returns:
        DataFrame com features agregadas de todos os arquivos.
    """
    audio_files = list(data_dir.glob("*.mp3"))
    print(f"Encontrados {len(audio_files)} arquivos de áudio")

    all_features = []

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
            print(f"  - Plot salvo: {plot_path.name}")

            # Stats básicas
            print(f"  - f0 médio: {df_voiced['f0'].mean():.1f} Hz")
            print(f"  - f0 range: {df_voiced['f0'].min():.1f} - {df_voiced['f0'].max():.1f} Hz")
            print(f"  - HNR médio: {df_voiced['hnr'].mean():.1f} dB")
            print(f"  - CPPS global: {features['cpps_global']:.2f}")

        except Exception as e:
            print(f"  ERRO: {e}")
            continue

    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)
        csv_path = output_dir / "processed" / "ademilde_features.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"\nFeatures salvas em: {csv_path}")
        return df_all

    return pd.DataFrame()


def main() -> None:
    """Ponto de entrada principal."""
    # vocal-analysis/src/vocal_analysis/preprocessing/ -> vocal-analysis/
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"

    # Garantir que diretórios existem
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    df = process_audio_files(data_dir, output_dir)

    if not df.empty:
        print("\n" + "=" * 50)
        print("RESUMO GERAL - Ademilde Fonseca")
        print("=" * 50)
        print(f"Total de frames voiced: {len(df)}")
        print(f"f0 médio global: {df['f0'].mean():.1f} Hz")
        print(f"f0 desvio padrão: {df['f0'].std():.1f} Hz")
        print(f"HNR médio global: {df['hnr'].mean():.1f} dB")

        # Stats por música
        print("\nPor música:")
        print(df.groupby("song")[["f0", "hnr"]].agg(["mean", "std"]).round(1))


if __name__ == "__main__":
    main()
