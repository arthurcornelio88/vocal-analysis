"""Regenera plot de validação de separação usando dados cacheados.

Uso:
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot

    # Música específica
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot \
        --song "Apanhei-te Cavaquinho"

    # Todas as músicas com cache
    uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all
"""

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from vocal_analysis.features.extraction import extract_bioacoustic_features
from vocal_analysis.preprocessing.separation import HTDEMUCS_SAMPLE_RATE
from vocal_analysis.visualization.plots import plot_separation_validation


def _normalize_song_name(name: str) -> str:
    """Normaliza nome da música para match com arquivos."""
    return re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")


def _parse_excerpt_interval(interval_str: str) -> tuple[float, float] | None:
    """Parseia intervalo no formato 'MMSS-MMSS' para segundos."""
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


def _get_excerpt_from_env(song_stem: str, project_root: Path) -> tuple[float, float] | None:
    """Busca intervalo de excerpt do .env para uma música."""
    song_key = _normalize_song_name(song_stem).upper()
    env_path = project_root / ".env"

    if not env_path.exists():
        return None

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.upper().startswith("EXCERPT_"):
                env_song = key.upper().replace("EXCERPT_", "")
                if (
                    song_key == env_song
                    or song_key.startswith(env_song)
                    or env_song.startswith(song_key)
                ):
                    return _parse_excerpt_interval(value.split("#")[0].strip())

    return None


def regenerate_plot(
    song_name: str,
    project_root: Path,
    use_praat_f0: bool = True,
) -> Path | None:
    """Regenera plot de validação para uma música usando cache.

    Args:
        song_name: Nome da música (ex: "Apanhei-te Cavaquinho")
        project_root: Raiz do projeto
        use_praat_f0: Se True, usa Praat (rápido). Se False, usa CREPE (lento, preciso).

    Returns:
        Path do plot gerado ou None se falhou.
    """
    song_stem = _normalize_song_name(song_name)

    # Encontrar arquivos
    audio_files = list((project_root / "data" / "raw").glob("*.mp3"))
    audio_path = None
    for f in audio_files:
        if _normalize_song_name(f.stem) == song_stem:
            audio_path = f
            break

    if not audio_path:
        print(f"❌ Áudio não encontrado para: {song_name}")
        return None

    cache_file = project_root / "data" / "cache" / "separated" / f"{song_stem}_vocals.npy"
    if not cache_file.exists():
        print(f"❌ Cache não encontrado: {cache_file}")
        return None

    output_path = project_root / "outputs" / "plots" / f"{song_stem}_separation_validation.png"

    # Buscar excerpt do .env
    excerpt = _get_excerpt_from_env(song_stem, project_root)
    start_time, end_time = excerpt if excerpt else (None, None)

    if excerpt:
        print(f"📊 {song_name} (excerpt: {start_time:.0f}s-{end_time:.0f}s)")
    else:
        print(f"📊 {song_name} (áudio completo)")

    # Extrair features do original
    print("  Extraindo features do áudio original...")
    orig_features = extract_bioacoustic_features(
        audio_path,
        skip_formants=True,
        skip_jitter_shimmer=True,
        skip_cpps=True,
        use_praat_f0=use_praat_f0,
    )

    # Carregar voz separada do cache
    print("  Carregando voz separada do cache...")
    vocals = np.load(cache_file)

    # Criar WAV temporário para extração
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, vocals, HTDEMUCS_SAMPLE_RATE)
        temp_path = Path(f.name)

    try:
        print("  Extraindo features da voz separada...")
        sep_features = extract_bioacoustic_features(
            temp_path,
            skip_formants=True,
            skip_jitter_shimmer=True,
            skip_cpps=True,
            use_praat_f0=use_praat_f0,
        )

        print("  Gerando plot...")
        plot_separation_validation(
            time_original=orig_features["time"],
            f0_original=orig_features["f0"],
            confidence_original=orig_features["confidence"],
            time_separated=sep_features["time"],
            f0_separated=sep_features["f0"],
            confidence_separated=sep_features["confidence"],
            title=f"Validação Separação - {song_stem}",
            save_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"  ✅ Plot salvo: {output_path.name}")
        return output_path

    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    """Ponto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Regenera plots de validação de separação usando dados cacheados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--song",
        type=str,
        help="Nome da música (ex: 'Apanhei-te Cavaquinho'). Lista músicas se omitido.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerar plots para todas as músicas com cache disponível",
    )
    parser.add_argument(
        "--use-crepe",
        action="store_true",
        help="Usar CREPE ao invés de Praat para f0 (mais lento, mais preciso)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / "data" / "cache" / "separated"

    if not cache_dir.exists():
        print("❌ Diretório de cache não encontrado. Execute primeiro:")
        print("   uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals")
        return

    cached_files = list(cache_dir.glob("*_vocals.npy"))
    if not cached_files:
        print("❌ Nenhum cache de separação encontrado.")
        return

    if args.all:
        print(f"Regenerando plots para {len(cached_files)} músicas...\n")
        for cache_file in cached_files:
            song_stem = cache_file.stem.replace("_vocals", "")
            regenerate_plot(song_stem, project_root, use_praat_f0=not args.use_crepe)
            print()
    elif args.song:
        regenerate_plot(args.song, project_root, use_praat_f0=not args.use_crepe)
    else:
        print("Músicas com cache disponível:")
        for cache_file in cached_files:
            song_stem = cache_file.stem.replace("_vocals", "")
            print(f"  - {song_stem}")
        print("\nUso:")
        print("  uv run python -m vocal_analysis.scripts.regenerate_validation_plot --song 'nome'")
        print("  uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all")


if __name__ == "__main__":
    main()
