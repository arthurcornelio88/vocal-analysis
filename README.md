[Portugues](docs/pt-BR/README.md) | **English**

# Vocal Analysis - Bioacoustic M1/M2 Analysis

Computational analysis of laryngeal mechanisms (M1/M2) in Brazilian Choro recordings, focusing on the voice of Ademilde Fonseca.

## Goal

Challenge the traditional "Fach" vocal classification system through a physiological analysis of laryngeal mechanisms, extracting explainable features (f0, HNR, CPPS) from singing audio.

## Stack

- **torchcrepe**: SOTA f0 (pitch) extraction
- **parselmouth** (Praat): HNR, CPPS, Jitter, Shimmer
- **xgboost**: Tabular M1/M2 classification
- **seaborn/matplotlib**: Academic visualizations
- **google-generativeai**: Multimodal narrative report generation with Gemini 2.0 Flash

## Setup

### Requirements

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (package manager)
- (Optional) Google Gemini API Key for AI-powered reports

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd vocal-analysis

# Install dependencies
uv sync

# Install development dependencies (ruff, pytest, jupyter)
uv sync --extra dev
```

### Configure Gemini (optional)

To generate AI-powered narrative reports:

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Set the environment variable:

```bash
export GEMINI_API_KEY=your_key_here
```

Or add it to your `.bashrc`/`.zshrc` to persist.

**Using a `.env` file:**

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your settings
3. Load into environment:
   ```bash
   source .env
   ```

### Configure Report Language

The language of generated reports (analysis_report.md, vmi_analysis.md, llm_report.md) is controlled by the `REPORT_LANG` variable:

```bash
# English reports (default)
REPORT_LANG=en

# Portuguese reports
REPORT_LANG=pt-BR
```

### Configure Excerpts (optional)

You can define specific time intervals for each song for analysis and validation plots.
Useful for focusing on vocal passages without instrumental introductions.

In `.env`, use the format `EXCERPT_<NAME>=MMSS-MMSS`:

```bash
# From second 22 to 1:03
EXCERPT_DELICADO="0022-0103"

# From second 33 to 1:04
EXCERPT_BRASILEIRINHO="0033-0104"

# From second 7 to 23
EXCERPT_APANHEITE_CAVAQUINHO="0007-0023"
```

Excerpts are automatically used in validation plots (`--validate-separation`).

## Project Structure

```
vocal-analysis/
├── src/vocal_analysis/
│   ├── preprocessing/
│   │   ├── audio.py              # load_audio(), normalize_audio()
│   │   ├── separation.py         # Source separation (HTDemucs)
│   │   └── process_ademilde.py   # Feature extraction script
│   ├── features/
│   │   ├── extraction.py         # Hybrid Crepe + Praat pipeline
│   │   ├── spectral.py           # Spectral features (Alpha Ratio, H1-H2, etc.)
│   │   ├── vmi.py                # Vocal Mechanism Index computation
│   │   └── articulation.py       # Articulatory agility features
│   ├── analysis/
│   │   ├── exploratory.py        # M1/M2 analysis, clustering
│   │   ├── run_analysis.py       # Full analysis script
│   │   └── llm_report.py         # Gemini-powered report generation
│   ├── modeling/
│   │   └── classifier.py         # XGBoost for M1/M2 classification
│   ├── scripts/
│   │   └── regenerate_validation_plot.py  # Regenerate plots without reprocessing
│   ├── visualization/
│   │   └── plots.py              # Academic plots
│   └── utils/
│       └── pitch.py              # Hz <-> Note conversion (A4, C5, etc.)
├── data/
│   ├── raw/                      # Original audio files (.mp3)
│   └── processed/                # CSVs, JSONs, logs
├── docs/
│   ├── en/                       # English documentation
│   └── pt-BR/                    # Portuguese documentation
├── outputs/
│   ├── plots/                    # Generated charts
│   └── models/                   # Trained models
└── tests/
```

## Platform Usage

### macOS
- **Quick validation**: Use `--use-praat-f0` to test the pipeline
- **Limitation**: CREPE full may crash due to memory (32GB+ recommended)
- **Recommendation**: Use macOS for validation only, process with CREPE on Colab/Windows

```bash
# Quick validation on macOS (Praat F0)
uv run python -m vocal_analysis.preprocessing.process_ademilde --use-praat-f0
```

### Windows/Linux (32GB+ RAM)
- **Full processing**: CREPE full with all features
- **GPU (NVIDIA)**: Use `--device cuda` for acceleration (~10x faster)

```bash
# Windows/Linux with CPU
uv run python -m vocal_analysis.preprocessing.process_ademilde

# Windows/Linux with NVIDIA GPU
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda
```

### Google Colab (Recommended!)
- **Free T4 GPU**: ~12-15h/day of usage
- **Fast processing**: 3 songs (~7min each) in ~10 minutes
- **Zero configuration**: Environment ready to go

**Full guide**: [docs/en/COLAB_SETUP.md](docs/en/COLAB_SETUP.md)

**Quick start**:
```python
# In Colab with T4 GPU enabled
!git clone https://github.com/arthurcornelio88/vocal-analysis.git
%cd vocal-analysis
!pip install uv && uv pip install --system -e .

# Verify installation
!python -c "import vocal_analysis; print('Installed!')"

# Process with CREPE + GPU
!python src/vocal_analysis/preprocessing/process_ademilde.py --device cuda
```

---

## Usage

### 1. Add audio files

Place MP3 files in `data/raw/`:

```
data/raw/
├── Apanhei-te Cavaquinho.mp3
├── brasileirinho.mp3
└── delicado.mp3
```

### 2. Process audio (extract features)

```bash
# Full processing with CREPE (requires GPU or 32GB+ RAM)
uv run python -m vocal_analysis.preprocessing.process_ademilde

# With GPU (Google Colab, Windows/Linux with NVIDIA)
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda

# Fast mode with Praat (macOS, validation)
uv run python -m vocal_analysis.preprocessing.process_ademilde --use-praat-f0
```

**Available options:**
- `--device cuda`: Use GPU (requires CUDA)
- `--use-praat-f0`: Use Praat instead of CREPE (faster, less accurate)
- `--crepe-model {tiny,small,full}`: Choose CREPE model (default: full)
- `--skip-formants`: Skip formant extraction (~30% faster)
- `--skip-jitter-shimmer`: Skip jitter/shimmer (~20% faster)
- `--skip-cpps`: Skip CPPS (avoids macOS hang)
- `--skip-plots`: Don't generate F0 plots
- `--limit N`: Process only N files (useful for testing)
- `--fast`: Enable all optimizations (Praat + no formants/jitter/shimmer/cpps/plots)
- `--no-separate-vocals`: Disable source separation (HTDemucs). **By default**, vocal separation is enabled to improve pitch detection in complex arrangements

**Generated outputs:**
- `data/processed/ademilde_features.csv` - Per-frame features
- `data/processed/ademilde_metadata.json` - Structured metadata
- `data/processed/processing_log.md` - Human-readable log
- `outputs/plots/*_f0.png` - Pitch contours

### 2.1. Regenerate validation plots (without reprocessing)

If you've already processed the audio and just want to regenerate validation plots:

```bash
# List songs with available cache
uv run python -m vocal_analysis.scripts.regenerate_validation_plot

# Regenerate plot for a specific song
uv run python -m vocal_analysis.scripts.regenerate_validation_plot --song "Apanhei-te Cavaquinho"

# Regenerate all plots
uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all

# Use CREPE instead of Praat (slower, more accurate)
uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all --use-crepe
```

The script uses separated vocal data cached in `data/cache/separated/` and excerpt intervals defined in `.env`.

### 3. Run exploratory analysis

```bash
uv run python -m vocal_analysis.analysis.run_analysis
```

**Generated outputs:**
- `outputs/plots/mechanism_analysis.png` - 4 M1/M2 analysis plots (threshold)
- `outputs/plots/mechanism_clusters.png` - GMM clustering
- `outputs/plots/xgb_mechanism_timeline.png` - Temporal contour colored by XGBoost prediction
- `outputs/xgb_predictions.csv` - Per-frame predictions (GMM + XGBoost)
- `outputs/analysis_report.md` - Structured report (includes XGBoost classification report)
- `outputs/vmi_analysis.md` - VMI report (if spectral features available)
- `outputs/llm_report.md` - Narrative report with Gemini (if API configured)

#### Multimodal LLM Report

If `GEMINI_API_KEY` is configured, the script generates a narrative report using Gemini 2.0 Flash with:

- **Multimodal analysis**: The LLM receives plots alongside numerical data
- **Clickable links**: Graph references include markdown links (e.g., `[brasileirinho_f0](plots/brasileirinho_f0.png)`)
- **Figure index**: Complete list of visualizations at the end of the report

### 4. Generate LLM report only (optional)

To generate just the narrative report with Gemini (without re-running the full analysis):

```bash
uv run python -m vocal_analysis.analysis.llm_report
```

**Optional parameters:**
- `--metadata`: Path to metadata file (default: `data/processed/ademilde_metadata.json`)
- `--stats`: Path to M1/M2 statistics JSON (optional)
- `--output`: Output path for the report (default: `outputs/llm_report.md`)
- `--plots-dir`: Directory with PNG plots for multimodal analysis (default: `outputs/plots/`)
- `--lang`: Report language (`en` or `pt-BR`, default: `en`)

**Prerequisites:**
1. Gemini API key configured: `export GEMINI_API_KEY=your_key_here`
2. Processed data files available (after running steps 1-3)
3. Generated plots (optional, for multimodal analysis)

**Common errors:**
1. **"API key not valid"**: Check that the API key is correct and active at [Google AI Studio](https://aistudio.google.com/apikey)
2. **"quota exceeded"**: The free Gemini tier has usage limits. Wait for quota reset or upgrade.
3. **Deprecation warning**: The `google-generativeai` package is deprecated but still functional. Ignore the warning.

### 5. Extracted features

| Column | Description |
|--------|-------------|
| `time` | Timestamp in seconds |
| `f0` | Fundamental frequency (Hz) |
| `confidence` | Pitch estimation confidence (0-1) |
| `hnr` | Harmonic-to-Noise Ratio (dB) |
| `energy` | RMS energy |
| `f1, f2, f3, f4` | Formants 1-4 (Hz) |
| `song` | Song name |
| `cpps_global` | Cepstral Peak Prominence (global value per song) |
| `jitter` | Jitter ppq5 - period instability (%) |
| `shimmer` | Shimmer apq11 - amplitude instability (%) |

## Utilities

### Hz <-> Note Conversion

```python
from vocal_analysis.utils.pitch import hz_to_note, note_to_hz

hz_to_note(440.0)        # "A4"
hz_to_note(261.63)       # "C4"
note_to_hz("G5")         # 783.99
```

## Development

### Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Tests

```bash
uv run pytest -v
```

## Documentation

### Introductory Guide

To understand the bioacoustic concepts used in the analysis (f0, HNR, formants, jitter, shimmer) and why each matters, without prior technical background:

**[docs/en/BIOACOUSTIC_GLOSSARY.md](docs/en/BIOACOUSTIC_GLOSSARY.md)**

### Detailed Methodology

For an in-depth understanding of methodological choices, technical parameters, and academic justifications for each pipeline component:

**[docs/en/METHODOLOGY.md](docs/en/METHODOLOGY.md)**

This document covers:
- CREPE vs traditional autocorrelation methods
- Preprocessing parameters (normalization, hop length, thresholds)
- Bioacoustic feature details (f0, HNR, CPPS, Jitter, Shimmer, Formants)
- Articulatory agility features (f0 velocity, syllabic rate)
- M1/M2 classification methods (Threshold, GMM, XGBoost, VMI)
- Data structure and execution workflow
- Recognized limitations and academic compliance

## References

- **CREPE**: [Kim et al., 2018 - CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182)
- **Praat**: [Boersma & Weenink - Praat: doing phonetics by computer](https://www.praat.org)
- **Laryngeal Mechanisms**: Roubeau, B., Henrich, N., & Castellengo, M. (2009). Laryngeal vibratory mechanisms
