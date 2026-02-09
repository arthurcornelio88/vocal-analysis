[Portugues](../pt-BR/COLAB_SETUP.md) | **English**

# Running the Project on Google Colab with GPU

This guide shows how to process audio files with CREPE using Google Colab's free GPU.

## Step by step

### 1. Open a new notebook on Colab
1. Go to: https://colab.research.google.com
2. File -> New Notebook
3. **IMPORTANT**: Runtime -> Change runtime type -> GPU (T4)

### 2. Clone the repository and install dependencies

```python
# IMPORTANT: Run THIS ENTIRE cell BEFORE continuing!
# Clone + Installation (required for imports to work)

!git clone https://github.com/arthurcornelio88/vocal-analysis.git
%cd vocal-analysis

# Install uv and the package in system mode (no venv)
!pip install uv
!uv pip install --system -e .

# VERIFICATION: If installed correctly, it should display the version
!python -c "import vocal_analysis; print(f'vocal_analysis installed! Version: {vocal_analysis.__version__}')"

# Check GPU availability
import torch
print(f"\nGPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

**Why `--system`?**
- `uv sync` creates a virtual environment that Colab does not use automatically
- `uv pip install --system` installs directly into the system Python
- Faster than `pip` but compatible with Colab

### 3. Upload audio files

```python
from google.colab import files
import os

# Create data/raw directory if it does not exist
os.makedirs('data/raw', exist_ok=True)

# Upload MP3 files
print("Upload your MP3 files:")
uploaded = files.upload()

# Move to data/raw
for filename in uploaded.keys():
    !mv "{filename}" data/raw/

!ls -lh data/raw/
```

### 4. Configure excerpts (optional)

```python
# 1. Set the environment variables in the Colab system
%env EXCERPT_DELICADO="0022-0103"
%env EXCERPT_BRASILEIRINHO="0033-0104"
%env EXCERPT_APANHEITE_CAVAQUINHO="0007-0023"
%env GEMINI_API_KEY="<YOUR-API-KEY>"
```

### 5. Process with CREPE (GPU)

```python
# Basic processing with CREPE + GPU (source separation enabled by default)
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --validate-separation

# WITH spectral features for VMI (recommended)
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --extract-spectral

# WITHOUT source separation (simple arrangements or a cappella vocals)
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --no-separate-vocals

# Check outputs
!ls -lh data/processed/
```

**Expected time**: ~5-10 minutes for 3 songs (~7min each) with GPU T4
- Source separation (default): already included in the time above
- `--extract-spectral`: adds ~1-2 min
- `--no-separate-vocals`: ~30% faster (but less accurate for complex arrangements)

### 6. Generate analyses

```python
# Run exploratory analysis
!!GEMINI_API_KEY=<YOUR-API-KEY> python src/vocal_analysis/analysis/run_analysis.py

# List outputs
!ls -lh outputs/plots/
!ls -lh outputs/*.md
```

### 7. Download results

```python
from google.colab import files

# Download CSV with features
files.download('data/processed/ademilde_features.csv')

# Download metadata JSON
files.download('data/processed/ademilde_metadata.json')

# Download reports
files.download('outputs/analysis_report.md')
files.download('outputs/vmi_analysis.md')

# Download XGBoost predictions
files.download('outputs/xgb_predictions.csv')

# Download plots (zip first)
!zip -r outputs_plots.zip outputs/plots/
files.download('outputs_plots.zip')

# Download audio excerpts
!zip -r excerpts.zip outputs/excerpt_*.wav
files.download('excerpts.zip')
```

## Useful commands

### Process only 1 file (quick test)
```python
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --limit 1
```

### Process without plots (faster)
```python
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --skip-plots
```

### Use a smaller CREPE model (faster, less accurate)
```python
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --crepe-model tiny
```

### Adjust batch size (if you get OOM)
```python
# If T4 gives "Out of Memory", reduce batch size
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --batch-size 512
```

### Extract spectral features for VMI
```python
# Alpha Ratio, H1-H2, Spectral Tilt (fast, recommended)
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --extract-spectral
```

**Note**: The `--cpps-per-frame` flag exists but is **extremely slow** (~40+ min per song). The global CPPS is sufficient for most analyses.

## Tips

1. **Free GPU T4**: ~12-15 hours/day of usage
2. **Save progress**: Download your files before the session expires
3. **Reprocessing**: If needed, files remain saved for ~12h on Colab
4. **Batch size**: Default is now 2048 (ideal for GPU T4). If you get OOM, use `--batch-size 512`

## Troubleshooting

**"ModuleNotFoundError: No module named 'vocal_analysis'"**:
- Run step 2: `!uv pip install --system -e .`
- NEVER use `!uv sync` alone (it creates a venv that Colab does not use)
- Verify with: `!python -c "import vocal_analysis; print('OK!')"`

**"GPU not available"**:
- Check: Runtime -> Change runtime type -> GPU (T4)

**"Out of memory"**:
- Use `--crepe-model small` or `--crepe-model tiny`

**"Process killed"**:
- Reduce batch_size by editing `extraction.py` line 110

## Expected output

After processing, you will have:
- `ademilde_features.csv` - All features (f0, formants, jitter, shimmer, etc.)
- `ademilde_metadata.json` - Metadata and statistics
- `analysis_report.md` - Technical report
- `xgb_predictions.csv` - Mechanism predictions (M1/M2 + VMI)
- Plots of F0, excerpts, mechanisms, etc.
- Excerpt audio files (`.wav`)

**With `--extract-spectral`**, the CSV will also contain:
- `alpha_ratio` - Energy ratio 0-1kHz vs 1-5kHz
- `h1_h2` - Difference between 1st and 2nd harmonic
- `spectral_tilt` - Spectral tilt

**With `--cpps-per-frame`** (very slow, not recommended):
- `cpps_per_frame` - CPPS calculated per frame (~40+ min/song)

**After `run_analysis.py`**, it adds:
- `vmi` - Vocal Mechanism Index (0-1)
- `vmi_label` - Categorical label (M1_HEAVY, M1_LIGHT, MIX_PASSAGGIO, M2_REINFORCED, M2_LIGHT)

## Ready for the paper!

With data from Colab (CREPE + GPU), you will have accurate F0 for academic analysis.
