[Portugues](../pt-BR/METODOLOGIA.md) | **English**

# Computational Methodology - Bioacoustic Analysis of Laryngeal Mechanisms

**Version:** 2.0.4
**Date:** 2026-02-09
**Context:** Computational analysis for academic paper on vocal "Fach" classification in Choro

> **New to the topic?** First read the [bioacoustic glossary](BIOACOUSTIC_GLOSSARY.md) — it explains the concepts and analysis logic in an accessible way, without technical jargon.

---

## 1. Context and Objectives

This document describes the computational methodology implemented for the physiological analysis of laryngeal mechanisms (M1/M2) in recordings of Choro singing. The goal is to provide quantitative evidence that challenges the traditional vocal "Fach" classification system, demonstrating through explainable bioacoustic features that singers use both mechanisms fluidly.

### 1.1 Laryngeal Mechanisms

- **M1 (Mechanism 1)**: Chest/modal voice. Characteristics: greater vocal fold mass, higher spectral energy in low harmonics, typically elevated HNR.
- **M2 (Mechanism 2)**: Head voice/falsetto. Characteristics: lower vibratory mass, energy concentrated in high harmonics, characteristic phase transitions.

### 1.2 Technical Challenge

Historical Choro recordings (1940s-1960s) present:
- Elevated background noise
- Low signal-to-noise ratio (SNR)
- Intense vibrato and rapid ornamentations (glissandi, portamenti)
- Degraded spectral quality

These characteristics require robust methods for pitch extraction and vocal quality features.

### 1.3 Generalization to Other Genres

Although developed for Choro, this pipeline is **genre-agnostic** and can be applied to any solo vocal repertoire: opera, MPB, fado, vocal jazz, sacred music, etc. The modular architecture (source separation -> feature extraction -> mechanism classification) is reusable without structural modification.

**Required adaptations per genre:**

| Parameter | Choro (current) | Suggested adaptation |
|-----------|-----------------|----------------------|
| `fmin` / `fmax` | 50-800 Hz | Adjust according to tessitura (e.g., operatic bass: 50-400 Hz; soprano: 200-1200 Hz) |
| Source separation | HTDemucs (mixed arrangement) | Unnecessary for *a cappella* voice; essential for orchestral arrangements |
| Threshold M1/M2 | 400 Hz (~G4, female voices) | Adjust by tessitura (e.g., ~300 Hz for male voices) |
| VMI | Fixed weights | The spectral weights (alpha ratio, H1-H2, spectral tilt) are tessitura-independent by construction |

The VMI (section 6.1, Method 4) is particularly portable, as it classifies mechanisms based on spectral features rather than absolute frequency thresholds.

---

## 2. Processing Pipeline

### 2.1 Audio Preprocessing

**Module:** `src/vocal_analysis/preprocessing/audio.py`

```python
load_audio(audio_path, sr=44100, mono=True, normalize=True, target_db=-3.0)
```

#### Critical Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Sample Rate** | 44.1 kHz | CD quality standard, supports fmax up to 22.05 kHz |
| **Mono** | True | Human voice is a point source, stereo unnecessary |
| **Normalization** | -3 dBFS | Standardizes amplitude across recordings, avoids clipping |

**Normalization Implementation:**
```python
target_amplitude = 10 ** (target_db / 20)  # -3dB = 0.708 in linear amplitude
audio_normalized = audio * (target_amplitude / max(abs(audio)))
```

### 2.2 Source Separation (HTDemucs)

**Module:** `src/vocal_analysis/preprocessing/separation.py`

In complex Choro arrangements (7-string guitar, cavaquinho, pandeiro, flute), pitch detection may capture instruments instead of the voice. Vocal source separation via HTDemucs is **enabled by default** since v2.0, as it significantly improves pitch detection in dense arrangements. To disable it (e.g., *a cappella* recordings), use `--no-separate-vocals`.

#### Why HTDemucs?

| Aspect | HTDemucs | Spleeter |
|--------|----------|----------|
| **Architecture** | Hybrid (time + frequency) with Transformer | Simple U-Net |
| **SDR on vocals** | 7-9 dB | 5-6 dB |
| **Quality in dense arrangements** | Excellent | Artifacts in overlapping instruments |
| **PyTorch Integration** | Via torchaudio.pipelines | Requires separate package |

**Reference:** Defossez, A., et al. (2021). Hybrid Spectrogram and Waveform Source Separation. *ISMIR*.

#### Implementation

```python
from vocal_analysis.preprocessing.separation import separate_vocals

# Separate vocals (with automatic caching)
vocals, sr = separate_vocals(
    audio_path,
    device="cuda",           # or "cpu"
    cache_dir=Path("data/cache/separated")
)
# vocals: np.ndarray mono, sr: 44100 Hz
```

#### Flow with Separation

```
Audio MP3 -> HTDemucs (separate vocals) -> Temporary WAV -> CREPE + Praat -> Features
                |
         .npy Cache (avoids reprocessing)
```

#### Visual Validation

To confirm that the separation is capturing the voice (and not the cavaquinho), the pipeline generates comparative plots with `--validate-separation`:

- Left Y-axis: Frequency (Hz)
- Right Y-axis: Musical notes (A#4, C5, etc.)
- Coloring: CREPE confidence (0-1)

If the melody after separation is more continuous and at the expected notes for female voice (~200-600 Hz), the separation is working.

#### CLI Flags

```bash
--no-separate-vocals       # Disable source separation (not recommended for complex arrangements)
--separation-device        # cpu or cuda (default: same as --device)
--no-separation-cache      # Force reprocessing
--validate-separation      # Generate comparative plot Hz + notes
```

**Note:** Separation is enabled by default. It is not necessary to pass `--separate-vocals`.

### 2.3 Hybrid Feature Extraction (Crepe + Praat)

**Module:** `src/vocal_analysis/features/extraction.py`

The pipeline combines:
1. **CREPE (CNN)** for robust f0 extraction
2. **Praat/Parselmouth** for spectral features (gold standard in vocal analysis)

---

## 3. Fundamental Frequency (f0) Extraction

### 3.1 Choice of CREPE

**Method:** Convolutional Neural Network trained on annotated pitch data
**Reference:** Kim et al. (2018) - "CREPE: A Convolutional Representation for Pitch Estimation"

#### Why CREPE instead of Autocorrelation (Praat)?

| Aspect | CREPE (CNN) | Praat (Autocorrelation) |
|--------|-------------|------------------------|
| **Intense vibrato** | Robust | May confuse with subharmonics |
| **Background noise** | Learns to ignore | Degrades autocorrelation peaks |
| **Rapid ornamentations** | High temporal resolution | Depends on windowing |
| **Historical recordings** | Generalizes to low SNR | Requires SNR > 20dB |

**Implementation:**
```python
f0, confidence = torchcrepe.predict(
    audio_tensor,
    sample_rate=44100,
    hop_length=220,        # 5ms @ 44.1kHz — capture rapid ornaments
    fmin=50.0,             # ~G1 (lower limit of human voice)
    fmax=800.0,            # ~G5 (covers M1 and M2)
    model='full',          # Maximum precision for academic context
    decoder=torchcrepe.decode.weighted_argmax,  # Preserves real high-pitched notes
    return_periodicity=True
)
```

### 3.2 Temporal Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **hop_length** | 220 samples (5ms) | High resolution to capture rapid Choro ornaments (glissandi, portamenti) |
| **model** | `full` | Maximum precision; `tiny` model showed errors in M1<->M2 transitions |
| **decoder** | `weighted_argmax` | Preserves real high-pitched notes in M2; `viterbi` tends to over-smooth, "swallowing" peaks |
| **CREPE Windowing** | ~25ms (internal) | Not configurable, optimized by CNN architecture |
| **Filtering** | fmin/fmax + periodicity | Frames with f0 outside [50, 800] Hz are discarded; confidence based on periodicity (not fixed threshold) |

**Note on decoder:** The `weighted_argmax` may generate more instantaneous variance than `viterbi`, but it better preserves rapid register transitions and high-pitched notes in M2 — essential for laryngeal mechanism analysis.

**Note on windowing:** CREPE internally uses its own windowing (~25ms) that is not user-configurable. This architectural choice has been validated in MIR benchmarks and outperforms autocorrelation-based methods.

---

## 4. Vocal Quality Features (Praat/Parselmouth)

### 4.1 Harmonicity-to-Noise Ratio (HNR)

**Definition:** Ratio between harmonic energy and noise energy (dB)
**Interpretation:**
- HNR > 15 dB -> "Clean" voice, efficient glottal closure (typical M1)
- HNR < 10 dB -> Breathiness, aspirative noise (typical M2 or pathologies)

**Extraction:**
```python
harmonicity = sound.to_harmonicity(time_step=0.005)  # 5ms (= hop_length/sr)
hnr_values = harmonicity.values[0]  # Temporal array
```

### 4.2 Cepstral Peak Prominence Smoothed (CPPS)

**Definition:** Smoothed cepstral peak prominence, proxy for vocal fold vibration regularity
**Application:** Differentiation of modal phonation (M1) vs breathiness (M2)

**Extraction:**
```python
power_cepstrogram = parselmouth.praat.call(sound, "To PowerCepstrogram", fmin, time_step, 5000, 50)
cpps = parselmouth.praat.call(power_cepstrogram, "Get CPPS", ...)
```

**Limitation:** Recordings with elevated background noise compromise the measurement. Fallback to mean HNR when extraction fails.

**Note on breathiness in M1:** It is possible to produce breathy sounds in M1 (e.g., intentional breathy phonation, *voix soufflee*). Our model treats low HNR and CPPS as **probabilistic indicators**, not deterministic, of M2. The model is approximate and improvable — the low CPPS -> M2 correlation is a statistical tendency, not an absolute rule.

### 4.3 Jitter (ppq5) and Shimmer (apq11)
**Jitter (Period Perturbation Quotient):**
Measures instability of glottal vibration frequency across 5 consecutive periods.

**Shimmer (Amplitude Perturbation Quotient):**
Measures amplitude variation across 11 consecutive periods.

**Current Relevance & Limitations:**
While historically important, Jitter and Shimmer are **less reliable** for continuous singing analysis, especially in historical recordings, for two reasons:
1. **Confounds with Vibrato:** Vibrato is essentially a periodic frequency modulation (jitter) and amplitude modulation (shimmer). Algorithms often confuse artistic vibrato with pathological instability.
2. **Noise Sensitivity:** Background noise (low SNR) degrades the precise period-to-period detection required for these metrics.

**Methodological Decision:**
We include them for compatibility with traditional literature (Teixeira et al., 2013), but interpret them with caution. **CPPS is our primary metric** for periodicity and voice quality, as it is robust to continuous speech, vibrato, and noise (Maryn & Weenink, 2015).

**Extraction:**
```python
point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", fmin, fmax)
jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
shimmer_apq11 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
```

**Global vs. Frame-level Note:**
These metrics are calculated as **global scalars** (one single value per song), unlike HNR or f0 which are time-series. In the dataset, they are repeated for every frame but do not carry temporal information. Therefore, **they are not used** in the frame-by-frame XGBoost classification or VMI, serving only as descriptive statistics for the entire recording.

### 4.4 Spectral Energy (RMS)

**Definition:** Root Mean Square of amplitude, proxy for vocal intensity.

**Application:** Essential feature for the XGBoost classifier (M1 is typically more energetic than M2).

**Extraction:**
```python
energy = librosa.feature.rms(y=audio, frame_length=int(0.025 * sr), hop_length=220)[0]
```

### 4.5 Formants F1-F4

**Definition:** Vocal tract resonances extracted via Linear Predictive Coding (Burg method).

**Application:**
- Detect formant proximity ("speech zone")
- Differentiate M1 vs M2 timbres
- Identify resonance strategies (*vowel tuning*)

**Practical example:** In M1, F1 typically lies above f0, allowing free resonance of the 1st harmonic. At the passaggio, f0 approaches F1, and trained singers adjust the vocal tract shape (*vowel tuning* / *formant tuning*) to maintain acoustic efficiency — for example, opening the jaw wider to raise F1 and keep it above f0 (Bozeman, 2013). This strategy is common both in high M1 and in the transition to M2.

**References:**
- Bozeman, K. W. (2013). *Practical Vocal Acoustics.* Pendragon Press.
- Bozeman, K. W. (2017). *Kinesthetic Voice Pedagogy.* Inside View Press.
- Sundberg, J. (1987). *The Science of the Singing Voice.* Northern Illinois University Press.

**Extraction:**
```python
formants = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5500)
f1 = formants.get_value_at_time(1, time)
f2 = formants.get_value_at_time(2, time)
# ... F3, F4
```

### 4.6 Spectral Features for VMI

**Module:** `src/vocal_analysis/features/spectral.py`

These features are used to compute the **VMI (Vocal Mechanism Index)**, enabling mechanism classification independent of tessitura. All are extracted with `hop_length=220` (5ms), temporally aligned with CREPE's f0.

**Summary of spectral features:**

| Feature | What it measures | M1 (chest) | M2 (head) | Main reference |
|---------|-----------------|------------|-----------|----------------|
| **Alpha Ratio** | Energy distribution: low vs high frequencies | High (more energy in high frequencies) | Low (energy in low frequencies) | Sundberg & Nordenberg (2006) |
| **H1-H2** | Glottal adduction pattern | Low (firm adduction) | High (light adduction) | Hanson (1997) |
| **Spectral Tilt** | Rate of spectral decay | Steep (negative) | Shallow (near zero) | Fant (1995); Gauffin & Sundberg (1989) |
| **CPPS per-frame** | Vibration periodicity | High (periodic) | High if well produced; low if breathy | Maryn & Weenink (2015) |

---

#### 4.6.1 Alpha Ratio

**Definition:** Spectral energy ratio between the high band (1-5 kHz) and the low band (50 Hz-1 kHz), in dB.

**Reference:** Sundberg, J., & Nordenberg, M. (2006). Effects of vocal loudness variation on spectrum balance as reflected by the alpha measure of long-term-average spectra of speech. *J. Acoust. Soc. Am.*, 120(1), 453-457.

**Physical basis:** In M1, glottal closure is rapid and firm, generating strong upper harmonics (more energy above 1 kHz). In M2/falsetto, closure is gentle, with energy concentrated in the fundamental and first harmonics.

**Interpretation:**
- High Alpha Ratio -> more energy in upper harmonics -> typical of M1
- Low Alpha Ratio -> energy concentrated in low frequencies -> typical of M2/falsetto

**Extraction:**
```python
from vocal_analysis.features.spectral import compute_alpha_ratio
alpha_ratio = compute_alpha_ratio(audio, sr, hop_length=220, low_band=(50, 1000), high_band=(1000, 5000))
```

---

#### 4.6.2 H1-H2 (Harmonic Difference)

**Definition:** Amplitude difference (dB) between the 1st harmonic (H1 = f0) and the 2nd harmonic (H2 = 2xf0).

**References:**
- Hanson, H. M. (1997). Glottal characteristics of female speakers: Acoustic correlates. *J. Acoust. Soc. Am.*, 101(1), 466-481.
- Kreiman, J., Gerratt, B. R., Garellek, M., Samlan, R., & Zhang, Z. (2014). Toward a unified theory of voice production and perception. *Loquens*, 1(1), e009.

**Physical basis:** H1-H2 is the most direct acoustic correlate of the glottal adduction pattern. Firm adduction (M1) produces an abrupt airflow closure, generating strong harmonics — H2 approaches H1 in amplitude (low H1-H2). Light adduction (M2) produces a dominant fundamental with weak harmonics (high H1-H2).

**Interpretation:**
- Low H1-H2 -> firm adduction, steep glottal slope -> M1
- High H1-H2 -> light adduction, gentle glottal slope -> M2

**Limitation:** When f0 > 350 Hz, H1 may coincide with F1 (first vocal tract resonance), contaminating the measurement. For this reason, we use Spectral Tilt as a complement.

**Extraction:**
```python
from vocal_analysis.features.spectral import compute_h1_h2
h1_h2 = compute_h1_h2(audio, sr, f0, hop_length=220, n_fft=4096, harmonic_tolerance_hz=50.0)
```

---

#### 4.6.3 Spectral Tilt

**Definition:** Slope of the linear regression fitted to the power spectrum (log-frequency vs amplitude in dB), in the 50-5000 Hz range.

**References:**
- Fant, G. (1995). The LF-model revisited: Transformations and frequency domain analysis. *STL-QPSR*, 2-3/1995, 119-156.
- Gauffin, J., & Sundberg, J. (1989). Spectral correlates of glottal voice source waveform characteristics. *J. Speech Hear. Res.*, 32(3), 556-565.

**Physical basis — intuitive explanation:** Spectral tilt measures the rate at which vocal energy decays in high frequencies. It works like a "brightness control" for the voice:

- **Steep tilt (very negative):** Energy drops rapidly with frequency. Low frequencies dominate, high frequencies are weak. The voice sounds "dark", "covered". Physically, this occurs when the vocal folds close *slowly and gently* (typical of M2/falsetto): the glottal pulse is smooth, without abrupt discontinuities, and therefore generates few high-frequency harmonics.

- **Shallow tilt (near zero):** Energy is distributed more uniformly. High-frequency harmonics are relatively strong. The voice sounds "bright", "projected". Physically, this occurs when the vocal folds close *rapidly and firmly* (typical of M1/chest): the abrupt closure creates discontinuities in airflow, which in turn generate strong harmonics at high frequencies (just as a square wave has more overtones than a sine wave).

**Causal chain:** glottal closure pattern -> airflow pulse shape -> spectral tilt.

**Advantage over H1-H2:** Spectral tilt captures the *global* pattern of energy distribution, without depending on precise detection of individual harmonics. It is more robust in the upper register (f0 > 350 Hz), where H1-H2 becomes unstable.

**Extraction:**
```python
from vocal_analysis.features.spectral import compute_spectral_tilt
spectral_tilt = compute_spectral_tilt(audio, sr, hop_length=220, fmin=50.0, fmax=5000.0)
```

---

#### 4.6.4 CPPS per-frame

**Definition:** Smoothed cepstral peak prominence, computed per frame (not global).

**Reference:** Maryn, Y., & Weenink, D. (2015). Objective dysphonia measures in the program Praat. *Journal of Voice*.

**Interpretation:**
- High CPPS -> periodic, clean voice
- Low CPPS -> noise, aperiodicity

**Note:** High CPPS occurs both in dense M1 and in reinforced M2 (*voix mixte*) — both are periodic and clean. Low CPPS indicates breathiness or voice breaks, not a specific mechanism. For this reason, in the VMI, CPPS has a neutral contribution (it does not strongly discriminate M1/M2).

**Extraction:**
```python
from vocal_analysis.features.spectral import compute_cpps_per_frame
cpps = compute_cpps_per_frame(audio_path, hop_length=220, window_duration=0.04)
```

---

#### 4.6.5 Features -> VMI Mapping

| Configuration | Alpha Ratio | CPPS | H1-H2 | Spectral Tilt | VMI |
|---------------|-------------|------|--------|---------------|-----|
| **Dense M1** | High | High | Low | Steep (negative) | 0.0-0.2 |
| **Light M1** | Moderate | High | Moderate | Moderate | 0.2-0.4 |
| **Passaggio/Mix** | Variable | Variable | Unstable | Transition | 0.4-0.6 |
| **Reinforced M2** | Moderate | High | High | Gentle | 0.6-0.8 |
| **Light M2** | Low | Moderate | Very high | Very gentle | 0.8-1.0 |

**Theoretical Notes:**

1. **High CPPS in Reinforced M2:** A well-produced M2 (*voix mixte*) has **high** CPPS because it is periodic and clean. Low CPPS indicates noise/aperiodicity, not reinforced resonance.

2. **Unstable H1-H2 at the passaggio:** When f0 > 350 Hz, H1 may coincide with F1, making H1-H2 less reliable (Hanson, 1997). For this reason we include Spectral Tilt as a complementary feature.

3. **F0-F1 does not indicate mechanism:** F0-F1 proximity is a resonance strategy (*vowel tuning*; Bozeman, 2013). Sopranos in M2 and tenors in M1 can use the same strategy. Do not use directly in VMI.

---

## 5. Articulatory Agility Features

**Module:** `src/vocal_analysis/features/articulation.py`

### 5.1 Pitch Change Rate (f0 velocity)

**Definition:** Rate of change of fundamental frequency (Hz/s)

```python
f0_velocity = Δf0 / Δt
```

**Application:** Quantify rapid ornamentations (glissandi, portamenti) characteristic of Choro.

### 5.2 Pitch Acceleration (f0 acceleration)

**Definition:** Rate of change of pitch velocity (Hz/s^2)

```python
f0_acceleration = Δ(f0_velocity) / Δt
```

**Application:** Detect abrupt register transitions (M1->M2 breaks).

### 5.3 Syllabic Rate

**Definition:** Estimate of syllables per second via energy peak detection.

**Method:**
1. Find local peaks in the RMS energy signal
2. Apply minimum distance of 100ms between peaks (avoid double counting)
3. Normalize by total time

```python
from scipy.signal import find_peaks
peaks, _ = find_peaks(energy, distance=int(0.1 / time_step))
syllable_rate = len(peaks) / duration
```

**Application:** Proxy for singer's technical agility.

---

## 6. M1/M2 Mechanism Classification

### 6.1 Hybrid Approach

The pipeline implements **4 complementary methods**:

#### Method 1: Heuristic Threshold
```python
mechanism = "M1" if f0 < 400 Hz else "M2"
```
**Justification:** 400 Hz (~G4) is the empirical passaggio threshold for female voices.
**Limitation:** Ignores covariance with HNR/energy.

#### Method 2: Gaussian Mixture Model (GMM)
```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

features = df_voiced[["f0", "hnr"]].values
scaler = RobustScaler()  # Uses median/IQR, robust to outliers
features_norm = scaler.fit_transform(features)

gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(features_norm)
```
**Advantage:** Unsupervised, discovers natural clusters.
**Normalization:** RobustScaler (median/IQR) instead of StandardScaler (mean/std), more robust to outliers in historical recordings.

![Mechanism Clusters (GMM)](../../outputs/plots/mechanism_clusters.png)

#### Method 3: XGBoost (Supervised with Pseudo-Labels)
```python
import xgboost as xgb
# Use GMM labels as pseudo-labels
# Base features + formants if available in CSV
X = (f0, HNR, energy, f0_velocity, f0_acceleration, f1, f2, f3, f4)
y = gmm_labels  # 0=M1, 1=M2
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X, y)
# Prediction applied over all voiced frames
predictions = model.predict(X)
```

**Advantage:** Learns non-linear interactions between features.
**Application:** Robust classification for new data. Classification report saved in the `outputs/analysis_report.md` report.

**Theoretical justification for pseudo-labeling:** Using labels generated by an unsupervised model (GMM) to train a supervised classifier (XGBoost) is an established technique in semi-supervised learning (Lee, 2013; Nigam et al., 2000). The idea is that the GMM discovers the natural data structure (M1/M2 clusters in the f0 x HNR space), and XGBoost learns more complex decision boundaries using additional features (energy, formants, pitch velocity).

**Caveats:**
1. **Confirmation bias:** XGBoost inherits GMM errors — it cannot be *better* than its pseudo-labels on average.
2. **Gaussian assumption:** The GMM assumes Gaussian clusters, which may not reflect the real distribution of mechanisms.
3. **Absence of ground truth:** Without laryngoscopic or electroglottographic (EGG) validation, the true accuracy is unknown.

The VMI (Method 4) offers a complementary approach: based on acoustic theory and spectral features, it does not depend on pseudo-labels.

![XGBoost Prediction: M1 vs M2 over time](../../outputs/plots/xgb_mechanism_timeline.png)

#### Method 4: VMI (Vocal Mechanism Index) — Tessitura-Agnostic

**Module:** `src/vocal_analysis/features/vmi.py`

The VMI is a continuous metric (0-1) that replaces the arbitrary G4 threshold with analysis based on **spectral features**:

```python
from vocal_analysis.features.vmi import compute_vmi_fixed, vmi_to_label

vmi = compute_vmi_fixed(
    alpha_ratio=df["alpha_ratio"].values,
    cpps=df["cpps_per_frame"].values,
    h1_h2=df["h1_h2"].values,
    spectral_tilt=df["spectral_tilt"].values,
)
labels = vmi_to_label(vmi)  # M1_HEAVY, M1_LIGHT, MIX_PASSAGGIO, M2_REINFORCED, M2_LIGHT
```

**VMI Scale:**

| VMI | Label | Description |
|-----|-------|-------------|
| 0.0-0.2 | `M1_HEAVY` | Heavy mechanism, firm adduction, full chest voice |
| 0.2-0.4 | `M1_LIGHT` | Thin-edge M1, common in tenors/mid register |
| 0.4-0.6 | `MIX_PASSAGGIO` | Passage zone, acoustic instability, mixed voice |
| 0.6-0.8 | `M2_REINFORCED` | M2 with glottal adduction, frontal resonance |
| 0.8-1.0 | `M2_LIGHT` | Light mechanism, falsetto, piano M2 |

**Advantage:** Does not depend on fixed frequencies like G4 — works for any tessitura.
**Application:** Gradual identification of vocal mechanism and passaggio.

### 6.2 Features Used in XGBoost

| Feature | Type | Expected Importance | Justification |
|---------|------|---------------------|---------------|
| **f0** | Base | High | Primary separation (M1 low, M2 high) |
| **HNR** | Base | Medium | M1 > M2 in modal phonation |
| **energy** | Base | Medium-High | M1 more energetic than M2 |
| **f0_velocity** | Derivative | Medium-High | M1->M2 transitions are rapid ornaments (glissandi) |
| **f0_acceleration** | Derivative | Medium | Abrupt register breaks indicate mechanism change |
| **f1, f2, f3, f4** | Formants | High | Vocal tract resonances directly differentiate M1 vs M2 |
| **alpha_ratio** | Spectral (VMI) | High | Spectral energy ratio: M1 brighter than M2 |
| **h1_h2** | Spectral (VMI) | Medium-High | Direct correlate of glottal adduction pattern |
| **spectral_tilt** | Spectral (VMI) | Medium | Robust in upper register (f0 > 350 Hz) |
| **cpps_per_frame** | Spectral (VMI) | Medium | Per-frame vibration regularity |

**Note:** F1-F4 are automatically included if available in the CSV (processing without `--skip-formants`). Spectral features (alpha_ratio, h1_h2, spectral_tilt, cpps_per_frame) require `--extract-spectral` during processing or are computed on-the-fly by `run_analysis`. `cpps_global`, `jitter`, and `shimmer` are scalar values per song (not per frame) and therefore do not generate useful variation for per-frame classification.

---

## 7. Visualizations and Reports

### 7.1 Academic Plots

**Module:** `src/vocal_analysis/visualization/plots.py`

- **Temporal f0 contour** (`{song}_f0.png`): Visual identification of M1<->M2 transitions
- **Mechanism analysis** (`mechanism_analysis.png`): 4 subplots — histogram, f0 vs HNR scatter, boxplot, temporal
- **GMM clusters** (`mechanism_clusters.png`): f0 vs HNR scatter colored by cluster
- **XGBoost timeline** (`xgb_mechanism_timeline.png`): Temporal f0 contour colored by XGBoost prediction (M1=blue, M2=coral)
- **VMI scatter** (`vmi_scatter.png`): f0 vs Alpha Ratio colored by VMI [0-1] with RdBu_r scale
- **VMI analysis** (`vmi_analysis.png`): 4 subplots — scatter, distribution, contour, boxplot by VMI category
- **Excerpts per song** (`excerpt_{song}.png`): 5-second excerpts at the densest window, with musical note axis — for manual inspection (human eval)
- **Separation validation** (`{song}_separation_validation.png`): Before/after source separation comparison

**Aesthetics:** Seaborn (`whitegrid`), `viridis` / `RdBu_r` (VMI) palette, DPI 150 for publication.

Excerpts automatically generated at the window of highest frame density per song:

![Apanhei-te Cavaquinho — excerpt](../../outputs/plots/excerpt_apanheite_cavaquinho.png)

![delicado — excerpt](../../outputs/plots/excerpt_delicado.png)

![brasileirinho — excerpt](../../outputs/plots/excerpt_brasileirinho.png)

VMI scatter — f0 vs Alpha Ratio colored by VMI:

![VMI Scatter](../../outputs/plots/vmi_scatter.png)

### 7.2 Narrative Report with AI

**Module:** `src/vocal_analysis/analysis/llm_report.py`

Uses **Gemini Multimodal** (Google) to:
1. Analyze generated plots
2. Interpret descriptive statistics
3. Generate contextualized academic narrative

**Input:** Plot images + JSON with statistics
**Output:** Markdown report with qualitative insights

---

## 8. Academic Rigor Compliance

### 8.1 Reproducibility

| Aspect | Guarantee |
|--------|-----------|
| **Random seed** | `random_state=42` in all models |
| **Fixed versions** | `pyproject.toml` with locked dependencies |
| **Documented parameters** | Values justified in inline comments |

### 8.2 Validation

- **Manual auditory inspection:** Verify M1/M2 classification in ambiguous passages
- **Cross-validation:** K-fold (k=5) on XGBoost to estimate generalization
- **Ablation study:** Evaluate individual importance of each feature

### 8.3 Acknowledged Limitations

1. **Historical recordings:** Background noise limits CPPS precision
2. **M1/M2 threshold:** 400 Hz is a heuristic, may vary between individuals
3. **Pseudo-labels:** GMM does not guarantee 100% accuracy for training XGBoost
4. **Absence of ground truth:** No validation by laryngoscopic analysis

---

## 9. Data Structure

### 9.1 Features DataFrame

**File:** `data/processed/ademilde_features.csv` (generated by `process_ademilde`)

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Timestamp in seconds |
| `f0` | float | Fundamental frequency (Hz) |
| `confidence` | float | CREPE detection confidence (0-1) |
| `hnr` | float | Harmonic-to-Noise Ratio (dB) |
| `energy` | float | RMS energy |
| `f1, f2, f3, f4` | float | Formants 1-4 (Hz) |
| `cpps_global` | float | CPPS (global value per song) |
| `jitter` | float | Jitter ppq5 (%) - global value per song |
| `shimmer` | float | Shimmer apq11 (%) - global value per song |
| `song` | string | Song name |
| `alpha_ratio` | float | Energy ratio 0-1kHz vs 1-5kHz (dB) — if `--extract-spectral` |
| `h1_h2` | float | H1-H2 difference (dB) — if `--extract-spectral` |
| `spectral_tilt` | float | Spectral tilt — if `--extract-spectral` |
| `cpps_per_frame` | float | CPPS per frame — if `--cpps-per-frame` |

**Spectral features** (computed in `process_ademilde` with `--extract-spectral` OR on-the-fly in `run_analysis` if absent from CSV):

| Column | Type | Description | Origin |
|--------|------|-------------|--------|
| `alpha_ratio` | float | Energy ratio 0-1kHz vs 1-5kHz (dB) | process_ademilde (`--extract-spectral`) or run_analysis |
| `h1_h2` | float | H1-H2 difference (dB) | process_ademilde (`--extract-spectral`) or run_analysis |
| `spectral_tilt` | float | Spectral tilt | process_ademilde (`--extract-spectral`) or run_analysis |
| `cpps_per_frame` | float | CPPS per frame | process_ademilde (`--cpps-per-frame`) |

**Derived features** (computed exclusively by `run_analysis`, present in `xgb_predictions.csv`):

| Column | Type | Description |
|--------|------|-------------|
| `f0_velocity` | float | Pitch change rate (Hz/s) |
| `f0_acceleration` | float | Pitch acceleration (Hz/s^2) |
| `syllable_rate` | float | Syllabic rate (syllables/s) |
| `cluster` | int | GMM cluster (0/1) |
| `mechanism` | string | GMM label (M1/M2) |
| `xgb_mechanism` | string | XGBoost prediction (M1/M2) |
| `vmi` | float | Vocal Mechanism Index (0-1) — if spectral features available |
| `vmi_label` | string | VMI label (M1_HEAVY, M1_LIGHT, MIX_PASSAGGIO, M2_REINFORCED, M2_LIGHT) |

**Predictions file:** `outputs/xgb_predictions.csv` (generated by `run_analysis`, contains all columns above)

### 9.2 Metadata JSON

**File:** `data/processed/ademilde_metadata.json`

```json
{
  "artist": "Ademilde Fonseca",
  "processed_at": "2025-01-21T10:30:00",
  "n_songs": 5,
  "global": {
    "f0_mean_hz": 285.3,
    "f0_range_notes": "C4-G5",
    "hnr_mean_db": 12.4
  },
  "songs": [...]
}
```

---

## 10. Execution Workflow

### Step 1: Audio Processing

Source separation (HTDemucs) is enabled by default.

```bash
# Recommended: full processing with spectral features for VMI
uv run python -m vocal_analysis.preprocessing.process_ademilde --extract-spectral --device cuda

# WITH visual validation of separation (generates Hz + notes plots)
uv run python -m vocal_analysis.preprocessing.process_ademilde --extract-spectral --validate-separation --limit 1

# WITH per-frame CPPS (slower, better VMI precision)
uv run python -m vocal_analysis.preprocessing.process_ademilde --extract-spectral --cpps-per-frame --device cuda

# WITHOUT source separation (not recommended for complex arrangements)
uv run python -m vocal_analysis.preprocessing.process_ademilde --no-separate-vocals --extract-spectral
```

**Output:**
- `data/processed/ademilde_features.csv`
- `data/processed/ademilde_metadata.json`
- `outputs/plots/{song}_f0.png` (one per song)
- `data/cache/separated/{song}_vocals.npy` (separation cache)
- `outputs/plots/{song}_separation_validation.png` (if `--validate-separation`)

### Step 2: Exploratory Analysis + Classification + VMI

```bash
# Full analysis (VMI enabled by default if spectral features available)
uv run python -m vocal_analysis.analysis.run_analysis
```

The environment variable `USE_VMI=true` (default) enables VMI analysis when spectral features are available in the CSV or in the audio files.

**Output:**
- `outputs/plots/mechanism_analysis.png` (threshold)
- `outputs/plots/mechanism_clusters.png` (GMM)
- `outputs/plots/xgb_mechanism_timeline.png` (temporal contour by XGBoost prediction)
- `outputs/plots/vmi_scatter.png` (f0 vs Alpha Ratio colored by VMI)
- `outputs/plots/vmi_analysis.png` (4 VMI subplots)
- `outputs/plots/excerpt_{song}.png` (5-second excerpts per song, note by note, for human eval)
- `outputs/xgb_predictions.csv` (per-frame predictions: GMM + XGBoost + VMI)
- `outputs/analysis_report.md` (basic report, includes XGBoost classification report)
- `outputs/vmi_analysis.md` (VMI report with distribution by category)
- `outputs/llm_report.md` (narrative report, requires `GEMINI_API_KEY`)

---

## 11. Methodological References

### Laryngeal Mechanisms M1/M2

1. **Roubeau, B., Henrich, N., & Castellengo, M. (2009).** Laryngeal Vibratory Mechanisms: the notion of vocal register revisited. *Journal of Voice*, 23(4), 425-438.

2. **Henrich, N., d'Alessandro, C., Doval, B., & Castellengo, M. (2004).** Glottal open quotient in singing: measurements and correlation with laryngeal mechanisms, vocal intensity, and fundamental frequency. *J. Acoust. Soc. Am.*, 117(3), 1417-1430.

3. **Henrich, N. (2006).** Mirroring the voice from Garcia to the present day: some insights into singing voice registers. *Logopedics Phoniatrics Vocology*, 31(1), 3-14.

4. **Henrich, N., et al. (2014).** Vocal tract resonances in singing: variation with laryngeal mechanism for male operatic singers. *J. Acoust. Soc. Am.*, 135(1), 491-501.

### Pitch and Vocal Analysis

5. **Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018).** CREPE: A Convolutional Representation for Pitch Estimation. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. arXiv:1802.06182.

6. **Boersma, P., & Weenink, D. (2023).** Praat: doing phonetics by computer [Computer program]. Version 6.3.

7. **Maryn, Y., & Weenink, D. (2015).** Objective dysphonia measures in the program Praat: Smoothed cepstral peak prominence and acoustic voice quality index. *Journal of Voice*.

8. **Teixeira, J. P., Oliveira, C., & Lopes, C. (2013).** Vocal Acoustic Analysis -- Jitter, Shimmer and HNR Parameters. *Procedia Technology*, 9, 1112-1122.

### Spectral Features

9. **Sundberg, J., & Nordenberg, M. (2006).** Effects of vocal loudness variation on spectrum balance as reflected by the alpha measure of long-term-average spectra of speech. *J. Acoust. Soc. Am.*, 120(1), 453-457.

10. **Hanson, H. M. (1997).** Glottal characteristics of female speakers: Acoustic correlates. *J. Acoust. Soc. Am.*, 101(1), 466-481.

11. **Kreiman, J., Gerratt, B. R., Garellek, M., Samlan, R., & Zhang, Z. (2014).** Toward a unified theory of voice production and perception. *Loquens*, 1(1), e009.

12. **Fant, G. (1995).** The LF-model revisited: Transformations and frequency domain analysis. *STL-QPSR*, 2-3/1995, 119-156.

13. **Gauffin, J., & Sundberg, J. (1989).** Spectral correlates of glottal voice source waveform characteristics. *J. Speech Hear. Res.*, 32(3), 556-565.

### Vocal Quality and Physiology

14. **Bourne, T., & Garnier, M. (2012).** Physiological and acoustic characteristics of four qualities in the female music theatre voice. *J. Acoust. Soc. Am.*, 131(2), 1586-1594.

15. **Behlau, M., & Ziemer, R. (1988).** *Voz: o livro do especialista.* Rio de Janeiro: Revinter.

### Vocal Pedagogy and Formants

16. **Bozeman, K. W. (2013).** *Practical Vocal Acoustics: Pedagogic Applications for Teachers and Singers.* Pendragon Press.

17. **Bozeman, K. W. (2017).** *Kinesthetic Voice Pedagogy: Motivating Acoustic Efficiency.* Inside View Press.

18. **Sundberg, J. (1987).** *The Science of the Singing Voice.* Northern Illinois University Press.

19. **Sundberg, J. (1974).** Articulatory interpretation of the "singing formant". *J. Acoust. Soc. Am.*, 55(4), 838-844.

20. **Miller, R. (2000).** *Training Soprano Voices.* New York: Oxford University Press.

21. **Alku, P., Kadiri, S. R., & Gowda, D. (2023).** Refining a deep learning-based formant tracker using linear prediction methods. *Computer Speech & Language*, 81.

### Machine Learning and Register Classification

22. **Kim, A., & Botha, C. (2025).** Machine learning approaches to vocal register classification in contemporary male pop music. *arXiv preprint arXiv:2505.11378*.

23. **Boratto, T., et al. (2025).** Machine Learning with Evolutionary Parameter Tuning for Singing Registers Classification. *Signals*, 6(1), 9.

24. **Hinrichs, R., et al. (2026).** A Dataset for Automatic Vocal Mode Classification. *arXiv preprint arXiv:2601.18339*.

25. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

26. **Lee, D.-H. (2013).** Pseudo-Label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML 2013 Workshop: Challenges in Representation Learning (WREPL)*.

27. **Nigam, K., McCallum, A. K., Thrun, S., & Mitchell, T. (2000).** Text classification from labeled and unlabeled documents using EM. *Machine Learning*, 39, 103-134.

### Source Separation

28. **Defossez, A., Usunier, N., Bottou, L., & Bach, F. (2021).** Hybrid Spectrogram and Waveform Source Separation. *Proceedings of the ISMIR 2021 Conference*.

29. **Rouard, S., Massa, F., & Défossez, A. (2023).** Hybrid Transformers for Music Source Separation. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

### Context: Choro and Fach Classification

30. **Rezende, D. (2016).** *A Voz e o Choro: aspectos técnicos vocais e o repertório de Choro cantado como ferramenta de estudo no canto popular.* Olinda: Livro Rápido.

31. **Cotton, S. (2007).** *Voice Classification and Fach: usage and inconsistencies.* Thesis (Doctor of Musical Arts) — University of North Carolina, Greensboro.

---

## 12. Contact and Contributions

**Author:** Arthur Cornélio (arthur.Cornélio@gmail.com)
**Project:** Bioacoustic Analysis — Laryngeal Mechanisms M1/M2 in Choro
**Version:** 2.0.4
**Stack:** Python 3.10+, torchcrepe, parselmouth, xgboost, torchaudio (HTDemucs), google-generativeai (Gemini)

For methodological questions or improvement suggestions, open an issue in the repository.

---

**Last Updated:** 2026-02-12
**Status:** Validated pipeline (v2.0.4). Source separation enabled by default. VMI (Vocal Mechanism Index) integrated for tessitura-agnostic classification.
