[Portugues](../pt-BR/GLOSSARIO_BIOACUSTICO.md) | **English**

# Bioacoustic Glossary -- Reading Guide

A companion document for understanding the concepts and logic behind the
analysis described in [METHODOLOGY.md](METHODOLOGY.md). It assumes a basic
familiarity with sound and voice but no background in digital signal processing.

---

## Why does this analysis exist?

The traditional vocal classification system known as **Fach** divides
singers into rigid categories (soprano, mezzo, contralto, etc.) based
on the range of notes they can sing. This analysis aims to show that
such a division is artificial: in practice, Choro singers use **two
laryngeal mechanisms** -- low and high -- fluidly within the same song.
The goal is to measure this quantitatively.

The two mechanisms are:

- **M1 (chest voice):** the vocal folds vibrate with greater mass,
  producing a lower-pitched, "fuller" sound. Think of the way you
  normally speak.
- **M2 (head voice):** the vocal folds vibrate with less mass,
  producing a higher-pitched, "thinner" sound. Think of what happens
  when you try to sing very high and your voice "breaks" into a
  different register.

The transition between M1 and M2 is what we call the **passaggio**.
Measuring when and how this transition occurs is the heart of the
analysis.

Below is the temporal f0 contour color-coded by the classifier's
prediction: blue = M1 (chest voice), coral = M2 (head voice). Each
song shows this fluid alternation within the same passage:

![XGBoost prediction: M1 vs M2 over time](../../outputs/plots/xgb_mechanism_timeline.png)

---

## Key concepts

### f0 -- Fundamental Frequency

**What it is:** The basic frequency of a sound. It determines the
"pitch" you hear -- lower or higher.

**How it works:** When you sing a note, your vocal folds vibrate N
times per second. That N, measured in Hz (hertz), is the f0.
An A4 note (the tuning-fork A) has f0 = 440 Hz.

**Why it matters here:** f0 is the most important feature for
separating M1 from M2. Notes below ~400 Hz tend to be M1; above that
they tend to be M2. But that is not the whole story -- which is why
we need the other features as well.

**In the code:** extracted by CREPE (a neural network), not by Praat,
because CREPE is more robust on recordings with background noise (such
as the historic Choro recordings).

---

### HNR -- Harmonic-to-Noise Ratio

**What it is:** The ratio between the "musical" part of a sound
(harmonics) and the "dirty" part (noise). Measured in dB.

**How to think about it:** High HNR = clean, well-defined voice.
Low HNR = breathier or noisier voice. Like the difference between a
guitar with a new string (bright) and one with an old string (dull).

**Why it matters here:** M1 tends to have a higher HNR than M2
because the vocal folds close more efficiently in the chest-voice
mechanism. HNR therefore helps confirm what f0 alone cannot: whether
an intermediate f0 value truly belongs to M1 or M2.

**A note on historic recordings:** The average HNR in this project is
negative (~-2 dB) because the Choro recordings carry a lot of
background noise. The absolute values do not follow the "normal"
clinical benchmarks (HNR > 15 dB = healthy voice). What matters here
is the **relative difference** between M1 and M2 within the same
recording.

---

### Energy (RMS)

**What it is:** The average intensity of the audio signal over a time
window. It measures "how loud" the sound is.

**Why it matters here:** M1 is typically more energetic than M2.
When the singer shifts to the upper register, energy usually drops.
This provides a second confirmation beyond f0.

---

### Formants F1, F2, F3, F4

**What they are:** Resonance frequencies of the vocal tract (throat,
mouth, nasal cavity). They are not determined by the vocal folds --
they are determined mainly by the shape of the mouth, tongue, and
soft palate.

**How to think about them:** Imagine that the vocal folds are like a
buzzer generating a "raw" sound. The vocal tract is like a pipe that
amplifies certain frequencies of that sound. F1, F2, F3, F4 are the
frequencies that this "pipe" amplifies.

**Why they matter here:** When the singer switches from M1 to M2,
the vocal tract also reconfigures -- especially F1 and F2 shift
position. So formants help capture the register transition from a
different angle than f0.

---

### Jitter and Shimmer

**Jitter:** Period instability between consecutive vibration cycles of
the vocal folds. Think of a metronome that is not perfectly regular.

**Shimmer:** Amplitude instability between consecutive cycles. Like a
metronome that varies in the force of each beat.

**Why they matter here:** These are global values per song (not per
frame), so they do not feed directly into the frame-by-frame
classifier. However, they are useful for describing the singer's
overall vocal quality across recordings -- especially relevant because
these are historic recordings with variable noise conditions.

---

### f0 velocity and f0 acceleration

**What they are:** The velocity and acceleration of the f0 curve over
time. If f0 is "where the note is right now," velocity is "how fast
it is changing," and acceleration is "how fast that change is
speeding up."

**Why they matter here:** M1-to-M2 transitions in Choro are not
instantaneous -- they happen through rapid ornaments such as glissandi
(smooth slides between notes). A register transition produces a
velocity/acceleration pattern that is very different from a normal
vibrato. The classifier uses this.

---

## Spectral features (for VMI)

The features below feed the **VMI** (Vocal Mechanism Index), a
continuous metric that does not depend on fixed frequencies like G4.

### Alpha Ratio

**What it is:** The ratio between the energy in the low frequencies
(0-1 kHz) and the high frequencies (1-5 kHz), measured in dB.

**How to think about it:** The higher the Alpha Ratio, the more
"brightness" and upper harmonics the voice has. M1 typically has a
higher Alpha Ratio (more energy distributed across upper harmonics),
while M2 concentrates energy on the fundamental.

---

### H1-H2 (Harmonic Difference)

**What it is:** The amplitude difference between the 1st harmonic
(the fundamental frequency) and the 2nd harmonic (twice the
fundamental).

**How to think about it:** Vocal folds that close firmly produce a
"square-like" waveform with many harmonics (low H1-H2). Vocal folds
that close gently produce a more "sinusoidal" waveform with fewer
harmonics (high H1-H2).

**Why it matters here:** M1 tends to have low H1-H2 (firm adduction),
while M2 tends to have high H1-H2 (light adduction).

**Limitation:** When the voice goes very high (f0 > 350 Hz), the
1st harmonic can coincide with the first formant (F1), which distorts
the measurement. That is why we also use Spectral Tilt.

---

### Spectral Tilt

**What it is:** The "slope" of the frequency spectrum. Imagine a
graph with frequency on the X axis and intensity on the Y axis.
Spectral Tilt measures whether that line drops quickly (negative) or
stays flatter (close to zero).

**How to think about it:** A spectrum that drops quickly = a "darker,"
"fuller" voice (M1). A flatter spectrum = a "brighter," "thinner"
voice (M2).

**Advantage:** It is more robust than H1-H2 at high pitches because
it does not depend on specific harmonics.

---

### CPPS (Cepstral Peak Prominence Smoothed)

**What it is:** A measure of how "periodic" and "clean" the vocal
fold vibration is. Technically, it measures the prominence of the peak
in the cepstrum (a transform of the spectrum).

**How to think about it:** High CPPS = well-defined voice, vocal
folds vibrating regularly. Low CPPS = breathy, hoarse, or noisy
voice.

**Important:** High CPPS does not specifically mean M1 or M2. Both a
well-produced M1 and a well-produced M2 (voix mixte, for example)
have high CPPS. Low CPPS indicates breathiness or vocal breaks,
regardless of the mechanism.

**Per-frame variant:** The pipeline can compute CPPS at each frame
(10 ms) instead of a single global value per song, enabling fine
temporal analysis.

---

## VMI -- Vocal Mechanism Index

**What it is:** A continuous metric from 0 to 1 indicating the
"weight" of the vocal mechanism, without depending on fixed
frequencies like G4 (400 Hz).

**How it works:** It combines Alpha Ratio, H1-H2, Spectral Tilt, and
CPPS into a single number. The closer to 0, the more "dense M1"
(full chest voice). The closer to 1, the more "light M2" (falsetto).

**Scale:**

| VMI | Label | Description |
|-----|-------|-------------|
| 0.0-0.2 | M1_DENSE | Full chest voice, firm adduction |
| 0.2-0.4 | M1_LIGHT | Lighter M1, thin edge |
| 0.4-0.6 | MIX_PASSAGGIO | Transition zone, mixed voice |
| 0.6-0.8 | M2_REINFORCED | M2 with glottic adduction, light belting |
| 0.8-1.0 | M2_LIGHT | Falsetto, light mechanism |

**Key advantage:** A high tenor in M1 and a low soprano in M2 can
sing the same note (e.g., A4 = 440 Hz). A fixed G4 threshold would
classify both as M2 because the note is above 400 Hz. VMI can
distinguish them by their spectral characteristics.

---

## How the concepts connect

```
Vocal folds vibrate              ->  generate f0 (fundamental frequency)
Vocal tract shape                ->  determines F1-F4 (formants)
Closure efficiency               ->  determines HNR (sound clarity)
Vibration intensity              ->  determines Energy (RMS)
Cycle-to-cycle stability         ->  determines Jitter and Shimmer
Rapid f0 change                  ->  determines velocity and acceleration

Glottic closure (firm/light)     ->  determines H1-H2 and Alpha Ratio
Spectral distribution            ->  determines Spectral Tilt
Vibration periodicity            ->  determines CPPS
Combination of features above    ->  yields VMI (0 = dense M1, 1 = light M2)
```

The pipeline offers **two classification approaches**:

1. **XGBoost + threshold:** Combines f0, HNR, energy, formants, and
   velocity/acceleration. It works well but uses f0 as the dominant
   feature -- making it sensitive to the singer's tessitura.

2. **VMI (Vocal Mechanism Index):** Combines Alpha Ratio, H1-H2,
   Spectral Tilt, and CPPS. It does not depend on f0 directly --
   it works for any tessitura.

The scatter plot below shows this separation in the f0 x HNR space --
the two clusters found by the GMM correspond to the two mechanisms:

![Mechanism Clusters (GMM)](../../outputs/plots/mechanism_clusters.png)

---

## Recommended reading

To understand the full analysis, read in the following order:

1. This document (glossary)
2. `METHODOLOGY.md` -- sections 1, 3, 4, 6 (pipeline and classification)
3. `outputs/analysis_report.md` -- concrete results
