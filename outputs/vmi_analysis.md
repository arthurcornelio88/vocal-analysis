# VMI Analysis (Vocal Mechanism Index) - Ademilde Fonseca

## Methodology

This report uses the **VMI (Vocal Mechanism Index)**, a continuous metric that
replaces the arbitrary G4 threshold (400 Hz) with spectral feature-based analysis:

- **Alpha Ratio**: Energy ratio 0-1kHz vs 1-5kHz
- **H1-H2**: Difference between 1st and 2nd harmonic (glottal slope)
- **Spectral Tilt**: Power spectrum slope
- **CPPS**: Cepstral peak prominence (periodicity)

VMI ranges from **0.0 (Dense M1)** to **1.0 (Light M2)**, enabling gradual
identification of vocal mechanism without relying on fixed frequencies.

---

## Global Summary

| Metric | Value | Note |
|--------|-------|------|
| **Mean F0** | 395.3 Hz | G4 |
| **Min F0** | 179.3 Hz | F3 |
| **Max F0** | 781.2 Hz | G5 |
| **Range** | F3 – G5 | ~2.1 octaves |
| **Mean VMI** | 0.428 | – |
| **Total frames** | 20266 | – |

---

## VMI Category Analysis

| Category | Frames | % | Mean VMI | Mean F0 | Alpha Ratio | H1-H2 |
|----------|--------|---|----------|---------|-------------|-------|
| **M1_HEAVY** | 995 | 4.9% | 0.154 | 372.3 Hz | 1.8 dB | -20.9 dB |
| **M1_LIGHT** | 7828 | 38.6% | 0.313 | 378.7 Hz | -5.0 dB | -3.9 dB |
| **MIX_PASSAGGIO** | 8977 | 44.3% | 0.492 | 407.8 Hz | -13.0 dB | 12.2 dB |
| **M2_REINFORCED** | 2331 | 11.5% | 0.664 | 413.6 Hz | -19.9 dB | 29.5 dB |
| **M2_LIGHT** | 135 | 0.7% | 0.851 | 389.8 Hz | -33.3 dB | 46.4 dB |

### Category Interpretation

- **M1_HEAVY (VMI 0.0-0.2)**: Heavy mechanism, firm adduction, full chest voice
- **M1_LIGHT (VMI 0.2-0.4)**: Thin-edge M1, common in tenors/middle register
- **MIX_PASSAGGIO (VMI 0.4-0.6)**: Passaggio zone, acoustic instability, mixed voice
- **M2_REINFORCED (VMI 0.6-0.8)**: M2 with glottic adduction, frontal resonance
- **M2_LIGHT (VMI 0.8-1.0)**: Light mechanism, falsetto, piano M2

---

## Per Song Analysis

### apanheite_cavaquinho

- Mean F0: 370.1 Hz (F#4)
- Mean VMI: 0.431
- Distribution: {'MIX_PASSAGGIO': 48.54784954769084, 'M1_LIGHT': 37.97809871448976, 'M2_REINFORCED': 10.03015394381844, 'M1_HEAVY': 3.3804158070147596, 'M2_LIGHT': 0.06348198698619266}

### delicado

- Mean F0: 382.9 Hz (G4)
- Mean VMI: 0.413
- Distribution: {'M1_LIGHT': 44.033593841129125, 'MIX_PASSAGGIO': 36.94156071386912, 'M2_REINFORCED': 11.43123760643882, 'M1_HEAVY': 6.275516155371514, 'M2_LIGHT': 1.318091683191415}

### brasileirinho

- Mean F0: 444.6 Hz (A4)
- Mean VMI: 0.449
- Distribution: {'MIX_PASSAGGIO': 51.02002967359051, 'M1_LIGHT': 30.78635014836795, 'M2_REINFORCED': 13.334569732937684, 'M1_HEAVY': 4.525222551928784, 'M2_LIGHT': 0.33382789317507416}

---

## VMI Advantages

1. **Tessitura-agnostic**: Does not depend on fixed frequencies like G4
2. **Continuous**: Captures gradations between mechanisms (passaggio)
3. **Multi-dimensional**: Combines multiple spectral features
4. **Interpretable**: Each feature has clear physiological meaning

## Limitations

1. **Fixed weights**: Current version uses default weights, not trained
2. **Global CPPS**: Ideally would use CPPS per-frame (slower)
3. **Unstable H1-H2**: May be less accurate for F0 > 350Hz

## Next Steps

1. Train VMI weights via XGBoost with GMM pseudo-labels
2. Validate with manual annotations on known passages
3. Add temporal regularization for stability
