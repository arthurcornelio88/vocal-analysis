# Bioacoustic Analysis - Ademilde Fonseca

## Global Summary

| Metric | Value | Note |
|--------|-------|------|
| **Mean f0** | 395.3 Hz | G4 |
| **Min f0** | 179.3 Hz | F3 |
| **Max f0** | 781.2 Hz | G5 |
| **Range** | F3 – G5 | ~2.1 octaves |
| **Mean HNR** | 16.9 dB | – |
| **Total frames** | 20266 | – |

## Mechanism Analysis

### M1 (Chest)

| Metric | Value | Note |
|--------|-------|------|
| **Frames** | 11617 (57.3%) | – |
| **Mean f0** | 315.9 Hz | D#4 |
| **f0 Std Dev** | 48.0 Hz | – |
| **Range** | F3 – G4 | – |
| **Mean HNR** | 16.9 dB | – |

### M2 (Head)

| Metric | Value | Note |
|--------|-------|------|
| **Frames** | 8649 (42.7%) | – |
| **Mean f0** | 502.1 Hz | B4 |
| **f0 Std Dev** | 75.4 Hz | – |
| **Range** | G4 – G5 | – |
| **Mean HNR** | 16.8 dB | – |

## Per Song

### apanheite_cavaquinho

- Mean f0: 370.1 Hz (F#4)
- Range: F#3 – F#5
- Mean HNR: 16.1 dB

### delicado

- Mean f0: 382.9 Hz (G4)
- Range: F3 – F5
- Mean HNR: 17.0 dB

### brasileirinho

- Mean f0: 444.6 Hz (A4)
- Range: G#3 – G5
- Mean HNR: 17.5 dB

## XGBoost Classification (GMM Pseudo-Labels)

Features used: `f0`, `hnr`, `energy`, `f0_velocity`, `f0_acceleration`, `f1`, `f2`, `f3`, `f4`
Training labels: GMM clusters (unsupervised)
Split: 80% train / 20% test

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2492
           1       1.00      1.00      1.00      1562

    accuracy                           1.00      4054
   macro avg       1.00      1.00      1.00      4054
weighted avg       1.00      1.00      1.00      4054
```

## Interpretation

### Bimodal Pattern

The f0 contour shows clear alternation between two regions:
- **Low region (M1)**: Mechanism 1 / chest voice
- **High region (M2)**: Mechanism 2 / head voice

### Implications for "Fach" Classification

The analysis suggests that traditional "voice type" classification does not capture
the physiological reality of laryngeal mechanisms. The singer uses both mechanisms
fluidly, contradicting rigid categorizations.

### Limitations

- Historical recordings with low quality (reduced HNR)
- M1/M2 classification via GMM (sensitive to training data)
- CPPS compromised by background noise

## Next Steps

1. Analyze transitions between mechanisms (register breaks)
2. Compare with contemporary singers (high-quality recordings)
3. Validate VMI with manual annotations from a speech-language pathologist
