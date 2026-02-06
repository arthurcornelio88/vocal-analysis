# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 395.3 Hz | G4 |
| **f0 mínimo** | 179.3 Hz | F3 |
| **f0 máximo** | 781.2 Hz | G5 |
| **Extensão** | F3 – G5 | ~2.1 oitavas |
| **HNR médio** | 16.9 dB | – |
| **Total frames** | 20266 | – |

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 11617 (57.3%) | – |
| **f0 médio** | 315.9 Hz | D#4 |
| **f0 desvio** | 48.0 Hz | – |
| **Extensão** | F3 – G4 | – |
| **HNR médio** | 16.9 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 8649 (42.7%) | – |
| **f0 médio** | 502.1 Hz | B4 |
| **f0 desvio** | 75.4 Hz | – |
| **Extensão** | G4 – G5 | – |
| **HNR médio** | 16.8 dB | – |

## Por Música

### apanheite_cavaquinho

- f0 médio: 370.1 Hz (F#4)
- Extensão: F#3 – F#5
- HNR médio: 16.1 dB

### delicado

- f0 médio: 382.9 Hz (G4)
- Extensão: F3 – F5
- HNR médio: 17.0 dB

### brasileirinho

- f0 médio: 444.6 Hz (A4)
- Extensão: G#3 – G5
- HNR médio: 17.5 dB

## Classificação XGBoost (Pseudo-Labels GMM)

Features utilizadas: `f0`, `hnr`, `energy`, `f0_velocity`, `f0_acceleration`, `f1`, `f2`, `f3`, `f4`
Labels de treinamento: clusters do GMM (não-supervisionado)
Split: 80% treino / 20% teste

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2492
           1       1.00      1.00      1.00      1562

    accuracy                           1.00      4054
   macro avg       1.00      1.00      1.00      4054
weighted avg       1.00      1.00      1.00      4054
```

## Interpretação

### Padrão Bimodal

O contorno de f0 mostra alternância clara entre duas regiões:
- **Região grave (M1)**: Mecanismo 1 / voz de peito
- **Região aguda (M2)**: Mecanismo 2 / voz de cabeça

### Implicações para Classificação "Fach"

A análise sugere que a classificação tradicional de "tipo vocal" não captura
a realidade fisiológica dos mecanismos laríngeos. A cantora utiliza ambos
os mecanismos de forma fluida, contradizendo categorizações rígidas.

### Limitações

- Gravações históricas com baixa qualidade (HNR reduzido)
- Classificação M1/M2 via GMM (sensível a dados de treino)
- CPPS comprometido pelo ruído de fundo

## Próximos Passos

1. Analisar transições entre mecanismos (quebras de registro)
2. Comparar com cantoras contemporâneas (gravações de alta qualidade)
3. Validar VMI com anotações manuais de fonoaudiólogo
