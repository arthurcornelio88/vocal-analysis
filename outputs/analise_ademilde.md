# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 396.4 Hz | G4 |
| **f0 mínimo** | 136.0 Hz | C#3 |
| **f0 máximo** | 784.0 Hz | G5 |
| **Extensão** | C#3 – G5 | ~2.5 oitavas |
| **HNR médio** | 16.7 dB | – |
| **Total frames** | 24124 | – |

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 13593 (56.3%) | – |
| **f0 médio** | 314.6 Hz | D#4 |
| **f0 desvio** | 49.5 Hz | – |
| **Extensão** | C#3 – G4 | – |
| **HNR médio** | 16.7 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 10531 (43.7%) | – |
| **f0 médio** | 502.0 Hz | B4 |
| **f0 desvio** | 75.6 Hz | – |
| **Extensão** | G4 – G5 | – |
| **HNR médio** | 16.7 dB | – |

## Por Música

### apanheite_cavaquinho

- f0 médio: 370.1 Hz (F#4)
- Extensão: F#3 – F#5
- HNR médio: 15.8 dB

### brasileirinho

- f0 médio: 444.5 Hz (A4)
- Extensão: C#3 – G5
- HNR médio: 17.4 dB

### delicado

- f0 médio: 384.5 Hz (G4)
- Extensão: F3 – F5
- HNR médio: 17.0 dB

## Classificação XGBoost (Pseudo-Labels GMM)

Features utilizadas: `f0`, `hnr`, `energy`, `f0_velocity`, `f0_acceleration`, `f1`, `f2`, `f3`, `f4`
Labels de treinamento: clusters do GMM (não-supervisionado)
Split: 80% treino / 20% teste

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3094
           1       0.99      0.99      0.99      1731

    accuracy                           1.00      4825
   macro avg       1.00      0.99      1.00      4825
weighted avg       1.00      1.00      1.00      4825
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
- Threshold M1/M2 baseado em heurística (400 Hz)
- CPPS comprometido pelo ruído de fundo

## Próximos Passos

1. Validar threshold com clustering não supervisionado
2. Analisar transições entre mecanismos (quebras de registro)
3. Comparar com cantoras contemporâneas (gravações de alta qualidade)
