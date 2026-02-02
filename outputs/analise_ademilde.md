# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 182.3 Hz | F#3 |
| **f0 mínimo** | 50.0 Hz | G1 |
| **f0 máximo** | 797.6 Hz | G5 |
| **Extensão** | G1 – G5 | ~4.0 oitavas |
| **HNR médio** | 5.3 dB | – |
| **Total frames** | 83636 | – |

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 73471 (87.8%) | – |
| **f0 médio** | 137.4 Hz | C#3 |
| **f0 desvio** | 98.5 Hz | – |
| **Extensão** | G1 – G4 | – |
| **HNR médio** | 4.9 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 10165 (12.2%) | – |
| **f0 médio** | 506.9 Hz | B4 |
| **f0 desvio** | 82.9 Hz | – |
| **Extensão** | G4 – G5 | – |
| **HNR médio** | 8.0 dB | – |

## Por Música

### apanheite_cavaquinho

- f0 médio: 220.4 Hz (A3)
- Extensão: G1 – G5
- HNR médio: 6.2 dB

### brasileirinho

- f0 médio: 183.5 Hz (F#3)
- Extensão: G1 – G5
- HNR médio: 5.3 dB

### delicado

- f0 médio: 159.6 Hz (D#3)
- Extensão: G1 – F5
- HNR médio: 4.7 dB

## Classificação XGBoost (Pseudo-Labels GMM)

Features utilizadas: `f0`, `hnr`, `energy`, `f0_velocity`, `f0_acceleration`, `f1`, `f2`, `f3`, `f4`
Labels de treinamento: clusters do GMM (não-supervisionado)
Split: 80% treino / 20% teste

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     11019
           1       1.00      1.00      1.00      5709

    accuracy                           1.00     16728
   macro avg       1.00      1.00      1.00     16728
weighted avg       1.00      1.00      1.00     16728
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
