# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 348.8 Hz | F4 |
| **f0 mínimo** | 50.8 Hz | G#1 |
| **f0 máximo** | 774.6 Hz | G5 |
| **Extensão** | G#1 – G5 | ~3.9 oitavas |
| **HNR médio** | 5.2 dB | – |
| **Total frames** | 20928 | – |

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 15853 (75.8%) | – |
| **f0 médio** | 303.3 Hz | D#4 |
| **f0 desvio** | 58.7 Hz | – |
| **Extensão** | G#1 – G4 | – |
| **HNR médio** | 5.0 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 5075 (24.2%) | – |
| **f0 médio** | 491.1 Hz | B4 |
| **f0 desvio** | 81.1 Hz | – |
| **Extensão** | G4 – G5 | – |
| **HNR médio** | 6.0 dB | – |

## Por Música

### apanheite_cavaquinho

- f0 médio: 350.7 Hz (F4)
- Extensão: A1 – G5
- HNR médio: 6.0 dB

### brasileirinho

- f0 médio: 388.1 Hz (G4)
- Extensão: D2 – G5
- HNR médio: 5.6 dB

### delicado

- f0 médio: 329.8 Hz (E4)
- Extensão: G#1 – F5
- HNR médio: 4.5 dB

## Classificação XGBoost (Pseudo-Labels GMM)

Features utilizadas: `f0`, `hnr`, `energy`, `f0_velocity`, `f0_acceleration`, `f1`, `f2`, `f3`, `f4`
Labels de treinamento: clusters do GMM (não-supervisionado)
Split: 80% treino / 20% teste

```
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      2919
           1       1.00      0.98      0.99      1267

    accuracy                           0.99      4186
   macro avg       0.99      0.99      0.99      4186
weighted avg       0.99      0.99      0.99      4186
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
