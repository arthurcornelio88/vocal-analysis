# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 364.7 Hz | F#4 |
| **f0 mínimo** | 74.0 Hz | D2 |
| **f0 máximo** | 796.5 Hz | G5 |
| **Extensão** | D2 – G5 | ~3.4 oitavas |
| **HNR médio** | 8.0 dB | – |
| **Total frames** | 5143 | – |

![Análise por Mecanismo — distribuição, scatter, HNR e contorno temporal](plots/mechanism_analysis.png)

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 3562 (69.3%) | – |
| **f0 médio** | 311.2 Hz | D#4 |
| **f0 desvio** | 49.1 Hz | – |
| **Extensão** | D2 – G4 | – |
| **HNR médio** | 7.3 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 1581 (30.7%) | – |
| **f0 médio** | 485.2 Hz | B4 |
| **f0 desvio** | 76.5 Hz | – |
| **Extensão** | G4 – G5 | – |
| **HNR médio** | 9.6 dB | – |

## Por Música

### Apanhei-te Cavaquinho

- f0 médio: 362.5 Hz (F#4)
- Extensão: G3 – G5
- HNR médio: 7.6 dB

![Apanhei-te Cavaquinho — contorno f0](plots/apanhei-te Cavaquinho_f0.png)

![Apanhei-te Cavaquinho — excerpt M1/M2](plots/excerpt_Apanhei-te Cavaquinho.png)

### delicado

- f0 médio: 341.9 Hz (F4)
- Extensão: D3 – F5
- HNR médio: 7.8 dB

![delicado — contorno f0](plots/delicado_f0.png)

![delicado — excerpt M1/M2](plots/excerpt_delicado.png)

### brasileirinho

- f0 médio: 415.9 Hz (G#4)
- Extensão: D2 – F#5
- HNR médio: 9.1 dB

![brasileirinho — contorno f0](plots/brasileirinho_f0.png)

![brasileirinho — excerpt M1/M2](plots/excerpt_brasileirinho.png)

![Clusters de Mecanismo (GMM)](plots/mechanism_clusters.png)

## Classificação XGBoost (Pseudo-Labels GMM)

Features utilizadas: `f0`, `hnr`, `energy`, `f0_velocity`, `f0_acceleration`, `f1`, `f2`, `f3`, `f4`
Labels de treinamento: clusters do GMM (não-supervisionado)
Split: 80% treino / 20% teste

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       776
           1       0.99      0.99      0.99       253

    accuracy                           0.99      1029
   macro avg       0.99      0.99      0.99      1029
weighted avg       0.99      0.99      0.99      1029
```

![Predição XGBoost: M1 vs M2 ao longo do tempo](plots/xgb_mechanism_timeline.png)

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
