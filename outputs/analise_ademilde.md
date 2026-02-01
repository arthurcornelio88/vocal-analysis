# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 219.3 Hz | A3 |
| **f0 mínimo** | 50.0 Hz | G1 |
| **f0 máximo** | 780.9 Hz | G5 |
| **Extensão** | G1 – G5 | ~4.0 oitavas |
| **HNR médio** | -1.9 dB | – |
| **Total frames** | 10896 | – |

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 9195 (84.4%) | – |
| **f0 médio** | 171.2 Hz | F3 |
| **f0 desvio** | 111.3 Hz | – |
| **Extensão** | G1 – G4 | – |
| **HNR médio** | -2.6 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 1701 (15.6%) | – |
| **f0 médio** | 479.1 Hz | A#4 |
| **f0 desvio** | 52.2 Hz | – |
| **Extensão** | G4 – G5 | – |
| **HNR médio** | 1.9 dB | – |

## Por Música

### Apanhei-te Cavaquinho

- f0 médio: 219.3 Hz (A3)
- Extensão: G1 – G5
- HNR médio: -1.9 dB

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
