# Análise Bioacústica - Ademilde Fonseca

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **f0 médio** | 435.0 Hz | A4 |
| **f0 mínimo** | 101.4 Hz | G#2 |
| **f0 máximo** | 796.4 Hz | G5 |
| **Extensão** | G#2 – G5 | ~3.0 oitavas |
| **HNR médio** | 2.4 dB | – |
| **Total frames** | 65 | – |

## Análise por Mecanismo

### M1 (Peito/Chest)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 32 (49.2%) | – |
| **f0 médio** | 280.8 Hz | C#4 |
| **f0 desvio** | 82.4 Hz | – |
| **Extensão** | G#2 – G4 | – |
| **HNR médio** | 2.8 dB | – |

### M2 (Cabeça/Head)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Frames** | 33 (50.8%) | – |
| **f0 médio** | 584.5 Hz | D5 |
| **f0 desvio** | 113.2 Hz | – |
| **Extensão** | G#4 – G5 | – |
| **HNR médio** | 2.0 dB | – |

## Por Música

### Apanhei-te Cavaquinho

- f0 médio: 459.0 Hz (A#4)
- Extensão: G3 – G5
- HNR médio: 3.8 dB

### delicado

- f0 médio: 412.1 Hz (G#4)
- Extensão: D3 – F5
- HNR médio: 2.4 dB

### brasileirinho

- f0 médio: 432.4 Hz (A4)
- Extensão: G#2 – F#5
- HNR médio: 0.8 dB

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
