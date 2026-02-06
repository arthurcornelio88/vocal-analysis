# Análise VMI (Vocal Mechanism Index) - Ademilde Fonseca

## Metodologia

Este relatório utiliza o **VMI (Vocal Mechanism Index)**, uma métrica contínua que
substitui o threshold arbitrário de G4 (400 Hz) por análise baseada em features espectrais:

- **Alpha Ratio**: Razão de energia 0-1kHz vs 1-5kHz
- **H1-H2**: Diferença entre 1º e 2º harmônico (inclinação glotal)
- **Spectral Tilt**: Inclinação do espectro de potência
- **CPPS**: Proeminência do pico cepstral (periodicidade)

O VMI varia de **0.0 (M1 Denso)** a **1.0 (M2 Ligeiro)**, permitindo identificação
gradual do mecanismo vocal sem depender de frequências fixas.

---

## Resumo Global

| Métrica | Valor | Nota |
|---------|-------|------|
| **F0 médio** | 395.3 Hz | G4 |
| **F0 mínimo** | 179.3 Hz | F3 |
| **F0 máximo** | 781.2 Hz | G5 |
| **Extensão** | F3 – G5 | ~2.1 oitavas |
| **VMI médio** | 0.428 | – |
| **Total frames** | 20266 | – |

---

## Análise por Categoria VMI

| Categoria | Frames | % | VMI médio | F0 médio | Alpha Ratio | H1-H2 |
|-----------|--------|---|-----------|----------|-------------|-------|
| **M1_DENSO** | 995 | 4.9% | 0.154 | 372.3 Hz | 1.8 dB | -20.9 dB |
| **M1_LIGEIRO** | 7828 | 38.6% | 0.313 | 378.7 Hz | -5.0 dB | -3.9 dB |
| **MIX_PASSAGGIO** | 8977 | 44.3% | 0.492 | 407.8 Hz | -13.0 dB | 12.2 dB |
| **M2_REFORCADO** | 2331 | 11.5% | 0.664 | 413.6 Hz | -19.9 dB | 29.5 dB |
| **M2_LIGEIRO** | 135 | 0.7% | 0.851 | 389.8 Hz | -33.3 dB | 46.4 dB |

### Interpretação das Categorias

- **M1_DENSO (VMI 0.0-0.2)**: Mecanismo pesado, adução firme, voz de peito plena
- **M1_LIGEIRO (VMI 0.2-0.4)**: M1 de borda fina, comum em tenores/registro médio
- **MIX_PASSAGGIO (VMI 0.4-0.6)**: Zona de passagem, instabilidade acústica, voz mista
- **M2_REFORCADO (VMI 0.6-0.8)**: M2 com adução glótica, ressonância frontal
- **M2_LIGEIRO (VMI 0.8-1.0)**: Mecanismo leve, falsete, piano M2

---

## Análise por Música

### apanheite_cavaquinho

- F0 médio: 370.1 Hz (F#4)
- VMI médio: 0.431
- Distribuição: {'MIX_PASSAGGIO': 48.54784954769084, 'M1_LIGEIRO': 37.97809871448976, 'M2_REFORCADO': 10.03015394381844, 'M1_DENSO': 3.3804158070147596, 'M2_LIGEIRO': 0.06348198698619266}

### delicado

- F0 médio: 382.9 Hz (G4)
- VMI médio: 0.413
- Distribuição: {'M1_LIGEIRO': 44.033593841129125, 'MIX_PASSAGGIO': 36.94156071386912, 'M2_REFORCADO': 11.43123760643882, 'M1_DENSO': 6.275516155371514, 'M2_LIGEIRO': 1.318091683191415}

### brasileirinho

- F0 médio: 444.6 Hz (A4)
- VMI médio: 0.449
- Distribuição: {'MIX_PASSAGGIO': 51.02002967359051, 'M1_LIGEIRO': 30.78635014836795, 'M2_REFORCADO': 13.334569732937684, 'M1_DENSO': 4.525222551928784, 'M2_LIGEIRO': 0.33382789317507416}

---

## Vantagens do VMI

1. **Agnóstico à tessitura**: Não depende de frequências fixas como G4
2. **Contínuo**: Captura gradações entre mecanismos (passaggio)
3. **Multi-dimensional**: Combina múltiplas features espectrais
4. **Interpretável**: Cada feature tem significado fisiológico claro

## Limitações

1. **Pesos fixos**: Versão atual usa pesos default, não treinados
2. **CPPS global**: Ideal seria CPPS per-frame (mais lento)
3. **H1-H2 instável**: Pode ser menos preciso para F0 > 350Hz

## Próximos Passos

1. Treinar pesos VMI via XGBoost com pseudo-labels GMM
2. Validar com anotações manuais em trechos conhecidos
3. Adicionar regularização temporal para estabilidade
