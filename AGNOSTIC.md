# Vocal Mechanism Index (VMI) - Pipeline Agnóstico de Classificação Vocal

Refatorar o pipeline atual de análise vocal para substituir o threshold arbitrário de G4 por uma abordagem baseada no **Espectro Absoluto** e na relação fonte-filtro, permitindo a identificação da "voz mista" e do "passaggio" de forma independente da tessitura do cantor.

---

## 1. Auditoria e Extração de Features

### Features Existentes (manter)
- **F0 (CREPE):** Extração robusta de pitch, especialmente em registros de passagem.
- **HNR:** Razão harmônico-ruído via Praat.
- **Formantes (F1-F4):** Extração LPC-Burg para análise de ressonância.
- **Energy (RMS):** Envelope de amplitude via librosa.

### Novas Features a Implementar

| Feature | Descrição | Implementação |
|---------|-----------|---------------|
| **Alpha Ratio** | Razão de energia 0-1kHz vs 1-5kHz. Indica densidade espectral - valores altos sugerem mais energia em harmônicos superiores (M1). | FFT por frame, calcular energia em bandas |
| **H1-H2** | Diferença de amplitude entre 1º e 2º harmônico. Indica inclinação glotal - valores baixos = adução firme (M1), valores altos = adução leve (M2). | Peak picking nos primeiros 2 harmônicos baseado em F0 |
| **Spectral Tilt** | Inclinação da regressão linear no espectro de potência. Mais robusto que H1-H2 em registros agudos (F0 > 350Hz) onde H1 pode coincidir com F1. | Regressão linear log-freq vs amplitude |
| **CPPS per-frame** | Proeminência do pico cepstral suavizada, calculada por frame. Indica periodicidade/limpeza vocal. | Praat PowerCepstrogram com sliding window |
| **F0-F1 Distance** | Distância em semitons entre F0 e F1. Indica estratégia de ressonância (vowel tuning), não mecanismo. Útil como feature auxiliar. | Derivado de F0 e formants existentes |

---

## 2. Lógica de Classificação: Vocal Mechanism Index (VMI)

### Princípios
- **Remoção de Limites Fixos:** Eliminar qualquer lógica baseada em frequências fixas (ex: G4 = 400Hz).
- **Output Contínuo:** VMI entre 0.0 e 1.0, onde a transição é uma curva probabilística, não um salto binário.
- **Pesos Treináveis:** Usar XGBoost regressor para aprender pesos ótimos das features.
- **Normalização Global:** Z-score global para permitir comparação entre cantores.

### Lógica Corrigida de Classificação

| Configuração | Alpha Ratio | CPPS | H1-H2 | Spectral Tilt | VMI |
|--------------|-------------|------|-------|---------------|-----|
| **M1 Denso** | Alta | Alto | Baixo | Íngreme (negativo) | 0.0 - 0.2 |
| **M1 Ligeiro** | Moderada | Alto | Moderado | Moderado | 0.2 - 0.4 |
| **Passaggio/Mix** | Variável | Variável | Instável | Transição | 0.4 - 0.6 |
| **M2 Reforçado** | Moderada | Alto | Alto | Suave | 0.6 - 0.8 |
| **M2 Ligeiro** | Baixa | Moderado | Muito alto | Muito suave | 0.8 - 1.0 |

### Notas Teóricas Importantes

1. **CPPS Alto em M2 Reforçado:** Um M2 bem produzido (voix mixte, belting leve) tem CPPS **alto** porque é periódico e limpo. CPPS baixo indica ruído/aperiodicidade, não ressonância reforçada.

2. **H1-H2 instável no passaggio:** Quando F0 > 350Hz, H1 pode coincidir com F1, tornando H1-H2 menos confiável. Por isso incluímos Spectral Tilt como feature complementar.

3. **F0-F1 não indica mecanismo:** A proximidade F0↔F1 é estratégia de ressonância (vowel tuning). Sopranos em M2 e tenores em M1 podem usar a mesma estratégia. Não usar diretamente no VMI.

---

## 3. Mapeamento de Terminologia (Output)

O sistema classifica segmentos de áudio utilizando a escala VMI:

| VMI | Label | Descrição |
|-----|-------|-----------|
| 0.0 - 0.2 | `M1_DENSO` | Mecanismo pesado, adução firme, voz de peito plena |
| 0.2 - 0.4 | `M1_LIGEIRO` | M1 de borda fina, comum em tenores ou registros médios |
| 0.4 - 0.6 | `MIX_PASSAGGIO` | Zona de passagem, instabilidade acústica, ajuste de voz mista |
| 0.6 - 0.8 | `M2_REFORCADO` | M2 com adução glótica, ressonância frontal, "voix mixte" |
| 0.8 - 1.0 | `M2_LIGEIRO` | Mecanismo leve, falsete, piano M2 |

---

## 4. Visualização e Validação

### Plot Principal: F0 vs Alpha Ratio
- **Eixo X:** F0 (Hz) com escala de notas musicais
- **Eixo Y:** Alpha Ratio (dB ou razão linear)
- **Cor:** VMI (colormap divergente: azul=M1, branco=mix, vermelho=M2)
- **Objetivo:** Identificar visualmente o "Turning Point" onde a configuração laríngea muda

### Plots Secundários
- Timeline VMI por música (similar ao plot de mecanismo atual)
- Histograma de distribuição VMI
- Correlação entre features espectrais

---

## 5. Arquitetura de Implementação

### Novos Arquivos
```
src/vocal_analysis/features/
├── spectral.py      # Alpha Ratio, H1-H2, Spectral Tilt
└── vmi.py           # Cálculo e treinamento do VMI
```

### Arquivos a Modificar
```
src/vocal_analysis/
├── features/extraction.py      # Integrar novas features
├── analysis/exploratory.py     # Substituir threshold por VMI
├── analysis/run_analysis.py    # Pipeline VMI
└── visualization/plots.py      # plot_vmi_scatter()
```

---

## 6. Dependências Técnicas

- **parselmouth:** CPPS per-frame via PowerCepstrogram
- **scipy.fft:** Cálculo de Alpha Ratio e análise espectral
- **numpy:** Operações numéricas, regressão linear para Spectral Tilt
- **xgboost:** Treinamento de pesos VMI (já existente no projeto)

---

## 7. Estratégia de Treinamento do VMI

Como os pesos são treináveis via ML, precisamos de ground truth:

1. **Pseudo-labels GMM:** Usar clusters do GMM atual como ponto de partida
2. **Consistência temporal:** Regularização para que VMI não oscile frame-a-frame
3. **Validação manual:** Selecionar trechos conhecidos (ex: passaggio claro) para validação

---

## 8. Critérios de Sucesso

1. **Correlação negativa VMI↔F0:** Em média, notas agudas devem ter VMI mais alto
2. **Clusters distintos:** Plot F0 vs Alpha Ratio deve mostrar separação visual
3. **Estabilidade temporal:** VMI não deve oscilar rapidamente em regiões estáveis
4. **Sensibilidade ao passaggio:** Zona 0.4-0.6 deve coincidir com regiões de transição conhecidas
