# TODO - Conformidade com Metodologia do Artigo

**Data**: 2025-01-21
**Status**: Análise completa da implementação vs metodologia descrita no artigo

---

## ✅ O QUE JÁ ESTÁ IMPLEMENTADO

- [x] **CREPE para f0**: Extração via torchcrepe com modelo CNN (extraction.py:54)
- [x] **Parselmouth/Praat para HNR**: Harmonicity temporal (extraction.py:74)
- [x] **CPPS**: Cepstral Peak Prominence Smoothed via Praat (extraction.py:80-83)
- [x] **Mono 44.1kHz**: Conversão automática no load (audio.py:24)
- [x] **XGBoost**: Estrutura de classificador pronta (classifier.py)
- [x] **Clustering GMM**: Classificação não-supervisionada M1/M2 (exploratory.py:95)
- [x] **Plots acadêmicos**: Visualizações com seaborn (plots.py, exploratory.py)
- [x] **Relatório LLM**: Geração narrativa com Gemini multimodal (llm_report.py)

---

## ⚠️ DIVERGÊNCIAS CRÍTICAS (Implementado diferente da metodologia)

### 1. Normalização de Áudio
- **Metodologia**: `-3 dBFS de pico`
- **Implementado**: Função existe em `audio.py:28` mas **NÃO É CHAMADA**
- **Impacto**: Afeta energia espectral e comparações entre músicas
- **Correção**: Adicionar `normalize_audio()` no pipeline de `load_audio()`

### 2. Hop Length
- **Metodologia**: `10 ms` (441 samples @ 44.1kHz)
- **Implementado**: `20 ms` (882 samples @ 44.1kHz) - `extraction.py:26`
- **Impacto**: Resolução temporal 2x menor (menos frames, menos precisão em ornamentos rápidos)
- **Correção**: Mudar `hop_length: int = 882` para `hop_length: int = 441`

### 3. Threshold de Confiança (f0)
- **Metodologia**: `> 0.8`
- **Implementado**: `> 0.5` - `exploratory.py:45` e `process_ademilde.py:47`
- **Impacto**: Pode incluir detecções de pitch menos confiáveis
- **Correção**: Mudar `df["confidence"] > 0.5` para `df["confidence"] > 0.8` em todos os arquivos

### 4. Janelamento Hanning 25ms
- **Metodologia**: Explícito "janelamento de Hanning com frames de 25 ms"
- **Implementado**: CREPE faz internamente (não configurável pelo usuário)
- **Impacto**: Baixo (CREPE usa janelamento próprio otimizado)
- **Correção**: Documentar no código que CREPE usa seu próprio janelamento

---

## ❌ FEATURES NÃO IMPLEMENTADAS (Citadas na metodologia)

### 1. Jitter (ppq5)
- **Descrição**: Period Perturbation Quotient (5 períodos) - mede instabilidade da vibração das pregas
- **Uso**: Quantificar estabilidade glótica em M1 vs M2
- **Implementação necessária**:
```python
# Em extraction.py, adicionar à função extract_bioacoustic_features()
point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", fmin, fmax)
jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
```
- **Arquivo**: `src/vocal_analysis/features/extraction.py`

### 2. Shimmer (apq11)
- **Descrição**: Amplitude Perturbation Quotient (11 períodos) - mede variação de amplitude
- **Uso**: Quantificar regularidade de amplitude em M1 vs M2
- **Implementação necessária**:
```python
shimmer_apq11 = parselmouth.praat.call(point_process, "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
```
- **Arquivo**: `src/vocal_analysis/features/extraction.py`

### 3. Formantes F1-F4 via LPC
- **Descrição**: Primeiros 4 formantes via Linear Predictive Coding (método de Burg)
- **Uso**: Detectar aproximação de formantes na "zona de fala"
- **Implementação necessária**:
```python
# Via Praat/Parselmouth
formants = sound.to_formant_burg(time_step=time_step, max_number_of_formants=5, maximum_formant=5500)
f1_values = formants.to_array(formant_number=1)  # Array temporal
f2_values = formants.to_array(formant_number=2)
f3_values = formants.to_array(formant_number=3)
f4_values = formants.to_array(formant_number=4)
```
- **Arquivo**: `src/vocal_analysis/features/extraction.py`
- **Nota**: Arrays temporais precisarão ser sincronizados com f0/HNR

### 4. Energia Espectral (Energy)
- **Descrição**: Energia RMS por frame
- **Uso**: Feature usada no XGBoost (`classifier.py:29` espera coluna 'energy')
- **Implementação necessária**:
```python
# Via librosa ou cálculo manual
import librosa
energy = librosa.feature.rms(y=audio, frame_length=int(0.025*sr), hop_length=hop_length)[0]
```
- **Arquivo**: `src/vocal_analysis/features/extraction.py`
- **CRÍTICO**: Classifier já espera essa feature!

### 5. VAD (Voice Activity Detection) - webrtcvad
- **Metodologia**: "algoritmo de detecção de atividade de voz (VAD) baseado em energia (biblioteca webrtcvad)"
- **Implementado**: Usa threshold de `confidence > 0.5` do CREPE
- **Impacto**: Moderado (CREPE já filtra silêncios razoavelmente bem)
- **Implementação necessária**:
```python
# Pré-processamento adicional (opcional)
import webrtcvad
vad = webrtcvad.Vad(mode=3)  # Modo 3 = mais agressivo
# Processar em frames de 10/20/30ms e filtrar silêncios
```
- **Arquivo**: Novo módulo `src/vocal_analysis/preprocessing/vad.py`
- **Prioridade**: BAIXA (funcionalidade já existe via confidence)

---

## 🔧 MELHORIAS PARA AGILIDADE ARTICULATÓRIA

**Problema**: CPPS não mede agilidade articulatória (canto rápido).

**Features necessárias** (não citadas na metodologia mas úteis para a análise):

### 1. Taxa de Mudança de Pitch (f0 velocity)
```python
f0_velocity = np.diff(f0) / np.diff(time)  # Hz/s
f0_acceleration = np.diff(f0_velocity) / np.diff(time[1:])
```

### 2. Detecção de Notas e Durações
```python
# Segmentar f0 em notas estáveis vs transições
note_onsets = detect_onsets(f0, threshold_change=20)  # 20 Hz de mudança
note_durations = np.diff(note_onsets) * time_step
mean_note_duration = np.mean(note_durations)
```

### 3. Taxa Silábica (Syllable Rate)
```python
# Proxy: contar picos de energia
from scipy.signal import find_peaks
syllable_peaks = find_peaks(energy, distance=int(0.1/time_step))[0]
syllable_rate = len(syllable_peaks) / total_duration  # sílabas/segundo
```

**Arquivo**: Novo módulo `src/vocal_analysis/features/articulation.py`
**Prioridade**: MÉDIA (ajudaria a justificar aspectos do Choro)

---

## 📊 INTEGRAÇÃO XGBoost (4.2.4 - Classificação M1/M2)

**Status atual**: Código existe mas não está integrado ao pipeline.

### O que falta:
1. ✅ Modelo XGBoost implementado (`classifier.py`)
2. ❌ Extração de `energy` (feature faltando)
3. ❌ Labels de treinamento (rótulos M1/M2)
4. ❌ Integração no pipeline de análise

### Opções de implementação:

**Opção A - Clustering como Pseudo-Labels** (já funciona parcialmente):
```python
# exploratory.py já faz isso com GMM
# Substituir GMM por XGBoost treinado em labels do GMM
labels_gmm = gmm.fit_predict(features)
model_xgb = xgb.XGBClassifier()
model_xgb.fit(features, labels_gmm)
```

**Opção B - Rotulagem Manual** (ideal mas trabalhoso):
- Ouvir trechos e rotular M1/M2 manualmente
- Treinar XGBoost supervisionado

**Opção C - Threshold Híbrido** (mais simples):
```python
# Usar f0 + HNR + CPPS + energy
# M1: f0 < 400 Hz AND (HNR > threshold OR CPPS > threshold)
# M2: f0 >= 400 Hz OR (HNR low AND CPPS low)
```

---

## 🎯 PRIORIDADES DE IMPLEMENTAÇÃO

### ALTA PRIORIDADE (Divergências críticas)
1. [ ] Corrigir hop_length para 441 samples (10ms)
2. [ ] Integrar normalização -3dBFS no pipeline
3. [ ] Mudar threshold confiança para 0.8
4. [ ] **Implementar extração de Energia** (classifier espera!)

### MÉDIA PRIORIDADE (Features citadas na metodologia)
5. [ ] Implementar Jitter (ppq5)
6. [ ] Implementar Shimmer (apq11)
7. [ ] Implementar Formantes F1-F4
8. [ ] Integrar XGBoost no pipeline de classificação

### BAIXA PRIORIDADE (Melhorias/Opcional)
9. [ ] Implementar webrtcvad (confidence CREPE já funciona)
10. [ ] Features de agilidade articulatória (f0 velocity, taxa silábica)
11. [ ] Documentar janelamento CREPE

---

## 📁 ARQUIVOS A MODIFICAR

```
src/vocal_analysis/
├── features/
│   ├── extraction.py         # Adicionar: Jitter, Shimmer, Formantes, Energy
│   └── articulation.py       # NOVO: f0 velocity, taxa silábica
├── preprocessing/
│   ├── audio.py              # Integrar normalize_audio no load_audio
│   └── vad.py                # NOVO (opcional): webrtcvad
├── modeling/
│   └── classifier.py         # Integrar no pipeline (run_analysis.py)
├── analysis/
│   ├── exploratory.py        # Atualizar threshold confiança
│   └── run_analysis.py       # Adicionar etapa XGBoost
└── preprocessing/
    └── process_ademilde.py   # Atualizar threshold confiança, hop_length
```

---

## 🧪 TESTES NECESSÁRIOS

Após implementar as correções:

```bash
# 1. Reprocessar áudios com novos parâmetros
uv run python -m vocal_analysis.preprocessing.process_ademilde

# 2. Verificar que todas features foram extraídas
# CSV deve ter colunas: time, f0, confidence, hnr, cpps_global, jitter, shimmer, f1, f2, f3, f4, energy

# 3. Rodar análise completa
uv run python -m vocal_analysis.analysis.run_analysis

# 4. Validar classificação M1/M2
# Verificar se separação faz sentido perceptualmente
```

---

## 📚 REFERÊNCIAS PENDENTES

- [ ] Henrich et al., 2014 - Para validar premissa CPPS em M1 vs M2
- [ ] Kim et al., 2018 - Referência do CREPE (já citado corretamente)
- [ ] Boersma & Weenink, 2023 - Praat (já citado corretamente)

---

## ⏭️ PRÓXIMOS PASSOS (para amanhã)

1. **Corrigir divergências críticas** (hop_length, normalização, threshold)
2. **Implementar Energy** (urgente - classifier precisa)
3. **Implementar Jitter e Shimmer** (metodologia exige)
4. **Testar pipeline completo** com dados reais
5. **Decidir sobre Formantes** (são necessários?)
6. **Integrar XGBoost** ou manter GMM?

---

**Nota**: O código atual funciona e gera resultados válidos, mas diverge da metodologia escrita. Essas correções garantirão conformidade acadêmica com o artigo.
