# Metodologia Computacional - Análise Bioacústica de Mecanismos Laríngeos

**Versão:** 1.0
**Data:** 2025-01-21
**Contexto:** Análise computacional para artigo acadêmico sobre classificação vocal "Fach" no Choro

> **Novo ao tema?** Leia primeiro o [glossário bioacústico](glossario_bioacustico.md) — explica os conceitos e a lógica da análise de forma acessível, sem jargão técnico.

---

## 1. Contexto e Objetivos

Este documento descreve a metodologia computacional implementada para a análise fisiológica de mecanismos laríngeos (M1/M2) em gravações de canto do gênero Choro. O objetivo é fornecer evidências quantitativas que desafiem o sistema tradicional de classificação vocal "Fach", demonstrando através de features bioacústicas explicáveis que cantores utilizam ambos os mecanismos de forma fluida.

### 1.1 Mecanismos Laríngeos

- **M1 (Mecanismo 1)**: Voz de peito/modal. Características: maior massa das pregas vocais, maior energia espectral em harmônicos baixos, HNR tipicamente elevado.
- **M2 (Mecanismo 2)**: Voz de cabeça/falsete. Características: menor massa vibratória, energia concentrada em harmônicos altos, transições de fase características.

### 1.2 Desafio Técnico

Gravações históricas do Choro (décadas de 1940-1960) apresentam:
- Ruído de fundo elevado
- Baixa razão sinal-ruído (SNR)
- Vibrato intenso e ornamentações rápidas (glissandi, portamenti)
- Qualidade espectral degradada

Essas características exigem métodos robustos de extração de pitch e features de qualidade vocal.

---

## 2. Pipeline de Processamento

### 2.1 Pré-processamento de Áudio

**Módulo:** `src/vocal_analysis/preprocessing/audio.py`

```python
load_audio(audio_path, sr=44100, mono=True, normalize=True, target_db=-3.0)
```

#### Parâmetros Críticos

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Sample Rate** | 44.1 kHz | Padrão CD quality, suporta fmax até 22.05 kHz |
| **Mono** | True | Voz humana é fonte pontual, stereo desnecessário |
| **Normalização** | -3 dBFS | Padroniza amplitude entre gravações, evita clipping |

**Implementação da Normalização:**
```python
target_amplitude = 10 ** (target_db / 20)  # -3dB = 0.708 em amplitude linear
audio_normalized = audio * (target_amplitude / max(abs(audio)))
```

### 2.2 Source Separation (HTDemucs) — Opcional

**Módulo:** `src/vocal_analysis/preprocessing/separation.py`

Em arranjos complexos de Choro (violão 7 cordas, cavaquinho, pandeiro, flauta), a detecção de pitch pode captar instrumentos ao invés da voz. Para isolar a voz antes da análise, o pipeline oferece **source separation** via HTDemucs.

#### Por que HTDemucs?

| Aspecto | HTDemucs | Spleeter |
|---------|----------|----------|
| **Arquitetura** | Híbrida (tempo + frequência) com Transformer | U-Net simples |
| **SDR em vocals** | 7-9 dB | 5-6 dB |
| **Qualidade em arranjos densos** | ✅ Excelente | ⚠️ Artefatos em instrumentos sobrepostos |
| **Integração PyTorch** | ✅ Via torchaudio.pipelines | ❌ Requer pacote separado |

**Referência:** Défossez, A., et al. (2021). Hybrid Spectrogram and Waveform Source Separation. *ISMIR*.

#### Implementação

```python
from vocal_analysis.preprocessing.separation import separate_vocals

# Separar voz (com cache automático)
vocals, sr = separate_vocals(
    audio_path,
    device="cuda",           # ou "cpu"
    cache_dir=Path("data/cache/separated")
)
# vocals: np.ndarray mono, sr: 44100 Hz
```

#### Fluxo com Separação

```
Áudio MP3 → HTDemucs (separar voz) → WAV temporário → CREPE + Praat → Features
                ↓
         Cache .npy (evita reprocessamento)
```

#### Validação Visual

Para confirmar que a separação está captando a voz (e não o cavaquinho), o pipeline gera plots comparativos com `--validate-separation`:

- Eixo Y esquerdo: Frequência (Hz)
- Eixo Y direito: Notas musicais (A#4, C5, etc.)
- Coloração: Confiança CREPE (0-1)

Se a melodia após separação for mais contínua e nas notas esperadas para voz feminina (~200-600 Hz), a separação está funcionando.

#### Flags CLI

```bash
--separate-vocals          # Habilitar source separation
--separation-device        # cpu ou cuda (default: mesmo que --device)
--no-separation-cache      # Forçar reprocessamento
--validate-separation      # Gerar plot comparativo Hz + notas
```

### 2.3 Extração de Features Híbrida (Crepe + Praat)

**Módulo:** `src/vocal_analysis/features/extraction.py`

O pipeline combina:
1. **CREPE (CNN)** para extração robusta de f0
2. **Praat/Parselmouth** para features espectrais (gold standard em análise vocal)

---

## 3. Extração de Frequência Fundamental (f0)

### 3.1 Escolha do CREPE

**Método:** Convolutional Neural Network treinada em dados de pitch anotados
**Referência:** Kim et al. (2018) - "CREPE: A Convolutional Representation for Pitch Estimation"

#### Por que CREPE ao invés de Autocorrelação (Praat)?

| Aspecto | CREPE (CNN) | Praat (Autocorrelação) |
|---------|-------------|------------------------|
| **Vibrato intenso** | ✅ Robusto | ⚠️ Pode confundir com subharmônicos |
| **Ruído de fundo** | ✅ Aprende a ignorar | ❌ Degrada picos de autocorrelação |
| **Ornamentações rápidas** | ✅ Alta resolução temporal | ⚠️ Depende de janelamento |
| **Gravações históricas** | ✅ Generaliza para baixa SNR | ❌ Requer SNR > 20dB |

**Implementação:**
```python
f0, confidence = torchcrepe.predict(
    audio_tensor,
    sample_rate=44100,
    hop_length=441,        # 10ms @ 44.1kHz
    fmin=50.0,             # ~G1 (limite inferior voz humana)
    fmax=800.0,            # ~G5 (cobre M1 e M2)
    model='tiny',          # Balanceio velocidade/precisão
    decoder=torchcrepe.decode.viterbi,  # Suaviza curva de pitch
    return_periodicity=True
)
```

### 3.2 Parâmetros Temporais

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **hop_length** | 441 samples (10ms) | Resolução suficiente para ornamentos rápidos, compromisso com custo computacional |
| **Janelamento CREPE** | ~25ms (interno) | Não configurável, otimizado pela arquitetura CNN |
| **Threshold confiança** | 0.8 | Filtragem rigorosa, descarta detecções ambíguas (silêncios, ruído) |

**Nota sobre janelamento:** O CREPE utiliza internamente janelamento próprio (~25ms) que não é configurável pelo usuário. Essa escolha arquitetural foi validada em benchmarks MIR e supera métodos baseados em autocorrelação.

---

## 4. Features de Qualidade Vocal (Praat/Parselmouth)

### 4.1 Harmonicity-to-Noise Ratio (HNR)

**Definição:** Razão entre energia harmônica e energia de ruído (dB)
**Interpretação:**
- HNR > 15 dB → Voz "limpa", fechamento glótico eficiente (típico M1)
- HNR < 10 dB → Soprosidade, ruído aspirativo (típico M2 ou patologias)

**Extração:**
```python
harmonicity = sound.to_harmonicity(time_step=0.01)  # 10ms
hnr_values = harmonicity.values[0]  # Array temporal
```

### 4.2 Cepstral Peak Prominence Smoothed (CPPS)

**Definição:** Proeminência do pico cepstral suavizado, proxy para regularidade de vibração das pregas vocais
**Aplicação:** Diferenciação de fonação modal (M1) vs soprosidade (M2)

**Extração:**
```python
power_cepstrogram = parselmouth.praat.call(sound, "To PowerCepstrogram", fmin, time_step, 5000, 50)
cpps = parselmouth.praat.call(power_cepstrogram, "Get CPPS", ...)
```

**Limitação:** Gravações com ruído de fundo elevado comprometem a medida. Fallback para HNR médio quando extração falha.

### 4.3 Jitter (ppq5) e Shimmer (apq11)

**Jitter (Period Perturbation Quotient):**
Mede instabilidade da frequência de vibração glótica entre 5 períodos consecutivos.

**Shimmer (Amplitude Perturbation Quotient):**
Mede variação de amplitude entre 11 períodos consecutivos.

**Interpretação:**
- Valores baixos (jitter < 1%, shimmer < 3%) → Fonação estável
- Valores altos → Vibrato, patologia, ou transição de registro

**Extração:**
```python
point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", fmin, fmax)
jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
shimmer_apq11 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
```

### 4.4 Energia Espectral (RMS)

**Definição:** Root Mean Square da amplitude, proxy para intensidade vocal.

**Aplicação:** Feature essencial para o classificador XGBoost (M1 tipicamente mais energético que M2).

**Extração:**
```python
energy = librosa.feature.rms(y=audio, frame_length=int(0.025 * sr), hop_length=441)[0]
```

### 4.5 Formantes F1-F4

**Definição:** Ressonâncias do trato vocal extraídas via Linear Predictive Coding (método de Burg).

**Aplicação:**
- Detectar aproximação de formantes ("zona de fala")
- Diferenciar timbres de M1 vs M2
- Identificar estratégias de ressonância

**Extração:**
```python
formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
f1 = formants.get_value_at_time(1, time)
f2 = formants.get_value_at_time(2, time)
# ... F3, F4
```

### 4.6 Features Espectrais para VMI

**Módulo:** `src/vocal_analysis/features/spectral.py`

Estas features são usadas para calcular o VMI (Vocal Mechanism Index), permitindo classificação de mecanismo independente da tessitura.

#### Alpha Ratio

**Definição:** Razão de energia espectral entre banda baixa (0-1kHz) e banda alta (1-5kHz), em dB.

**Interpretação:**
- Alpha Ratio alta → mais energia em harmônicos superiores → típico de M1
- Alpha Ratio baixa → energia concentrada em graves → típico de M2/falsete

**Extração:**
```python
from vocal_analysis.features.spectral import compute_alpha_ratio
alpha_ratio = compute_alpha_ratio(audio, sr, hop_length=220)
```

#### H1-H2 (Diferença de Harmônicos)

**Definição:** Diferença de amplitude (dB) entre o 1º harmônico (H1 = f0) e o 2º harmônico (H2 = 2×f0).

**Interpretação:**
- H1-H2 baixo → adução firme, inclinação glotal íngreme → M1
- H1-H2 alto → adução leve, inclinação glotal suave → M2

**Limitação:** Quando f0 > 350Hz, H1 pode coincidir com F1, tornando a medida instável. Por isso, usamos Spectral Tilt como complemento.

**Extração:**
```python
from vocal_analysis.features.spectral import compute_h1_h2
h1_h2 = compute_h1_h2(audio, sr, f0, hop_length=220)
```

#### Spectral Tilt (Inclinação Espectral)

**Definição:** Inclinação da regressão linear no espectro de potência (log-frequência vs amplitude em dB).

**Interpretação:**
- Tilt negativo (steep) → espectro decai rapidamente → M1
- Tilt próximo de zero (flat) → espectro mais plano → M2

**Vantagem:** Mais robusto que H1-H2 em registros agudos.

**Extração:**
```python
from vocal_analysis.features.spectral import compute_spectral_tilt
spectral_tilt = compute_spectral_tilt(audio, sr, hop_length=220)
```

#### CPPS per-frame

**Definição:** Proeminência do pico cepstral suavizada, calculada por frame (não global).

**Interpretação:**
- CPPS alto → voz periódica, limpa
- CPPS baixo → ruído, aperiodicidade

**Nota:** CPPS alto ocorre tanto em M1 denso quanto em M2 reforçado (voix mixte). CPPS baixo indica soprosidade ou quebra de voz.

**Extração:**
```python
from vocal_analysis.features.spectral import compute_cpps_per_frame
cpps = compute_cpps_per_frame(audio_path, hop_length=220)
```

---

## 5. Features de Agilidade Articulatória

**Módulo:** `src/vocal_analysis/features/articulation.py`

### 5.1 Velocidade de Mudança de Pitch (f0 velocity)

**Definição:** Taxa de mudança da frequência fundamental (Hz/s)

```python
f0_velocity = Δf0 / Δt
```

**Aplicação:** Quantificar ornamentações rápidas (glissandi, portamenti) características do Choro.

### 5.2 Aceleração de Pitch (f0 acceleration)

**Definição:** Taxa de mudança da velocidade de pitch (Hz/s²)

```python
f0_acceleration = Δ(f0_velocity) / Δt
```

**Aplicação:** Detectar transições abruptas de registro (quebras M1→M2).

### 5.3 Taxa Silábica

**Definição:** Estimativa de sílabas por segundo via detecção de picos de energia.

**Método:**
1. Encontrar picos locais no sinal de energia RMS
2. Aplicar distância mínima de 100ms entre picos (evitar dupla contagem)
3. Normalizar pelo tempo total

```python
from scipy.signal import find_peaks
peaks, _ = find_peaks(energy, distance=int(0.1 / time_step))
syllable_rate = len(peaks) / duration
```

**Aplicação:** Proxy para agilidade técnica do cantor.

---

## 6. Classificação de Mecanismos M1/M2

### 6.1 Abordagem Híbrida

O pipeline implementa **4 métodos complementares**:

#### Método 1: Threshold Heurístico
```python
mechanism = "M1" if f0 < 400 Hz else "M2"
```
**Justificativa:** 400 Hz (~G4) é limiar empírico de passaggio para vozes femininas.
**Limitação:** Ignora covariância com HNR/energia.

#### Método 2: Gaussian Mixture Model (GMM)
```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

features = df_voiced[["f0", "hnr"]].values
scaler = RobustScaler()  # Usa mediana/IQR, robusto a outliers
features_norm = scaler.fit_transform(features)

gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(features_norm)
```
**Vantagem:** Não-supervisionado, descobre clusters naturais.
**Normalização:** RobustScaler (mediana/IQR) em vez de StandardScaler (mean/std), mais robusto a outliers em gravações históricas.

![Clusters de Mecanismo (GMM)](../outputs/plots/mechanism_clusters.png)

#### Método 3: XGBoost (Supervisionado com Pseudo-Labels)
```python
import xgboost as xgb
# Usar labels do GMM como pseudo-labels
# Features base + formantes se disponíveis no CSV
X = (f0, HNR, energy, f0_velocity, f0_acceleration, f1, f2, f3, f4)
y = gmm_labels  # 0=M1, 1=M2
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X, y)
# Predição aplicada sobre todos os frames voiced
predictions = model.predict(X)
```

**Vantagem:** Aprende interações não-lineares entre features.
**Aplicação:** Classificação robusta para novos dados. Classification report salvo no relatório `outputs/analise_ademilde.md`.

![Predição XGBoost: M1 vs M2 ao longo do tempo](../outputs/plots/xgb_mechanism_timeline.png)

#### Método 4: VMI (Vocal Mechanism Index) — Agnóstico à Tessitura

**Módulo:** `src/vocal_analysis/features/vmi.py`

O VMI é uma métrica contínua (0-1) que substitui o threshold arbitrário de G4 por análise baseada em **features espectrais**:

```python
from vocal_analysis.features.vmi import compute_vmi_fixed, vmi_to_label

vmi = compute_vmi_fixed(
    alpha_ratio=df["alpha_ratio"].values,
    cpps=df["cpps_per_frame"].values,
    h1_h2=df["h1_h2"].values,
    spectral_tilt=df["spectral_tilt"].values,
)
labels = vmi_to_label(vmi)  # M1_DENSO, M1_LIGEIRO, MIX_PASSAGGIO, M2_REFORCADO, M2_LIGEIRO
```

**Escala VMI:**

| VMI | Label | Descrição |
|-----|-------|-----------|
| 0.0-0.2 | `M1_DENSO` | Mecanismo pesado, adução firme, voz de peito plena |
| 0.2-0.4 | `M1_LIGEIRO` | M1 de borda fina, comum em tenores/registro médio |
| 0.4-0.6 | `MIX_PASSAGGIO` | Zona de passagem, instabilidade acústica, voz mista |
| 0.6-0.8 | `M2_REFORCADO` | M2 com adução glótica, ressonância frontal |
| 0.8-1.0 | `M2_LIGEIRO` | Mecanismo leve, falsete, piano M2 |

**Vantagem:** Não depende de frequências fixas como G4 — funciona para qualquer tessitura.
**Aplicação:** Identificação gradual do mecanismo vocal e do passaggio.

### 6.2 Features Utilizadas no XGBoost

| Feature | Tipo | Importância Esperada | Justificativa |
|---------|------|---------------------|---------------|
| **f0** | Base | Alta | Separação primária (M1 grave, M2 agudo) |
| **HNR** | Base | Média | M1 > M2 em fonação modal |
| **energy** | Base | Média-Alta | M1 mais energético que M2 |
| **f0_velocity** | Derivada | Média-Alta | Transições M1→M2 são ornamentos rápidos (glissandi) |
| **f0_acceleration** | Derivada | Média | Quebras abruptas de registro indicam mudança de mecanismo |
| **f1, f2, f3, f4** | Opcional | Alta | Ressonâncias do trato vocal diferenciam diretamente M1 vs M2 |

**Nota:** F1-F4 são incluídas automaticamente se disponíveis no CSV (processamento sem `--skip-formants`). `cpps_global`, `jitter` e `shimmer` são valores escalares por música (não por frame) e portanto não geram variação útil para classificação por frame.

---

## 7. Visualizações e Relatórios

### 7.1 Plots Acadêmicos

**Módulo:** `src/vocal_analysis/visualization/plots.py`

- **Contorno de f0 temporal:** Identificação visual de transições M1↔M2
- **Scatter f0 vs HNR:** Separação de clusters por mecanismo
- **Histogramas de distribuição:** Evidência de bimodalidade
- **Timeline colorido:** Mapeamento temporal de mecanismos
- **Excerpts por música:** Trechos de 5s na janela mais densa, com eixo de notas musicais — para inspeção manual (eval humano)

**Estética:** Seaborn (`whitegrid`), paleta `viridis`, DPI 150 para publicação.

Excerpts gerados automaticamente pela janela de maior densidade de frames por música:

![Apanhei-te Cavaquinho — excerpt](../outputs/plots/excerpt_Apanhei-te_Cavaquinho.png)

![delicado — excerpt](../outputs/plots/excerpt_delicado.png)

![brasileirinho — excerpt](../outputs/plots/excerpt_brasileirinho.png)

### 7.2 Relatório Narrativo com IA

**Módulo:** `src/vocal_analysis/analysis/llm_report.py`

Utiliza **Gemini Multimodal** (Google) para:
1. Analisar plots gerados
2. Interpretar estatísticas descritivas
3. Gerar narrativa acadêmica contextualizada

**Input:** Imagens de plots + JSON com estatísticas
**Output:** Relatório markdown com insights qualitativos

---

## 8. Conformidade com Rigor Acadêmico

### 8.1 Reprodutibilidade

| Aspecto | Garantia |
|---------|----------|
| **Seed aleatória** | `random_state=42` em todos os modelos |
| **Versões fixadas** | `pyproject.toml` com dependencies travadas |
| **Parâmetros documentados** | Valores justificados em comentários inline |

### 8.2 Validação

- **Inspeção auditiva manual:** Verificar classificação M1/M2 em trechos ambíguos
- **Cross-validation:** K-fold (k=5) no XGBoost para estimar generalização
- **Ablation study:** Avaliar importância individual de cada feature

### 8.3 Limitações Reconhecidas

1. **Gravações históricas:** Ruído de fundo limita precisão do CPPS
2. **Threshold M1/M2:** 400 Hz é heurística, pode variar entre indivíduos
3. **Pseudo-labels:** GMM não garante 100% de acurácia para treinar XGBoost
4. **Ausência de ground truth:** Sem validação por análise laringoscópica

---

## 9. Estrutura de Dados

### 9.1 DataFrame de Features

**Arquivo:** `data/processed/ademilde_features.csv` (gerado pelo `process_ademilde`)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `time` | float | Timestamp em segundos |
| `f0` | float | Frequência fundamental (Hz) |
| `confidence` | float | Confiança da detecção CREPE (0-1) |
| `hnr` | float | Harmonic-to-Noise Ratio (dB) |
| `energy` | float | Energia RMS |
| `f1, f2, f3, f4` | float | Formantes 1-4 (Hz) |
| `cpps_global` | float | CPPS (valor global por música) |
| `jitter` | float | Jitter ppq5 (%) - valor global por música |
| `shimmer` | float | Shimmer apq11 (%) - valor global por música |
| `song` | string | Nome da música |
| `alpha_ratio` | float | Razão de energia 0-1kHz vs 1-5kHz (dB) — se `--extract-spectral` |
| `h1_h2` | float | Diferença H1-H2 (dB) — se `--extract-spectral` |
| `spectral_tilt` | float | Inclinação espectral — se `--extract-spectral` |
| `cpps_per_frame` | float | CPPS por frame — se `--cpps-per-frame` |

**Features derivadas** (calculadas pelo `run_analysis`, não presentes no CSV acima):

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `f0_velocity` | float | Velocidade de mudança de pitch (Hz/s) |
| `f0_acceleration` | float | Aceleração de pitch (Hz/s²) |
| `syllable_rate` | float | Taxa silábica (sílabas/s) |
| `mechanism` | string | Cluster do GMM (M1/M2) |
| `xgb_mechanism` | string | Predição do XGBoost (M1/M2) |
| `vmi` | float | Vocal Mechanism Index (0-1) — se VMI habilitado |
| `vmi_label` | string | Label VMI (M1_DENSO, M1_LIGEIRO, MIX_PASSAGGIO, M2_REFORCADO, M2_LIGEIRO) |

**Arquivo de predições:** `outputs/xgb_predictions.csv` (gerado pelo `run_analysis`, contém todas as colunas acima)

### 9.2 Metadata JSON

**Arquivo:** `data/processed/ademilde_metadata.json`

```json
{
  "artist": "Ademilde Fonseca",
  "processed_at": "2025-01-21T10:30:00",
  "n_songs": 5,
  "global": {
    "f0_mean_hz": 285.3,
    "f0_range_notes": "C4-G5",
    "hnr_mean_db": 12.4
  },
  "songs": [...]
}
```

---

## 10. Workflow de Execução

### Passo 1: Processamento de Áudio

```bash
# Processamento padrão (sem separação)
uv run python -m vocal_analysis.preprocessing.process_ademilde

# COM source separation (recomendado para arranjos complexos)
uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals --device cuda

# COM validação visual (gera plots Hz + notas para conferir separação)
uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals --validate-separation --limit 1

# COM features espectrais para VMI (recomendado)
uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals --extract-spectral --device cuda

# COM CPPS per-frame (mais lento, melhor precisão VMI)
uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals --extract-spectral --cpps-per-frame --device cuda
```

**Output:**
- `data/processed/ademilde_features.csv`
- `data/processed/ademilde_metadata.json`
- `outputs/plots/{song}_f0.png` (um por música)
- `data/cache/separated/{song}_vocals.npy` (se `--separate-vocals`)
- `outputs/plots/{song}_separation_validation.png` (se `--validate-separation`)

### Passo 2: Análise Exploratória + Classificação

```bash
uv run python -m vocal_analysis.analysis.run_analysis
```

**Output:**
- `outputs/plots/mechanism_analysis.png` (threshold)
- `outputs/plots/mechanism_clusters.png` (GMM)
- `outputs/plots/xgb_mechanism_timeline.png` (contorno temporal pela predição XGBoost)
- `outputs/plots/excerpt_{song}.png` (trechos de 5s por música, nota a nota, para eval humano)
- `outputs/xgb_predictions.csv` (predições por frame: GMM + XGBoost)
- `outputs/analise_ademilde.md` (relatório básico, inclui classification report do XGBoost)
- `outputs/relatorio_llm.md` (relatório narrativo, requer `GEMINI_API_KEY`)

---

## 11. Referências Metodológicas

1. **Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018).** CREPE: A Convolutional Representation for Pitch Estimation. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

2. **Boersma, P., & Weenink, D. (2023).** Praat: doing phonetics by computer [Computer program]. Version 6.3.

3. **Henrich, N., et al. (2014).** On the use of electroglottography for characterization of the laryngeal mechanisms. *Proceedings of Stockholm Music Acoustics Conference*.

4. **Maryn, Y., & Weenink, D. (2015).** Objective dysphonia measures in the program Praat: Smoothed cepstral peak prominence and acoustic voice quality index. *Journal of Voice*.

5. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

6. **Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2021).** Hybrid Spectrogram and Waveform Source Separation. *Proceedings of the ISMIR 2021 Conference*.

---

## 12. Contato e Contribuições

**Autor:** ML Engineer & Músico Profissional
**Projeto:** Análise Computacional - Voz no Choro
**Stack:** Python 3.11+, torchcrepe, parselmouth, xgboost

Para dúvidas metodológicas ou sugestões de melhorias, abra uma issue no repositório.

---

**Última Atualização:** 2026-02-05
**Status:** Pipeline validado. Inclui source separation opcional (HTDemucs) para arranjos complexos.
