[English](../../README.md) | **Portugues**

# Vocal Analysis - Analise Bioacustica M1/M2

Analise computacional de mecanismos laringeos (M1/M2) em gravacoes de Choro brasileiro, com foco na voz de Ademilde Fonseca.

## Objetivo

Criticar o sistema de classificacao vocal "Fach" atraves de uma analise fisiologica dos mecanismos laringeos. O pipeline extrai features bioacusticas explicaveis (f0, HNR, CPPS, razoes de energia espectral) e classifica mecanismos vocais com quatro metodos complementares: limiar de frequencia, clustering GMM, XGBoost com pseudo-labels e o **VMI (Vocal Mechanism Index)** — uma metrica continua agnostica de tessitura baseada em features espectrais.

## Stack

- **torchcrepe**: Extracao SOTA de f0 (pitch) via CNN
- **parselmouth** (Praat): HNR, CPPS, Jitter, Shimmer, Formantes
- **numpy/scipy**: Features espectrais (Alpha Ratio, H1-H2, Spectral Tilt)
- **xgboost**: Classificacao tabular M1/M2 com pseudo-labels
- **scikit-learn**: Clustering GMM para deteccao nao-supervisionada de mecanismos
- **seaborn/matplotlib**: Visualizacoes academicas
- **google-generativeai**: Geracao de relatorios narrativos multimodais (Gemini 2.0 Flash)

## Setup

### Requisitos

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (gerenciador de pacotes)
- (Opcional) API Key do Google Gemini para relatorios com IA

### Instalacao

```bash
# Clonar o repositorio
git clone <repo-url>
cd vocal-analysis

# Instalar dependencias
uv sync

# Instalar dependencias de desenvolvimento (ruff, pytest, jupyter)
uv sync --extra dev
```

### Configurar Gemini (opcional)

Para gerar relatorios narrativos com IA:

1. Acesse [Google AI Studio](https://aistudio.google.com/apikey)
2. Clique em "Create API Key"
3. Configure a variavel de ambiente:

```bash
export GEMINI_API_KEY=sua_chave_aqui
```

Ou adicione ao seu `.bashrc`/`.zshrc` para persistir.

**Usando arquivo `.env`:**

1. Copie o template:
   ```bash
   cp .env.example .env
   ```
2. Edite `.env` com suas configuracoes
3. Carregue no ambiente:
   ```bash
   source .env
   ```

### Configurar Idioma dos Relatorios

O idioma dos relatorios gerados (analysis_report.md, vmi_analysis.md, llm_report.md) e controlado pela variavel `REPORT_LANG`:

```bash
# Relatorios em portugues
REPORT_LANG=pt-BR

# Relatorios em ingles (padrao)
REPORT_LANG=en
```

### Configurar Excerpts (opcional)

Voce pode definir trechos especificos de cada musica para analise e plots de validacao.
Util para focar em passagens vocais sem introducoes instrumentais.

No `.env`, use o formato `EXCERPT_<NOME>=MMSS-MMSS`:

```bash
# Do segundo 22 ao 1:03
EXCERPT_DELICADO="0022-0103"

# Do segundo 33 ao 1:04
EXCERPT_BRASILEIRINHO="0033-0104"

# Do segundo 7 ao 23
EXCERPT_APANHEITE_CAVAQUINHO="0007-0023"
```

Os excerpts sao usados automaticamente nos plots de validacao (`--validate-separation`).

## Estrutura do Projeto

```
vocal-analysis/
├── src/vocal_analysis/
│   ├── preprocessing/
│   │   ├── audio.py              # load_audio(), normalize_audio()
│   │   ├── separation.py         # Source separation (HTDemucs)
│   │   └── process_ademilde.py   # Script de extracao de features
│   ├── features/
│   │   ├── extraction.py         # Pipeline hibrido Crepe + Praat
│   │   ├── spectral.py           # Features espectrais (Alpha Ratio, H1-H2, etc.)
│   │   ├── vmi.py                # Calculo do Vocal Mechanism Index
│   │   └── articulation.py       # Features de agilidade articulatoria
│   ├── analysis/
│   │   ├── exploratory.py        # Analise M1/M2, clustering
│   │   ├── run_analysis.py       # Script de analise completa
│   │   └── llm_report.py         # Geracao de relatorio com Gemini
│   ├── modeling/
│   │   └── classifier.py         # XGBoost para classificacao M1/M2
│   ├── scripts/
│   │   └── regenerate_validation_plot.py  # Regenerar plots sem reprocessar
│   ├── visualization/
│   │   └── plots.py              # Plots academicos
│   └── utils/
│       └── pitch.py              # Conversao Hz <-> Notas (A4, C5, etc)
├── data/
│   ├── raw/                      # Audios originais (.mp3)
│   └── processed/                # CSVs, JSONs, logs
├── docs/
│   ├── en/                       # English documentation
│   └── pt-BR/                    # Documentacao em portugues
├── outputs/
│   ├── plots/                    # Graficos gerados
│   └── models/                   # Modelos treinados
└── tests/
```

## Uso por Plataforma

### macOS
- **Validacao rapida**: Use `--use-praat-f0` para testar o pipeline
- **Limitacao**: CREPE full trava por falta de memoria (32GB+ recomendado)
- **Recomendacao**: Use macOS apenas para validacao, processe com CREPE no Colab/Windows

```bash
# Validacao rapida no macOS (Praat F0)
uv run python -m vocal_analysis.preprocessing.process_ademilde --use-praat-f0
```

### Windows/Linux (32GB+ RAM)
- **Processamento completo**: CREPE full com todas features
- **GPU (NVIDIA)**: Use `--device cuda` para aceleracao (~10x mais rapido)

```bash
# Windows/Linux com CPU
uv run python -m vocal_analysis.preprocessing.process_ademilde

# Windows/Linux com GPU NVIDIA
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda
```

### Google Colab (Recomendado!)
- **GPU T4 gratuita**: ~12-15h/dia de uso
- **Processamento rapido**: 3 musicas (~7min cada) em ~10 minutos
- **Zero configuracao**: Ambiente ja pronto

**Guia completo**: [COLAB_SETUP.md](COLAB_SETUP.md)

**Quick start**:
```python
# No Colab com GPU T4 habilitada
!git clone https://github.com/arthurcornelio88/vocal-analysis.git
%cd vocal-analysis
!pip install uv && uv pip install --system -e .

# Verificar instalacao
!python -c "import vocal_analysis; print('Instalado!')"

# Processar com CREPE + GPU
!python src/vocal_analysis/preprocessing/process_ademilde.py --device cuda
```

---

## Uso

### 1. Adicionar audios

Coloque os arquivos MP3 em `data/raw/`:

```
data/raw/
├── Apanhei-te Cavaquinho.mp3
├── brasileirinho.mp3
└── delicado.mp3
```

### 2. Processar audios (extrair features)

```bash
# Processamento completo com CREPE (requer GPU ou 32GB+ RAM)
uv run python -m vocal_analysis.preprocessing.process_ademilde

# Com GPU (Google Colab, Windows/Linux com NVIDIA)
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda

# Modo rapido com Praat (macOS, validacao)
uv run python -m vocal_analysis.preprocessing.process_ademilde --use-praat-f0
```

**Opcoes disponiveis:**
- `--device cuda`: Usar GPU (requer CUDA)
- `--use-praat-f0`: Usar Praat em vez de CREPE (mais rapido, menos preciso)
- `--crepe-model {tiny,small,full}`: Escolher modelo CREPE (default: full)
- `--skip-formants`: Pular extracao de formantes (~30% mais rapido)
- `--skip-jitter-shimmer`: Pular jitter/shimmer (~20% mais rapido)
- `--skip-cpps`: Pular CPPS (evita travamento no macOS)
- `--skip-plots`: Nao gerar plots de F0
- `--limit N`: Processar apenas N arquivos (util para testes)
- `--fast`: Ativa todas otimizacoes (Praat + sem formants/jitter/shimmer/cpps/plots)
- `--no-separate-vocals`: Desabilitar source separation (HTDemucs). **Por padrao**, a separacao de voz e habilitada para melhorar a deteccao de pitch em arranjos complexos

**Outputs gerados:**
- `data/processed/ademilde_features.csv` - Features por frame
- `data/processed/ademilde_metadata.json` - Metadados estruturados
- `data/processed/processing_log.md` - Log legivel
- `outputs/plots/*_f0.png` - Contornos de pitch

### 2.1. Regenerar plots de validacao (sem reprocessar)

Se voce ja processou os audios e quer apenas regenerar os plots de validacao:

```bash
# Listar musicas com cache disponivel
uv run python -m vocal_analysis.scripts.regenerate_validation_plot

# Regenerar plot de uma musica especifica
uv run python -m vocal_analysis.scripts.regenerate_validation_plot --song "Apanhei-te Cavaquinho"

# Regenerar todos os plots
uv run python -m vocal_analysis.scripts.regenerate_validation_plot --all
```

### 3. Rodar analise exploratoria

```bash
uv run python -m vocal_analysis.analysis.run_analysis
```

O script executa quatro metodos de classificacao em sequencia:

1. **Limiar de frequencia** (400 Hz / G4) — divisao binaria simples
2. **Clustering GMM** — descoberta nao-supervisionada de clusters M1/M2 no espaco f0 x HNR
3. **XGBoost** — classificador supervisionado treinado com pseudo-labels do GMM
4. **VMI (Vocal Mechanism Index)** — metrica continua 0-1 baseada em features espectrais (Alpha Ratio, H1-H2, CPPS, Spectral Tilt), independente de limiares fixos de frequencia

**Outputs gerados:**
- `outputs/plots/mechanism_analysis.png` - Analise M1/M2 por limiar
- `outputs/plots/mechanism_clusters.png` - Clustering GMM
- `outputs/plots/vmi_scatter.png` - Distribuicao VMI por features espectrais
- `outputs/plots/xgb_mechanism_timeline.png` - Contorno temporal pela predicao XGBoost
- `outputs/xgb_predictions.csv` - Predicoes por frame (todos os metodos)
- `outputs/analysis_report.md` - Relatorio estruturado com metricas de classificacao
- `outputs/vmi_analysis.md` - Relatorio VMI com detalhamento por musica
- `outputs/llm_report.md` - Relatorio narrativo com Gemini (se API configurada)

### 4. Rodar geracao de relatorio LLM (opcional)

```bash
uv run python -m vocal_analysis.analysis.llm_report
```

**Parametros opcionais:**
- `--metadata`: Caminho para o arquivo de metadados (padrao: `data/processed/ademilde_metadata.json`)
- `--stats`: Caminho para JSON com estatisticas M1/M2 (opcional)
- `--output`: Caminho de saida para o relatorio (padrao: `outputs/llm_report.md`)
- `--plots-dir`: Diretorio com plots PNG para analise multimodal (padrao: `outputs/plots/`)
- `--lang`: Idioma do relatorio (`en` ou `pt-BR`, padrao: `en`)

### 5. Features extraidas

**Features base** (por frame, de `process_ademilde`):

| Coluna | Descricao |
|--------|-----------|
| `time` | Timestamp em segundos |
| `f0` | Frequencia fundamental (Hz) |
| `confidence` | Confianca da estimativa de pitch (0-1) |
| `hnr` | Harmonic-to-Noise Ratio (dB) |
| `energy` | Energia RMS |
| `f1, f2, f3, f4` | Formantes 1-4 (Hz) |
| `song` | Nome da musica |
| `cpps_global` | Cepstral Peak Prominence (global por musica) |
| `jitter` | Jitter ppq5 - instabilidade de periodo (%) |
| `shimmer` | Shimmer apq11 - instabilidade de amplitude (%) |

**Features espectrais** (adicionadas por `run_analysis`):

| Coluna | Descricao |
|--------|-----------|
| `alpha_ratio` | Razao de energia 0-1 kHz vs 1-5 kHz (dB) |
| `h1_h2` | Diferenca entre 1o e 2o harmonico (inclinacao glotal, dB) |
| `spectral_tilt` | Inclinacao do espectro de potencia (dB/oitava) |
| `cpps_per_frame` | Cepstral Peak Prominence por frame |

**Classificacao e VMI** (adicionadas por `run_analysis`):

| Coluna | Descricao |
|--------|-----------|
| `mechanism` | Label do GMM (M1/M2) |
| `xgb_mechanism` | Predicao XGBoost (M1/M2) |
| `vmi` | Vocal Mechanism Index (0.0 - 1.0) |
| `vmi_label` | Categoria VMI (M1_HEAVY, M1_LIGHT, MIX_PASSAGGIO, M2_REINFORCED, M2_LIGHT) |
| `f0_velocity` | Taxa de mudanca de pitch (Hz/s) |
| `f0_acceleration` | Aceleracao de pitch (Hz/s^2) |
| `syllable_rate` | Taxa silabica (silabas/s) |

## Utilitarios

### Conversao Hz <-> Notas

```python
from vocal_analysis.utils.pitch import hz_to_note, note_to_hz

hz_to_note(440.0)        # "A4"
hz_to_note(261.63)       # "C4"
note_to_hz("G5")         # 783.99
```

## Desenvolvimento

### Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Testes

```bash
uv run pytest -v
```

## Documentacao

- **[Glossario Bioacustico](GLOSSARIO_BIOACUSTICO.md)** — Introducao acessivel a f0, HNR, formantes, jitter, shimmer, VMI e por que cada um importa
- **[Metodologia](METODOLOGIA.md)** — Referencia tecnica completa: preprocessamento, extracao de features, 4 metodos de classificacao, features espectrais, calculo do VMI, estrutura de dados, limitacoes e referencias academicas
- **[Setup Colab](COLAB_SETUP.md)** — Guia passo a passo para rodar no Google Colab com GPU gratuita
