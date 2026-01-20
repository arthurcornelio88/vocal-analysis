# Vocal Analysis - Análise Bioacústica M1/M2

Análise computacional de mecanismos laríngeos (M1/M2) em gravações de Choro brasileiro, com foco na voz de Ademilde Fonseca.

## Objetivo

Criticar o sistema de classificação vocal "Fach" através de uma análise fisiológica dos mecanismos laríngeos, extraindo features explicáveis (f0, HNR, CPPS) de áudios de canto.

## Stack

- **torchcrepe**: Extração SOTA de f0 (pitch)
- **parselmouth** (Praat): HNR, CPPS, Jitter, Shimmer
- **xgboost**: Classificação tabular M1/M2
- **seaborn/matplotlib**: Visualizações acadêmicas

## Setup

### Requisitos

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (gerenciador de pacotes)

### Instalação

```bash
# Clonar o repositório
git clone <repo-url>
cd vocal-analysis

# Instalar dependências
uv sync

# Instalar dependências de desenvolvimento (ruff, pytest, jupyter)
uv sync --extra dev
```

## Estrutura do Projeto

```
vocal-analysis/
├── src/vocal_analysis/
│   ├── preprocessing/
│   │   ├── audio.py              # load_audio(), normalize_audio()
│   │   └── process_ademilde.py   # Script principal de processamento
│   ├── features/
│   │   └── extraction.py         # Pipeline híbrido Crepe + Praat
│   ├── modeling/
│   │   └── classifier.py         # XGBoost para classificação M1/M2
│   └── visualization/
│       └── plots.py              # Plots acadêmicos (f0 contour, clusters)
├── data/
│   ├── raw/                      # Áudios originais (.mp3)
│   └── processed/                # CSVs com features extraídas
├── outputs/
│   ├── plots/                    # Gráficos gerados
│   └── models/                   # Modelos treinados
└── tests/
```

## Uso

### 1. Adicionar áudios

Coloque os arquivos MP3 em `data/raw/`:

```
data/raw/
├── Apanhei-te Cavaquinho.mp3
├── brasileirinho.mp3
└── delicado.mp3
```

### 2. Processar áudios

```bash
uv run python -m vocal_analysis.preprocessing.process_ademilde
```

Este comando:
- Extrai f0 (pitch) usando CREPE
- Extrai HNR e CPPS usando Praat
- Filtra frames com confiança > 0.5
- Gera plots de contorno de f0 em `outputs/plots/`
- Salva features em `data/processed/ademilde_features.csv`

### 3. Analisar resultados

O CSV gerado contém:

| Coluna | Descrição |
|--------|-----------|
| `time` | Timestamp em segundos |
| `f0` | Frequência fundamental (Hz) |
| `confidence` | Confiança da estimativa de pitch |
| `hnr` | Harmonic-to-Noise Ratio (dB) |
| `song` | Nome da música |
| `cpps_global` | Cepstral Peak Prominence (global) |

### 4. Classificação M1/M2 (futuro)

Quando houver dados rotulados:

```python
from vocal_analysis.modeling import train_mechanism_classifier
import pandas as pd

df = pd.read_csv("data/processed/ademilde_features.csv")
df["mechanism_label"] = ...  # 0=M1, 1=M2

model = train_mechanism_classifier(df)
```

## Desenvolvimento

### Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Testes

```bash
uv run pytest
```

## Referências

- **CREPE**: [Kim et al., 2018 - CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182)
- **Praat**: [Boersma & Weenink - Praat: doing phonetics by computer](https://www.praat.org)
