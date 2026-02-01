# Vocal Analysis - Análise Bioacústica M1/M2

Análise computacional de mecanismos laríngeos (M1/M2) em gravações de Choro brasileiro, com foco na voz de Ademilde Fonseca.

## Objetivo

Criticar o sistema de classificação vocal "Fach" através de uma análise fisiológica dos mecanismos laríngeos, extraindo features explicáveis (f0, HNR, CPPS) de áudios de canto.

## Stack

- **torchcrepe**: Extração SOTA de f0 (pitch)
- **parselmouth** (Praat): HNR, CPPS, Jitter, Shimmer
- **xgboost**: Classificação tabular M1/M2
- **seaborn/matplotlib**: Visualizações acadêmicas
- **google-generativeai**: Geração de relatórios narrativos multimodais com Gemini 2.0 Flash (⚠️ deprecated mas funcional)

## Setup

### Requisitos

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (gerenciador de pacotes)
- (Opcional) API Key do Google Gemini para relatórios com IA

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

### Configurar Gemini (opcional)

Para gerar relatórios narrativos com IA:

1. Acesse [Google AI Studio](https://aistudio.google.com/apikey)
2. Clique em "Create API Key"
3. Configure a variável de ambiente:

```bash
export GEMINI_API_KEY=sua_chave_aqui
```

Ou adicione ao seu `.bashrc`/`.zshrc` para persistir.

**Usando arquivo `.env`:**
O projeto inclui um arquivo `.env` na raiz. Para usar:
1. Edite o arquivo `.env` e adicione sua API Key:
   ```
   GEMINI_API_KEY=sua_chave_aqui
   ```
2. Certifique-se de que a variável está carregada no ambiente atual:
   ```bash
   source .env
   ```

**Nota sobre o pacote:**
O projeto usa `google-generativeai` que está deprecated mas ainda funcional. Você verá um warning ao executar, mas o código funciona normalmente. Para migrar para o novo pacote `google-genai` no futuro, será necessário atualizar o código em `src/vocal_analysis/analysis/llm_report.py`.

## Estrutura do Projeto

```
vocal-analysis/
├── src/vocal_analysis/
│   ├── preprocessing/
│   │   ├── audio.py              # load_audio(), normalize_audio()
│   │   └── process_ademilde.py   # Script de extração de features
│   ├── features/
│   │   └── extraction.py         # Pipeline híbrido Crepe + Praat
│   ├── analysis/
│   │   ├── exploratory.py        # Análise M1/M2, clustering
│   │   ├── run_analysis.py       # Script de análise completa
│   │   └── llm_report.py         # Geração de relatório com Gemini
│   ├── modeling/
│   │   └── classifier.py         # XGBoost para classificação M1/M2
│   ├── visualization/
│   │   └── plots.py              # Plots acadêmicos
│   └── utils/
│       └── pitch.py              # Conversão Hz ↔ Notas (A4, C5, etc)
├── data/
│   ├── raw/                      # Áudios originais (.mp3)
│   └── processed/                # CSVs, JSONs, logs
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

### 2. Processar áudios (extrair features)

```bash
uv run python -m vocal_analysis.preprocessing.process_ademilde
```

**Outputs gerados:**
- `data/processed/ademilde_features.csv` - Features por frame
- `data/processed/ademilde_metadata.json` - Metadados estruturados
- `data/processed/processing_log.md` - Log legível
- `outputs/plots/*_f0.png` - Contornos de pitch

### 3. Rodar análise exploratória

```bash
uv run python -m vocal_analysis.analysis.run_analysis
```

**Outputs gerados:**
- `outputs/plots/mechanism_analysis.png` - 4 plots de análise M1/M2 (threshold)
- `outputs/plots/mechanism_clusters.png` - Clustering GMM
- `outputs/plots/xgb_mechanism_timeline.png` - Contorno temporal pela predição XGBoost
- `outputs/xgb_predictions.csv` - Predições por frame (GMM + XGBoost)
- `outputs/analise_ademilde.md` - Relatório estruturado (inclui classification report do XGBoost)
- `outputs/relatorio_llm.md` - Relatório narrativo com Gemini (se API configurada)

#### Relatório LLM Multimodal

Se `GEMINI_API_KEY` estiver configurada, o script gera um relatório narrativo usando Gemini 2.0 Flash com:

- **Análise multimodal**: O LLM recebe os plots junto com os dados numéricos
- **Links clicáveis**: Referências a gráficos incluem links markdown (ex: `[brasileirinho_f0](plots/brasileirinho_f0.png)`)
- **Índice de figuras**: Lista completa de visualizações no final do relatório

### 4. Rodar geração de relatório LLM (opcional)

Para gerar apenas o relatório narrativo com Gemini (sem rodar toda a análise novamente):

```bash
uv run python -m vocal_analysis.analysis.llm_report
```

**Parâmetros opcionais:**
- `--metadata`: Caminho para o arquivo de metadados (padrão: `data/processed/ademilde_metadata.json`)
- `--stats`: Caminho para JSON com estatísticas M1/M2 (opcional)
- `--output`: Caminho de saída para o relatório (padrão: `outputs/relatorio_llm.md`)
- `--plots-dir`: Diretório com plots PNG para análise multimodal (padrão: `outputs/plots/`)

**Exemplo com parâmetros customizados:**
```bash
uv run python -m vocal_analysis.analysis.llm_report \
  --metadata data/processed/ademilde_metadata.json \
  --output outputs/relatorio_customizado.md \
  --plots-dir outputs/plots/
```

**Pré-requisitos:**
1. API Key do Gemini configurada: `export GEMINI_API_KEY=sua_chave_aqui`
2. Arquivos de dados processados disponíveis (após executar os passos 1-3)
3. Plots gerados (opcional, para análise multimodal)

**Verificação da API Key:**
Para verificar se a API Key está configurada corretamente:
```bash
echo $GEMINI_API_KEY
```
Se não mostrar nada, configure a variável de ambiente:
```bash
export GEMINI_API_KEY=sua_chave_aqui
```

**Nota importante:** O script `run_analysis.py` só invocará automaticamente a geração do relatório LLM se a variável `GEMINI_API_KEY` estiver configurada no ambiente. Caso contrário, ele mostrará apenas uma mensagem informativa e não gerará o relatório.

**Erros comuns:**
1. **"API key not valid"**: Verifique se a API key está correta e ativa no [Google AI Studio](https://aistudio.google.com/apikey)
2. **"quota exceeded" / "You exceeded your current quota"**: A conta gratuita do Gemini tem limites de uso. Espere o reset da quota ou atualize para um plano pago.
3. **Warning sobre pacote deprecated**: O pacote `google-generativeai` está deprecated mas ainda funciona. Ignore o warning.

**Nota:** Este comando é útil quando você já executou a análise completa e deseja:
- Regenerar o relatório LLM com diferentes parâmetros
- Testar diferentes prompts ou configurações
- Gerar relatórios específicos para apresentações
- Executar a geração do relatório quando a API Key não estava configurada durante a execução do `run_analysis.py`

### 5. Features extraídas

| Coluna | Descrição |
|--------|-----------|
| `time` | Timestamp em segundos |
| `f0` | Frequência fundamental (Hz) |
| `confidence` | Confiança da estimativa de pitch (0-1) |
| `hnr` | Harmonic-to-Noise Ratio (dB) |
| `energy` | Energia RMS |
| `f1, f2, f3, f4` | Formantes 1-4 (Hz) |
| `song` | Nome da música |
| `cpps_global` | Cepstral Peak Prominence (valor global por música) |
| `jitter` | Jitter ppq5 - instabilidade de período (%) |
| `shimmer` | Shimmer apq11 - instabilidade de amplitude (%) |

### 6. Classificação M1/M2

Com dados rotulados:

```python
from vocal_analysis.modeling import train_mechanism_classifier
import pandas as pd

df = pd.read_csv("data/processed/ademilde_features.csv")
df["mechanism_label"] = ...  # 0=M1 (peito), 1=M2 (cabeça)

model = train_mechanism_classifier(df)
```

## Utilitários

### Conversão Hz ↔ Notas

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

## Documentação

### Guia Introdutório

Para entender os conceitos bioacústicos usados na análise (f0, HNR, formantes, jitter, shimmer) e por que cada um importa, sem precisar de background técnico prévio:

**[docs/glossario_bioacustico.md](docs/glossario_bioacustico.md)**

### Metodologia Detalhada

Para entender em profundidade as escolhas metodológicas, parâmetros técnicos e justificativas acadêmicas de cada componente do pipeline, consulte:

**[docs/METODOLOGIA.md](docs/METODOLOGIA.md)**

Este documento descreve:
- Escolha do CREPE vs métodos tradicionais de autocorrelação
- Parâmetros de pré-processamento (normalização, hop length, thresholds)
- Detalhamento de cada feature bioacústica (f0, HNR, CPPS, Jitter, Shimmer, Formantes)
- Features de agilidade articulatória (f0 velocity, taxa silábica)
- Métodos de classificação M1/M2 (Threshold, GMM, XGBoost)
- Estrutura de dados e workflow de execução
- Limitações reconhecidas e conformidade acadêmica

## Referências

- **CREPE**: [Kim et al., 2018 - CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182)
- **Praat**: [Boersma & Weenink - Praat: doing phonetics by computer](https://www.praat.org)
- **Mecanismos Laríngeos**: Roubeau, B., Henrich, N., & Castellengo, M. (2009). Laryngeal vibratory mechanisms
