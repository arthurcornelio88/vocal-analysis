# 🚀 Rodando o Projeto no Google Colab com GPU

Este guia mostra como processar os áudios com CREPE usando GPU gratuita do Google Colab.

## 📋 Passo a passo

### 1. Abrir novo notebook no Colab
1. Acesse: https://colab.research.google.com
2. File → New Notebook
3. **IMPORTANTE**: Runtime → Change runtime type → GPU (T4)

### 2. Clonar o repositório e instalar dependências

```python
# 🚨 IMPORTANTE: Execute TODA esta célula ANTES de continuar!
# Clone + Instalação (necessário para imports funcionarem)

!git clone https://github.com/arthurcornelio88/vocal-analysis.git
%cd vocal-analysis

# Instalar uv e pacote em modo system (não venv)
!pip install uv
!uv pip install --system -e .

# ✅ VERIFICAÇÃO: Se instalado corretamente, deve mostrar versão
!python -c "import vocal_analysis; print(f'✅ vocal_analysis instalado! Versão: {vocal_analysis.__version__}')"

# Verificar GPU disponível
import torch
print(f"\nGPU disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

**💡 Por que `--system`?**
- `uv sync` cria ambiente virtual que o Colab não usa automaticamente
- `uv pip install --system` instala direto no Python do sistema
- Mais rápido que `pip` mas compatível com Colab

### 3. Upload dos arquivos de áudio

```python
from google.colab import files
import os

# Criar diretório data/raw se não existir
os.makedirs('data/raw', exist_ok=True)

# Upload dos MP3
print("Faça upload dos arquivos MP3:")
uploaded = files.upload()

# Mover para data/raw
for filename in uploaded.keys():
    !mv "{filename}" data/raw/

!ls -lh data/raw/
```

### 4. Configurar excerpts (opcional)

```python
# Criar arquivo .env com os trechos desejados
with open('.env', 'w') as f:
    f.write('''EXCERPT_DELICADO="0022-0103"
EXCERPT_BRASILEIRINHO="0033-0104"
EXCERPT_APANHEITE_CAVAQUINHO="0007-0023"
''')
```

### 5. Processar com CREPE (GPU)

```python
# Processamento completo com CREPE full + GPU
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda

# Verificar outputs
!ls -lh data/processed/
```

**Tempo esperado**: ~5-10 minutos para 3 músicas (~7min cada) com GPU T4

### 6. Gerar análises

```python
# Rodar análise exploratória
!python src/vocal_analysis/analysis/run_analysis.py

# Listar outputs
!ls -lh outputs/plots/
!ls -lh outputs/*.md
```

### 7. Download dos resultados

```python
from google.colab import files

# Download CSV com features
files.download('data/processed/ademilde_features.csv')

# Download metadata JSON
files.download('data/processed/ademilde_metadata.json')

# Download relatórios
files.download('outputs/analise_ademilde.md')

# Download plots (zip primeiro)
!zip -r outputs_plots.zip outputs/plots/
files.download('outputs_plots.zip')

# Download excerpts de áudio
!zip -r excerpts.zip outputs/excerpt_*.wav
files.download('excerpts.zip')
```

## 🎯 Comandos úteis

### Processar apenas 1 arquivo (teste rápido)
```python
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --limit 1
```

### Processar sem plots (mais rápido)
```python
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --skip-plots
```

### Usar modelo CREPE menor (mais rápido, menos preciso)
```python
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --crepe-model tiny
```

### Ajustar batch size (se tiver OOM)
```python
# Se T4 der "Out of Memory", reduza batch size
!python src/vocal_analysis/preprocessing/process_ademilde.py \
    --device cuda \
    --batch-size 512
```

## 💡 Dicas

1. **GPU T4 gratuita**: ~12-15 horas/dia de uso
2. **Salvar progresso**: Download dos arquivos antes da sessão expirar
3. **Reprocessar**: Se precisar, os arquivos ficam salvos por ~12h no Colab
4. **Batch size**: Padrão agora é 2048 (ideal para GPU T4). Se tiver OOM, use `--batch-size 512`

## 🐛 Troubleshooting

**"ModuleNotFoundError: No module named 'vocal_analysis'"**:
- Execute na seção 2: `!uv pip install --system -e .`
- NUNCA use `!uv sync` sozinho (cria venv que o Colab não usa)
- Verifique com: `!python -c "import vocal_analysis; print('✅ OK!')"`

**"GPU not available"**:
- Verifique: Runtime → Change runtime type → GPU (T4)

**"Out of memory"**:
- Use `--crepe-model small` ou `--crepe-model tiny`

**"Process killed"**:
- Reduza batch_size editando `extraction.py` linha 110

## 📊 Output esperado

Após processar, você terá:
- ✅ `ademilde_features.csv` - Todas as features (f0, formants, jitter, shimmer, etc.)
- ✅ `ademilde_metadata.json` - Metadados e estatísticas
- ✅ `analise_ademilde.md` - Relatório técnico
- ✅ `xgb_predictions.csv` - Predições de mecanismo (M1/M2)
- ✅ Plots de F0, excerpts, mecanismos, etc.
- ✅ Áudios dos excerpts (`.wav`)

## 🚀 Ready para o artigo!

Com os dados do Colab (CREPE + GPU), você terá F0 preciso para análise acadêmica! 🎵
