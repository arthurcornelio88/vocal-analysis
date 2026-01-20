# CLAUDE.md - Protocolo de Análise Bioacústica e ML (Choro/Canto)

## Contexto do Projeto
Este repositório contém o código para a análise computacional de um artigo acadêmico sobre a voz no Choro. O objetivo é criticar o sistema de classificação vocal "Fach" através de uma análise fisiológica dos mecanismos laríngeos (M1/M2).
- **Autor:** ML Engineer & Músico Profissional.
- **Domínio:** Bioacústica, Processamento Digital de Sinais (DSP) e Machine Learning.
- **Goal:** Extrair features explicáveis (Jitter, Shimmer, HNR, f0) de áudios de canto e classificar os mecanismos M1 (peito) e M2 (cabeça).

## Stack Tecnológica & Diretrizes
- **Linguagem:** Python 3.10+
- **Bibliotecas Principais:**
  - `torchcrepe`: Para extração SOTA de f0 (crucial para vibrato/instabilidade).
  - `parselmouth` (Praat wrapper): GOLD STANDARD para Jitter, Shimmer e HNR. Não usar librosa para essas métricas específicas.
  - `xgboost` / `sklearn`: Para classificação tabular dos mecanismos.
  - `seaborn` / `matplotlib`: Plots com estética acadêmica (artigo).

## Comandos Úteis
- Setup: `pip install torchcrepe parselmouth-praat librosa xgboost pandas seaborn`
- Linting: Manter código limpo e tipado (`typing.List`, `np.ndarray`).

## Architecture Overview
1. **Preprocessing:** Normalização de áudio (-3dB), conversão mono, 44.1kHz.
2. **Feature Extraction (Híbrida):**
   - Usar `torchcrepe` para a curva de pitch (f0) temporalmente precisa.
   - Usar `parselmouth` nos mesmos timestamps para extrair qualidade vocal (HNR, CPPs).
3. **Data Aggregation:** Criar DataFrames onde cada linha é um frame de áudio (ex: 10ms) com suas features.
4. **Modelling:** Treinar XGBoost para separar clusters M1/M2 baseados em Energia vs HNR.

## Code Snippets Importantes

### 1. Extração Híbrida (Crepe + Praat)
Use este padrão para garantir que temos a precisão do Crepe com a física do Praat.

```python
import torchcrepe
import parselmouth
import numpy as np
import librosa

def extract_bioacoustic_features(audio_path: str, hop_length: int = 441):
    """
    Pipeline Híbrido:
    1. Crepe para f0 (SOTA em robustez de pitch).
    2. Parselmouth para métricas espectrais (Rigor acadêmico).
    """
    # Carregar áudio para o Crepe
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)

    # 1. Extração de f0 com CREPE (Melhor que autocorrelação para canto)
    # batch_size ajustável conforme VRAM. decoder='viterbi' suaviza a curva.
    f0, confidence = torchcrepe.predict(
        torch.from_numpy(audio).unsqueeze(0),
        sr,
        hop_length=hop_length,
        fmin=50,
        fmax=800,
        model='full',
        decoder=torchcrepe.decode.viterbi,
        batch_size=2048,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    f0 = f0.squeeze().numpy()

    # 2. Extração de Timbre com Parselmouth (Praat)
    s = parselmouth.Sound(audio_path)

    # Harmonicity (HNR) - Proxy para "limpeza" da voz (M1 costuma ser mais alto)
    harmonicity = s.to_harmonicity(time_step=hop_length/sr)
    hnr_values = harmonicity.values[0]  # Array de HNR em dB

    # Cepstral Peak Prominence (CPPS) - Proxy para soprosidade/fechamento glótico
    power_cepstrum = s.to_power_cepstrum()
    cpps = power_cepstrum.get_peak_prominence(fmin=60, fmax=3300)

    # Ajustar tamanhos dos arrays (Crepe e Praat podem divergir por 1-2 frames)
    min_len = min(len(f0), len(hnr_values))

    return {
        "f0": f0[:min_len],
        "confidence": confidence.squeeze().numpy()[:min_len],
        "hnr": hnr_values[:min_len],
        "cpps_global": cpps  # Valor global do arquivo
    }

```

### 2. Configuração do Classificador (XGBoost)

Setup inicial para diferenciar M1/M2.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_mechanism_classifier(df_features):
    """
    Assume que df_features tem colunas: ['f0', 'hnr', 'energy', 'zcr']
    Target: 0 (M1/Peito), 1 (M2/Cabeça) - se houver dados rotulados.
    Se não houver rótulos, usar K-Means ou GMM para clusterização inicial.
    """
    X = df_features[['f0', 'hnr', 'energy']]
    y = df_features['mechanism_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='binary:logistic'
    )

    model.fit(X_train, y_train)

    print("Relatório de Classificação M1/M2:")
    print(classification_report(y_test, model.predict(X_test)))

    return model

```

### 3. Plotagem Acadêmica (Style Guide)

Para os gráficos do artigo.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mechanism_clusters(df):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # O "Etoile" do artigo: Scatter plot de Pitch vs HNR
    sns.scatterplot(
        data=df,
        x='f0',
        y='hnr',
        hue='mechanism',
        alpha=0.6,
        palette='viridis'
    )

    plt.title('Distribuição Espectral: Mecanismo 1 vs Mecanismo 2')
    plt.xlabel('Frequência Fundamental (Hz)')
    plt.ylabel('Harmonic-to-Noise Ratio (dB)')
    plt.show()
```

# Ultimas instrucoes

- usamos UV (pyproject.toml)
- ruff e lint de CI quando push master
