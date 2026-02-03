# Plano: Adicionar Source Separation ao Pipeline de Análise Vocal

## Contexto

O CREPE não está conseguindo detectar pitch bem em arranjos complexos de choro (muitos instrumentos). Solução: adicionar HTDemucs para isolar a voz ANTES de aplicar CREPE + Praat.

**Pipeline atual:**
```
Áudio MP3 → load_audio → extract_bioacoustic_features → CSV + plots
```

**Pipeline proposto:**
```
Áudio MP3 → HTDemucs (separar voz) → extract_bioacoustic_features → CSV + plots
```

---

## Arquivos a Modificar

| Arquivo | Ação |
|---------|------|
| `pyproject.toml` | Adicionar `torchaudio>=2.0.0` |
| `src/vocal_analysis/preprocessing/separation.py` | **CRIAR** - módulo VocalSeparator |
| `src/vocal_analysis/preprocessing/__init__.py` | Exportar `separate_vocals` |
| `src/vocal_analysis/preprocessing/process_ademilde.py` | Adicionar flags CLI e integração |
| `src/vocal_analysis/visualization/plots.py` | Adicionar `plot_separation_validation()` |
| `tests/test_separation.py` | **CRIAR** - testes unitários |

---

## Implementação

### 1. Novo módulo: `preprocessing/separation.py`

```python
class VocalSeparator:
    """Separador de voz usando HTDemucs via torchaudio."""

    def __init__(self, device: str = "cpu", segment_seconds: float = 10.0):
        # Lazy loading do modelo HDEMUCS_HIGH_MUSDB_PLUS

    def extract_vocals(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        # Retorna array mono com voz isolada

def separate_vocals(audio_path: Path, device: str, cache_dir: Path | None) -> tuple[np.ndarray, int]:
    # Função de conveniência com cache em .npy
```

**Por que HTDemucs via torchaudio:**
- SOTA para voz em arranjos densos (7-9 dB SDR)
- Já integrado com torch (projeto já usa PyTorch)
- 44.1kHz nativo (mesmo sample rate do projeto)
- ~1GB de modelo, baixado automaticamente

### 2. Novos flags CLI em `process_ademilde.py`

```bash
--separate-vocals          # Habilitar source separation
--separation-device        # cpu ou cuda (default: mesmo que --device)
--no-separation-cache      # Forçar reprocessamento
--validate-separation      # Gerar plot comparativo original vs separado (Hz + notas)
```

### 3. Cache de áudio separado

```
data/cache/separated/
├── brasileirinho_vocals.npy
├── delicado_vocals.npy
└── ...
```

- Evita reprocessar (separação leva ~1.5x duração em CPU)
- Formato numpy binário para I/O rápido

### 4. Integração no loop de processamento

```python
for audio_path in audio_files:
    if config.separate_vocals:
        vocals, sr = separate_vocals(audio_path, device, cache_dir)
        temp_wav = criar_wav_temporario(vocals, sr)  # Praat precisa de arquivo
        audio_para_features = temp_wav
    else:
        audio_para_features = audio_path

    features = extract_bioacoustic_features(audio_para_features, ...)
```

---

## Uso Final

```bash
# Processamento normal (sem separação - comportamento atual)
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda

# COM source separation
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda --separate-vocals

# COM validação visual (gera plots Hz + notas para conferir)
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda --separate-vocals --validate-separation

# Separação em CPU, CREPE em GPU
uv run python -m vocal_analysis.preprocessing.process_ademilde --device cuda --separate-vocals --separation-device cpu
```

---

## Validação Visual

Adicionar função `plot_separation_validation()` em `visualization/plots.py` para confirmar que estamos analisando a voz e não o cavaquinho:

```python
def plot_separation_validation(
    time_original: np.ndarray,
    f0_original: np.ndarray,
    time_separated: np.ndarray,
    f0_separated: np.ndarray,
    title: str = "Validação: Original vs Voz Separada",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot comparativo mostrando f0 antes/depois da separação.
    Eixo Y: Hz (esquerda) + Notas inglesas (direita).
    """
```

**Características do plot:**
- Dois subplots empilhados: "Original (mix)" e "Voz Separada"
- Eixo Y esquerdo: Hz
- Eixo Y direito: Notas em formato inglês (A#4, C5, etc.) usando `hz_to_note`
- Grid horizontal nas notas para fácil identificação
- Permite "bater o martelo" visualmente

**Uso:**
- Flag `--validate-separation` gera plot comparativo
- Salvo em `outputs/plots/{song}_separation_validation.png`
- Se a melodia "limpa" (contínua, sem saltos) após separação = voz
- Se mantiver saltos erráticos = ainda captando instrumentos

---

## Verificação

1. **Testar separação isolada:**
   ```bash
   uv run python -c "from vocal_analysis.preprocessing.separation import separate_vocals; ..."
   ```

2. **Validação visual:**
   ```bash
   uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals --validate-separation --limit 1
   ```
   - Olhar os plots gerados em `outputs/plots/*_separation_validation.png`
   - Confirmar que a melodia extraída corresponde à voz (notas esperadas para canto feminino)

3. **Comparar métricas antes/depois:**
   - Confidence média do CREPE deve aumentar
   - HNR deve aumentar (menos ruído de instrumentos)
   - Contorno de f0 mais suave e contínuo

4. **Rodar testes:**
   ```bash
   uv run pytest tests/test_separation.py -v
   ```

5. **Processamento completo:**
   ```bash
   uv run python -m vocal_analysis.preprocessing.process_ademilde --separate-vocals --limit 1
   ```

---

## Notas

- **Fallback:** Se separação falhar, usa áudio original automaticamente
- **GPU:** HTDemucs precisa ~3GB VRAM; funciona em CPU mas é mais lento
- **Não precisa do pacote `demucs`:** torchaudio já inclui o modelo
