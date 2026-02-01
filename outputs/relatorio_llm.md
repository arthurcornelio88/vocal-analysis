# Análise Bioacústica - Ademilde Fonseca

*Relatório gerado com auxílio de IA (Gemini 2.0 Flash)*

*Análise multimodal com 9 visualizações*

---

## Análise Vocal de Ademilde Fonseca: Uma Crítica ao Sistema Fach

### 1. Caracterização Vocal

A voz de Ademilde Fonseca demonstra uma extensão vocal notável, abrangendo de D2 (74.0 Hz) a G5 (796.5 Hz), com uma frequência fundamental (f0) média de 364.7 Hz (F#4). Essa amplitude, aliada ao desvio padrão de 99.6 Hz na f0, sugere uma cantora com domínio considerável sobre sua emissão e capacidade de executar passagens melódicas complexas. No entanto, a análise da qualidade vocal, através do HNR médio de 8.0 dB e CPPS de 0.33, indica uma voz com relativa presença de ruído e menor definição comparada a cantores líricos. Os valores de instabilidade, com Jitter (ppq5) em 2.117% e Shimmer (apq11) em 15.613%, revelam uma vibração vocal perceptível, o que, no contexto do choro, pode ser interpretado como expressividade e não necessariamente como uma deficiência técnica. A análise dos formantes (F1: 658.6 Hz, F2: 1610.7 Hz, F3: 2770.0 Hz, F4: 3873.5 Hz) oferece indícios sobre o timbre, com F1 relativamente alto, sugerindo uma produção aberta e com ênfase na região central do trato vocal.

### 2. Análise de Mecanismos

A distribuição dos mecanismos laríngeos revela uma predominância do mecanismo M1 (voz de peito), responsável por 69.3% dos frames analisados, com uma f0 média de 311.2 Hz (D#4) e alcance de D2 a G4. O mecanismo M2 (voz de cabeça), por sua vez, corresponde a 30.7% dos frames, com uma f0 média significativamente mais alta, de 485.2 Hz (B4), e extensão vocal de G4 a G5. Essa distribuição, visualizada no gráfico [mechanism_clusters](plots/mechanism_clusters.png), demonstra uma clara separação entre os mecanismos em termos de f0 e HNR, com o M2 exibindo, em geral, um HNR mais elevado (média de 9.6 dB) em comparação ao M1 (média de 7.3 dB). O gráfico [mechanism_analysis](plots/mechanism_analysis.png) detalha essa relação, mostrando que o mecanismo M2 tende a ocupar frequências mais altas e exibir maior "limpeza" vocal. A análise do timeline em [xgb_mechanism_timeline](plots/xgb_mechanism_timeline.png) mostra o uso dinâmico dos mecanismos ao longo das peças, o que enfatiza a agilidade e o controle vocal da cantora.

### 3. Implicações para o Sistema Fach

O sistema Fach, tradicionalmente utilizado para classificar vozes em categorias rígidas baseadas em extensão, tessitura e timbre, enfrenta desafios ao ser aplicado a cantores como Ademilde Fonseca. A análise demonstra uma cantora que transita com fluidez entre os mecanismos M1 e M2, explorando uma extensão vocal ampla e utilizando uma variedade de recursos expressivos que não se encaixam facilmente em uma única categoria. Por exemplo, observamos em [apanhei-te_Cavaquinho_f0](plots/apanhei-te_Cavaquinho_f0.png), [delicado_f0](plots/delicado_f0.png) e [brasileirinho_f0](plots/brasileirinho_f0.png) a rápida alternância entre regiões tonais, que dificilmente se encaixam em uma categoria Fach. A ênfase na expressividade e na interpretação musical, características do choro, muitas vezes se sobrepõe à busca por uma perfeição técnica estéril, o que torna a categorização Fach inadequada para capturar a riqueza e a complexidade da voz de Ademilde Fonseca. As estatísticas globais e por música revelam uma extensão vocal impressionante, porém, a análise detalhada dos mecanismos laríngeos e da instabilidade vocal sugere uma voz que prioriza a expressividade e a agilidade em detrimento da potência e da estabilidade encontradas em vozes líricas.

### 4. Limitações

É importante reconhecer as limitações inerentes à análise de gravações históricas. A qualidade do áudio pode influenciar a precisão das medidas de f0, HNR, Jitter e Shimmer. A ausência de gravações com equipamentos modernos e controle de variáveis (acústica, microfones, etc.) pode introduzir vieses na análise. Adicionalmente, a segmentação automática dos mecanismos laríngeos, embora baseada em algoritmos robustos, pode apresentar erros, especialmente em passagens com transições rápidas entre M1 e M2. Portanto, as conclusões aqui apresentadas devem ser interpretadas com cautela, considerando as limitações dos dados e dos métodos de análise.


---

## Figuras

### apanhei-te_Cavaquinho_f0

![Contorno de f0 - apanhei-te_Cavaquinho](plots/apanhei-te_Cavaquinho_f0.png)

*Contorno de f0 - apanhei-te_Cavaquinho*

### brasileirinho_f0

![Contorno de f0 - brasileirinho](plots/brasileirinho_f0.png)

*Contorno de f0 - brasileirinho*

### delicado_f0

![Contorno de f0 - delicado](plots/delicado_f0.png)

*Contorno de f0 - delicado*

### excerpt_Apanhei-te Cavaquinho

![excerpt_Apanhei-te Cavaquinho](plots/excerpt_Apanhei-te Cavaquinho.png)

*excerpt_Apanhei-te Cavaquinho*

### excerpt_brasileirinho

![excerpt_brasileirinho](plots/excerpt_brasileirinho.png)

*excerpt_brasileirinho*

### excerpt_delicado

![excerpt_delicado](plots/excerpt_delicado.png)

*excerpt_delicado*

### mechanism_analysis

![Análise de mecanismos M1/M2 (histograma, scatter, boxplot, temporal)](plots/mechanism_analysis.png)

*Análise de mecanismos M1/M2 (histograma, scatter, boxplot, temporal)*

### mechanism_clusters

![Clustering GMM dos mecanismos laríngeos](plots/mechanism_clusters.png)

*Clustering GMM dos mecanismos laríngeos*

### xgb_mechanism_timeline

![xgb_mechanism_timeline](plots/xgb_mechanism_timeline.png)

*xgb_mechanism_timeline*



---

## Dados Brutos

```json
{
  "stats": {
    "M1": {
      "count": 3562,
      "f0_mean": 311.2354673464346,
      "f0_std": 49.077085524591595,
      "f0_min": 74.03129,
      "f0_max": 399.9976,
      "hnr_mean": 7.251753429490397,
      "note_mean": "D#4",
      "note_range": "D2 – G4"
    },
    "M2": {
      "count": 1581,
      "f0_mean": 485.1935146932321,
      "f0_std": 76.52496986233103,
      "f0_min": 400.0083,
      "f0_max": 796.48145,
      "hnr_mean": 9.612680907662764,
      "note_mean": "B4",
      "note_range": "G4 – G5"
    }
  },
  "global": {
    "total_voiced_frames": 5143,
    "f0_mean_hz": 364.70001220703125,
    "f0_mean_note": "F#4",
    "f0_min_hz": 74.0,
    "f0_max_hz": 796.5,
    "f0_range_notes": "D2 – G5",
    "f0_std_hz": 99.5999984741211,
    "hnr_mean_db": 8.0
  }
}
```
