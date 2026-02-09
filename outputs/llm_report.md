# Bioacoustic Analysis - Ademilde Fonseca

*Report generated with AI assistance (Gemini 2.0 Flash)*

*Multimodal analysis with 13 visualizations*

---

## Vocal Analysis of Ademilde Fonseca: A Challenge to the Fach System

This study analyzes the vocal characteristics of Ademilde Fonseca, a prominent figure in Brazilian Choro, through acoustic analysis and physiological interpretation. Our objective is to critique the traditional "Fach" vocal classification system by examining her laryngeal mechanisms and vocal qualities.

### Vocal Characterization

Ademilde Fonseca exhibits a versatile vocal profile, evidenced by a mean f0 of 395.3 Hz (G4) and a range spanning from F3 to G5. The f0 standard deviation of 110.6 Hz points to a high degree of pitch variability, indicative of complex melodic ornamentation characteristic of Choro. Globally, HNR averages 16.9 dB and CPPS 1.88, suggesting a relatively clear and stable vocal production, despite the presence of Jitter (1.020%) and Shimmer (9.220%), indicators of slight instability in the vocal folds. These instability features, however, could also reflect the stylistic choices inherent in Choro performance. Mean formant values (F1: 658.6 Hz, F2: 1622.8 Hz, F3: 2798.4 Hz, F4: 3802.3 Hz) provide a baseline for timbral qualities, requiring further investigation in relation to specific vowel articulations. Alpha Ratio, H1-H2 and Spectral Tilt point to a generally M1 dominant style, but the interaction with f0 will be further discussed.

### Mechanism Analysis

The distribution of laryngeal mechanisms reveals a prevalence of M1 (chest voice), accounting for 57.3% of analyzed frames, with a mean f0 of 315.9 Hz (D#4). M2 (head voice) represents 42.7% with a mean f0 of 502.1 Hz (B4). The [mechanism_analysis](plots/mechanism_analysis.png) plot illustrates this distribution, highlighting the regions where each mechanism is dominant. Notice that the two mechanisms overlap in the G4 region. Analysis of excerpt [excerpt_apanheite_cavaquinho](plots/excerpt_apanheite_cavaquinho.png) shows this region of overlap in a specific musical excerpt. The [mechanism_clusters](plots/mechanism_clusters.png) plot provides a more granular view via GMM clustering of HNR and f0 values, revealing that some overlap exists between the mechanisms, especially in terms of HNR values. This indicates Fonseca's ability to maintain a relatively consistent vocal quality across different registers.

### VMI Analysis

The Vocal Mechanism Index (VMI) provides a nuanced perspective on register usage, moving beyond the binary M1/M2 classification. The [vmi_scatter](plots/vmi_scatter.png) plot displays f0 against alpha ratio, color-coded by VMI, showing a gradual transition from "Dense M1" (lower f0, negative alpha ratio) to "Light M2" (higher f0, less negative or positive alpha ratio). The [vmi_analysis](plots/vmi_analysis.png) plot's VMI distribution demonstrates a concentration around the "Mix/Passaggio" region, which is supported by temporal view and alpha ratio relationship. This observation highlights Fonseca's adeptness at navigating the passaggio, employing a mixed voice to create a seamless transition between registers. The boxplot by VMI category shows that although the data tends to separate well by VMI range, there is considerable overlap. This is likely a result of rapid changes in f0 and vocal mode that are inherent to Choro.

### Implications for the Fach System

The data challenges the rigid categorization of the Fach system. While a traditional classification might attempt to pigeonhole Fonseca into a specific voice type, her fluid register transitions and frequent use of mixed voice complicate such an assignment. The [vmi_scatter](plots/vmi_scatter.png) plot underscores the limitations of defining vocal categories based solely on pitch range or register, as Fonseca effectively manipulates vocal color and resonance across her entire range. The continuous spectrum of VMI values better reflects her vocal capabilities than discrete Fach categories. Her performance exemplifies the limitations of the Fach system in accounting for vocal flexibility and stylistic nuance, particularly in non-classical genres.

### Limitations

This study acknowledges certain limitations. The use of historical recordings may introduce artifacts that affect acoustic analysis. Additionally, the automated mechanism detection and VMI calculation, while advanced, are not perfect and may introduce some error. Despite this, the observed trends provide valuable insights into Fonseca's vocal production and the challenges of applying traditional vocal classification systems to diverse vocal styles.


---

## Figures

### apanheite_cavaquinho_f0

![f0 Contour - apanheite_cavaquinho](plots/apanheite_cavaquinho_f0.png)

*f0 Contour - apanheite_cavaquinho*

### apanheite_cavaquinho_separation_validation

![apanheite_cavaquinho_separation_validation](plots/apanheite_cavaquinho_separation_validation.png)

*apanheite_cavaquinho_separation_validation*

### brasileirinho_f0

![f0 Contour - brasileirinho](plots/brasileirinho_f0.png)

*f0 Contour - brasileirinho*

### delicado_f0

![f0 Contour - delicado](plots/delicado_f0.png)

*f0 Contour - delicado*

### delicado_separation_validation

![delicado_separation_validation](plots/delicado_separation_validation.png)

*delicado_separation_validation*

### excerpt_apanheite_cavaquinho

![excerpt_apanheite_cavaquinho](plots/excerpt_apanheite_cavaquinho.png)

*excerpt_apanheite_cavaquinho*

### excerpt_brasileirinho

![excerpt_brasileirinho](plots/excerpt_brasileirinho.png)

*excerpt_brasileirinho*

### excerpt_delicado

![excerpt_delicado](plots/excerpt_delicado.png)

*excerpt_delicado*

### mechanism_analysis

![M1/M2 mechanism analysis (histogram, scatter, boxplot, temporal)](plots/mechanism_analysis.png)

*M1/M2 mechanism analysis (histogram, scatter, boxplot, temporal)*

### mechanism_clusters

![GMM clustering of laryngeal mechanisms](plots/mechanism_clusters.png)

*GMM clustering of laryngeal mechanisms*

### vmi_analysis

![VMI analysis (F0 vs Alpha Ratio, distribution, temporal contour)](plots/vmi_analysis.png)

*VMI analysis (F0 vs Alpha Ratio, distribution, temporal contour)*

### vmi_scatter

![Scatter F0 vs Alpha Ratio colored by VMI](plots/vmi_scatter.png)

*Scatter F0 vs Alpha Ratio colored by VMI*

### xgb_mechanism_timeline

![xgb_mechanism_timeline](plots/xgb_mechanism_timeline.png)

*xgb_mechanism_timeline*



---

## Raw Data

```json
{
  "stats": {
    "M1": {
      "count": 11617,
      "f0_mean": 315.86244535594386,
      "f0_std": 47.97150934842754,
      "f0_min": 179.32782,
      "f0_max": 396.98135,
      "hnr_mean": 16.908787590253315,
      "note_mean": "D#4",
      "note_range": "F3 – G4"
    },
    "M2": {
      "count": 8649,
      "f0_mean": 502.0948230512198,
      "f0_std": 75.37779412004222,
      "f0_min": 401.03928,
      "f0_max": 781.1527,
      "hnr_mean": 16.799579622279683,
      "note_mean": "B4",
      "note_range": "G4 – G5"
    }
  },
  "global": {
    "total_voiced_frames": 20266,
    "f0_mean_hz": 395.29998779296875,
    "f0_mean_note": "G4",
    "f0_min_hz": 179.3,
    "f0_max_hz": 781.2,
    "f0_range_notes": "F3 – G5",
    "f0_std_hz": 110.6,
    "hnr_mean_db": 16.9
  }
}
```
