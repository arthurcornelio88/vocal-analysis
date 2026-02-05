# Glossário Bioacústico — Guia de Leitura

Documento de apoio para compreender os conceitos e a lógica da análise
descrita em [METODOLOGIA.md](METODOLOGIA.md). Assume conhecimento básico de sons e voz,
mas não de processamento digital de sinal.

---

## Por que essa análise existe?

O sistema tradicional de classificação vocal chamado **Fach** divide
cantores em categorias rígidas (soprano, mezzo, contralto, etc.) baseado
na extensão de notas que conseguem cantar. Essa análise pretende mostrar
que essa divisão é artificial: na prática, cantoras do Choro usam **dois
mecanismos laríngeos** — grave e agudo — de forma fluida dentro da
mesma música. O objetivo é medir isso quantitativamente.

Os dois mecanismos são:

- **M1 (peito):** pregas vocais vibram com mais massa, produz som grave
  e mais "cheia". Pense na forma como você fala normalmente.
- **M2 (cabeça):** pregas vibram com menos massa, produz som agudo e
  mais "fino". Pense no que acontece quando você tenta cantar muito
  alto e a voz "quebra" para um registro diferente.

A transição entre M1 e M2 é o que chamamos de **passaggio**. Medir
quando e como essa transição acontece é o coração da análise.

Abaixo, o contorno temporal de f0 colorido pela predição do classificador:
azul = M1 (peito), coral = M2 (cabeça). Cada música mostra essa
alternância fluida dentro do mesmo trecho:

![Predição XGBoost: M1 vs M2 ao longo do tempo](../outputs/plots/xgb_mechanism_timeline.png)

---

## Os conceitos-chave

### f0 — Frequência Fundamental

**O que é:** A frequência básica de um som. Determina o "tom" que você
ouve — mais grave ou mais agudo.

**Como funciona:** Quando você canta uma nota, suas pregas vocais
vibram N vezes por segundo. Esse N, medido em Hz (hertz), é o f0.
Uma nota A4 (o "lá" do diapasão) tem f0 = 440 Hz.

**Por que importa aqui:** f0 é a feature mais importante para
separar M1 de M2. Notas abaixo de ~400 Hz tendem a ser M1, acima
tendem a ser M2. Mas não é só isso — por isso precisamos das outras
features também.

**No código:** extraído pelo CREPE (uma rede neural), não pelo Praat,
porque CREPE é mais robusto em gravações com ruído de fundo (como as
históricas do Choro).

---

### HNR — Harmonic-to-Noise Ratio

**O que é:** A proporção entre a parte "musical" do som (harmônicos)
e a parte "suja" (ruído). Medido em dB.

**Como pensar:** HNR alto = voz limpa e bem definida. HNR baixo = voz
com mais "sopro" ou ruído. Como a diferença entre uma guitarra com
corda nova (limpa) e uma com corda velha (abafada).

**Por que importa aqui:** M1 tende a ter HNR maior que M2 porque o
fechamento das pregas vocais é mais eficiente no mecanismo de peito.
Então HNR ajuda a confirmar o que o f0 sozinho não consegue: quando
um f0 intermediário é realmente M1 ou M2.

**Atenção com gravações históricas:** O HNR médio neste projeto é
negativo (~-2 dB) porque as gravações do Choro têm muito ruído de
fundo. Os valores absolutos não seguem os parâmetros clínicos
"normais" (HNR > 15 dB = voz saudável). O que importa aqui é a
**diferença relativa** entre M1 e M2 dentro da mesma gravação.

---

### Energia (RMS)

**O que é:** Intensidade média do sinal de áudio em um intervalo de
tempo. Mede "quão alto" está o som.

**Por que importa aqui:** M1 é tipicamente mais energético que M2.
Quando a cantora passa para o registro agudo, a energia costuma cair.
Isso dá uma segunda confirmação além do f0.

---

### Formantes F1, F2, F3, F4

**O que são:** Frequências de ressonância do trato vocal (garganta,
boca, cavidade nasal). Não são determinadas pelas pregas vocais —
são determinadas pela forma como você posiciona a boca, língua e palato mole principalmente.

**Como pensar:** Imagine que as pregas vocais são como um buzzer que
gera um som "base". O trato vocal é como um cano que amplifica
certas frequências desse som. F1, F2, F3, F4 são as frequências que
esse "cano" amplifica.

**Por que importam aqui:** Quando a cantora muda de M1 para M2, o
trato vocal também se reconfigura — especialmente F1 e F2 mudam de
posição. Então formantes ajudam a capturar a transição de registro
de um ângulo diferente do f0.

---

### Jitter e Shimmer

**Jitter:** Instabilidade de período entre ciclos consecutivos de
vibração das pregas vocais. Pensa como um metrônomo que não está
perfeitamente regular.

**Shimmer:** Instabilidade de amplitude entre ciclos consecutivos.
Como um metrônomo que varia na força de cada toque.

**Por que importam aqui:** São valores globais por música (não por
frame), então não entram diretamente no classificador frame a frame.
Mas são úteis para descrever a qualidade vocal geral da cantora nas gravações — especialmente relevante porque são gravações históricas com condições de ruído variáveis.

---

### f0 velocity e f0 acceleration

**O que são:** Velocidade e aceleração da curva de f0 ao longo do
tempo. Se f0 é "onde está a nota agora", velocity é "quão rápido
está mudando", e acceleration é "quão rápido essa mudança está
acelerando".

**Por que importam aqui:** Transições M1→M2 no Choro não são
instantâneas — acontecem através de ornamentos rápidos como
glissandi (deslizamento suave entre notas). Uma transição de registro
produz um padrão de velocity/acceleration muito diferente de um
vibrato normal. O classificador usa isso.

---

## Features espectrais (para VMI)

As features abaixo alimentam o **VMI** (Vocal Mechanism Index), uma
métrica contínua que não depende de frequências fixas como G4.

### Alpha Ratio

**O que é:** Razão entre a energia nos graves (0-1 kHz) e nos agudos
(1-5 kHz), medida em dB.

**Como pensar:** Quanto maior o Alpha Ratio, mais "brilho" e
harmônicos altos a voz tem. M1 tipicamente tem Alpha Ratio mais alta
(mais energia distribuída em harmônicos superiores), enquanto M2
concentra energia na fundamental.

---

### H1-H2 (Diferença de Harmônicos)

**O que é:** Diferença de amplitude entre o 1º harmônico (a
frequência fundamental) e o 2º harmônico (o dobro da fundamental).

**Como pensar:** Pregas vocais que fecham com força produzem uma
onda "quadrada" com muitos harmônicos (H1-H2 baixo). Pregas que
fecham suavemente produzem onda mais "senoidal" com menos
harmônicos (H1-H2 alto).

**Por que importa aqui:** M1 tende a ter H1-H2 baixo (adução firme),
M2 tende a ter H1-H2 alto (adução leve).

**Limitação:** Quando a voz fica muito aguda (f0 > 350 Hz), o 1º
harmônico pode coincidir com o primeiro formante (F1), o que
distorce a medida. Por isso usamos Spectral Tilt junto.

---

### Spectral Tilt (Inclinação Espectral)

**O que é:** A "inclinação" do espectro de frequências. Imagina um
gráfico com frequência no eixo X e intensidade no eixo Y. Spectral
Tilt mede se essa linha cai rápido (negativo) ou fica mais plana
(próximo de zero).

**Como pensar:** Espectro que cai rápido = voz mais "escura" e
"cheia" (M1). Espectro mais plano = voz mais "clara" e "fina" (M2).

**Vantagem:** É mais robusto que H1-H2 em notas agudas porque não
depende de harmônicos específicos.

---

### CPPS (Cepstral Peak Prominence Smoothed)

**O que é:** Uma medida de quão "periódica" e "limpa" é a vibração
das pregas vocais. Tecnicamente, mede a proeminência do pico no
cepstro (uma transformação do espectro).

**Como pensar:** CPPS alto = voz bem definida, pregas vibrando de
forma regular. CPPS baixo = voz soprosa, rouca ou com ruído.

**Importante:** CPPS alto não significa M1 ou M2 especificamente.
Tanto M1 bem produzido quanto M2 bem produzido (voix mixte, por
exemplo) têm CPPS alto. CPPS baixo indica soprosidade ou quebra de
voz, independente do mecanismo.

**Variante per-frame:** O pipeline pode calcular CPPS a cada frame
(10ms) em vez de um valor global por música, permitindo análise
temporal fina.

---

## VMI — Vocal Mechanism Index

**O que é:** Uma métrica contínua de 0 a 1 que indica o "peso" do
mecanismo vocal, sem depender de frequências fixas como G4 (400 Hz).

**Como funciona:** Combina Alpha Ratio, H1-H2, Spectral Tilt e CPPS
em um único número. Quanto mais próximo de 0, mais "M1 denso"
(voz de peito cheia). Quanto mais próximo de 1, mais "M2 ligeiro"
(falsete).

**Escala:**

| VMI | Label | Descrição |
|-----|-------|-----------|
| 0.0-0.2 | M1_DENSO | Voz de peito plena, adução firme |
| 0.2-0.4 | M1_LIGEIRO | M1 mais leve, borda fina |
| 0.4-0.6 | MIX_PASSAGGIO | Zona de passagem, voz mista |
| 0.6-0.8 | M2_REFORCADO | M2 com adução glótica, belting leve |
| 0.8-1.0 | M2_LIGEIRO | Falsete, mecanismo leve |

**Vantagem principal:** Um tenor agudo em M1 e uma soprano grave em
M2 podem cantar a mesma nota (ex: A4 = 440 Hz). O threshold fixo de
G4 classificaria ambos como M2 por estar acima de 400 Hz. O VMI
consegue distingui-los pelas características espectrais.

---

## Como os conceitos se conectam

```
Pregas vocais vibram          →  geram f0 (frequência fundamental)
Forma do trato vocal          →  determina F1-F4 (formantes)
Eficiência do fechamento      →  determina HNR (limpeza do som)
Intensidade da vibração       →  determina Energia (RMS)
Estabilidade entre ciclos     →  determina Jitter e Shimmer
Mudança rápida de f0          →  determina velocity e acceleration

Fechamento glótico (firme/leve)  →  determina H1-H2 e Alpha Ratio
Distribuição espectral           →  determina Spectral Tilt
Periodicidade da vibração        →  determina CPPS
Combinação das features acima    →  gera VMI (0 = M1 denso, 1 = M2 leve)
```

O pipeline oferece **duas abordagens** de classificação:

1. **XGBoost + threshold:** Combina f0, HNR, energia, formantes e
   velocity/acceleration. Funciona bem, mas usa f0 como feature
   dominante — sensível à tessitura do cantor.

2. **VMI (Vocal Mechanism Index):** Combina Alpha Ratio, H1-H2,
   Spectral Tilt e CPPS. Não depende de f0 diretamente — funciona
   para qualquer tessitura.

O scatter abaixo mostra essa separação no espaço f0 × HNR — os dois
clusters que o GMM encontra correspondem aos dois mecanismos:

![Clusters de Mecanismo (GMM)](../outputs/plots/mechanism_clusters.png)

---

## Leitura recomendada

Para entender a análise completa, leia na seguinte ordem:

1. Este documento (glossário)
2. `METODOLOGIA.md` — seções 1, 3, 4, 6 (pipeline e classificação)
3. `outputs/analise_ademilde.md` — resultados concretos
