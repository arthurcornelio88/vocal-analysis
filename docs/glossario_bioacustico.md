# Glossário Bioacústico — Guia de Leitura

Documento de apoio para compreender os conceitos e a lógica da análise
descrita em `METODOLOGIA.md`. Assume conhecimento básico de sons e voz,
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
são determinadas pela forma como você posiciona a boca e a língua.

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
Mas são úteis para descrever a qualidade vocal geral da cantora nasgravaçoes — especialmente relevante porque são gravações históricas com condições de ruído variáveis.

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

## Como os conceitos se conectam

```
Pregas vocais vibram          →  geram f0 (frequência fundamental)
Forma do trato vocal          →  determina F1-F4 (formantes)
Eficiência do fechamento      →  determina HNR (limpeza do som)
Intensidade da vibração       →  determina Energia (RMS)
Estabilidade entre ciclos     →  determina Jitter e Shimmer
Mudança rápida de f0          →  determina velocity e acceleration
```

O classificador XGBoost combina todas essas features para decidir,
frame por frame (a cada 10ms), se a cantora está em M1 ou M2.
Features mais fiáveis (f0, HNR, energia) formam a base. Features
complementares (formantes, velocity) resolvem os casos ambíguos.

---

## Leitura recomendada

Para entender a análise completa, leia na seguinte ordem:

1. Este documento (glossário)
2. `METODOLOGIA.md` — seções 1, 3, 4, 6 (pipeline e classificação)
3. `outputs/analise_ademilde.md` — resultados concretos
