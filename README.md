# Uge 10 — Machine Learning

## Introduktion
Der er i dette to-ugers projekt tre forskellige opgaver.
Den første er supervised learning på MNIST-datasættet, hvor man ud fra taggede billeder skal træne en model, der kan genkende og kategorisere tøjtyper.
I den anden opgave skal man lave en model, der kan løse Wordle-opgaver.

## Setup
1. Hent en kopi af repositoriet: `git clone https://github.com/symegac/uge_10.git`
2. Gå ind i mappen: `cd uge_10`
3. Opsæt virtuelt miljø (alt. `python3` eller `python.exe` istf. `py`): `py -m venv .venv`
4. Aktivér virtult miljø (her *bash*): `. .venv/Scripts/activate`
5. Installér påkrævede pakker: `pip install -r requirements.txt`
6. Kør main-scriptet (alt. `python3` eller `python.exe` istf. `py`): `py main.py`

## FashionMNIST (Opgave 1)


## Wordle (Opgave 2)
### Regler
Man har seks forsøg til at gætte et engelsk ord på fem bogstaver. Ca. 2000 ord er gyldige svar, og ca. 13000 ord er gyldige input. For hvert input får man oplysninger, om hvor mange bogstaver stod på korrekte sted i ordet (grønne bogstaver), hvor mange bogstaver stod på et forkert sted men kan findes andetsteds i ordet (gule bogstaver), og hvor mange bogstaver slet ikke findes i ordet (grå/hvide bogstaver). Ud fra disse oplysninger kan man nedsnævre feltet til det korrekte ord.

### Actions og States
#### Actions
I min kode har jeg lavet funktioner for flere forskellige strategier, som det neurale netværk kan vælge imellem:
1. Prædefineret
    * Udvalgt pga. deres indhold af de hyppigste bogstaver i engelske, hvilket giver god information om ordets fonotaktiske struktur.
    * `WordleSolver.default_guess()`
    * `WordleSolver.rand_default_guess()`
2. Logisk
    * Her kigges der der på gættehistorikken, resultathukommelsen og nuværende tilstand, og et ord udvælges ud fra en liste af mulige kandidater.
    * `WordleSolver.logic_guess()`
    * `WordleSolver.rand_logic_guess()`
3. Statistisk
    * Her kigges der kun på resultathukommelsen, og et ord udvælges ud fra dets forekomstfrekvens i engelsksprogede tekster.
    * `WordleSolver.stat_guess()`
    * `WordleSolver.rand_stat_guess()`
4. Brute-force
    * Her kigges der kun på gættehistorikken, og det mest hyppige endnu ikke-gættede ord vælges.
    * `WordleSolver.brute_guess()`
    * `WordleSolver.rand_brute_guess()`
5. Tilfældig
    * Her vælges der fuldstændig tilfældigt et ord ud fra listen af gyldige gæt.
    * `WordleSolver.rand_guess()`

Der er altså i alt 9 forskellige strategier.

#### States
##### Gættehistorik
En hukommelse over allerede brugte ord.

`history: list[str]`

$[$ BLAND, HORSE $]$

<details>
<summary>I kodeform:</summary>

```py
history = ["bland", "horse"]
```

</details>

##### Resultathukommelse
En hukommelse over bogstavers gyldighed på forskellige pladser.

`results: dict[int, dict[str, int]]`

| 0 | 1 | 2 | 3 | 4 |
|:-:|:-:|:-:|:-:|:-:|
|⬜|⬜|⬜|⬜|⬜|
| B | L | A | N | D |
|⬜|🟨|🟩|🟨|⬜|
| H | O | R | S | E |

<details>
<summary>I kodeform:</summary>

```py
results = {
    0: {
        'b': 0,
        'h': 0
    },
    1: {
        'l': 0,
        'o': 1
    },
    2: {
        'a': 0,
        'r': 2
    },
    3: {
        'n': 0,
        's': 0
    },
    4: {
        'd': 0,
        'e': 0
    },
}
```

</details>

##### Nuværende situation
Et overblik over korrekt gættede bogstaver.

`found: dict[int, str]`

| 0 | 1 | 2 | 3 | 4 |
|:-:|:-:|:-:|:-:|:-:|
|⬜|⬜|🟩|⬜|⬜|
| ? | ? | R | ? | ? |

Et overblik over fundne bogstaver, hvis plads endnu ikke kendes.

`lost: list[str]`

$[$ 🟨 O, S $]$

<details>
<summary>I kodeform:</summary>

```py
found = {
    2: 'r'
}

lost = ['o', 's']
```

</details>

### Reward og Punishment
Succes måles ud fra om ordet blev gættet, og hvor mange bogstaver der blev gættet.
**Ordet**: Sejr belønnes eksponentielt, ud fra hvor hurtigt ordet blev gættet. Tab straffes med en fast værdi på to gange ordets længde.
**Bogstaverne**: For hvert korrekt gættet bogstav gives der 2 point, og for hvert tilbageværende ugættet bogstav, tages 1 point fra. Gule bogstaver gør ingen forskel.


| Mål  | Værdi            |
|------|------------------|
| Sejr | $+2^{(6-gæt)}-6$ |
| Tabt | $-2*5$           |
| Grøn | $+2$             |
| Gul  | $\pm 0$          |
| Grå  | $-1$             |

**Reward**
Den vandrette akse viser antal gæt, der skulle til for at finde det rigtige ord.
|   1 |   2 |   3 |   4 |   5 |   6 |
|----:|----:|----:|----:|----:|----:|
|  36 |  20 |  12 |   8 |   6 |   5 |

Værdierne beregnes ud fra formlen $2 * N_{grøn} + 2^{(gæt_{maks} - gæt_{total})} - gæt_{maks}$

**Punishment**
Den lodrette akse er antal grønne bogstaver i sidste runde, mens den vandrette akse er antal gule bogstaver i sidste runde.
|       |   0 |   1 |   2 |   3 |   4 |   5 |
|------:|----:|----:|----:|----:|----:|----:|
| **0** | -15 | -14 | -13 | -12 | -11 | -10 |
| **1** | -12 | -11 | -10 |  -9 |  -8 |   — |
| **2** |  -9 |  -8 |  -7 |  -6 |   — |   — |
| **3** |  -6 |  -5 |  -4 |   — |   — |   — |
| **4** |  -3 |  -2 |   — |   — |   — |   — |

Værdierne beregnes ud fra formlen $2 * N_{grøn} - N_{grå} - 2 * N_{alle}$