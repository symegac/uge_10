import os
import re
import enum
import random
import typing
from collections import Counter
from string import ascii_lowercase
from math import ceil, log10

# TODO: valider type og indhold for parameterinputs, så logikken altid virker
# TODO: hent ordlister for andre længder ord
# TODO: spil mod maskinen og/eller mod AI'en
# TODO: beregn ordene med flest hyppige bogstaver
    # Find først regex [etaoinshrd]{5}
    # Fjern ord med dubletbogstaver i
    # Giv de resterende vægt efter hyppighed, f.eks. 'toner' = 1 + 3 + 5 + 0 + 8 = 17
    # Gentag:
        # Vælg et ord og find derefter alle ord med de fem resterende bogstaver
        # Tilføj disse par som en tuple[str, str] til en liste
    # Herefter kan man så kigge efter højst summeret ordfrekvens
    # Og herefter sortere dette mest frekvente ordpar
        # efter hvilket af de to der har laveste vægtning
        # eller hvliket der har højeste frekvens
    # dette burde være de optimale startgæt
    # man kan måske også gøre hele denne procedure med 3 eller endda 4 ord (hvis der er nok vokaler til det...)
# TODO: find bogstavfrekvens for hver af de fem pladser i ordet


# Parametre som AI skal optimere?
DEFAULT_GUESSES: list[str] = ["toner", "dashi"] # fra 0 til 5 elementer # standard: toner, dashi
DEFAULT_THRESHOLD: int = 2 # fra 1 til .word_length()+1 # standard: 3
# CANDIDATE_THRESHOLD: int = 100 # fra 1 til # standard: 100

def load_words(
    data_dir: str = "data",
    filename: str = "wordle_words.csv",
    appeared: bool = False
) -> dict[str, float]:
    with open(os.path.join(data_dir, filename), 'r', encoding="utf-8") as file:
        file.readline()
        if appeared:
            dictionary = {line.split(',')[0]: float(line.split(',')[1]) for line in file.readlines() if line.split(',')[2].strip()}
        else:
            dictionary = {line.split(',')[0]: float(line.split(',')[1]) for line in file.readlines()}
    return dictionary

class State(enum.IntEnum):
    no_info_yet = -1
    letter_not_in_word = 0
    letter_in_wrong_pos = 1
    correct_letter = 2

class Wordle:
    letters = tuple(ascii_lowercase)

    def __init__(
        self,
        word: str | None = None,
        dictionary: dict[str, float] | None = None,
        allowed_guesses: int = 6,
        *,
        valid_chars: str | None = None,
    ) -> None:
        # Regler
        self.valid_chars = tuple(set(valid_chars)) if valid_chars is not None else self.letters
        self.dictionary = dictionary if dictionary is not None else load_words()
        self.word = word if word is not None else self.get_random_word()
        self.word_length = len(self.word)
        # Historik
        self.rounds = 0
        self.allowed_guesses = allowed_guesses
        self.history = []
        self.guess = ''
        # Resultat
        self.result = tuple({char: State.no_info_yet for char in self.valid_chars} for _ in range(self.word_length))
        self.found = ['?'] * self.word_length

    def get_random_word(self) -> str:
        return random.choice(list(self.dictionary.keys()))

    def log(self, colorize: bool = True) -> str:
        return '\n'.join([self.colorize_guess(guess) if colorize else guess for guess in self.history])

    def colorize_guess(self, guess: str) -> str:
        colorized = ''
        for pos, letter in enumerate(guess):
            if letter not in self.valid_chars:
                colorized += f"\033[30m{letter}\u001B[0m"
                continue
            status = self.result[pos][letter]
            # Grøn hvis korrekt
            if status == State.correct_letter:
                colorized += f"\033[32m{letter}\u001B[0m"
            # Gul hvis halvt korrekt
            elif status == State.letter_in_wrong_pos:
                colorized += f"\033[33m{letter}\u001B[0m"
            else:
                colorized += letter
        return colorized

    def guess_word(self) -> str | bool:
        if self.rounds >= self.allowed_guesses:
            return False
        guess = input("Indtast et gæt på 5 bogstaver: ").strip()
        if len(guess) != self.word_length:
            guess = guess[:5]
        if all(char in self.valid_chars for char in guess):
            return self.compare(guess)

    def compare(self, guess: str) -> str:
        self.guess = guess
        self.history.append(self.guess)
        self.rounds += 1
        for pos, char in enumerate(guess):
            # Definer som 0
            if char not in self.word:
                self.result[pos][char] = State.letter_not_in_word
            # Definer som 2
            elif char == self.word[pos]:
                self.result[pos][char] = State.correct_letter
                self.found[pos] = char
            # Definer som 1
            elif char in self.word:
                self.result[pos][char] = State.letter_in_wrong_pos
        return self.log()

    def play(self) -> None:
        while self.guess != self.word:
            result = self.guess_word()
            if result:
                print(result)
            else:
                input(f"GAME OVER! Ordet var {self.word}.")
                break
        else:
            input(f"TILLYKKE! Du gættede ordet {self.word} på {self.rounds} gæt.")

class WordleSolver(Wordle):
    freq = tuple("etaoinshrdlcumwfgypbvkjxqz")

    def __init__(
        self,
        word: str | None = None,
        dictionary: dict[str, float] | None = None,
        allowed_guesses: int = 6,
        *,
        valid_chars: str | None = None,
        frequency: str | None = None,
        default_guesses: list[str] = [],
        default_threshold: int = 3,
        # candidate_threshold: int = 100
    ):
        # Regler
        self.valid_chars = tuple(set(valid_chars)) if valid_chars is not None else self.letters
        self.dictionary = dictionary if dictionary is not None else load_words()
        self.word = word if word is not None else self.get_random_word()
        self.word_length = len(self.word)
        self.frequency = tuple(frequency) if frequency is not None else self.freq
        self.inventory = sorted(self.valid_chars, key=lambda x: self.frequency.index(x) if x in self.frequency else ord(x))
        # Historik
        self.rounds = 0
        self.allowed_guesses = allowed_guesses
        self.history = []
        # Strategi
        self.default_guesses = default_guesses
        self.default_threshold = default_threshold
        # self.candidate_threshold = candidate_threshold
        # Resultat
        self.result = tuple({char: State.no_info_yet for char in self.valid_chars} for _ in range(self.word_length))
        self.found = ['?'] * self.word_length
        self.green = 0
        self.yellow = 0
        self.lost = []
        # Beregning
        self.pattern = ''
        self.candidates = self._initial_candidates()
        self.guess = ''

    def log(self, colorize: bool = True) -> str:
        return '->'.join([self.colorize_guess(guess) if colorize else guess for guess in self.history])

    def generate_random(self) -> str:
        guess = []
        for _ in range(self.word_length):
            guess.append(random.choice(self.valid_chars))
        return ''.join(guess)

    # action 0
    def default_guess(self) -> str:
        """Gætter med foruddefinerede ord."""
        if self.default_guesses:
            if self.rounds < len(self.default_guesses):
                return self.default_guesses[self.rounds]
            else:
                return self.rand_default_guess(referred=True)
        else:
            return self.rand_guess()

    # action 1
    def rand_default_guess(self, referred: bool = False) -> str:
        if self.default_guesses or referred:
            return random.choice(self.default_guesses)
        else:
            return self.rand_guess()

    # action 2
    def logic_guess(self) -> str:
        """Gætter med det mest frekvente ord fra kandidatlisten, som ikke allerede er blevet brugt,
        og som kan lade sig gøre ud fra forrige resultater."""
        self._logic_guess()
        # Find første kandidat, der ikke allerede er blevet gættet
        for candidate in self.candidates:
            if candidate not in self.history:
                return candidate
        else:
            return self.rand_logic_guess(referred=True)

    # action 3
    def rand_logic_guess(self, referred: bool = False) -> str:
        """Gætter med et tilfældigt ord fra kandidatlisten, som ikke allerede er blevet brugt,
        og som kan lade sig gøre ud fra forrige resultater."""
        if not referred:
            self._logic_guess()
        return random.choice(self.candidates)

    # action 4
    def stat_guess(self) -> str:
        """Gætter med det mest frekvente ord fra kandidatlisten,
        som kan lade sig gøre ud fra forrige resultater."""
        self.candidates = self._sort_candidate_list(self._generate_candidate_list(self._find_candidates()))
        if self.candidates:
            return self.candidates[0]
        else:
            return self.rand_stat_guess(referred=True)

    # def lfreq_cand_guess(self) -> str:
    #     """Gætter med ordet med mest frekvente bogstaver fra kandidatlisten."""
    #     return self.logic_guess()

    # action 5
    def rand_stat_guess(self, referred: bool = False) -> str:
        """Gætter med et tilfældigt ord fra kandidatlisten,
        som kan lade sig gøre ud fra forrige resultater."""
        if not referred:
            self.candidates = list(self._generate_candidate_list(self._find_candidates()))
        return random.choice(self.candidates)

    # action 6
    def brute_guess(self) -> str:
        """Gætter med det mest frekvente ord fra listen over gyldige inputs,
        så længe det ikke er blevet brugt før."""
        self.candidates = sorted(self.dictionary.keys(), key=lambda x: -self.dictionary[x])
        for word in self.candidates:
            if word not in self.history:
                return word
        else:
            return self.rand_brute_guess()

    # def lfreq_brute_guess(self) -> str:
    #     """Gætter med ordet med mest frekvente bogstaver fra listen over gyldige inputs."""
    #     return self.logic_guess()

    # action 7
    def rand_brute_guess(self) -> str:
        """Gætter med et tilfældigt ord fra listen over gyldige inputs,
        så længe det ikke er blevet brugt før."""
        dictionary = tuple(self.dictionary.keys())
        while (guess := random.choice(dictionary)) in self.history:
            guess = random.choice(dictionary)
        return guess

    # action 8
    def rand_guess(self) -> str:
        """Gætter med et tilfældigt ord fra listen over gyldige inputs."""
        dictionary = tuple(self.dictionary.keys())
        return random.choice(dictionary)

    def generate_guess(self) -> str:
        # Angivne standardgæt
        if self.default_guesses and self.rounds < len(self.default_guesses):
            if len(self.lost) < self.default_threshold: # or len(self.candidates) > self.candidate_threshold:
                return self.default_guesses[self.rounds]
        # Efter ordfrekvens
        return self.logic_guess()

    def _lost_and_found(self) -> typing.Pattern:
        # Hvis fem unikke værdier udgør lost&found
        return re.compile(''.join(char if char != '?' else f"[{''.join(self.lost)}]" for char in self.found))

    def _find_candidates(self) -> typing.Pattern:
        pattern = ''
        for pos in range(self.word_length):
            if (status := self.found[pos]) != '?':
                pattern += status
                continue
            # Tjekker gyldige bogstaver på denne placering
            spot_check = (char for char in self.inventory if self.result[pos][char] not in (State.letter_not_in_word, State.letter_in_wrong_pos))
            pattern += f"[{''.join(spot_check)}]"
        return re.compile(pattern)

    def _find_basic_candidates(self) -> typing.Pattern:
        return re.compile(f"[{''.join(char for char in self.inventory)}]+")

    def _generate_candidate_list(self, pattern: typing.Pattern) -> set[str]:
        return {word for word in self.dictionary if pattern.fullmatch(word)}

    def _check_candidate_list(self, candidates: typing.Iterable) -> set[str]:
        """Fjerner kandidater, der ikke indeholder kendte tegn m. ukendt placering"""
        candidates = tuple(candidates)
        false_hope = {index for index, candidate in enumerate(candidates) for letter in self.lost if letter not in candidate}
        return {candidate for index, candidate in enumerate(candidates) if index not in false_hope}

    def _sort_candidate_list(self, candidates: typing.Iterable) -> list[str]:
        """Sorterer kandidatliste efter frekvens"""
        return sorted(candidates, key=lambda x: -self.dictionary[x])

    def _initial_candidates(self) -> list[str]:
        return self._sort_candidate_list(self._generate_candidate_list(self._find_basic_candidates()))

    def _logic_guess(self) -> None:
        # Kigger på nuværende tilstand - er der nok grønne og gule til at indsnævre antallet af kandidater?
        if self.lost and self.green + len(self.lost) == self.word_length:
            pattern = self._lost_and_found()
        else:
            pattern = self._find_candidates()
        candidate_list = self._generate_candidate_list(pattern)
        # Se, om der kan fjernes kandidater, hvis de ikke 
        if self.lost:
            candidate_list = self._check_candidate_list(candidate_list)
        # Sortér efter frekvens
        self.candidates = self._sort_candidate_list(candidate_list)

    def compare(self, guess: str) -> None:
        # Opdatér historik
        self.guess = guess
        self.history.append(self.guess)
        self.rounds += 1
        # Analysér gæt
        green = yellow = 0
        for pos, char in enumerate(guess):
            # Definer som <0>
            if char not in self.word:
                self.result[pos][char] = State.letter_not_in_word
                if char in self.inventory:
                    self.inventory.pop(self.inventory.index(char))
            # Definer som <2>
            elif char == self.word[pos]:
                self.result[pos][char] = State.correct_letter
                green += 1
                self.found[pos] = char
                if char in self.lost:
                    self.lost.pop(self.lost.index(char))
            # Definer som <1>
            elif char in self.word:
                self.result[pos][char] = State.letter_in_wrong_pos
                yellow += 1
                if char not in self.lost and char not in self.found:
                    self.lost.append(char)
        self.green, self.yellow = green, yellow

    def solve(self, full: bool = False, colorize: bool = True):
        while self.guess != self.word:
            if full:
                msg = f"{self.rounds + 1}. gæt"
                print(len(msg) * '-')
                print(msg)
                print(len(msg) * '-')

                candidates = self.candidates
                if len(candidates) < 11:
                    print(f"candidates: {candidates}")
                else:
                    print(f"candidates: {candidates[:9]} ... (+{len(candidates)-8})")

            guess = self.generate_guess()
            self.compare(guess)

            if full:
                lost = ','.join([f"\033[33m{char}\u001B[0m" if colorize else char for char in self.lost])
                print(f"[ {self.found if not colorize else self.colorize_guess(self.found)} ] <{lost}>")
                print(self.log(colorize=colorize))
            else:
                print(self.colorize_guess(guess) if colorize else guess)
        # self.green = len(tuple(char for char in self.found if char != '?'))
        if full:
            print(f"LØST! Ordet var '{self.word}'. Det tog {self.rounds} gæt at løse.")

def freq_exp(frequency: str) -> int | str:
    try:
        freq = float(frequency)
    except:
        raise ValueError("Frekvensen skal kunne laves til en float.")

    if freq:
        return ceil(abs(log10(freq)))
    else:
        return "hapax_legomenon"

def freq_distribution(word_list: dict[str, float]) -> Counter:
    return Counter(freq_exp(word_list[word]) for word in word_list)

def counter_avg(counter: Counter) -> float:
    return sum(key * count for key, count in counter.items()) / counter.total()

if __name__ == "__main__":
    # Ord, der er gyldige svar
    training_words = load_words(appeared=True)
    words = list(training_words.keys())
    total = len(words)
    # Ord, der er gyldige inddata
    testing_words = load_words()
    testwords = list(testing_words.keys())
    testtotal = len(testwords)

    # training_freq = freq_distribution(training_words)
    # testing_freq = freq_distribution(testing_words)

    # print(training_freq, testing_freq)
    # input()

    solve_steps = {"all": Counter()}
    solve_steps.update({k: Counter() for k in range(3,13)})
    solve_steps.update({"hapax_legomenon": Counter()})

    wm = WordleSolver()
    print(wm.wfreq_dict_guess())
    input()

    for i, word in enumerate(words):
        wg = WordleSolver(
            word=word,
            dictionary=testing_words,
            default_guesses=DEFAULT_GUESSES,
            default_threshold=DEFAULT_THRESHOLD,
            # candidate_threshold=CANDIDATE_THRESHOLD
        )
        exp = freq_exp(testing_words[wg.word])
        print(f"{i+1}/{total}")
        wg.solve(full=False, colorize=True)

        solve_steps["all"][wg.rounds] += 1
        solve_steps[exp][wg.rounds] += 1
        # if wg.rounds > 6:
        #     input(wg.history)

    stats = {}
    for counter in solve_steps:
        if len(solve_steps[counter]):
            if isinstance(counter, str):
                name = counter
            else:
                name = f"10e-{counter}"
            stats[name] = counter_avg(solve_steps[counter])
    print(stats)


# toner dashi  [('all', 4.3), ('10e-2', 2.0), ('10e-3', 2.6666666666666665), ('10e-4', 3.1538461538461537), ('10e-5', 3.7857142857142856), ('10e-6', 4.078947368421052), ('10e-7', 4.423076923076923), ('10e-8', 4.744186046511628), ('10e-9', 4.56), ('10e-10', 4.818181818181818)]
# dashi toner [('all', 4.38), ('10e-2', 3.0), ('10e-3', 2.6666666666666665), ('10e-4', 3.3076923076923075), ('10e-5', 3.857142857142857), ('10e-6', 4.052631578947368), ('10e-7', 4.480769230769231), ('10e-8', 4.930232558139535), ('10e-9', 4.68), ('10e-10', 4.7272727272727275)]
# trend hails [('all', 4.365), ('10e-2', 3.0), ('10e-3', 3.0), ('10e-4', 3.4615384615384617), ('10e-5', 3.9285714285714284), ('10e-6', 4.078947368421052), ('10e-7', 4.403846153846154), ('10e-8', 4.627906976744186), ('10e-9', 4.8), ('10e-10', 5.2727272727272725)]
# no defaults [('all', 4.685), ('10e-2', 2.0), ('10e-3', 3.0), ('10e-4', 3.6153846153846154), ('10e-5', 4.071428571428571), ('10e-6', 4.315789473684211), ('10e-7', 4.826923076923077), ('10e-8', 5.162790697674419), ('10e-9', 5.36), ('10e-10', 4.636363636363637)]


### Training data (N = 2315) ###
# d_g: ["toner", "dashi"], d_t: 1, c_t: 100
# {'all': 4.000863930885529, '10e-3': 2.3333333333333335, '10e-4': 2.7745098039215685, '10e-5': 3.48, '10e-6': 3.986826347305389, '10e-7': 4.337334933973589, '10e-8': 4.771929824561403}
# d_g: ["toner", "dashi"], d_t: 2, c_t: 100
# {'all': 3.876889848812095, '10e-3': 2.5, '10e-4': 2.9607843137254903, '10e-5': 3.428235294117647, '10e-6': 3.9005988023952094, '10e-7': 4.133253301320528, '10e-8': 4.394736842105263}
# {'all': 3.9719222462203025, '10e-3': 2.5, '10e-4': 2.9607843137254903, '10e-5': 3.468235294117647, '10e-6': 3.9676646706586824, '10e-7': 4.26530612244898, '10e-8': 4.719298245614035}
# d_g: ["toner", "dashi"], d_t: 3, c_t: 100
# {'all': 3.968466522678186, '10e-3': 3.0, '10e-4': 3.0686274509803924, '10e-5': 3.456470588235294, '10e-6': 3.9580838323353293, '10e-7': 4.250900360144057, '10e-8': 4.745614035087719}
# d_g: ["toner", "dashi", "clump"], d_t: 4, c_t: 100
# {'all': 4.210799136069115, '10e-3': 4.0, '10e-4': 3.872549019607843, '10e-5': 3.9411764705882355, '10e-6': 4.198802395209581, '10e-7': 4.342136854741897, '10e-8': 4.657894736842105}

### Testing data (N = 12972) ###

# tuples
# {'all': 4.33, '10e-3': 3.0, '10e-4': 3.0, '10e-5': 3.3076923076923075, '10e-6': 3.9285714285714284, '10e-7': 4.052631578947368, '10e-8': 4.384615384615385, '10e-9': 4.837209302325581, '10e-10': 4.48, '10e-11': 4.909090909090909}
# generators
# {'all': 5.585, '10e-3': 3.0, '10e-4': 3.3333333333333335, '10e-5': 4.538461538461538, '10e-6': 4.642857142857143, '10e-7': 5.447368421052632, '10e-8': 5.903846153846154, '10e-9': 5.8604651162790695, '10e-10': 5.84, '10e-11': 6.181818181818182

# Tidligere
# toner dashi {'all': 4.819611470860314, '10e-3': 2.5, '10e-4': 2.967479674796748, '10e-5': 3.4991974317817016, '10e-6': 4.059955588452998, '10e-7': 4.461437908496732, '10e-8': 4.842574257425743, '10e-9': 5.204118173679499, '10e-10': 5.517319704713231, '10e-11': 5.638655462184874, '10e-12': 5.6, 'hapax_legomenon': 5.242857142857143}
# toner dashi {'all': 4.775439407955597, '10e-3': 3.0, '10e-4': 3.065040650406504, '10e-5': 3.491171749598716, '10e-6': 4.05699481865285, '10e-7': 4.433115468409586, '10e-8': 4.786798679867987, '10e-9': 5.145926589077887, '10e-10': 5.45144804088586, '10e-11': 5.546218487394958, '10e-12': 5.6, 'hapax_legomenon': 5.228571428571429}