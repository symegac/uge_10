import os
import re
import enum
import random
from collections import Counter
from string import ascii_lowercase
from math import ceil, floor, log10

# TODO: hent ordlister for andre længder ord

# Parametre som AI skal optimere
DEFAULT_GUESSES: list[str] = ["toner", "dashi"] # fra 0 til 5 elementer # standard: toner dashi
DEFAULT_THRESHOLD: int = 2 # fra 1 til .word_length()+1 # standard: 2
CANDIDATE_THRESHOLD: int = 100 # fra 1 til # standard: 100
# d_g: ["toner", "dashi"], d_t: 2, c_t: 100
# {'all': 3.9719222462203025, '10e-3': 2.5, '10e-4': 2.9607843137254903, '10e-5': 3.468235294117647, '10e-6': 3.9676646706586824, '10e-7': 4.26530612244898, '10e-8': 4.719298245614035}

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
    letter_not_in_word = 0
    letter_in_wrong_pos = 1
    correct_letter = 2

class Wordle:
    letters = tuple(ascii_lowercase)

    def __init__(
        self,
        word: str = None,
        dictionary: str = None,
        *,
        valid_chars: str = None,
    ) -> None:
        # Regler
        self.valid_chars = tuple(set(valid_chars)) if valid_chars is not None else self.letters
        self.dictionary = dictionary if dictionary is not None else load_words()
        self.word = word if word is not None else self.get_random_word()
        self.word_length = len(self.word)
        # Historik
        self.rounds = 0
        self.history = []
        self.guess = ''
        # Resultat
        self.result = {k: {} for k in range(self.word_length)}
        self.found = {}

    def get_random_word(self) -> str:
        return random.choice(list(self.dictionary.keys()))

    def log(self):
        return "->".join([guess for guess in self.history])

    def current(self):
        return [self.found.get(k, '.') for k in range(self.word_length)]

    def guess_word(self):
        guess = input("Indtast et ord på 5 bogstaver").strip()
        if len(guess) != self.word_length:
            guess = guess[:5]

    def compare(self, guess: str) -> None:
        self.guess = guess
        self.history.append(self.guess)
        self.rounds += 1
        for p, l in enumerate(guess):
            # Definer som 0
            if l not in self.word:
                self.result[p].setdefault(l, State.letter_not_in_word)
            # Definer som 2
            elif l == self.word[p]:
                self.result[p].setdefault(l, State.correct_letter)
                self.found[p] = l
            # Definer som 1
            elif l in self.word:
                self.result[p].setdefault(l, State.letter_in_wrong_pos)
        return self.current()

class WordleSolver(Wordle):
    freq = tuple("etaoinshrdlcumwfgypbvkjxqz")

    def __init__(
        self,
        word: str = None,
        dictionary: str = None,
        *,
        valid_chars: str = None,
        frequency: str = None,
        default_guesses: list[str] = [],
        default_threshold: int = 2,
        candidate_threshold: int = 100
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
        self.history = []
        # Strategi
        self.default_guesses = default_guesses
        self.default_threshold = default_threshold
        self.candidate_threshold = candidate_threshold
        # Beregning
        self.pattern = ''
        self.candidates = []
        self.guess = ''
        # Resultat
        self.result = {k: {} for k in range(self.word_length)}
        self.found = {}
        self.lost = []

    def generate_random(self) -> str:
        guess = []
        for n in range(self.word_length):
            guess.append(random.choice(self.valid_chars))
        return ''.join(guess)

    def generate_guess(self) -> str:
        # Angivne standardgæt
        if self.default_guesses and self.rounds < len(self.default_guesses):
            if len(self.lost) < DEFAULT_THRESHOLD:
                return self.default_guesses[self.rounds]
        # Efter ordfrekvens
        candidates = self.find_candidates()
        if len(self.candidates) < CANDIDATE_THRESHOLD:
            for candidate in candidates:
                if candidate not in self.history:
                    return candidate
        # Fallback-gæt efter resultater og bogstavfrekvens
        # TODO: Ud fra fonotaks? (https://en.wikipedia.org/wiki/English_phonology#Phonotactics)
        guess = []
        while ''.join(guess) not in candidates:
            guess.clear()
            for n in range(self.word_length):
                # Indsætter allerede fundne bogstaver
                if self.found.get(n, False):
                    guess.append(self.found[n])
                    continue
                # Indsætter bogstav, der stod på forkert plads i tidligere gæt
                if self.lost and self.result[n].get(self.lost[-1], None) not in (State.letter_not_in_word, State.letter_in_wrong_pos):
                    guess.append(self.lost.pop())
                    continue
                # Indsætter efter bogstavfrekvens
                for i, lt in enumerate(self.inventory):
                    if self.result[i].get(lt, None) not in (State.letter_not_in_word, State.letter_in_wrong_pos):
                        guess.append(lt)
                        break
                # Indsætter tilfældigt bogstav (burde ikke nå hertil)
                if len(guess) <= n:
                    guess.append(random.choice(self.inventory))
            
        return ''.join(guess)

    def compare(self, guess: str) -> None:
        self.guess = guess
        self.history.append(self.guess)
        self.rounds += 1
        for p, l in enumerate(guess):
            # Fjern bogstav fra beholdning
            if l not in self.word and l in self.inventory:
                self.inventory.pop(self.inventory.index(l))
            # Definer som 0
            if l not in self.inventory:
                self.result[p].setdefault(l, State.letter_not_in_word)
            # Definer som 2
            elif l == self.word[p]:
                self.result[p].setdefault(l, State.correct_letter)
                self.found[p] = l
                if l in self.lost:
                    self.lost.pop(self.lost.index(l))
            # Definer som 1
            elif l in self.word:
                self.result[p].setdefault(l, State.letter_in_wrong_pos)
                if l not in self.lost and l not in self.found.values(): #len(self.found) < 4:
                    self.lost.append(l)

    def find_candidates(self):
        pattern = r''
        for n in range(self.word_length):
            # Hvis fem unikke værdier udgør lost&found
            if self.lost and len(self.found.values()) + len(self.lost) == self.word_length:
                pattern += f"{self.found.get(n, f"[{''.join(self.lost)}]")}"
                continue
            # Tjekker gyldige bogstaver på denne placering
            spot_check = tuple(char for char in self.inventory if self.result[n].get(char, None) not in (State.letter_not_in_word, State.letter_in_wrong_pos))
            pattern += f"{self.found.get(n, f"[{''.join(spot_check)}]")}"
        self.pattern = pattern
        candidates = [word for word in self.dictionary if re.match(pattern, word)]
        # Fjerner kandidater, der ikke indeholder kendte tegn m. ukendt placering
        false_hope = [index for index, candidate in enumerate(candidates) for letter in self.lost if letter not in candidate]
        for index, candidate in enumerate(candidates):
            for letter in self.lost:
                if letter not in candidate:
                    false_hope.append(index)
        candidates = [candidate for index, candidate in enumerate(candidates) if index not in false_hope]
        # print(len(self.dictionary), len(candidates))
        # sort regex result first by lost
        # then by frequency
        # weighted_order = tuple(dict.fromkeys([*self.lost, *self.frequency]).keys())
        # sorter = lambda x: tuple(weighted_order.index(c) for c in x)
        occurrence = lambda x: -self.dictionary[x]
        return sorted(candidates, key=occurrence)

    def solve(self, full: bool = False):
        while self.guess != self.word:
            if full:
                msg = f"{self.rounds + 1}. gæt"
                print(len(msg) * '-')
                print(msg)
                print(len(msg) * '-')

            candidates = self.find_candidates()

            if full:
                if len(candidates) < 11:
                    print(f"candidates: {candidates}")
                else:
                    print(f"candidates: {candidates[:9]} ... (+{len(candidates)-8})")

            guess = self.generate_guess()
            self.compare(guess)

            found = self.current()
            lost = ','.join([char for char in self.lost])
            print(f"[ {''.join(found)} ] <{lost}>")
            if full:
                print(f"gæt: '{self.guess}'\nlog: {self.log()}")
        print(f"LØST! Ordet var '{self.word}'. Det tog {self.rounds} gæt at løse.")


# Most frequent letters in English
# ETAOI NSHRD LCUMW FGYPB VKJXQ Z

# 10/10 letters in top ten
# TONER DASHI
# 9/10
# TREND HAILS
# RATES CHINO
# MOIST CRUDE

# Others
#       CHORD
#       ICHOR
# HIDER
# STOAI
# SHINE
# TRODE
# TRODS 
# ASHEN
# STERN (AOIDH)
# NITON SHARD

def freq_exp(frequency: str) -> int | str:
    try:
        freq = float(frequency)
    except:
        raise ValueError("Frekvensen skal kunne laves til en float.")
    
    if freq:
        return ceil(abs(log10(freq)))
    else:
        return "hapax_legomenon"

def counter_avg(counter: Counter) -> float:
    return sum(key * count for key, count in counter.items()) / counter.total()

if __name__ == "__main__":
    # Ord, der er gyldige svar
    training_words = load_words(appeared=True)
    words = list(training_words.keys())
    total = len(words)
    # Ord, der er gyldige inddata
    testing_words = load_words()

    freq = Counter()
    for word in training_words:
        exp = freq_exp(training_words[word])
        freq[exp] += 1

    print(freq)
    input()

    solve_steps = {"all": Counter(), "hapax_legomenon": Counter()}
    solve_steps.update({k: Counter() for k in range(2,13)})

    for i, word in enumerate(words):
        wg = WordleSolver(
            word=word,
            dictionary=training_words,
            default_guesses=DEFAULT_GUESSES,
            default_threshold=DEFAULT_THRESHOLD,
            candidate_threshold=CANDIDATE_THRESHOLD
        )
        exp = freq_exp(training_words[wg.word])
        print(f"{i+1}/{total}")
        while wg.guess != wg.word:
            candidates = wg.find_candidates()
            guess = wg.generate_guess()
            wg.compare(guess)
        solve_steps["all"][wg.rounds] += 1
        solve_steps[exp][wg.rounds] += 1
        # if wg.rounds > 7:
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

# full
# toner dashi [('all', 4.819611470860314), ('10e-2', 2.5), ('10e-3', 2.967479674796748), ('10e-4', 3.4991974317817016), ('10e-5', 4.059955588452998), ('10e-6', 4.461437908496732), ('10e-7', 4.842574257425743), ('10e-8', 5.204118173679499), ('10e-9', 5.517319704713231), ('10e-10', 5.638655462184874), ('10e-11', 5.6), ('10e-14', 5.242857142857143)]





    # def generate_word_list(self) -> list[str]:
    #     words = []
    #     for p1 in self.letters:
    #         for p2 in self.letters:
    #             for p3 in self.letters:
    #                 for p4 in self.letters:
    #                     for p5 in self.letters:
    #                         words.append(''.join((p1, p2, p3, p4, p5)))
    #                         print(words[-1])
    #     return words