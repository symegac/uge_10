from tabulate import tabulate

class Rewards:
    def __init__(
        self,
        word_length: int = 5,
        allowed_guesses: int = 6
    ):
        self.word_length = word_length
        self.allowed_guesses = allowed_guesses

    def get(
        self,
        guesses: int | None = None,
        green: int | None = None,
        yellow: int | None = None,
    ) -> int | None:
        if guesses is None:
            guesses = self.allowed_guesses
        if green is None:
            if yellow is None:
                green = self.word_length
            else:
                green = 0
        if yellow is None:
            yellow = 0

        if not all(isinstance(param, int) for param in (guesses, green, yellow)):
            raise TypeError("Alle inddata skal være heltal.")

        if guesses < 1 or guesses > self.allowed_guesses:
            print(f"Antal gæt kan kun være heltal fra 1 til {self.allowed_guesses}, ikke {guesses}.") # ValueError
            guesses = 6

        # Hvis alle bogstaver er grønne og antal gæt er inden for det tilladte,
        # så beregnes belønning
        if green == self.word_length:
            return self._calculate_reward(guesses)

        # Straf kan kun uddeles, hvis det maksimale antal gæt er brugt
        if guesses < self.allowed_guesses:
            raise ValueError(f"Man kan ikke måle strafpoint for et spil, der endnu ikke er slut. Antal gæt ({guesses}) er under maksimum ({self.allowed_guesses}), men wordlen er ikke løst.")

        # Mængden af grønne og gule bogstaver skal passe til ordlængden
        if any((
            green + yellow > self.word_length,
            yellow > self.word_length,
            green > self.word_length
        )):
            raise ValueError(f"Den angivne mængde af gule ({yellow}) og grønne ({green}) bogstaver må hverken hver især eller tilsammen overstige ordlængden ({self.word_length}).")
        if green < 0 or yellow < 0:
            raise ValueError("De angivne mængder for gule og grønne bogstaver skal være positive heltal.")

        return self._calculate_punishment(green, yellow)

    def _calculate_reward(
        self,
        guesses: int,
    ) -> int:
        return 2 * self.word_length + 2**(self.allowed_guesses - guesses) - self.allowed_guesses

    def _calculate_punishment(
        self,
        green: int,
        yellow: int,
    ) -> int:
        gray = self.word_length - green - yellow
        return 2 * green - gray - 2 * self.word_length

    def generate_reward_table(
        self,
        format: bool = False,
        **kwargs
    ) -> dict[int, int] | str:
        table = {guesses: self._calculate_reward(guesses) for guesses in range(1, self.allowed_guesses + 1)}
        if format:
            table = tabulate((table.keys(), table.values()), **kwargs)
        return table

    def generate_punishment_table(
        self,
        format: bool = False,
        **kwargs
    ) -> list[list[int | None]] | str:
        table = []
        for green in range(self.word_length):
            row = []
            for yellow in range(self.word_length + 1):
                if (green + yellow) > self.word_length:
                    row.append(None)
                    continue
                punishment = self._calculate_punishment(green, yellow)
                row.append(punishment)
            table.append(row)

        if format:
            table = tabulate(table, **kwargs)
        return table

if __name__ == "__main__":
    rewards = Rewards(word_length=5, allowed_guesses=6)
    markdown_fmt = {
        "headers":    "keys",
        "showindex":  True,
        "stralign":   "right",
        "missingval": '—',
        "tablefmt":   "pipe"
    }

    rtable = rewards.generate_reward_table(format=True, tablefmt="pipe", headers="firstrow")
    ptable = rewards.generate_punishment_table(format=True, **markdown_fmt)
    print(rtable, ptable, sep='\n')
    print(rewards.get(3, 5, 0)) # 3 gæt, alt korrekt        =  12
    print(rewards.get(6, 2, 1)) # 6 gæt, 2 grønne, 1 gul    =  -8
    print(rewards.get())        # 6 gæt, alt korrekt        =   5
    print(rewards.get(2))       # 2 gæt, alt korrekt        =  20
    print(rewards.get(6, 4))    # 6 gæt, 4 grønne           =  -3
    print(rewards.get(yellow=4))# 6 gæt, 0 grønne, 4 gule   = -11
