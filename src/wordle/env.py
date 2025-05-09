import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, OneOf, Sequence, Text, Tuple
from game import WordleSolver
from rewards import Rewards

class Char(Text):
    def __init__(self, charset: str):
        super().__init__(max_length=1, min_length=1, charset=charset)

class Null(Char):
    def __init__(self):
        super().__init__('?')

class Word(Text):
    def __init__(self, length: int, charset: str):
        super().__init__(max_length=length, min_length=length, charset=charset)

class WordleEnv(gym.Env):
    metadata = {"render_modes": "ansi"}

    def __init__(
        self,
        dictionary: dict[str, float] | None = None,
        render: str | None = None,
        **kwargs
    ) -> None:
        self._dictionary = dictionary
        self._settings = kwargs
        self.reset()

        assert render is None or render in self.metadata["render_modes"]
        self.render_mode = render

        self._rewards = Rewards(
            word_length=self._game.word_length,
            allowed_guesses=self._game.allowed_guesses
        )
        self._charmin = min(self._game.valid_chars)
        self._charmax = max(self._game.valid_chars)

        self.observation_space = Dict({
            # tuple[dict[str, int]]
            "result": Tuple(tuple(
                Dict(
                    {char: Discrete(4, start=-1) for char in self._game.valid_chars}
                ) for _ in range(self._game.word_length)
            )),
            # list[str]
            "history": Sequence(
                Word(self._game.word_length, self._game.valid_chars)
            ),
            # list[str]
            "found": Tuple(tuple(
                OneOf((
                    Char(self._game.valid_chars),
                    Null()
                )) for _ in range(self._game.word_length)
            )),
            # list[str]
            "lost": Sequence(
                Char(self._game.valid_chars)
            )
        })

        self.action_space = Discrete(len(self._action_to_strategy))

    def _get_obs(self, as_numbers: bool = True) -> dict[str, tuple[dict[str, int]] | list[str]]:
        # Cursed arrays for at kunne lave om til tensors
        observation = {
            "result": np.array(tuple(np.array(tuple(pos.values())) for pos in self._game.result)) if as_numbers else self._game.result,
            "history": np.array(list(np.array(tuple(ord(char) for char in word)) for word in self._game.history)) if as_numbers else self._game.history,
            "found": np.array(tuple(ord(char) if char != '?' else 0 for char in self._game.found)) if as_numbers else self._game.found,
            "lost": np.array(list(ord(char) for char in self._game.lost)) if as_numbers else self._game.lost
        }
        if as_numbers:
            observation = tuple(np.ravel(observation["result"]).tolist())
        return observation

    def reset(self) -> tuple[dict[str, tuple[dict[str, int]] | list[str]], dict[str, str]]:
        super().reset(seed=None)

        self._game = WordleSolver(
            dictionary=self._dictionary,
            **self._settings
        )
        self._action_to_strategy = {
            0: self._game.default_guess,
            1: self._game.logic_guess,
            2: self._game.wfreq_cand_guess,
            3: self._game.lfreq_cand_guess,
            4: self._game.rand_cand_guess,
            5: self._game.wfreq_dict_guess,
            6: self._game.lfreq_dict_guess,
            7: self._game.rand_dict_guess
        }

        observation = self._get_obs()
        info = {"strategy": self.reset.__name__, "word": self._game.word}
        return observation, info

    def step(self, action) -> tuple[dict[str, tuple[dict[str, int]] | list[str]], int, bool, bool, dict[str, str]]:
        # Genererer gæt ud fra strategi
        strategy = self._action_to_strategy[action]
        guess = strategy()
        self._game.compare(guess)

        if self.render_mode:
            self.render()

        # Info efter gættet er evalueret
        observation = self._get_obs()

        # Hvis ordet er gættet
        terminated = self._game.guess == self._game.word

        # Hvis det maksimale antal gæt er nået (eller overskredet) og ordet ikke er gættet
        truncated = self._game.rounds >= self._game.allowed_guesses and not terminated

        # Belønning fås kun, når runden er slut
        if terminated or truncated:
            reward = self._rewards.get(
                guesses=self._game.rounds,
                green=self._game.green,
                yellow=self._game.yellow
            )
        else:
            # reward = 0
            # Men man kan undersøge, om AI'en bliver bedre til at fastholde grønne bogstaver,
            # hvis man også giver rewards undervejs:
            reward = self._game.green

        info = {
            "strategy": strategy.__name__,
            "word": self._game.word
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> str:
        print(self._game.colorize_guess(self._game.guess))

if __name__ == "__main__":
    from collections import namedtuple
    Step = namedtuple("Step", ("observation", "reward", "terminated", "truncated", "info"))

    # print(fenv.step(0))

    env = WordleEnv(
        # render="ansi",
        default_guesses=["toner", "dashi"]
    )
    for _ in range(50):
        observation, info = env.reset()
        episode_over = False
        # print(env._game.candidates[:3], len(env._game.candidates), env._game.word)
        while not episode_over:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = Step(*env.step(action))
            print(env._action_to_strategy[action].__name__, env._game.candidates[:3], len(env._game.candidates), env._game.word)

            episode_over = terminated or truncated
        # print(observation)
