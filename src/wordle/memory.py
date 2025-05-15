import os
import json
import base64
import random
import torch
from torch import tensor, Tensor
from typing import NamedTuple
from collections import deque

class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor | None

class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def _check_memory(self, path: str, read: bool = True) -> bool:
        if read and not os.path.exists(path):
            raise FileNotFoundError("Den angivne replay-hukommelsesbank findes ikke.")
        if os.path.splitext(path)[1] != ".qmem":
            raise ValueError("Den angivne fil er ikke en .qmem-fil.")
        return True

    def _read_memory(self, path: str) -> tuple[Transition] | None:
        with open(path, "br") as file:
            if content := file.readline():
                raw: dict[str, list] = json.loads(base64.b64decode(content).decode())
                saved = tuple(Transition(
                    tensor(entry[0], dtype=torch.float32, device="cpu"), # state
                    tensor(entry[1], dtype=torch.int64, device="cpu"), # action
                    tensor(entry[2], dtype=torch.int64, device="cpu"), # reward
                    tensor(entry[3], dtype=torch.float32, device="cpu") if entry[3] is not None else None # next_state
                ) for entry in raw.values())
                return saved

    def save(self, path: str) -> None:
        if self._check_memory(path, read=False):
            try:
                with open(path, "bw") as file:
                    print("BS:", len(self.memory))
                    content = self._read_memory(path)
                    if content is not None:
                        self.memory.extendleft(content)
                        print("AS:", len(self.memory))
                    file.write(base64.b64encode(json.dumps({id: Transition(tr.state.tolist(), tr.action.tolist(), tr.reward.tolist(), None if tr.next_state is None else tr.next_state.tolist()) for id, tr in enumerate(self.memory)}).encode()))
            except:
                raise
            else:
                print("Replay-hukommelsesbank gemt.")

    def load(self, path: str) -> None:
        if self._check_memory(path):
            try:
                print("BL:", len(self.memory))
                content = self._read_memory(path)
                if content is not None:
                    self.memory.extend(content)
                print("AL:", len(self.memory))
                input()
            except:
                raise
                # print("Replay-hukommelsesbank kunne ikke indlæses. Bruger tom bank.")
            else:
                print("Replay-hukommelsesbank indlæst.")

    def clear(self, path: str = '') -> None:
        self.memory.clear()
        if path and self._check_memory(path):
            try:
                os.remove(path)
            except:
                raise
        print("Replay-hukommelsesbank slettet.")

    def __len__(self) -> int:
        return len(self.memory)

    def __repr__(self) -> str:
        return repr(self.memory)

    def __str__(self) -> str:
        return str(self.memory)

def main() -> None:
    rm = ReplayMemory(10000)
    print(len(rm))

if __name__ == "__main__":
    main()
