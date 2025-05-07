import gymnasium as gym
import os
import math
import json
import random
from wordle.wordle import WordleSolver
from collections import namedtuple, deque
from itertools import count
from network import DQN

import torch
from torch import nn, optim, accelerator
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

n_observations = []
n_actions = (
    WordleSolver.default_guess,
    WordleSolver.rand_cand_guess,
    WordleSolver.wfreq_cand_guess,
    WordleSolver.lfreq_cand_guess,
    WordleSolver.rand_dict_guess,
    WordleSolver.wfreq_dict_guess,
    WordleSolver.lfreq_dict_guess,
)

def main() -> None:
    # Config
    config: dict[str, dict[str, int | float]] = json.load(open("config.json"))
    batch_size, gamma, eps_start, eps_end, eps_decay, tau, learning_rate, data_dir, model_file = config["wordle"].values()
    model_path = os.path.join(data_dir, "output", model_file)

    device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(
        params=policy_net.parameters(),
        lr=learning_rate,
        amsgrad=True
    )
    memory = ReplayMemory(10000)

    steps_done = 0

    episode_durations = []

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimerer modellen
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    print(model_path)

if __name__ == "__main__":
    main()