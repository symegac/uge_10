import os
import math
import json
import random
import numpy as np
from collections import namedtuple, deque, defaultdict
from itertools import count

import torch
from torch import nn, optim, accelerator
import torchvision
import tensordict
from tensordict.tensordict import TensorDict
from network import DQN
from env import WordleEnv
from gymnasium import Env

import matplotlib.pyplot as plt

tensordict.set_list_to_stack(True).set()
Step = namedtuple("Step", ("observation", "reward", "terminated", "truncated", "info"))
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

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

class Model:
    def __init__(
        self,
        config: dict[str, float | int | str | dict[str, float | int]],
        observation_size: int,
        action_size: int
    ) -> None:
        # Config
        self.batch_size, self.learning_rate, *_, self.dqn = config.values()
        self.gamma, self.eps_start, self.eps_end, self.eps_decay, self.tau = self.dqn.values()
        self.observation_size = observation_size
        self.action_size = action_size
        self.steps_done = 0
        self.setup()

    def setup(self) -> None:
        """Opsætter modellen"""
        # Device
        self.device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"
        # Neurale netværk
        self.policy_net = DQN(self.observation_size, self.action_size).to(self.device)
        self.target_net = DQN(self.observation_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Optimizer
        self.optimizer = optim.AdamW(
            params=self.policy_net.parameters(),
            lr=self.learning_rate,
            amsgrad=True
        )
        # Loss-funktion
        self.loss_fn = nn.SmoothL1Loss()
        # Hukommelse
        self.memory = ReplayMemory(10000)

    def select_action(
        self,
        state: torch.Tensor,
        env: Env = WordleEnv()
    ):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([env.action_space.sample()], device=self.device, dtype=torch.float32)

    def optimize(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Optimerer modellen
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

def training_loop(
    env: Env,
    model: Model,
    episodes: int = 50
) -> list[tuple[int, int]]:
    stats = []
    wins = 0
    for n in range(episodes):
        # Opstart environment og få start-state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
        tstats = []
        # Kør derefter gennem resten af states
        for t in count():
            action = model.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=model.device)

            if terminated or truncated:
                next_state = None
                if terminated:
                    wins += 1
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=model.device).unsqueeze(0)

            # Gem overgangen
            model.memory.push(state, action, next_state, reward)
            tstats.append((n, t, reward.tolist()[0], wins))

            # Optimisér policy nn
            try:
                model.optimize()
            except:
                if next_state is None:
                    break

            # Opdatér target nn
            target_net_state_dict = model.target_net.state_dict()
            policy_net_state_dict = model.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * model.tau + target_net_state_dict[key] * (1 - model.tau)
            model.target_net.load_state_dict(target_net_state_dict)

            # Videre
            if next_state is None:
                break
            state = next_state

        print(f"Episode {n} -- belønning: {reward}, gættet: {terminated}, info: {info}")
        stats.append(tstats)
    print("Træning afsluttet")
    return stats

def testing_loop() -> None:
    pass

def main() -> None:
    # Config
    config: dict[str, dict[str, int | float]] = json.load(open("config.json"))
    data_dir = config["wordle"]["data_dir"]
    model_file = config["wordle"]["model_file"]
    model_path = os.path.join(data_dir, "output", model_file)

    env = WordleEnv(
        # render="ansi",
        default_guesses=["toner", "dashi"]
    )
    state, info = env.reset()
    observation_size = len(state)
    action_size = env.action_space.n
    model = Model(
        config=config["wordle"],
        observation_size=observation_size,
        action_size=action_size
    )

    stats = training_loop(env, model, episodes=250)

    wins = []
    rewards = []
    for n, data in enumerate(stats):
        wins.append(data[-1][-1])
        for t, step in enumerate(data):
            rewards.append(step[2])
    # print(wins, rewards, sep='\n')
    total_rewards = list(np.cumsum(rewards))

    # plt.plot(rewards)
    plt.plot(wins)
    # plt.plot(total_rewards)
    plt.show()


if __name__ == "__main__":
    main()






    # def select_action_b(
    #     self,
    #     state,
    #     action_size,
    #     Q,
    # ):
    #     probabilities = np.ones(action_size, dtype = float) * self.eps_start / action_size
    #     sample = random.random()
    #     eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
    #     self.steps_done += 1
    #     if sample > eps_threshold:
    #         best_action = Q[state].argmax()
    #         probabilities[best_action] += (1.0 - self.eps_start)
    #     return probabilities

    # def training_loop_b(
    #     env: Env,
    #     model: Model,
    #     episodes: int = 50
    # ):
    #     Q = defaultdict(lambda: np.zeros(8))
    #     policy = model.select_action_b
    #     for n in range(episodes):
    #         state, info = env.reset()
    #         for t in count():
    #             action_probabilities = policy(state, env.action_space.n, Q=Q)
    #             action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
    #             print("action is", action)
    #             input()
    #             next_state, reward, terminated, truncated, info = env.step(action)
    #             done = terminated or truncated
    #             print(terminated, truncated)
    #             next_state = sum(next_state)

    #             best_next_action = Q[next_state].argmax()
    #             print(best_next_action)
    #             td_target = reward + model.gamma * Q[next_state][best_next_action]
    #             td_delta = td_target - Q[state][action]
    #             Q[state][action] += 0.5 * td_delta

    #             if done:
    #                 break

    #             state = next_state
    #     return Q