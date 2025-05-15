import os
import math
import json
import random
import matplotlib.pyplot as plt

import torch
from torch import nn, optim, accelerator
from network import DQN
from memory import ReplayMemory, Transition
from env import WordleEnv, Step

from numpy import cumsum
from itertools import count

# Kommentarer med et initialt =1=, =2=, =...=, =14= betegner linjenummeret i den oprindelige DQN-algoritme fra Mnih et al. (2013), s. 5 (https://arxiv.org/abs/1312.5602)
class Model:
    def __init__(
        self,
        batch_size: int,
        learning_rate: int | float,
        gamma: int | float,
        eps_start: int | float,
        eps_end: int | float,
        eps_decay: int | float,
        tau: int | float,
        observation_size: int,
        action_size: int,
        model_dir: str = '',
        model_name: str = ''
    ) -> None:
        # Config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.observation_size = observation_size
        self.action_size = action_size
        self.policy_path = self._find_path(model_dir, model_name)
        self.target_path = self._find_path(model_dir, model_name, target=True)
        self.memory_path = self._find_path(model_dir, model_name, memory=True)
        self.set_up_model()
        # Denne værdi gør måske, at modellen kører i hak og ikke udvikler sig,
        # når tilstrækkeligt mange steps er taget, da epsilon så bliver fanget
        # ved eps_end-værdien
        # self.steps_done = len(self.memory)
        self.steps_done = 0

    def set_up_model(self) -> None:
        """Opsætter modellen"""
        # Device
        self.device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"
        # Hukommelse og neurale netværk
        # =1, 2=
        self.load()
        # Optimizer
        self.optimizer = optim.AdamW(
            params=self.policy_net.parameters(),
            lr=self.learning_rate,
            amsgrad=True
        )
        # Loss-funktion
        self.loss_fn = nn.SmoothL1Loss()

        print("Indlæsning af model fuldført.")

    def select_action(self, state: torch.Tensor):
        # Beregner 𝜖
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # Find handling ud fra sandsynlighed 𝜖
        sample = random.random()
        if sample > eps_threshold:
            # =7= otherwise select 𝑎ₜ = maxₐ 𝑄∗(𝜙(𝑠ₜ), 𝑎; 𝜃)
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # TODO: Brug torch.unsqueeze istf. [[]] for at ændre tensorens dimensioner
            # evt. 
            # =6= With probability 𝜖 select a random action 𝑎ₜ
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.int64)

    def optimize(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        # =11= Sample random minibatch of transitions (𝜙ⱼ , 𝑎ⱼ , 𝑟ⱼ , 𝜙ⱼ₊₁) from 𝒟
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # =12, 13=
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

        next_state_values: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

    def _find_path(
        self,
        dir: str, name: str,
        target: bool = False,
        memory: bool = False
    ) -> str:
        addon = ''
        if not memory:
            addon = "target" if target else "policy"
        extension = "qmem" if memory else "pt"
        filename = '_'.join((name, addon)) if addon else name
        file = '.'.join((filename, extension))
        path = os.path.join(dir, "output", file)
        return path

    def save(self) -> None:
        torch.save(self.policy_net.state_dict(), self.policy_path)
        torch.save(self.target_net.state_dict(), self.target_path)
        print("Vægte gemt.")

        self.memory.save(self.memory_path)

    def load(self) -> None:
        # =2= Initialize action-value function 𝑄 with random weights
        self.policy_net = DQN(self.observation_size, self.action_size).to(self.device)
        self.target_net = DQN(self.observation_size, self.action_size).to(self.device)

        self._load_weights(self.policy_path)
        self._load_weights(self.target_path, target=True)
        print("Alle vægte indlæst.")

        self._load_memory(self.memory_path)

    def _load_weights(
        self,
        path: str,
        target: bool = False
    ):
        network = self.target_net if target else self.policy_net
        if os.path.exists(path):
            network.load(path, "target" if target else "policy")
        elif target:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Vægte for target-netværket findes ikke. Bruger vægte fra policy-netcærket i stedet.")
            return
        else:
            print("Vægte for policy-netværket findes ikke. Bruger tilfældige vægte i stedet.")
            return

    def _load_memory(self, path: str) -> None:
        # =1= Initialize replay memory 𝒟 to capacity 𝑁
        self.memory = ReplayMemory(10000)
        if os.path.exists(path):
            self.memory_path = path
            self.memory.load(self.memory_path)

def training_loop(
    env: WordleEnv,
    model: Model,
    episodes: int = 50,
    render: bool = False,
    save: bool = True
) -> list[tuple[int, int]]:
    stats = []
    wins = 0
    # =3= for episode = 1, 𝑀 do
    for n in range(episodes):
        tstats = []
        tinfo = []
        # Opstart environment og få start-state
        # =4= Initialise sequence 𝑠₁ = {𝑥₁} and preprocessed sequenced 𝜙₁ = 𝜙(𝑠₁)
        initial_observation, info = env.reset()
        word = info["word"]
        state = torch.tensor(initial_observation, dtype=torch.float32, device=model.device).unsqueeze(0)
        # Kør derefter gennem resten af states
        # =5= for 𝑡 = 1, 𝑇 do
        for t in count():
            # =6, 7=
            action = model.select_action(state)
            # =8= Execute action 𝑎ₜ in emulator and observe reward 𝑟ₜ and image 𝑥ₜ₊₁
            observation, reward, terminated, truncated, info = env.step(action.item())
            # TODO: Brug torch.unsqueeze her istf. [[]] for at ændre på tensorens dimensioner
            reward = torch.tensor([reward], device=model.device)
            done = terminated or truncated

            # Se om det var et terminalt step
            if done:
                next_state = None
                if terminated:
                    wins += 1
            else:
                # =9= Set 𝑠ₜ₊₁ = 𝑠ₜ, 𝑎ₜ, 𝑥ₜ₊₁ and preprocess 𝜙ₜ₊₁ = 𝜙(𝑠ₜ₊₁)
                next_state = torch.tensor(observation, dtype=torch.float32, device=model.device).unsqueeze(0)

            # Gem overgangen
            # =10= Store transition (𝜙ₜ, 𝑎ₜ, 𝑟ₜ, 𝜙ₜ₊₁) in 𝒟
            model.memory.push(state, action, reward, next_state)

            # Til egen brug, har ikke noget med algoritmen at gøre
            tinfo.append((info["strategy"], env._game.history[-1], reward.tolist()[0]))
            tstats.append((n, t, reward.tolist()[0], wins))

            # Optimér policy nn
            model.optimize()

            # Opdatér target net
            target_net_state_dict = model.target_net.state_dict()
            policy_net_state_dict = model.policy_net.state_dict()
            for key in policy_net_state_dict:
                # θ′ ← τ θ + (1 − τ)θ′
                target_net_state_dict[key] = policy_net_state_dict[key] * model.tau + target_net_state_dict[key] * (1 - model.tau)
            model.target_net.load_state_dict(target_net_state_dict)

            # Forbered til næste iteration
            state = next_state
            if next_state is None:
                break
        if render:
            print(f"Episode {n + 1} -- word: {word}, reward: {reward.tolist()[0]}, guessed: {terminated}, info: {tinfo}")
        else:
            print(n)
        stats.append(tstats)
    print("Træning afsluttet.")
    if save:
        model.save()
    return stats

def testing_loop() -> None:
    pass

def main() -> None:
    # Config
    config: dict[str, dict[str, int | float | str]] = json.load(open("config.json"))

    env = WordleEnv(
        # render="ansi",
        default_guesses=["toner", "dashi"]
    )
    state, info = env.reset()
    observation_size = len(state)
    action_size = env.action_space.n
    model = Model(
        observation_size=observation_size,
        action_size=action_size,
        **config["wordle"]
    )

    stats = training_loop(env, model, episodes=1000, render=True, save=False)
    print(model.steps_done, len(model.memory))

    wins = []
    rewards = []
    for n, data in enumerate(stats):
        wins.append(data[-1][-1])
        for t, step in enumerate(data):
            rewards.append(step[2])
    # print(wins, rewards, sep='\n')
    total_rewards = list(cumsum(rewards))

    plt.plot(rewards)
    plt.show()
    plt.plot(wins)
    plt.show()
    plt.plot(total_rewards)
    plt.show()

if __name__ == "__main__":
    main()
