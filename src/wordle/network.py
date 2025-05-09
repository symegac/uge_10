from torch import nn

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super(DQN, self).__init__()
        self.network_stack = nn.Sequential(
            nn.Linear(in_features=n_observations, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_actions),
        )

    def forward(self, x):
        output = self.network_stack(x)
        return output
