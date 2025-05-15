import torch
import pickle
from torch import nn, serialization

class WeightsUnpicklingError(pickle.UnpicklingError):
    pass

class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
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

    def load(self, path: str, network_type: str):
        try:
            with serialization.safe_globals(GLOBALS):
                weights = torch.load(path, weights_only=True)
        except:
            err = serialization.get_unsafe_globals_in_checkpoint(path)
            err_msg = f"Følgende globals skal godkendes for at kunne indlæse vægtene fra det gemte checkpoint: {err}."
            raise WeightsUnpicklingError(err_msg)
        else:
            print(f"Vægte for {network_type}-netværket hentet.")

        try:
            self.load_state_dict(weights)
            self.eval()
        except:
            raise

GLOBALS = [getattr, nn.Linear, nn.ReLU, nn.Sequential, DQN]

def main() -> None:
    pass

if __name__ == "__main__":
    main()
