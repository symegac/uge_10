from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten: nn.Flatten = nn.Flatten()
        self.network_stack: nn.Sequential = nn.Sequential(
            # Reducerer billede til 512 værdier
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            # Reducerer 512 dataværdier til 10 grupper
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x) -> nn.Sequential:
        x = self.flatten(x)
        output = self.network_stack(x)
        return output

def main():
    pass

if __name__ == "__main__":
    main()
