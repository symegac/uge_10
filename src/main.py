import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

config: dict[str, int | float] = json.load(open("config.json"))
epochs, learning_rate, batch_size = config.values()

training_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
testing_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

training_data = DataLoader(
    dataset=training_dataset,
    batch_size=batch_size
)
testing_data = DataLoader(
    dataset=testing_dataset,
    batch_size=batch_size
)

if __name__ == "__main__":
    print(training_dataset)
    print(len(training_data))
    print(testing_dataset)
    print(len(testing_data))
