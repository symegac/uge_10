import os
import json
import torch
from torch import accelerator, optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network import NeuralNetwork
from pickle import UnpicklingError

GLOBALS = [getattr, nn.Linear, nn.ReLU, nn.Sequential, nn.Flatten, NeuralNetwork]

class WeightsUnpicklingError(UnpicklingError):
    pass

def load_data(root: str, batch_size: int) -> tuple[DataLoader]:
    training_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    testing_dataset = datasets.FashionMNIST(
        root=root,
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

    return training_data, testing_data

def set_up_model(model_path: str = None) -> nn.Module:
    """
    Opsætter et neuralt netværk, og indlæser allerede gemte vægte.

    :param model_path: Filsti (og -navn) til en gemt model.
    :type model_path: str
    :raises WeightsUnpicklingError: Hvis unsafe globals ikke er blevet whitelisted.
    :return: Den indlæste model.
    :rtype: nn.Module
    """
    device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"
    model = NeuralNetwork().to(device)

    # Indlæser evt. gemte vægte
    if model_path and os.path.exists(model_path):
        try:
            with torch.serialization.safe_globals(GLOBALS):
                weights = torch.load(model_path, weights_only=True)
        except:
            err = torch.serialization.get_unsafe_globals_in_checkpoint(model_path)
            err_msg = f"Følgende globals skal godkendes for at kunne indlæse vægtene fra det gemte checkpoint: {err}."
            raise WeightsUnpicklingError(err_msg)
        else:
            print("Vægte hentet.")

        try:
            model.load_state_dict(weights)
            model.eval()
        except:
            raise
        else:
            print("Vægte indlæst.")

    return model

# Loops
def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    batch_size: int
) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Printer forløb
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.CrossEntropyLoss
) -> None:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def save_model(model: nn.Module, model_path: str) -> None:
    """
    Gemmer en models vægte.

    :param model: Modellen, hvis vægte skal gemmes.
    :type model: nn.Module
    :param model_path: Filstien (og -navnet), hvor vægtene skal gemmes.
    :type model_path: str
    """
    torch.save(model.state_dict(), model_path)

def main() -> None:
    # Config
    config: dict[str, dict[str, int | float]] = json.load(open("config.json"))
    epochs, learning_rate, batch_size, data_dir, model_file = config["fashionmnist"].values()
    model_path = os.path.join(data_dir, "output", model_file)

    # Setup
    training_data, testing_data = load_data(root=data_dir, batch_size=batch_size)
    model = set_up_model(model_path=model_path)

    # Opretter optimizer og loss-funktion
    # Adam er bedre end den gamle
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    # Udførsel
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(
            dataloader=training_data,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=batch_size
        )
        test_loop(
            dataloader=testing_data,
            model=model,
            loss_fn=loss_fn
        )
    print("Done!")

    # Gemning
    save_model(model=model, model_path=model_path)
    print("Model saved.")

if __name__ == "__main__":
    main()
