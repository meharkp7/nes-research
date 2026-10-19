import pickle
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)


class ResidualDataset(Dataset):

    def __init__(self, path):

        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x, y = self.data[idx]

        x = x.float()

        return x, torch.tensor(
            y,
            dtype=torch.float32
        )


class Detector(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                100000,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():

    dataset = ResidualDataset(
        "detector_dataset.pkl"
    )

    train_size = int(
        0.8 * len(dataset)
    )

    test_size = (
        len(dataset)
        - train_size
    )

    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
    )

    model = Detector()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.BCELoss()

    print("Training...")

    for epoch in range(50):

        total_loss = 0

        for x, y in train_loader:

            pred = (
                model(x)
                .squeeze()
            )

            loss = loss_fn(
                pred,
                y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}:",
            total_loss
        )

    print("\nEvaluating...")

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in test_loader:

            pred = (
                model(x)
                .squeeze()
                > 0.5
            )

            correct += (
                pred == y.bool()
            ).sum().item()

            total += len(y)

    accuracy = (
        correct / total
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print("Raw Accuracy:", accuracy)
    print("Flipped Accuracy:", 1 - accuracy)

if __name__ == "__main__":
    main()