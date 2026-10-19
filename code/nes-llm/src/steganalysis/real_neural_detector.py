import pickle
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

DEVICE = "cpu"


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
            dtype=torch.float32,
        )


class Detector(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        return self.net(x)


def main():

    dataset = ResidualDataset(
        "real_detector_dataset.pkl"
    )

    labels = [
        y for _, y in dataset
    ]

    num_stego = sum(
        int(y.item())
        for y in labels
    )

    num_clean = (
        len(labels)
        - num_stego
    )

    print(
        f"Clean Samples: {num_clean}"
    )

    print(
        f"Stego Samples: {num_stego}"
    )

    input_dim = len(
        dataset[0][0]
    )

    print(
        f"Input Dimension: {input_dim}"
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
        [
            train_size,
            test_size,
        ]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=32,
    )

    model = Detector(
        input_dim
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    loss_fn = nn.BCELoss()

    print("\nTraining...")

    for epoch in range(100):

        total_loss = 0

        model.train()

        for x, y in train_loader:

            pred = (
                model(x)
                .squeeze()
            )

            loss = loss_fn(
                pred,
                y,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        print(
            f"Epoch {epoch+1}: "
            f"{total_loss:.4f}"
        )

    print("\nEvaluating...")

    model.eval()

    correct = 0
    total = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.no_grad():

        for x, y in test_loader:

            probs = (
                model(x)
                .squeeze()
            )

            pred = (
                probs > 0.5
            )

            correct += (
                pred == y.bool()
            ).sum().item()

            total += len(y)

            for p, t in zip(
                pred,
                y.bool()
            ):

                if p and t:
                    tp += 1

                elif p and not t:
                    fp += 1

                elif not p and t:
                    fn += 1

                else:
                    tn += 1

    accuracy = (
        correct / total
    )

    print("\n====================")
    print("RESULTS")
    print("====================")

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Flipped Accuracy: "
        f"{1-accuracy:.4f}"
    )

    print("\nCONFUSION MATRIX")

    print(
        f"TP: {tp}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"TN: {tn}"
    )

    positive_rate = (
        (tp + fp)
        /
        (tp + fp + tn + fn)
    )

    print(
        f"\nPositive Prediction Rate: "
        f"{positive_rate:.4f}"
    )

    print(
        f"Total Samples: "
        f"{tp+fp+tn+fn}"
    )


if __name__ == "__main__":
    main()