import json
from pathlib import Path

import torch

from attention import BLOCK_SIZE


PROJECT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 16
DEVICE = "cpu"


def load_data():
    text = (PROJECT_DIR / "rumi_english.txt").read_text(
        encoding="utf-8"
    )
    with (PROJECT_DIR / "vocab.json").open(
        "r",
        encoding="utf-8"
    ) as file:
        stoi = json.load(file)

    missing = sorted(set(text) - set(stoi))
    if missing:
        raise ValueError(f"Dataset characters missing from vocabulary: {missing!r}")

    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split_index = int(0.90 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
        raise ValueError("Dataset is too small for the configured BLOCK_SIZE.")

    return stoi, train_data, val_data


def get_batch(split, train_data, val_data):
    if split == "train":
        dataset = train_data
    elif split == "val":
        dataset = val_data
    else:
        raise ValueError("Split must be 'train' or 'val'.")

    indices = torch.randint(
        len(dataset) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )
    inputs = torch.stack([
        dataset[index:index + BLOCK_SIZE]
        for index in indices
    ])
    targets = torch.stack([
        dataset[index + 1:index + BLOCK_SIZE + 1]
        for index in indices
    ])

    return inputs.to(DEVICE), targets.to(DEVICE)


def main():
    stoi, train_data, val_data = load_data()
    inputs, targets = get_batch("train", train_data, val_data)

    assert torch.equal(inputs[0, 1:], targets[0, :-1])

    print("Vocabulary size:", len(stoi))
    print("Input and target shapes:", inputs.shape, targets.shape)
    print("Shifted target test passed!")


if __name__ == "__main__":
    main()
