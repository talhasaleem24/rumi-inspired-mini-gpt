import argparse
import csv
import json
from pathlib import Path

import torch

from attention import BLOCK_SIZE
from model import model
from tokenizer import encode


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "mini_gpt_word_latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "mini_gpt_word_best.pt"
TRAINING_LOG = PROJECT_DIR / "training_log_word.csv"

BATCH_SIZE = 4
MAX_ITERS = 20000
EVAL_INTERVAL = 500
EVAL_BATCHES = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DEVICE = "cpu"


def load_training_data():
    dataset_path = PROJECT_DIR / "rumi_english.txt"
    vocabulary_path = PROJECT_DIR / "vocab.json"
    text = dataset_path.read_text(encoding="utf-8")

    with vocabulary_path.open("r", encoding="utf-8") as file:
        stoi = json.load(file)

    if not text:
        raise ValueError("Training dataset is empty.")
    if len(stoi) != model.vocab_size:
        raise ValueError(
            "Vocabulary size does not match the model. "
            "Run tokenizer.py before train.py."
        )

    if "<unk>" not in stoi:
        raise ValueError("Vocabulary is missing the required <unk> token.")
    if len(set(stoi.values())) != len(stoi):
        raise ValueError("Vocabulary token IDs must be unique.")

    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    if len(data) <= BLOCK_SIZE:
        raise ValueError(
            "Tokenized dataset is too small for the configured BLOCK_SIZE."
        )

    split = int(0.90 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
        raise ValueError("Dataset is too small for the configured BLOCK_SIZE.")

    return stoi, train_data, val_data


def make_batch(dataset):
    indices = torch.randint(
        len(dataset) - BLOCK_SIZE,
        (BATCH_SIZE,)
    ).tolist()

    inputs = torch.stack([
        dataset[index:index + BLOCK_SIZE]
        for index in indices
    ])
    targets = torch.stack([
        dataset[index + 1:index + BLOCK_SIZE + 1]
        for index in indices
    ])

    return inputs.to(DEVICE), targets.to(DEVICE)


@torch.no_grad()
def estimate_loss(train_data, val_data):
    was_training = model.training
    model.eval()
    results = {}

    for split_name, dataset in (
        ("train", train_data),
        ("val", val_data)
    ):
        losses = []

        for _ in range(EVAL_BATCHES):
            inputs, targets = make_batch(dataset)
            _, loss = model(inputs, targets)
            losses.append(loss.item())

        results[split_name] = sum(losses) / len(losses)

    model.train(was_training)
    return results


def save_checkpoint(path, checkpoint):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def train_model(steps=MAX_ITERS, resume=False):
    if steps <= 0:
        raise ValueError("Training steps must be greater than zero.")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    stoi, train_data, val_data = load_training_data()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    start_step = 0
    best_val_loss = float("inf")
    best_checkpoint_exists = BEST_CHECKPOINT.exists()

    if resume:
        if not LATEST_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Cannot resume: {LATEST_CHECKPOINT} does not exist."
            )

        checkpoint = torch.load(
            LATEST_CHECKPOINT,
            map_location=DEVICE,
            weights_only=True
        )

        if checkpoint.get("stoi") != stoi:
            raise ValueError(
                "Checkpoint vocabulary differs from vocab.json; "
                "start a new training run instead of resuming."
            )

        checkpoint_config = checkpoint.get("config", {})
        expected_config = {
            "vocab_size": len(stoi),
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "tokenizer": "word"
        }
        mismatched_config = {
            key: (checkpoint_config.get(key), expected)
            for key, expected in expected_config.items()
            if checkpoint_config.get(key) != expected
        }
        if mismatched_config:
            raise ValueError(
                f"Checkpoint architecture is incompatible: "
                f"{mismatched_config}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]

        if BEST_CHECKPOINT.exists():
            best_checkpoint = torch.load(
                BEST_CHECKPOINT,
                map_location=DEVICE,
                weights_only=True
            )
            if best_checkpoint.get("stoi") != stoi:
                raise ValueError(
                    "Best checkpoint vocabulary differs from vocab.json."
                )
            if best_checkpoint.get("config", {}).get("tokenizer") != "word":
                raise ValueError(
                    "Best checkpoint uses an incompatible tokenizer."
                )
            if best_checkpoint.get("config", {}).get("block_size") != BLOCK_SIZE:
                raise ValueError(
                    "Best checkpoint context size differs from the latest "
                    "checkpoint. Start a fresh run."
                )
            best_val_loss = best_checkpoint["val_loss"]

    if steps <= start_step:
        raise ValueError(
            f"Requested final step {steps} is not after resume step "
            f"{start_step}."
        )

    print("Dataset loaded successfully!")
    print("Vocabulary size:", len(stoi))
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))
    print("Training steps:", start_step, "to", steps)

    initial_losses = estimate_loss(train_data, val_data)
    if not best_checkpoint_exists:
        best_val_loss = initial_losses["val"]

    if resume:
        initial_checkpoint = {
            "step": start_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": initial_losses["train"],
            "val_loss": initial_losses["val"],
            "stoi": stoi,
            "config": {
                "vocab_size": len(stoi),
                "embed_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "block_size": BLOCK_SIZE,
                "batch_size": BATCH_SIZE,
                "tokenizer": "word"
            }
        }
        save_checkpoint(LATEST_CHECKPOINT, initial_checkpoint)
        if not best_checkpoint_exists:
            save_checkpoint(BEST_CHECKPOINT, initial_checkpoint)

    log_mode = "a" if resume else "w"
    with TRAINING_LOG.open(
        log_mode,
        newline="",
        encoding="utf-8"
    ) as log_file:
        writer = csv.writer(log_file)
        if not resume:
            writer.writerow(["step", "train_loss", "val_loss"])

        model.train()
        for step in range(start_step + 1, steps + 1):
            inputs, targets = make_batch(train_data)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )
            optimizer.step()

            if (
                step % EVAL_INTERVAL == 0
                or step == 1
                or step == steps
            ):
                losses = estimate_loss(train_data, val_data)
                train_loss = losses["train"]
                val_loss = losses["val"]

                print(
                    f"Step {step:5d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Validation Loss: {val_loss:.4f}"
                )
                writer.writerow([step, train_loss, val_loss])
                log_file.flush()

                checkpoint = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "stoi": stoi,
                    "config": {
                        "vocab_size": len(stoi),
                        "embed_dim": 64,
                        "num_heads": 4,
                        "num_layers": 2,
                        "block_size": BLOCK_SIZE,
                        "batch_size": BATCH_SIZE,
                        "tokenizer": "word"
                    }
                }

                save_checkpoint(LATEST_CHECKPOINT, checkpoint)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(BEST_CHECKPOINT, checkpoint)
                    print("Best model checkpoint saved.")

    print("Training completed.")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Latest checkpoint:", LATEST_CHECKPOINT)
    print("Training history:", TRAINING_LOG)


def main():
    parser = argparse.ArgumentParser(
        description="Train the character-level poetry language model."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=MAX_ITERS,
        help=f"Final training step (default: {MAX_ITERS})."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the cleaned-data latest checkpoint."
    )
    args = parser.parse_args()
    train_model(steps=args.steps, resume=args.resume)


if __name__ == "__main__":
    main()
