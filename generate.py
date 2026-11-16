import json
from pathlib import Path

import torch

from model import model
from tokenizer import UNKNOWN_TOKEN, decode, encode


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = (
    PROJECT_DIR / "checkpoints" / "mini_gpt_word_best.pt",
    PROJECT_DIR / "checkpoints" / "mini_gpt_word_latest.pt",
    PROJECT_DIR / "checkpoints" / "mini_gpt_clean_best.pt",
    PROJECT_DIR / "checkpoints" / "mini_gpt_clean_latest.pt",
    PROJECT_DIR / "checkpoints" / "mini_gpt_best.pt",
    PROJECT_DIR / "checkpoints" / "mini_gpt_latest.pt"
)

DEVICE = "cpu"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 12


def load_checkpoint():
    checkpoint_path = next(
        (path for path in CHECKPOINT_CANDIDATES if path.exists()),
        None
    )

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No trained checkpoint found. Run prepare_rumi.py, "
            "tokenizer.py, and train.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    vocabulary_path = PROJECT_DIR / "vocab.json"
    with vocabulary_path.open("r", encoding="utf-8") as file:
        current_stoi = json.load(file)

    if checkpoint.get("stoi") != current_stoi:
        raise ValueError(
            "The checkpoint vocabulary does not match vocab.json. "
            "Run train.py to create a checkpoint for the current dataset."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")
    print("Checkpoint:", checkpoint_path.name)
    print("Training step:", checkpoint["step"])
    print("Validation loss:", checkpoint["val_loss"])
    print("Vocabulary size:", model.vocab_size)

    return checkpoint["stoi"]


def generate_poetry(
    prompt,
    stoi,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K
):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please provide a non-empty prompt.")

    prompt_tokens = encode(prompt, stoi)
    input_ids = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        forbidden_token_ids=[stoi[UNKNOWN_TOKEN]]
    )

    continuation_ids = generated_ids[0, len(prompt_tokens):].tolist()
    return prompt + decode(continuation_ids, stoi)


def main():
    stoi = load_checkpoint()

    print("\n================================")
    print("       MINI RUMI GPT")
    print("================================")
    print("\nEnter a poetry prompt. Example: The heart")

    while True:
        try:
            prompt = input("\nYour prompt (or 'exit' to quit): ")

            if prompt.strip().lower() == "exit":
                break
            if not prompt.strip():
                print("Please enter a prompt.")
                continue

            result = generate_poetry(prompt, stoi)
            print("\n========== GENERATED POETRY ==========\n")
            print(result)
            print("\n======================================")

        except ValueError as error:
            print("Input error:", error)
        except KeyboardInterrupt:
            print("\nGeneration stopped.")
            break

    print("\nMini Rumi GPT closed.")


if __name__ == "__main__":
    main()
