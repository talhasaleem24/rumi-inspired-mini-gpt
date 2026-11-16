import json
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
UNKNOWN_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(
    r"\n+|[ \t]+|[\w]+(?:[’'][\w]+)*|[^\w\s]",
    flags=re.UNICODE
)


def tokenize(text):
    return [
        " " if token.isspace() and "\n" not in token else token
        for token in TOKEN_PATTERN.findall(text)
    ]


def build_vocabulary(text, min_frequency=2):
    if not text:
        raise ValueError("Dataset is empty.")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least one.")

    counts = Counter(tokenize(text))
    tokens = sorted(
        (
            token for token, count in counts.items()
            if count >= min_frequency and token != UNKNOWN_TOKEN
        ),
        key=lambda token: (-counts[token], token)
    )
    return {
        token: index
        for index, token in enumerate([UNKNOWN_TOKEN, *tokens])
    }


def encode(text, stoi):
    unknown_id = stoi.get(UNKNOWN_TOKEN)
    if unknown_id is None:
        missing = sorted(set(tokenize(text)) - set(stoi))
        if missing:
            raise ValueError(f"Tokens not in vocabulary: {missing!r}")
        return [stoi[token] for token in tokenize(text)]
    return [
        stoi.get(token, unknown_id)
        for token in tokenize(text)
    ]


def decode(numbers, stoi):
    itos = {index: token for token, index in stoi.items()}

    try:
        decoded = []
        for index in numbers:
            token = itos[int(index)]
            decoded.append("[UNK]" if token == UNKNOWN_TOKEN else token)
        return "".join(decoded)
    except KeyError as error:
        raise ValueError(
            f"Token ID {error.args[0]} is not in the vocabulary."
        ) from error


def main():
    dataset_path = PROJECT_DIR / "rumi_english.txt"
    text = dataset_path.read_text(encoding="utf-8")
    stoi = build_vocabulary(text)

    sample = text[:1500]
    expected = "".join(
        token if token in stoi else "[UNK]"
        for token in tokenize(sample)
    )
    if decode(encode(sample, stoi), stoi) != expected:
        raise RuntimeError("Tokenizer round-trip check failed.")

    vocabulary_path = PROJECT_DIR / "vocab.json"
    vocabulary_path.write_text(
        json.dumps(stoi, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Dataset loaded successfully!")
    print("Total characters:", len(text))
    print("Total tokens:", len(tokenize(text)))
    print("Vocabulary size:", len(stoi))
    print("Minimum token frequency: 2")
    print("Tokenizer round-trip test passed!")
    print("Vocabulary saved to:", vocabulary_path)


if __name__ == "__main__":
    main()
