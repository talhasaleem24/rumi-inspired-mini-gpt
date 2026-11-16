import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def prepare_poetry(raw_text):
    start_match = re.search(
        r"PROEM\.\s*The Reed-Flute\.",
        raw_text,
        flags=re.IGNORECASE
    )

    if start_match is None:
        raise ValueError("Could not find the poetry section.")

    remaining_text = raw_text[start_match.start():]
    end_match = re.search(
        r"(?m)^\s*THE END\.\s*$",
        remaining_text
    )

    if end_match is None:
        raise ValueError("Could not find the end of the poetry.")

    poetry = remaining_text[:end_match.start()]
    poetry = re.sub(r"\[\d+\]", "", poetry)
    poetry = poetry.replace("_", "").replace("\r\n", "\n")

    clean_lines = []

    for raw_line in poetry.splitlines():
        if re.fullmatch(r"\s*\d{1,3}\.?\s*", raw_line):
            continue

        line = re.sub(r"\s+\d{1,3}\.?\s*$", "", raw_line)
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            clean_lines.append(line)
        elif clean_lines and clean_lines[-1]:
            clean_lines.append("")

    while clean_lines and not clean_lines[-1]:
        clean_lines.pop()

    clean_text = "\n".join(clean_lines)

    if not clean_text:
        raise ValueError("The poetry section is empty after cleaning.")

    return clean_text


def main():
    source_path = PROJECT_DIR / "source.txt"

    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found. Download the book first."
        )

    raw_text = source_path.read_text(encoding="utf-8")
    clean_text = prepare_poetry(raw_text)

    output_path = PROJECT_DIR / "rumi_english.txt"
    output_path.write_text(clean_text, encoding="utf-8")

    print("Dataset successfully prepared!")
    print("Total characters:", len(clean_text))
    print("Unique characters:", len(set(clean_text)))
    print("Total lines:", len(clean_text.splitlines()))
    print("Stanza breaks:", clean_text.count("\n\n"))
    print("Saved as:", output_path)


if __name__ == "__main__":
    main()
