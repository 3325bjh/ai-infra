"""Convert raw GSM8K JSONL data into the prompt/response format used by SFT.

The raw GSM8K format contains ``question`` and ``answer``.  The answer is
split at the final ``####`` marker: the explanation becomes the thinking
section and the text after the marker becomes the final answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_item(item: dict, prompt_template: str) -> dict:
    """Convert one raw GSM8K record."""
    question = item["question"].strip()
    raw_answer = item["answer"].strip()

    if "####" not in raw_answer:
        raise ValueError("answer does not contain the GSM8K '####' marker")

    reasoning, final_answer = raw_answer.rsplit("####", 1)
    reasoning = reasoning.strip()
    final_answer = final_answer.strip()
    if not reasoning or not final_answer:
        raise ValueError("reasoning or final answer is empty")

    prompt = prompt_template.replace("{question}", question)
    response = f"{reasoning}\n</think> <answer>{final_answer}</answer>"

    return {
        "prompt": prompt,
        "response": response,
        "question": question,
        "answer": raw_answer,
        "is_correct": True,
    }


def convert_file(input_path: Path, output_path: Path, prompt_path: Path) -> tuple[int, int]:
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    converted = 0
    skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                converted_item = convert_item(item, prompt_template)
            except (KeyError, ValueError) as exc:
                skipped += 1
                print(f"Skipping line {line_number}: {exc}")
                continue
            target.write(json.dumps(converted_item, ensure_ascii=False) + "\n")
            converted += 1

    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/gsm8k/train.jsonl"),
        help="Raw GSM8K JSONL input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gsm8k/train_sft.jsonl"),
        help="Converted SFT JSONL output file.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("cs336_alignment/prompts/r1_zero.prompt"),
        help="Prompt template containing the {question} placeholder.",
    )
    args = parser.parse_args()

    converted, skipped = convert_file(args.input, args.output, args.prompt)
    print(f"Converted: {converted}; skipped: {skipped}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
