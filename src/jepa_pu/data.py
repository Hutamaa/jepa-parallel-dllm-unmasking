from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from datasets import load_dataset

from .config import ensure_output_dirs, load_config
from .utils import read_jsonl, write_jsonl


NUMBER_RE = re.compile(
    r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[0-9]+)?(?:[eE][-+]?\d+)?"
)


def extract_answer(text: str | None) -> str | None:
    """Extract the final numeric answer, preferring GSM8K's #### marker."""
    if not text:
        return None
    has_marker = "####" in text
    tail = text.rsplit("####", 1)[-1] if has_marker else text
    matches = NUMBER_RE.findall(tail)
    if not matches and has_marker:
        matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace("$", "").replace(",", "").replace(" ", "")
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            value = Decimal(numerator) / Decimal(denominator)
        else:
            value = Decimal(raw)
        normalized = format(value.normalize(), "f")
        return "0" if normalized in ("-0", "") else normalized
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def prepare(cfg: dict, force: bool = False) -> None:
    ensure_output_dirs(cfg)
    out_dir = Path(cfg["paths"]["data_dir"])
    expected = [out_dir / f"gsm8k_{split}.jsonl" for split in ("train", "validation", "test")]
    if any(path.exists() for path in expected) and not force:
        raise FileExistsError("Prepared splits already exist; pass --force to replace them")

    dcfg = cfg["dataset"]
    dataset = load_dataset(dcfg["name"], dcfg.get("config", "main"))
    shuffled = dataset["train"].shuffle(seed=cfg["seed"])
    n_train = int(dcfg["train_examples"])
    n_val = int(dcfg["validation_examples"])
    if n_train + n_val > len(shuffled):
        raise ValueError("Requested train + validation size exceeds GSM8K train split")
    selected = {
        "train": shuffled.select(range(n_train)),
        "validation": shuffled.select(range(n_train, n_train + n_val)),
        "test": dataset["test"].shuffle(seed=int(cfg["seed"]) + 1).select(
            range(min(int(dcfg["test_examples"]), len(dataset["test"])))
        ),
    }
    for split, rows in selected.items():
        payload = []
        for i, row in enumerate(rows):
            payload.append({
                "id": f"gsm8k-{split}-{i:05d}",
                "question": row["question"],
                "answer": row["answer"],
                "gold": extract_answer(row["answer"]),
                "source_split": "test" if split == "test" else "train",
            })
        write_jsonl(out_dir / f"gsm8k_{split}.jsonl", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(load_config(args.config), args.force)


if __name__ == "__main__":
    main()
