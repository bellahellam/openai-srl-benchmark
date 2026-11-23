"""Download and parse PropBank-style SRL data from Hugging Face.

Source dataset: `cu-kairos/propbank_srl_seq2seq` (Apache-2.0), derived from
OntoNotes/PropBank and distributed as parquet, so we read it directly with
pandas/pyarrow and need no dataset loading script.

Raw format (two string columns):
    prompt   : "SRL for [predicate]: <sentence with the predicate bracketed>"
    response : "ARG-0: John | ARG-1: the door | ARG-M-TMP: yesterday"   (or "<unk>")

We canonicalize role spelling (ARG-0 -> ARG0, ARG-M-TMP -> ARGM-TMP), compute
character offsets from the un-bracketed sentence, drop <unk> rows, and write
train/validation/test JSONL under data/. The source has only train+validation,
so our test set is carved from the source validation split (disjoint).

Regenerate the data with:
    python -m src.data_loader --help
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

try:
    from .schema import Argument, SRLExample
except ImportError:  # run as a loose script from src/
    from schema import Argument, SRLExample  # type: ignore

HF_DATASET = "cu-kairos/propbank_srl_seq2seq"
PARQUET_BASE = (
    "https://huggingface.co/datasets/cu-kairos/propbank_srl_seq2seq/"
    "resolve/refs%2Fconvert%2Fparquet/default"
)
PARQUET_URLS = {
    "train": f"{PARQUET_BASE}/train/0000.parquet",
    "validation": f"{PARQUET_BASE}/validation/0000.parquet",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "_raw_cache"

PROMPT_RE = re.compile(r"^SRL for \[(?P<pred>.+?)\]:\s*(?P<sent>.*)$", re.DOTALL)
UNK = "<unk>"


def download_parquet(split: str, force: bool = False) -> Path:
    if split not in PARQUET_URLS:
        raise ValueError(f"Unknown split {split!r}; expected {list(PARQUET_URLS)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{split}.parquet"
    if dest.exists() and not force:
        return dest
    url = PARQUET_URLS[split]
    print(f"Downloading {split} from {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  saved {dest} ({len(resp.content):,} bytes)")
    return dest


def load_raw_split(split: str, force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(download_parquet(split, force=force))


def _canonical_role(raw: str) -> str:
    role = raw.strip().upper()
    role = role.replace("ARG-M-", "ARGM-")
    role = role.replace("ARG-", "ARG")
    return role


@dataclass
class _ParsedRow:
    sentence: str
    predicate: str
    predicate_start: int
    predicate_end: int
    arguments: list


def _locate_span(sentence: str, span: str, search_from: int):
    idx = sentence.find(span, search_from)
    if idx == -1:
        idx = sentence.find(span)
    if idx != -1:
        return idx, idx + len(span)
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in span.split()))
    m = pattern.search(sentence, search_from) or pattern.search(sentence)
    if m:
        return m.start(), m.end()
    return None, None


def _parse_arguments(sentence: str, response: str) -> list:
    args = []
    search_from = 0
    for chunk in response.split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        raw_role, raw_text = chunk.split(":", 1)
        role = _canonical_role(raw_role)
        text = raw_text.strip()
        if not text:
            continue
        start, end = _locate_span(sentence, text, search_from)
        if start is not None:
            search_from = end
        args.append(Argument(text=text, role=role, start=start, end=end))
    return args


def parse_row(prompt: str, response: str):
    if response is None or response.strip() == UNK:
        return None
    m = PROMPT_RE.match(prompt.strip())
    if not m:
        return None
    predicate = m.group("pred").strip()
    bracketed = m.group("sent").strip()
    marker = f"[{predicate}]"
    marker_idx = bracketed.find(marker)
    if marker_idx == -1:
        return None
    sentence = (
        bracketed[:marker_idx]
        + predicate
        + bracketed[marker_idx + len(marker):]
    )
    pred_start = marker_idx
    pred_end = marker_idx + len(predicate)
    arguments = _parse_arguments(sentence, response)
    if not arguments:
        return None
    return _ParsedRow(sentence, predicate, pred_start, pred_end, arguments)


def build_examples(df: pd.DataFrame, split: str) -> list:
    examples = []
    for i, row in enumerate(df.itertuples(index=False)):
        parsed = parse_row(getattr(row, "prompt"), getattr(row, "response"))
        if parsed is None:
            continue
        examples.append(
            SRLExample(
                example_id=f"{split}-{i}",
                sentence=parsed.sentence,
                predicate=parsed.predicate,
                predicate_start=parsed.predicate_start,
                predicate_end=parsed.predicate_end,
                arguments=parsed.arguments,
            )
        )
    return examples


def write_jsonl(examples: Iterable, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path) -> list:
    examples = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(SRLExample.model_validate_json(line))
    return examples


def prepare(
    train_size: Optional[int] = 2000,
    val_size: Optional[int] = 500,
    test_size: Optional[int] = 500,
    seed: int = 13,
    force_download: bool = False,
) -> dict:
    print("== Building SRL datasets ==")
    raw_train = load_raw_split("train", force=force_download)
    raw_val = load_raw_split("validation", force=force_download)

    train_ex = build_examples(raw_train, "train")
    val_pool = build_examples(raw_val, "val")

    rng = random.Random(seed)
    rng.shuffle(train_ex)
    rng.shuffle(val_pool)

    if train_size is not None:
        train_ex = train_ex[:train_size]

    if val_size is None and test_size is None:
        half = len(val_pool) // 2
        val_ex, test_ex = val_pool[:half], val_pool[half:]
    else:
        v = val_size or 0
        t = test_size or 0
        val_ex = val_pool[:v]
        test_ex = val_pool[v : v + t]

    counts = {
        "train": write_jsonl(train_ex, DATA_DIR / "train.jsonl"),
        "validation": write_jsonl(val_ex, DATA_DIR / "validation.jsonl"),
        "test": write_jsonl(test_ex, DATA_DIR / "test.jsonl"),
    }
    print("Wrote:", json.dumps(counts, indent=2))
    print(f"(usable: train pool={len(train_ex)}, val pool={len(val_pool)})")
    return counts


def _norm(n):
    return None if n is not None and n < 0 else n


def _build_cli():
    p = argparse.ArgumentParser(description="Prepare PropBank SRL JSONL datasets.")
    p.add_argument("--train-size", type=int, default=2000, help="-1 for all")
    p.add_argument("--val-size", type=int, default=500, help="-1 for all")
    p.add_argument("--test-size", type=int, default=500, help="-1 for all")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force-download", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()
    prepare(
        train_size=_norm(args.train_size),
        val_size=_norm(args.val_size),
        test_size=_norm(args.test_size),
        seed=args.seed,
        force_download=args.force_download,
    )
    sys.exit(0)
