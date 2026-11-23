"""Prompts, output schema, and few-shot demonstration builders for SRL.

The model is given a sentence and a specific predicate (marked by character
offsets) and must return that predicate's PropBank arguments as JSON. Keeping
the predicate fixed matches the dataset and isolates argument labeling, which is
what we evaluate.
"""

from __future__ import annotations

import json
from typing import List

try:
    from .schema import SRLExample
except ImportError:
    from schema import SRLExample  # type: ignore

# Role inventory shown to the model (PropBank core + common modifiers).
ROLE_INVENTORY = [
    "ARG0", "ARG1", "ARG2", "ARG3", "ARG4",
    "ARGM-TMP", "ARGM-LOC", "ARGM-MNR", "ARGM-CAU", "ARGM-DIR",
    "ARGM-GOL", "ARGM-PRP", "ARGM-ADV", "ARGM-DIS", "ARGM-MOD",
    "ARGM-NEG", "ARGM-EXT", "ARGM-PRD",
]

SYSTEM_PROMPT = """\
You are an expert NLP system specializing in PropBank-style Semantic Role Labeling.

You are given a sentence and ONE target predicate, identified by its character
offsets in the sentence. Label only the arguments of that target predicate.

For the target predicate:
- Identify all semantic arguments expressed in the sentence.
- Assign each argument a PropBank role: core roles ARG0-ARG4 and ARGM-* modifiers
  (for example ARGM-TMP time, ARGM-LOC location, ARGM-MNR manner, ARGM-CAU cause,
  ARGM-NEG negation, ARGM-MOD modal, ARGM-DIR direction, ARGM-GOL goal).
- Return the exact argument surface text and its character offsets (start
  inclusive, end exclusive) in the given sentence.

Rules:
- ARG0 is the proto-agent and ARG1 the proto-patient; these are semantic roles,
  not grammatical subject/object. Decide by the predicate's frame, not word order.
- Do not invent arguments that are not expressed in the sentence.
- Label arguments for the target predicate only; ignore other predicates.
- Return ONLY JSON matching the required schema, with no extra commentary.
"""

# JSON Schema used for Structured Outputs (Responses API `text.format`).
OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "predicate": {"type": "string"},
        "arguments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "role": {"type": "string", "enum": ROLE_INVENTORY},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["text", "role", "start", "end"],
            },
        },
    },
    "required": ["predicate", "arguments"],
}


def build_user_input(example: SRLExample) -> str:
    """Render one test example into the user message content."""
    return (
        f"Sentence: {example.sentence}\n"
        f"Target predicate: {example.predicate!r} "
        f"at offsets [{example.predicate_start}, {example.predicate_end})\n"
        f"Label the arguments of this predicate."
    )


def _example_to_demo(example: SRLExample) -> str:
    """Render a gold example as an input/output demonstration block."""
    target = {
        "predicate": example.predicate,
        "arguments": [
            {"text": a.text, "role": a.role, "start": a.start, "end": a.end}
            for a in example.arguments
        ],
    }
    return (
        f"{build_user_input(example)}\n"
        f"Output: {json.dumps(target, ensure_ascii=False)}"
    )


def build_few_shot_block(demos: List[SRLExample]) -> str:
    """Build the in-context demonstration text from gold examples."""
    if not demos:
        return ""
    blocks = [f"Example {i + 1}:\n{_example_to_demo(d)}" for i, d in enumerate(demos)]
    return (
        "Here are labeled examples. Follow the same conventions.\n\n"
        + "\n\n".join(blocks)
        + "\n\nNow label the following.\n"
    )


def select_demos(pool: List[SRLExample], k: int, seed: int = 7) -> List[SRLExample]:
    """Pick k demonstrations from the training pool, biased toward role coverage.

    Greedily chooses examples that introduce previously-unseen roles so the demos
    span a range of argument structures, then fills any remainder at random.
    Demonstrations are drawn from the TRAIN pool only (never test).
    """
    import random

    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    chosen: List[SRLExample] = []
    seen_roles: set = set()
    for ex in shuffled:
        roles = {a.role for a in ex.arguments}
        if roles - seen_roles:
            chosen.append(ex)
            seen_roles |= roles
        if len(chosen) >= k:
            break
    for ex in shuffled:
        if len(chosen) >= k:
            break
        if ex not in chosen:
            chosen.append(ex)
    return chosen[:k]
