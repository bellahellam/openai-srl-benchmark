"""Evaluation for PropBank-style SRL.

Primary metric: labeled argument span F1. An argument is correct only when its
span boundaries, its role, and its predicate all match the gold annotation.

We also report:
  - micro and macro P/R/F1,
  - a relaxed span mode (text match, offsets ignored) to separate span-boundary
    errors from role errors,
  - per-role F1,
  - schema validity rate (formatting vs semantics),
  - a role confusion matrix,
  - an error taxonomy for qualitative analysis.

Inputs are the result dicts produced by srl_inference.predict_batch plus the
gold SRLExample list they were produced from.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from .schema import SRLExample
except ImportError:
    from schema import SRLExample  # type: ignore


# --------------------------------------------------------------------------- #
# Tuple extraction
# --------------------------------------------------------------------------- #

def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).lower()


def gold_strict_tuples(ex: SRLExample) -> set:
    return ex.gold_tuples()


def gold_relaxed_tuples(ex: SRLExample) -> set:
    """(id, predicate, normalized_text, role) — offsets ignored."""
    return {
        (ex.example_id, ex.predicate, _norm_text(a.text), a.role)
        for a in ex.arguments
    }


def pred_strict_tuples(result: dict) -> set:
    eid = result["example_id"]
    ps, pe = result["predicate_start"], result["predicate_end"]
    out = set()
    for a in result["prediction"].get("arguments", []):
        if a.get("start") is not None and a.get("end") is not None:
            out.add((eid, ps, pe, a["start"], a["end"], a["role"]))
    return out


def pred_relaxed_tuples(result: dict) -> set:
    eid = result["example_id"]
    pred = result["predicate"]
    return {
        (eid, pred, _norm_text(a.get("text", "")), a["role"])
        for a in result["prediction"].get("arguments", [])
    }


# --------------------------------------------------------------------------- #
# Core P/R/F1
# --------------------------------------------------------------------------- #

def prf1(gold: set, predicted: set) -> Dict[str, float]:
    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _role_of(tup: tuple) -> str:
    return tup[-1]


def macro_f1_by_role(gold: set, predicted: set) -> Tuple[float, Dict[str, Dict]]:
    roles = {_role_of(t) for t in gold} | {_role_of(t) for t in predicted}
    per_role: Dict[str, Dict] = {}
    f1s = []
    for role in sorted(roles):
        g = {t for t in gold if _role_of(t) == role}
        p = {t for t in predicted if _role_of(t) == role}
        m = prf1(g, p)
        per_role[role] = m
        f1s.append(m["f1"])
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return macro, per_role


# --------------------------------------------------------------------------- #
# Top-level evaluate
# --------------------------------------------------------------------------- #

def _index_gold(gold_examples: List[SRLExample]) -> Dict[str, SRLExample]:
    return {ex.example_id: ex for ex in gold_examples}


def evaluate(results: List[dict], gold_examples: List[SRLExample]) -> dict:
    """Compute the full metric set for one configuration's predictions."""
    gold_by_id = _index_gold(gold_examples)

    g_strict, p_strict = set(), set()
    g_relaxed, p_relaxed = set(), set()
    schema_valid = 0
    pred_detected = 0  # the predicate was given, but did the model emit it back?
    n = 0

    for r in results:
        ex = gold_by_id.get(r["example_id"])
        if ex is None:
            continue
        n += 1
        schema_valid += 1 if r.get("schema_valid") else 0
        if _norm_text(r["prediction"].get("predicate", "")) == _norm_text(ex.predicate):
            pred_detected += 1
        g_strict |= gold_strict_tuples(ex)
        p_strict |= pred_strict_tuples(r)
        g_relaxed |= gold_relaxed_tuples(ex)
        p_relaxed |= pred_relaxed_tuples(r)

    micro_strict = prf1(g_strict, p_strict)
    micro_relaxed = prf1(g_relaxed, p_relaxed)
    macro_strict, per_role_strict = macro_f1_by_role(g_strict, p_strict)
    macro_relaxed, _ = macro_f1_by_role(g_relaxed, p_relaxed)

    return {
        "n_examples": n,
        "schema_valid_rate": schema_valid / n if n else 0.0,
        "predicate_echo_accuracy": pred_detected / n if n else 0.0,
        "strict": {
            "micro": micro_strict,
            "macro_f1": macro_strict,
            "per_role": per_role_strict,
        },
        "relaxed": {
            "micro": micro_relaxed,
            "macro_f1": macro_relaxed,
        },
    }


def summary_row(config_name: str, metrics: dict) -> dict:
    """Flatten the headline numbers for a comparison table."""
    s = metrics["strict"]["micro"]
    return {
        "configuration": config_name,
        "precision": round(s["precision"], 4),
        "recall": round(s["recall"], 4),
        "f1": round(s["f1"], 4),
        "macro_f1": round(metrics["strict"]["macro_f1"], 4),
        "relaxed_f1": round(metrics["relaxed"]["micro"]["f1"], 4),
        "predicate_echo_acc": round(metrics["predicate_echo_accuracy"], 4),
        "schema_valid_rate": round(metrics["schema_valid_rate"], 4),
        "n": metrics["n_examples"],
    }


# --------------------------------------------------------------------------- #
# Confusion matrix (role vs role), aligned by matching spans
# --------------------------------------------------------------------------- #

def role_confusion(results: List[dict], gold_examples: List[SRLExample]) -> Dict[Tuple[str, str], int]:
    """Count gold-role -> predicted-role for arguments whose spans match.

    Spans are aligned on (predicate, normalized text). Gold args with no span
    match are counted as gold_role -> 'NONE' (missed); predicted args with no
    gold match are 'NONE' -> pred_role (hallucinated).
    """
    gold_by_id = _index_gold(gold_examples)
    confusion: Dict[Tuple[str, str], int] = defaultdict(int)

    for r in results:
        ex = gold_by_id.get(r["example_id"])
        if ex is None:
            continue
        gold_map = {_norm_text(a.text): a.role for a in ex.arguments}
        pred_map: Dict[str, str] = {}
        for a in r["prediction"].get("arguments", []):
            pred_map[_norm_text(a.get("text", ""))] = a["role"]

        for text, grole in gold_map.items():
            prole = pred_map.get(text, "NONE")
            confusion[(grole, prole)] += 1
        for text, prole in pred_map.items():
            if text not in gold_map:
                confusion[("NONE", prole)] += 1
    return dict(confusion)


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #

ERROR_CATEGORIES = [
    "predicate_echo_error",
    "argument_omission",
    "argument_hallucination",
    "role_confusion",
    "boundary_error",
    "adjunct_confusion",
]


def classify_errors(results: List[dict], gold_examples: List[SRLExample]) -> List[dict]:
    """Produce per-error records for qualitative analysis.

    Categories:
      - predicate_echo_error : model did not echo the target predicate
      - argument_omission    : gold arg (by text) absent from prediction
      - argument_hallucination: predicted arg text not in gold
      - role_confusion       : same span/text, different role
      - boundary_error       : role correct, offsets differ from gold
      - adjunct_confusion    : one ARGM-* role confused with another ARGM-*
    """
    gold_by_id = _index_gold(gold_examples)
    rows: List[dict] = []

    for r in results:
        ex = gold_by_id.get(r["example_id"])
        if ex is None:
            continue
        eid = ex.example_id

        if _norm_text(r["prediction"].get("predicate", "")) != _norm_text(ex.predicate):
            rows.append(
                {
                    "example_id": eid,
                    "sentence": ex.sentence,
                    "predicate": ex.predicate,
                    "category": "predicate_echo_error",
                    "gold": ex.predicate,
                    "predicted": r["prediction"].get("predicate", ""),
                }
            )

        gold_by_text = {_norm_text(a.text): a for a in ex.arguments}
        pred_by_text = {_norm_text(a.get("text", "")): a for a in r["prediction"].get("arguments", [])}

        for text, ga in gold_by_text.items():
            pa = pred_by_text.get(text)
            if pa is None:
                rows.append(
                    {
                        "example_id": eid, "sentence": ex.sentence, "predicate": ex.predicate,
                        "category": "argument_omission", "gold": f"{ga.role}:{ga.text}", "predicted": "",
                    }
                )
                continue
            if pa["role"] != ga.role:
                cat = (
                    "adjunct_confusion"
                    if ga.role.startswith("ARGM") and pa["role"].startswith("ARGM")
                    else "role_confusion"
                )
                rows.append(
                    {
                        "example_id": eid, "sentence": ex.sentence, "predicate": ex.predicate,
                        "category": cat, "gold": f"{ga.role}:{ga.text}",
                        "predicted": f"{pa['role']}:{pa.get('text','')}",
                    }
                )
            elif (pa.get("start"), pa.get("end")) != (ga.start, ga.end):
                rows.append(
                    {
                        "example_id": eid, "sentence": ex.sentence, "predicate": ex.predicate,
                        "category": "boundary_error", "gold": f"{ga.role}[{ga.start}:{ga.end}]:{ga.text}",
                        "predicted": f"{pa['role']}[{pa.get('start')}:{pa.get('end')}]:{pa.get('text','')}",
                    }
                )

        for text, pa in pred_by_text.items():
            if text not in gold_by_text:
                rows.append(
                    {
                        "example_id": eid, "sentence": ex.sentence, "predicate": ex.predicate,
                        "category": "argument_hallucination", "gold": "",
                        "predicted": f"{pa['role']}:{pa.get('text','')}",
                    }
                )
    return rows


def error_category_counts(error_rows: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in error_rows:
        counts[row["category"]] += 1
    return dict(counts)
