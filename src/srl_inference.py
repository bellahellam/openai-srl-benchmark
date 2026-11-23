"""OpenAI inference for SRL: zero-shot, few-shot, and fine-tuned models.

Uses the Responses API with Structured Outputs (a JSON schema) so every reply
conforms to our schema. Model IDs and the API key come from the environment so
nothing secret is hard-coded; see .env.example.

Env vars:
    OPENAI_API_KEY      required
    SRL_MODEL           base model for prompting (default gpt-4o-mini-2024-07-18)
    SRL_FINETUNED_MODEL fine-tuned model id, set after a job completes

The fine-tuning helpers (export JSONL, upload, create/poll job) are thin wrappers
over the OpenAI SDK; run them only with course approval since training spends
credits differently from inference.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

try:
    from .schema import Argument, Prediction, SRLExample
    from .data_loader import _locate_span
    from .prompts import (
        OUTPUT_JSON_SCHEMA,
        SYSTEM_PROMPT,
        build_few_shot_block,
        build_user_input,
    )
except ImportError:
    from schema import Argument, Prediction, SRLExample  # type: ignore
    from data_loader import _locate_span  # type: ignore
    from prompts import (  # type: ignore
        OUTPUT_JSON_SCHEMA,
        SYSTEM_PROMPT,
        build_few_shot_block,
        build_user_input,
    )

DEFAULT_MODEL = os.environ.get("SRL_MODEL", "gpt-4o-mini-2024-07-18")
FINETUNED_MODEL = os.environ.get("SRL_FINETUNED_MODEL")
DEFAULT_FT_BASE = os.environ.get("SRL_FT_BASE_MODEL", "gpt-4o-mini-2024-07-18")


def get_client() -> OpenAI:
    """Create an OpenAI client, loading .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export the variable."
        )
    return OpenAI()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def _parse_response_json(raw: str, fallback_predicate: str) -> Prediction:
    """Parse model JSON into a Prediction; tolerate minor deviations."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return Prediction(predicate=fallback_predicate, arguments=[])
    args = []
    for a in data.get("arguments", []) or []:
        try:
            args.append(
                Argument(
                    text=str(a.get("text", "")),
                    role=str(a.get("role", "")),
                    start=a.get("start"),
                    end=a.get("end"),
                )
            )
        except Exception:
            continue
    return Prediction(predicate=data.get("predicate", fallback_predicate), arguments=args)



def _repair_offsets(sentence: str, pred: Prediction) -> Prediction:
    """Re-anchor each argument's offsets to where its text occurs in the sentence.

    Models reliably return correct argument TEXT but unreliable character
    positions. We relocate each span by searching for its text (exact, then
    whitespace-insensitive) using the same logic that places gold spans, so
    strict span F1 reflects span text + role, not the model's ability to count.
    If the text cannot be located, offsets are left as None (not credited).
    """
    repaired = []
    cursor = 0
    for a in pred.arguments:
        start, end = _locate_span(sentence, a.text, cursor)
        if start is not None:
            cursor = end
        repaired.append(Argument(text=a.text, role=a.role, start=start, end=end))
    return Prediction(predicate=pred.predicate, arguments=repaired)


def predict_one(
    client: OpenAI,
    example: SRLExample,
    *,
    model: str = DEFAULT_MODEL,
    demos: Optional[List[SRLExample]] = None,
    temperature: float = 0.0,
) -> tuple[Prediction, bool]:
    """Run SRL on one example. Returns (prediction, schema_valid)."""
    instructions = SYSTEM_PROMPT
    few_shot = build_few_shot_block(demos or [])
    user_input = (few_shot + "\n" + build_user_input(example)) if few_shot else build_user_input(example)

    kwargs = dict(
        model=model,
        instructions=instructions,
        input=user_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "srl_prediction",
                "strict": True,
                "schema": OUTPUT_JSON_SCHEMA,
            }
        },
    )
    # Fine-tuned snapshots and some base models accept temperature; reasoning
    # models reject it. Try with, fall back without.
    try:
        resp = client.responses.create(temperature=temperature, **kwargs)
    except Exception:
        resp = client.responses.create(**kwargs)

    raw = resp.output_text
    schema_valid = True
    try:
        json.loads(raw)
    except Exception:
        schema_valid = False
    prediction = _parse_response_json(raw, example.predicate)
    prediction = _repair_offsets(example.sentence, prediction)
    return prediction, schema_valid


def predict_batch(
    client: OpenAI,
    examples: List[SRLExample],
    *,
    model: str = DEFAULT_MODEL,
    demos: Optional[List[SRLExample]] = None,
    temperature: float = 0.0,
    progress: bool = True,
) -> List[dict]:
    """Run SRL over many examples. Returns a list of result dicts (one per example)."""
    iterator = examples
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(examples, desc=f"infer[{model}]")
        except Exception:
            pass

    results = []
    for ex in iterator:
        try:
            pred, valid = predict_one(
                client, ex, model=model, demos=demos, temperature=temperature
            )
            err = None
        except Exception as e:  # keep going; record the failure
            pred, valid, err = Prediction(predicate=ex.predicate), False, str(e)
        results.append(
            {
                "example_id": ex.example_id,
                "sentence": ex.sentence,
                "predicate": ex.predicate,
                "predicate_start": ex.predicate_start,
                "predicate_end": ex.predicate_end,
                "schema_valid": valid,
                "error": err,
                "prediction": pred.model_dump(),
            }
        )
    return results


def save_predictions(results: List[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_predictions(path) -> List[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #

def export_finetune_jsonl(examples: List[SRLExample], path) -> int:
    """Write chat-format JSONL for supervised fine-tuning."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for ex in examples:
            target = {
                "predicate": ex.predicate,
                "arguments": [
                    {"text": a.text, "role": a.role, "start": a.start, "end": a.end}
                    for a in ex.arguments
                ],
            }
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_input(ex)},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ]
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def upload_file(client: OpenAI, path) -> str:
    f = client.files.create(file=open(path, "rb"), purpose="fine-tune")
    return f.id


def create_finetune_job(
    client: OpenAI,
    training_file_id: str,
    *,
    validation_file_id: Optional[str] = None,
    base_model: str = DEFAULT_FT_BASE,
) -> str:
    kwargs = dict(training_file=training_file_id, model=base_model)
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    job = client.fine_tuning.jobs.create(**kwargs)
    return job.id


def poll_finetune_job(client: OpenAI, job_id: str, interval: int = 30) -> Optional[str]:
    """Block until a job finishes; return the fine-tuned model id or None."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  job {job_id}: {job.status}")
        if job.status in {"succeeded", "failed", "cancelled"}:
            return getattr(job, "fine_tuned_model", None)
        time.sleep(interval)
