"""End-to-end prompting experiment from the command line (no notebooks).

Steps: prepare data (if missing) -> zero-shot + few-shot inference over the test
set -> evaluate -> write results/metrics.json, predictions.csv, figures.

Fine-tuning is intentionally left to notebook 04 because it spends credits
differently and needs instructor approval.

    python run_experiment.py --shots 2 5 10
    python run_experiment.py --limit 25           # quick/cheap smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import data_loader as dl
import evaluation as ev
import prompts as pr
import srl_inference as si
import visualization as viz


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the SRL prompting benchmark.")
    ap.add_argument("--shots", type=int, nargs="*", default=[2, 5, 10])
    ap.add_argument("--limit", type=int, default=None, help="cap test examples (cost)")
    ap.add_argument("--skip-zero-shot", action="store_true")
    ap.add_argument("--prepare", action="store_true", help="force rebuild data")
    args = ap.parse_args()

    if args.prepare or not (Path("data/test.jsonl").exists() and Path("data/test.jsonl").stat().st_size):
        dl.prepare()

    train = dl.read_jsonl("data/train.jsonl")
    test = dl.read_jsonl("data/test.jsonl")
    if args.limit:
        test = test[: args.limit]
    print(f"train={len(train)} test={len(test)}")

    client = si.get_client()
    metrics = {}
    loaded = {}

    if not args.skip_zero_shot:
        print("== zero-shot ==")
        res = si.predict_batch(client, test, model=si.DEFAULT_MODEL)
        si.save_predictions(res, "results/predictions_zero_shot.jsonl")
        metrics["zero-shot"] = ev.evaluate(res, test)
        loaded["zero-shot"] = res

    for k in args.shots:
        print(f"== {k}-shot ==")
        demos = pr.select_demos(train, k, seed=7)
        res = si.predict_batch(client, test, model=si.DEFAULT_MODEL, demos=demos)
        si.save_predictions(res, f"results/predictions_few_shot_{k}.jsonl")
        metrics[f"{k}-shot"] = ev.evaluate(res, test)
        loaded[f"{k}-shot"] = res

    rows = [ev.summary_row(name, m) for name, m in metrics.items()]
    for r in rows:
        print(r)

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    if rows:
        viz.plot_overall_f1(rows)
        role_f1 = {n: {r: pr_["f1"] for r, pr_ in m["strict"]["per_role"].items()} for n, m in metrics.items()}
        viz.plot_role_f1_comparison(role_f1)
    print("wrote results/metrics.json and figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
