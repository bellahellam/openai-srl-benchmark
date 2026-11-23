# Benchmarking OpenAI Models on PropBank-style Semantic Role Labeling

A small, reproducible NLP experiment that measures how well OpenAI models perform
PropBank-style Semantic Role Labeling (SRL) under three settings:

- **RQ1 — Zero-shot:** instructions only.
- **RQ2 — Few-shot:** 2 / 5 / 10 in-context demonstrations.
- **RQ3 — Fine-tuning:** a model fine-tuned on labeled PropBank data.

All configurations are scored against gold annotations with labeled argument
span F1 (strict), plus relaxed F1, per-role F1, a confusion matrix, and an error
taxonomy.

## Data

The guide assumes a dataset supplied by the instructor. Here we instead fetch a
public, directly comparable PropBank SRL dataset from Hugging Face:
[`cu-kairos/propbank_srl_seq2seq`](https://huggingface.co/datasets/cu-kairos/propbank_srl_seq2seq)
(Apache-2.0, derived from OntoNotes/PropBank).

Each example is one sentence with one predicate (given to the model) and its gold
arguments. `src/data_loader.py` downloads the parquet shards, canonicalizes role
names (`ARG-0` -> `ARG0`, `ARG-M-TMP` -> `ARGM-TMP`), computes character offsets,
drops unannotated rows, and writes `data/{train,validation,test}.jsonl`. The
source has only train + validation splits, so the test set is carved from the
source validation split (disjoint from our validation set).

To swap in instructor-provided data later, produce JSONL matching `src/schema.py`
(`SRLExample`) and the rest of the pipeline is unchanged.

## Setup

Python 3.11 or 3.12 recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or requirements-lock.txt for exact pins
```

### API key (required for notebooks 02-05)

```bash
cp .env.example .env
# then edit .env and REPLACE the placeholder with your course-authorized key:
#   OPENAI_API_KEY=sk-...
```

The course API token is **not** your personal ChatGPT subscription. Ask Eric for
API access and permission to use the course's credits, and confirm which model
snapshot you are allowed to call (`SRL_MODEL` in `.env`).

## Running

Open the notebooks in order with the `Python 3 (srl .venv)` kernel:

1. `notebooks/01_data_exploration.ipynb` — fetch data, stats, role distribution. **No API key needed.**
2. `notebooks/02_zero_shot.ipynb` — zero-shot baseline (RQ1).
3. `notebooks/03_few_shot.ipynb` — 2/5/10-shot prompting (RQ2).
4. `notebooks/04_fine_tuning.ipynb` — fine-tuning (RQ3). **Spends credits; get approval.**
5. `notebooks/05_evaluation.ipynb` — comparison tables, figures, error analysis.

Or regenerate the datasets from the command line:

```bash
python -m src.data_loader --train-size 2000 --val-size 500 --test-size 500
# use -1 for any size to take everything available
```

## Project layout

```
data/            train/validation/test JSONL (+ parquet cache, gitignored)
notebooks/       01..05, the experiment workflow
src/
  schema.py        Pydantic schema shared across the pipeline
  data_loader.py   download + parse Hugging Face data -> JSONL
  prompts.py       system prompt, JSON output schema, few-shot builders
  srl_inference.py OpenAI calls (zero/few-shot, fine-tuning helpers)
  evaluation.py    P/R/F1 (micro/macro), spans, confusion, error taxonomy
  visualization.py role distribution, F1 comparison, confusion heatmap
results/         predictions.csv, metrics.json, error_analysis.csv, figures/
report.pdf       final write-up (placeholder; fill in with your findings)
```

## Metrics

Primary metric is **strict labeled argument span F1**: an argument counts as
correct only when its character span, its role, and its predicate all match the
gold annotation. Relaxed F1 (text match, offsets ignored) isolates boundary
errors from role errors. Schema-validity rate is reported separately so correct
JSON formatting is never mistaken for correct semantics.

## Methodology notes

- Demonstrations for few-shot come from the **train** split only, never test.
- Decoding is greedy (`temperature=0`) where the model allows it, for reproducibility.
- Record the exact model ID/snapshot and prompt version with your results.
- For a harder generalization test, build an unseen-predicate split (hold out
  predicate lemmas from training) and report it separately.

## Reproducibility / cost

Default split sizes (2000/500/500) keep a full prompting sweep cheap. A zero-shot
+ 2/5/10-shot run makes ~2000 API calls over the 500-sentence test set. Fine-tuning
has a different cost profile — get instructor approval before launching a job.
