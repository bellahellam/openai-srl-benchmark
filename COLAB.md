# Running this project on Google Colab

The code runs on Colab unchanged. You only add a setup cell at the top of each
notebook to (a) get the repo, (b) install dependencies, and (c) inject your API
key from Colab Secrets. The `.venv/` from local dev is not used on Colab.

## One-time: store your key in Colab Secrets
Left sidebar -> key icon (Secrets) -> add a secret named `OPENAI_API_KEY` with
your course-authorized key, and enable "Notebook access".

## Paste this as the FIRST cell of each notebook (before the existing bootstrap cell)

```python
# --- Colab setup ---
import os, sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # 1. Get the project. Either clone from your Git remote...
    if not os.path.exists("/content/openai-srl-benchmark"):
        !git clone <YOUR_REPO_URL> /content/openai-srl-benchmark
    %cd /content/openai-srl-benchmark

    # ...or, if you uploaded a zip instead, unzip it and %cd into it.

    # 2. Install dependencies (session is ephemeral; rerun each session).
    !pip install -q -r requirements.txt

    # 3. Inject the API key from Colab Secrets into the environment.
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    # Optional: override the model if your course uses a different snapshot.
    # os.environ["SRL_MODEL"] = "gpt-4o-mini-2024-07-18"
```

Then run the notebook's existing bootstrap cell and the rest as normal.

## Notes
- Colab runs Python 3.12; the code is compatible (no 3.11-only syntax).
- Data is fetched from Hugging Face at runtime, so nothing extra to upload for data.
- `data/`, `results/` are written inside the Colab VM and are lost when the
  session ends. Save anything you want to keep to Google Drive, e.g.:
  `from google.colab import drive; drive.mount('/content/drive')` then copy
  `results/` there.
- No GPU is needed: all computation is remote OpenAI API calls.
- Fine-tuning (notebook 04) still spends credits and needs instructor approval;
  it works the same on Colab.
