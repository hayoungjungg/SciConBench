# SciConBench

SciConBench is a large-scale, live benchmark for evaluating AI agents on *scientific conclusion synthesis*: a long-horizon task spanning web retrieval, evidence quality assessment, conflict reconciliation, and integration of heterogeneous findings into long-form, expert-level conclusions. 

This repository contains the code to reproduce our results and run benchmark evaluations using *SciConHarness*, our **clean-room evaluation protocol** that controls access to web search and browsing tools to prevent agents from retrieving reference ground-truth conclusions from the open web and maintain temporal integrity, ensuring valid capability measurement.

- 📄 **Paper**: [Can AI Agents Synthesize Scientific Conclusions?](https://arxiv.org/abs/XXXX.XXXXX)
- 🤗 **Dataset**: [hayoungjung/SciConBench on HuggingFace](https://huggingface.co/datasets/hayoungjung/SciConBench)

---

## 🗂️ Repository Structure

Each subdirectory marked `→ README.md` above contains a dedicated README guide.

```
SciConBench/
├── evaluate.py                  # End-to-end evaluation script (Steps 1–6)
├── pyproject.toml               # Package definition (pip install -e .)
├── requirements.txt             # All Python dependencies
├── .env.example                 # Template for API keys (copy → .env)
├── sciconharness/               # Evaluation harness package  → README.md
│   ├── harness.py               # SciConHarness Python API
│   ├── cli_scripts/             # Querying CLIs (single query, batch)
│   ├── mcp_client/              # Agentic loop, LLM providers, clean-room filters  → README.md
│   ├── mcp_server/              # Local MCP server (web search + browse tools)     → README.md
│   ├── remote_mcp_servers/      # HTTP MCP servers for OpenAI Deep Research agents → README.md
│   └── utils/
└── scripts/
    ├── download_data.py         # Download pre-computed data from Google Drive
    ├── data_preprocessing/
    │   ├── atomic_fact_generation/  # Decompose text into atomic facts  → README.md
    │   └── question_generation/     # Generate benchmark questions      → README.md
    ├── data_labeling/           # LLM judge for factual precision & recall  → README.md
    ├── analysis/                # Reproduce tables, figures, and plots       → README.md
    └── audits/                  # OpenEvidence, Google AI Overview, Google AI Mode audits  → README.md
```


---

## ⚙️ Setup

**Requirements:** Python ≥ 3.10

```bash
git clone https://github.com/your-org/SciConBench.git
cd SciConBench
pip install -e .
cp .env.example .env  # fill in API keys for the providers you want to use
```

Minimum keys in `.env` for a standard evaluation of an OpenAI model:

```
OPENAI_API_KEY=sk-...          # or AZURE_OPENAI_KEY + OPENAI_BASE_URL for Azure
SERPER_API_KEY=...             # Google web search
JINA_API_KEY=...               # Webpage fetching
S2_API_KEY=...                 # Semantic Scholar search
```

**An OpenAI API key is always required** — it is used for atomic fact generation and LLM-judge evaluation regardless of which model you benchmark. For evaluating Claude, Gemini, or Perplexity, add their respective keys as well (see `.env.example`).

The dataset is hosted at [`hayoungjung/SciConBench`](https://huggingface.co/datasets/hayoungjung/SciConBench) on HuggingFace and updated monthly with new samples.

```python
from datasets import load_dataset

ds = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")
print(f"{len(ds)} samples, {sum(len(f) for f in ds['all_facts'])} atomic facts total")
```

---

## 🚀 Quick Start — End-to-End Evaluation

`evaluate.py` walks through the full pipeline in six steps:

| Step | Description |
|------|-------------|
| 1 | Load the SciConBench dataset from HuggingFace |
| 2 | Explore the dataset and build clean-room filtering variables |
| 3 | Query a model via SciConHarness across all three configs (`no_tools`, `tools`, `tools_filter`) |
| 4 | Generate atomic facts from model conclusions (all three configs) |
| 5 | Label factual precision and recall (all three configs) |
| 6 | Compute macro-averaged Precision / Recall / F1 from labeled results |

```bash
# Run the full pipeline (default: 2 examples, gpt-5.1, openai provider)
python evaluate.py

# Use a different model / provider
python evaluate.py --model claude-sonnet-4-5 --provider claude
python evaluate.py --model gemini-3-pro-preview --provider gemini

# Run on more examples
python evaluate.py --n 10
```

---

### Python API Example 

Evaluating models & deep research agents using SciConHarness:

```python
import asyncio, os
from datasets import load_dataset
from sciconharness import SciConHarness

ds           = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")
all_titles   = list(ds["title"])
doi_to_title = {row["doi"]: row["title"] for row in ds}

harness = SciConHarness(
    provider         = "openai",
    model            = "gpt-5.1",
    api_key          = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY"),
    base_url         = os.environ.get("OPENAI_BASE_URL"),
    api_version      = os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview"),
    enable_tools     = True,
    enable_filtering = True,
    cochrane_titles  = all_titles,
    doi_to_title     = doi_to_title,
    save_results     = True,
)

async def main():
    async with harness:
        row = ds[0]
        response, usage = await harness.query(
            row["question"],
            doi              = row["doi"],
            publication_date = row["publication_date"],
        )
        print(response)

asyncio.run(main())
```

For batch evaluation using the CLI, see [`sciconharness/cli_scripts/README.md`](sciconharness/cli_scripts/README.md).


## 🤖 Supported Models

| Provider | Example models |
|----------|---------------|
| `openai` | `gpt-5.1`, `o4-mini-deep-research`, `o3-deep-research` |
| `claude` | `claude-sonnet-4-5`, `claude-haiku-4.5` |
| `gemini` | `gemini-3-pro-preview` |
| `perplexity` | `sonar-reasoning-pro`, `sonar-deep-research` |

See [`sciconharness/README.md`](sciconharness/README.md) for the full configuration reference, clean-room protocol details, model-specific behaviour, and extension guides (custom filters, new tools, new LLM providers & models). SciConHarness can be easily customized to support the evaluations of new frontier models and AI agents.


---

## ♻️ Reproducibility & Supporting Future Research

To reproduce paper results, support further research, and align/validate new LLM judges using expert annotations, we provide the paper's labeled facts, model responses, preprocessed atomic facts, and expert annotations on Google Drive. Download them with:

```bash
python scripts/download_data.py
```

See [`scripts/analysis/README.md`](scripts/analysis/README.md) to reproduce paper tables, figures, and plots.


---

## ✏️ Citation & License

Our GitHub repository employs the MIT License.

If you find our work helpful, please use the following citation.

```bibtex
Will be released soon
```

## 🤝 Contact

We welcome any contributions, pull requests, or issues! To do so, please either file a new pull request or issue and fill in the corresponding templates accordingly. We'll be sure to follow up shortly!

Contact: Hayoung Jung (Email: hayoung@cs.princeton.edu)
