# data_labeling — Factual Precision & Recall Judge

A modular LLM-as-a-judge package for evaluating **factual precision** (correctness of the generated conclusions) and **factual recall** (coverage/comprehensiveness of the generated conclusions).

## Setup

### 1. Install the project

From the repo root:

```bash
pip install -e .
```

This registers `data_labeling` as an importable package via the `pyproject.toml` `package-dir` mapping.

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file at the repo root and load it with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

For Azure OpenAI, set `OPENAI_BASE_URL` and `AZURE_OPENAI_KEY` instead.

---

## Quick Start

### Precision judge — are the LLM's facts grounded?

```python
from data_labeling import make_precision_judge

# gpt-5.4-mini · few-shot prompt · no reasoning · temperature 0.2
analyzer = make_precision_judge()

result = analyzer.compute_factual_precision(
    llm_atomic_facts=[
        "Aspirin significantly reduces fever in adults.",
        "Ibuprofen has no effect on pain.",
    ],
    ground_truth_text="Aspirin is a well-established antipyretic...",
)

print(result["factual_precision"])        # e.g. 0.5
print(result["all_precision_metrics"])    # all three scoring variants
```

### Recall judge — does the LLM cover the key facts?

```python
from data_labeling import make_recall_judge

# gpt-5.4-mini · zero-shot prompt · no reasoning · temperature 1.0
analyzer = make_recall_judge()

result = analyzer.compute_factual_recall(
    llm_response_text="The study found that aspirin reduces fever...",
    article_atomic_facts=[
        "Aspirin reduces fever.",
        "Pain relief was statistically significant.",
    ],
)

print(result["factual_recall"])    # e.g. 0.5
```

---

## Judge Configuration

| Parameter        | Precision judge          | Recall judge             |
|------------------|--------------------------|--------------------------|
| Model            | `gpt-5.4-mini`           | `gpt-5.4-mini`           |
| Prompt style     | Few-shot (6 examples)    | Zero-shot                |
| Reasoning        | `"none"` (responses API) | `"none"` (responses API) |
| Temperature      | 0.2                      | 1.0                      |
| Max tokens       | 1 024                    | 1 024                    |

---

## Using a Custom Judge Backend

`LLMJudgeBase` defines the interface: implement `complete()` to wrap any model API. For these custom implementations, you need to adapt the call signature to your provider's SDK — adjust authentication, hyperparameters, message formatting, streaming, and response parsing as needed. If your model requires a different prompt structure (e.g., no system role, no temperature, or a different JSON schema), subclass the relevant `Analyzer` and override `check_fact_judgment` / `check_fact_support` accordingly.

```python
from data_labeling import LLMJudgeBase, FactualPrecisionAnalyzer, FactualRecallAnalyzer
from typing import Dict, List, Optional, Tuple

class MyModelJudge(LLMJudgeBase):
    """Wrap your own model API here."""

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        # Call your provider SDK, return (text_response, token_usage_dict).
        # token_usage_dict keys: input_tokens, output_tokens, total_tokens.
        response = my_sdk.call(messages=messages, temperature=temperature)
        return response.text, {
            "input_tokens": response.usage.input,
            "output_tokens": response.usage.output,
            "total_tokens": response.usage.total,
        }

judge = MyModelJudge()
precision_analyzer = FactualPrecisionAnalyzer(llm_judge=judge, temperature=0.2)
recall_analyzer    = FactualRecallAnalyzer(llm_judge=judge, temperature=1.0)
```

For Claude, Gemini, or GPT-5 reasoning models backed by `sciconharness`, use `ModelJudgeFactory.create()`:

```python
from data_labeling import ModelJudgeFactory, FactualPrecisionAnalyzer

judge = ModelJudgeFactory.create(
    model="claude-sonnet-4-5",
    claude_thinking_mode="disabled",
)
analyzer = FactualPrecisionAnalyzer(llm_judge=judge, temperature=0.2)
```

---

## Evaluating Judge Performance Against Human Annotations

Human-annotated labels are stored in `data/llm-judge-human-annotations/`:

| File | Task | Labels |
|------|------|--------|
| `factual_precision_annotations.json` | Precision (task1) | `SUPPORTED` / `CONTRADICTED` / `NOT SUPPORTED` |
| `factual_recall_annotations.json`    | Recall (task2)    | `SUPPORTED` / `NOT SUPPORTED` |

Each record has fields: `fact`, `reference_text`, `consensus_label`, `disagreed`, `annotator_A_label`, `annotator_B_label`, `annotator_C_label` (if annotator_A and annotator_B disagreed). Primarily focus on `consensus_label` to evaluate LLM judge performance, along with `annotator_A_label` and `annotator_B_label` to measure whether your LLM judge agrees with one of the experts than the expert annotators agree between themselves.

### Excluding few-shot examples from evaluation

The few-shot prompts embed concrete `(source_text, atomic_fact)` pairs drawn directly from the annotation files. Since these examples are used for in-context learning, evaluating on them would be data leakage — the judge has effectively "seen" them. **Always filter out the keys that appear in your few-shot examples before computing any metrics**, regardless of whether you are running zero-shot or few-shot mode. The keys below correspond to the default prompts (`factual_precision_few_shot_prompt` / `factual_recall_few_shot_prompt`). If you have substituted your own few-shot examples, replace these sets with the keys from your own prompts instead.

**Precision** — keys to exclude `(doi, atomic_fact_index)`:

```python
FACTUAL_PRECISION_FEW_SHOT_DEMO_KEYS: frozenset[tuple[str, int]] = frozenset({
    ("10.1002/14651858.CD000247.pub4", 6),
    ("10.1002/14651858.CD002243.pub5", 2),
    ("10.1002/14651858.CD002122.pub3", 0),
    ("10.1002/14651858.CD002974.pub3", 1),
    ("10.1002/14651858.CD003333.pub4", 3),
    ("10.1002/14651858.CD003477.pub5", 9),
})
```

**Recall** — keys to exclude `(doi, atomic_fact_index)`:

```python
FACTUAL_RECALL_FEW_SHOT_DEMO_KEYS: frozenset[tuple[str, int]] = frozenset({
    ("10.1002/14651858.CD000247.pub4", 0),
    ("10.1002/14651858.CD000547.pub3", 3),
    ("10.1002/14651858.CD002042.pub6", 0),
    ("10.1002/14651858.CD001920.pub4", 1),
    ("10.1002/14651858.CD001920.pub4", 7),
    ("10.1002/14651858.CD002769.pub6", 4),
})
```

Example filtering before evaluation:

```python
import json

with open("data/llm-judge-human-annotations/factual_precision_annotations.json") as f:
    precision_rows = json.load(f)

eval_rows = [
    row for row in precision_rows
    if (row["doi"], row["atomic_fact_index"]) not in FACTUAL_PRECISION_FEW_SHOT_DEMO_KEYS
]
```

---

## Output Schema

### `compute_factual_precision` returns:

```python
{
    "factual_precision": float,          # primary score (main / total)
    "precision_function_used": str,      # e.g. "main"
    "all_precision_metrics": {           # all three scoring variants
        "main":           float,         # (E/T) × (1 - C/T)  ← primary
        "entailment":     float,         # supported / total
        "no_contradiction": float,       # (total - contradicted) / total
    },
    "total_llm_facts":      int,
    "supported_facts":      int,
    "contradicted_facts":   int,
    "not_supported_facts":  int,
    "supported_facts_list":     list[str],
    "contradicted_facts_list":  list[str],
    "not_supported_facts_list": list[str],
    "precision_details": [               # one entry per fact
        {
            "llm_fact":          str,
            "judgment":          str,    # "SUPPORTED" | "CONTRADICTED" | "NOT SUPPORTED"
            "excerpts":          list[str],
            "justification":     str,
            "raw_json_response": str,
            "token_usage":       {"input_tokens": int, "output_tokens": int, "total_tokens": int},
        },
        ...
    ],
    "ground_truth_source": str,          # always "reference"
    "token_usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
}
```

### `compute_factual_recall` returns:

```python
{
    "factual_recall":       float,       # supported_facts / total_article_facts
    "total_article_facts":  int,
    "supported_facts":      int,
    "not_supported_facts":  list[str],   # text of unsupported facts
    "coverage_details": [                # one entry per fact
        {
            "article_fact":  str,
            "is_supported":  bool,
            "excerpts":      list[str],  # passages from LLM response that support/refute
            "justification": str,
            "token_usage":   {"input_tokens": int, "output_tokens": int, "total_tokens": int},
        },
        ...
    ],
    "token_usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
}
```

---

## Environment Variables

| Variable                                   | Default | Description                                               |
|--------------------------------------------|---------|-----------------------------------------------------------|
| `OPENAI_API_KEY`                           | —       | OpenAI API key (required for default judges)              |
| `OPENAI_BASE_URL`                          | —       | Override base URL (Azure or compatible)                   |
| `AZURE_OPENAI_KEY`                         | —       | Azure API key (alternative to `OPENAI_API_KEY`)           |
| `OPENAI_API_VERSION`                       | `2025-04-01-preview` | Azure API version                            |
| `DATA_LABELING_JUDGE_DELAY_S`              | `0`     | Sleep (seconds) between per-fact API calls                |
| `DATA_LABELING_PRECISION_JUDGE_MAX_ATTEMPTS` | `3`   | Max retries when precision judge output is unparseable    |
| `DATA_LABELING_RECALL_JSON_REPAIR_MAX`     | `1`     | Max repair passes when recall judge output is unparseable |

---

## Package Structure

```
scripts/data_labeling/          ← importable as `data_labeling` after pip install -e .
├── __init__.py                 ← make_precision_judge(), make_recall_judge(), all exports
├── factual_precision_analyzer.py   ← FactualPrecisionAnalyzer
├── factual_recall_analyzer.py      ← FactualRecallAnalyzer
├── model_class/
│   ├── __init__.py
│   └── llm_judge_base.py       ← LLMJudgeBase, OpenAICompatibleJudge, ModelJudgeFactory, ...
└── utils/
    ├── __init__.py
    ├── prompts.py              ← all judge prompt templates
    └── precision_equations.py  ← main / entailment / no_contradiction scorers
```
