# Atomic Fact Generation

Decomposes long-form text into independent and complete **atomic facts** using our LLM pipeline.

---

## Pipeline

Each stage is configurable; stages 3–5 can be individually disabled.

| # | Stage | Description |
|---|-------|-------------|
| 0 | Preprocessing | Split text into sentences; merge bullet points; filter non-content |
| 1 | Decomposition | Extract initial atomic facts per sentence |
| 2 | Decontextualization | Resolve vague pronouns and references |
| 3 | Incomplete detection | Rewrite facts that depend on missing context *(optional)* |
| 4 | Irrelevant filtering | Remove facts not relevant to the question *(optional)* |
| 5 | Redundant filtering | Keep the most atomic, non-duplicate set per sentence *(optional)* |

---

## Setup

### Environment variables

Create a `.env` file (or export variables) before running:

```bash
# Azure OpenAI
OPENAI_BASE_URL=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>

# OpenAI (only if using provider='openai')
OPENAI_API_KEY=<your-key>
```

### Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # or your preferred spaCy model
```

---

## Usage

### Python API

```python
from atomic_fact_generation import AtomicFactGenerator

# Default config (reads config/model_config.yaml), which is the same as in the paper
generator = AtomicFactGenerator()

facts, para_breaks, metadata = generator.run(
    generation="Breast stimulation appears beneficial ...",
    question="What are the benefits and harms of breast stimulation for labour induction?",
)

for sentence, sentence_facts in facts:
    print(sentence)
    for f in sentence_facts:
        print(f"  - {f}")
```

#### Custom per-component models

While we experimented with a variety of configurations to balance cost and quality,
users may customize our LLM pipeline using different models, hyperparameters, etc.

```python
from atomic_fact_generation import AtomicFactGenerator
from atomic_fact_generation import ModelConfig

generator = AtomicFactGenerator(
    model_configs={
        "decomposition": ModelConfig(
            model="gpt-5.1", provider="azure",
            reasoning_effort="none", verbosity="low",
        ),
        "incomplete_detection": ModelConfig(
            model="gpt-5-mini", provider="azure",
            reasoning_effort="minimal", verbosity="low",
        ),
    }
)
```

Unspecified components fall back to `config/model_config.yaml`.

#### Disable optional stages

```python
facts, _, metadata = generator.run(
    generation="...",
    question="...",
    enable_incomplete_detection=False,
    enable_irrelevant_filtering=False,
    enable_redundant_filtering=False,
)
```

### CLI

Run from `scripts/data_preprocessing/` (or anywhere by providing full paths):

```bash
python -m atomic_fact_generation.generate_atomic_facts \
    --text     "Your scientific text here ..." \
    --question "What are the effects of X on Y?" \
    --output-file results/output.json
```

**Optional flags:**

| Flag | Description |
|------|-------------|
| `--model-configs <path>` | Path to a YAML or JSON model-config file |
| `--verbose` | Print step-by-step fact comparisons (requires logging configured) |
| `--disable-incomplete-detection` | Skip stage 3 |
| `--disable-irrelevant-filtering` | Skip stage 4 |
| `--disable-redundant-filtering` | Skip stage 5 |

---

## Configuration

### `config/model_config.yaml`

Defines the default model for each pipeline component:

```yaml
decomposition:
  model: gpt-5.1
  provider: azure
  reasoning_effort: none
  verbosity: low
  reasoning_summary: auto

incomplete_detection:
  model: gpt-5-mini
  provider: azure
  reasoning_effort: minimal
  verbosity: low
  reasoning_summary: auto
# ... etc.
```

### `ModelConfig` parameters

| Parameter | Applies to | Description |
|-----------|-----------|-------------|
| `model` | all | Deployment/model name |
| `provider` | all | `"azure"` or `"openai"` |
| `reasoning_effort` | Responses API (GPT-5 family) | `"none"` / `"minimal"` / `"low"` / `"medium"` / `"high"` |
| `verbosity` | Responses API (GPT-5 family) | `"low"` / `"medium"` / `"high"` |
| `reasoning_summary` | Responses API (GPT-5 family) | `"auto"` / `"none"` / `"brief"` / `"detailed"` |
| `temperature` | Chat Completions (gpt-5-chat, non-GPT-5) | `0`–`2` |

> **GPT-5 family** (`gpt-5`, `gpt-5.1`, `gpt-5-mini`, etc.) uses the **Responses API** — temperature is ignored.  
> **`gpt-5-chat` / `gpt-5-chat-latest`** uses **Chat Completions** — reasoning parameters are ignored.

### Loading configs programmatically

```python
from atomic_fact_generation import load_configs_from_yaml, load_configs_from_json

# From YAML
configs = load_configs_from_yaml("config/model_config.yaml")

# From JSON
configs = load_configs_from_json("my_configs.json")
```

---

## Output format

`generator.run()` returns `(final_facts_pairs, para_breaks, metadata)`.

Use `format_output_for_json()` to serialise everything:

```python
from atomic_fact_generation import format_output_for_json
import json

output = format_output_for_json(facts, para_breaks, metadata)
with open("output.json", "w") as f:
    json.dump(output, f, indent=2)
```

**Top-level keys:**

```
final_atomic_facts_pairs  — list of [sentence, [fact, ...]]
paragraph_breaks          — list of sentence-index boundaries
metadata/
  initial_atomic_facts_pairs
  decontextualized_atomic_facts_pairs
  rewritten_missing_atomic_facts_pairs
  dependent_facts_metadata   — {sent|||fact: [type, explanation, class, rewrite]}
  irrelevant_facts_metadata  — {sent|||fact: [question, statement, reasoning, class]}
  redundant_facts_metadata   — {sent|||fact: [statement, reasoning, class, redundant_with]}
  token_usage                — per-component and total token counts
```

---

## Example scripts

As additional examples, we provide useful, applied scripts using our pipeline in `example_scripts/`:

| Script | Description |
|--------|-------------|
| `decompose_generated_conclusions.py` | Process model query-log result files (`result.json`) |
| `decompose_cdsr_conclusions.py` | Batch-process CDSR review articles from `data.json` |
| `run_bulk_atomic_facts.py` | Shardable, parallel-friendly version of the above for `sciconharness/logs/` |

Both `decompose_*` scripts accept `--help` for full argument documentation and can be run directly:

```bash
# From scripts/data_preprocessing/atomic_fact_generation/example_scripts/

# Process ALL models in the logs directory
python decompose_generated_conclusions.py \
    --logs-dir  ../../../../data_querying/logs \
    --output-dir output/

# Process a single model
python decompose_generated_conclusions.py \
    --logs-dir  ../../../../data_querying/logs \
    --output-dir output/ \
    --model gpt-5.1_tools_filter

python decompose_cdsr_conclusions.py \
    --batch 0 \
    --data-path      ../../../../data/review_articles/data.json \
    --output-dir     output/ \
    --questions-path ../../../../data/preprocessed_qa/generated_questions.json
```

#### Bulk, sharded, parallel runs (`run_bulk_atomic_facts.py`)

For processing one `sciconharness/logs/<model>/` directory at scale, `run_bulk_atomic_facts.py`
splits the DOIs into shards that can be run as **separate parallel processes** (optionally against
different API keys/endpoints per shard), and writes a resumable JSONL file per shard:

```bash
# From scripts/data_preprocessing/atomic_fact_generation/example_scripts/

# See which model directories are available
python run_bulk_atomic_facts.py --logs-dir ../../../../sciconharness/logs --list-models

# Launch shards 0..3 in parallel (each can use a distinct key/endpoint)
for i in 0 1 2 3; do
    python run_bulk_atomic_facts.py \
        --logs-dir  ../../../../sciconharness/logs \
        --model     qwen_qwen3.5-9b_tools_filter \
        --api-key   "$AZURE_OPENAI_KEY" --base-url "$OPENAI_BASE_URL" \
        --shard-id  "$i" --num-shards 4 \
        --output-dir output/ &
done
wait

# Merge all 4 shards into the single dict-format qwen_qwen3.5-9b_tools_filter_atomic_facts.json
python run_bulk_atomic_facts.py \
    --model qwen_qwen3.5-9b_tools_filter \
    --num-shards 4 \
    --output-dir output/ \
    --merge
```

Re-running a shard with the same `--output-dir`/`--model`/`--shard-id`/`--num-shards` resumes
(skips DOIs already present in that shard's JSONL).

---

## Directory structure

```
atomic_fact_generation/
├── __init__.py                  # Public API
├── generate_atomic_facts.py     # AtomicFactGenerator + CLI
├── config/
│   ├── model_config.py          # ModelConfig, UnifiedLLMClient, config loaders
│   └── model_config.yaml        # Default per-component model settings
├── utils/
│   ├── prompts.py               # All LLM prompt strings
│   ├── helper_atomic_facts.py   # Parsing and preprocessing helpers
│   └── calculate_pricing.py     # Token cost estimation
└── example_scripts/
    ├── decompose_generated_conclusions.py
    └── decompose_cdsr_conclusions.py
```
