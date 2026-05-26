# Question Generation

Converts structured research paper abstracts into answerable research questions.
Supports **Azure OpenAI** and the standard **OpenAI API**, including reasoning models.

## Setup

### 1. Install dependencies

From the repo root:
```bash
pip install -e .
```

### 2. Configure environment variables

Edit `.env` with the credentials for whichever provider you use:

```dotenv
# Azure OpenAI
AZURE_OPENAI_KEY=<your-key>
OPENAI_BASE_URL=<your-endpoint>       # e.g. https://<name>.cognitiveservices.azure.com/
OPENAI_API_VERSION=<api-version>      # e.g. 2025-04-01-preview

# Standard OpenAI API
OPENAI_API_KEY=<your-key>
```

---

## Python API

### Quick start

By default, the code uses `gpt-5-chat` with zero-shot prompts, temperature of 0, and max token of 1024, which is the same configuration used in the paper. However, our code easily supports other models, hyperparameters, and configurations as they wish. For different model providers (e.g., Gemini, Anthropic), you need to adapt the code accordingly.

```python
from data_preprocessing.question_generation import QuestionGenerator

# Default instantiation — gpt-5-chat, Azure, zero-shot, temperature=0, max_tokens=1024 (used in paper)
gen = QuestionGenerator()

# Equivalent explicit form
gen = QuestionGenerator(model="gpt-5-chat", provider="azure")

# Standard OpenAI API
gen = QuestionGenerator(model="gpt-4o", provider="openai")

result = gen.run(
    objective="To assess the efficacy of X in patients with Y.",
    background_context="X has been used since...",  # optional
)
print(result["question"])
```

### Using a reasoning model (Responses API)

```python
from data_preprocessing.question_generation import QuestionGenerator, ModelConfig

gen = QuestionGenerator(
    model_config=ModelConfig(
        model="gpt-5-mini",
        provider="openai",          # or "azure"
        reasoning_effort="low",
        verbosity="low",
    )
)
```

### Batch processing

```python
results = gen.batch_run(
    objectives=["objective 1", "objective 2"],
    background_contexts=["background 1", "background 2"],  # optional
)
for r in results:
    print(r["question"])
```

### Constructor reference

| Argument | Type | Description |
|---|---|---|
| `model_config` | `ModelConfig` | Full config object — overrides all individual kwargs below |
| `model` | `str` | Model deployment/API name (default from config) |
| `provider` | `str` | `'azure'` or `'openai'` (default from config) |
| `include_few_shot` | `bool` | Use few-shot prompt examples (default from config) |
| `temperature` | `float` | Chat-completions only; 0 = deterministic |
| `max_tokens` | `int` | Chat-completions only; upper bound on response length |
| `reasoning_effort` | `str` | Responses-API only: `none`/`minimal`/`low`/`medium`/`high` |
| `verbosity` | `str` | Responses-API only: `low`/`medium`/`high` |
| `reasoning_summary` | `str` | Responses-API only: `auto`/`none`/`brief`/`detailed` |
| `min_question_length` | `int` | Override validation min-length (default from config) |
| `question_words` | `list[str]` | Override validation word list (default from config) |

---

## Batch script (CLI)

`example_scripts/batch_generate_questions.py` reads a JSON data file, generates a question for every unprocessed entry, and writes a `doi → question` JSON output.  Re-running is safe — already-processed entries are skipped.

### Required arguments

| Flag | Description |
|---|---|
| `--data-file` | Path to the input JSON file |
| `--output-file` | Path to the output JSON file |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/preprocessing_config.yaml` | Path to the YAML config file |
| `--provider` | value from config | `azure` or `openai` |
| `--model` | value from config | Model deployment/API name |
| `--few-shot` / `--no-few-shot` | value from config | Enable/disable few-shot prompting |
| `--batch-save-interval N` | `10` | Save progress every N questions |

### Examples

```bash
# Azure OpenAI (default)
python scripts/data_preprocessing/question_generation/example_scripts/batch_generate_questions.py \
    --data-file  data/review_articles/data.json \
    --output-file data/preprocessed_qa/generated_questions.json

# OpenAI API with a reasoning model
python scripts/data_preprocessing/question_generation/example_scripts/batch_generate_questions.py \
    --data-file  data/review_articles/data.json \
    --output-file data/preprocessed_qa/generated_questions.json \
    --provider openai \
    --model gpt-4o \
    --few-shot
```

---

## Configuration

`config/preprocessing_config.yaml` holds all defaults. However, constructor args and CLI flags always take precedence.

```yaml
question_generator:
  provider: azure          # azure | openai
  model: gpt-5-chat

  # Chat-completions models
  temperature: 0
  max_tokens: 1024

  # Responses-API / reasoning models
  reasoning_effort: low
  verbosity: low
  reasoning_summary: auto

validation:
  # Edit here to add/remove accepted question words — no code changes needed
  min_question_length: 10
  question_words: [what, how, why, ...]
```

---

## Module layout

```
question_generation/
├── generate_questions.py            # QuestionGenerator class
├── config/
│   ├── model_config.py              # ModelConfig dataclass + UnifiedLLMClient
│   └── preprocessing_config.yaml   # Default settings (single source of truth)
├── example_scripts/
│   └── batch_generate_questions.py  # CLI script for batch processing
├── utils/
│   ├── helper_questions.py          # Response parsing & question validation
│   └── prompts.py                   # Zero-shot and few-shot prompt templates
└── README.md
```

## Input data format

The input JSON file must be a list of article objects, each with:

```json
{
  "doi": "10.1002/14651858.CD015092.pub3",
  "abstract": [
    { "heading": "Background",  "text": "..." },
    { "heading": "Objectives",  "text": "To assess..." }
  ]
}
```

Only entries with an `"Objectives"` heading are processed; others are skipped with a warning.
