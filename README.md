# EduQuestion3 — Minimal Reproduction Package

This package contains the complete pipeline for the Honours thesis
**"Evaluating Large Language Models Across Bloom's Taxonomy"**
(University of British Columbia Okanagan, 2026).

It covers three steps: labelling questions with Bloom's taxonomy levels,
querying LLMs with those questions, and scoring the responses.

## Project structure

```
.
├── config.py         # All settings: model IDs, study runs, evaluation configs
├── run.py            # Unified CLI entry point (label + query + evaluate)
├── labeler.py        # Assigns Bloom's taxonomy levels to questions
├── clients.py        # API client initialisation for each LLM provider
├── prompts.py        # Domain-aware prompt builder
├── model_api.py      # Per-provider query functions + dispatcher
├── data_loader.py    # Question CSV/Excel loader and validator
├── querier.py        # Sends questions to LLMs, saves raw responses
├── judge.py          # LLM-as-judge prompt, API call, and response parsing
├── metrics.py        # BLEU, ROUGE, and keyword overlap metrics
├── evaluator.py      # Scores responses with LLM judge and lexical metrics
├── requirements.txt  # Python dependencies
├── .env.template     # Copy to .env and fill in your API keys
└── tests/            # 153 unit tests — run with: python -m pytest tests/ -v
```

## Setup

**1. Create and activate a Python virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set up API keys**

```bash
cp .env.template .env
```

Open `.env` and fill in your keys. You only need the keys for the providers
you intend to use. The pipeline prints a warning and skips any model whose
key is missing rather than crashing.

**4. Place your question files in the project directory**

One file is expected: `questions.csv` — all benchmark questions.

Required columns: `question_id`, `domain`, `difficulty`, `bloom_level`,
`bloom_name`, `question_text`, `ground_truth_answer`

Optional columns: `question_type`, `context`, `choices_json`, `dataset_name`

If your questions don't yet have Bloom labels, run the labeler first (see below).

---

## Step 1 — Label questions with Bloom's taxonomy

If your questions already have `bloom_level` and `bloom_name` columns populated,
skip this step.

```bash
python run.py label --questions questions.csv
```

This sends each unlabelled question to GPT-4.1-mini, which classifies it into
one of the six Bloom's Revised Taxonomy levels:

| Level | Name | What it tests |
|---|---|---|
| 1 | Remember | Recall of facts, definitions, formulas |
| 2 | Understand | Explaining, summarising, paraphrasing |
| 3 | Apply | Using a procedure to solve a problem |
| 4 | Analyze | Breaking down, comparing, identifying relationships |
| 5 | Evaluate | Judging, critiquing, justifying a position |
| 6 | Create | Designing, constructing, producing something new |

The labeler writes `bloom_level`, `bloom_name`, `bloom_confidence`, and
`bloom_justification` columns back into the same CSV file. Questions that
already have a `bloom_level` are skipped automatically.

To re-classify everything from scratch:

```bash
python run.py label --questions questions.csv --relabel
```

To use a different classifier model:

```bash
python run.py label --questions questions.csv --model gpt-4o-mini
```

---

## Step 2 — Query LLMs

**Using a named study run (defined in `config.py`)**

```bash
python run.py query --run gpt_gemini
python run.py query --run claude
python run.py query --run llama
python run.py query --run deepseek
python run.py query --run all        # runs every defined study run
```

**Custom query**

```bash
python run.py query \
    --models gpt-4o-mini claude-sonnet-4.5 \
    --questions questions.csv \
    --output my_responses.csv
```

---

## Step 3 — Evaluate responses

**Using a named evaluation config**

```bash
python run.py evaluate --config gpt_gemini
python run.py evaluate --config all
```

**Custom evaluation**

```bash
python run.py evaluate \
    --responses my_responses.csv \
    --questions questions.csv \
    --output my_scores.csv \
    --judge gpt-4.1-mini
```

---

## Resuming interrupted runs

All three steps support resuming. If an output CSV already exists, completed
rows are detected and skipped automatically — just re-run the same command.

---

## Output schema

The final scores CSV contains one row per (question, model) pair:

| Column | Description |
|---|---|
| `score_id` | Unique score identifier |
| `response_id` | Links back to the response |
| `question_id` | Links back to the question |
| `model_name` | Model that produced the response |
| `domain` | Question domain (math, science, reading, computer_science) |
| `difficulty` | easy / medium / hard |
| `bloom_level` | Bloom's level integer (1–6) |
| `bloom_name` | Bloom's level name (Remember, Understand, …) |
| `bloom_confidence` | Labeler's confidence score (0–1) |
| `correctness` | Judge score 0–10 |
| `completeness` | Judge score 0–10 |
| `clarity` | Judge score 0–10 |
| `cognitive_alignment` | Judge score 0–10 |
| `overall_score` | Judge score 0–10 (mean of above if not given) |
| `justification` | Judge's explanation |
| `bleu` | BLEU score (non-MCQ only) |
| `rouge1/rouge2/rougeL` | ROUGE F1 scores (non-MCQ only) |
| `keyword_precision/recall/f1` | Keyword overlap scores (non-MCQ only) |

---

## Running the tests

```bash
python -m pytest tests/ -v
```

153 tests, no API keys or data files needed — all external calls are mocked.

| Test file | What it covers |
|---|---|
| `tests/test_config.py` | Model IDs, study run definitions, output schema |
| `tests/test_data_loader.py` | CSV/Excel loading, deduplication, validation |
| `tests/test_labeler.py` | Bloom labeling prompt, response parsing, skip/relabel logic |
| `tests/test_prompts.py` | Prompt formatting for all domains and question types |
| `tests/test_metrics.py` | BLEU, ROUGE, keyword overlap correctness and edge cases |
| `tests/test_judge.py` | Judge prompt, response parsing, retry logic |
| `tests/test_querier.py` | Querying loop, resume logic, rate limit handling |
| `tests/test_evaluator.py` | Scoring loop, MCQ/non-MCQ logic, resume, error rows |

---

## Notes

- `questions.csv` is not included — supply your own from the original experiment.
- `OPENAI_API_KEY` is used for labeling, GPT model querying, and all judge calls.
- Token counts are word-count approximations, not exact API token counts.
