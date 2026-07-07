# Helping Educators Choose AI Tools: A Study of Cognitive Complexity, Task Difficulty, and Domain Effects

This repository contains the complete implementation accompanying the paper:

> **Helping Educators Choose AI Tools: A Study of Cognitive Complexity, Task Difficulty, and Domain Effects**

The repository contains the full experimental pipeline used to evaluate large language models across Bloom's Revised Taxonomy, including Bloom classification, model querying, rubric-based evaluation, lexical metrics, and statistical analysis.

---

# Repository Overview

This repository reproduces the evaluation presented in the paper.

The benchmark consists of:

- **4,476 questions**
- **9 open-source datasets**
- **4 subject domains**
- **6 Bloom's Revised Taxonomy levels**
- **3 difficulty levels**
- **6 evaluated language models**
- **26,856 evaluated responses**

The study investigates whether **cognitive complexity** explains model performance better than traditional task difficulty.

---

# Repository Structure

```
.
├── config.py                 # Central experiment configuration
├── run.py                    # Main command-line interface
├── labeler.py                # Bloom taxonomy classification
├── prompts.py                # Prompt templates
├── clients.py                # API client initialization
├── model_api.py              # Model-specific API wrappers
├── querier.py                # Query pipeline
├── evaluator.py              # LLM-as-judge evaluation
├── judge.py                  # Rubric scoring implementation
├── metrics.py                # BLEU / ROUGE / lexical metrics
├── data_loader.py            # Dataset loading utilities
├── questions.csv             # Benchmark questions
├── requirements.txt
├── tests/
└── outputs/
```

---

# Experimental Pipeline

The repository follows the five-stage evaluation pipeline described in the paper.

## Phase 1 - Difficulty Stratification

Questions are sampled according to their predefined difficulty labels while preserving the original benchmark distributions.

---

## Phase 2 - Bloom Classification

Questions are automatically classified into Bloom's Revised Taxonomy levels using **GPT-4.1-mini** with few-shot prompting.

Classification is performed **after sampling** to preserve the natural cognitive distribution of the benchmark.

Bloom labels are **never shown** to the evaluated models.

---

## Phase 3 - Model Evaluation

Each model receives identical zero-shot prompts.

The evaluated models are:

- GPT-4o-mini
- Gemini 2.5 Flash
- Claude Sonnet 4.5
- DeepSeek V3.2
- LLaMA 3.3 70B
- LLaMA 3.1 8B

All generations are produced using:

- Temperature = 0.0

---

## Phase 4 - Rubric-Based Scoring

Responses are evaluated using GPT-4.1-mini acting as an LLM judge.

Each response receives four independent scores:

- Correctness
- Completeness
- Clarity
- Cognitive Alignment

The overall score is the arithmetic mean of these four dimensions.

Lexical metrics (BLEU and ROUGE) are additionally reported for comparison but are not used as the primary evaluation metric.

---

## Phase 5 - Human Validation

Both Bloom classifications and LLM-as-judge scores were validated using independent human raters.

Reported agreement:

- Bloom classification
  - Cohen's κ = 0.774
  - Human agreement = 83.6%

- Rubric scoring
  - Cohen's κ = 0.719
  - Human agreement = 80.4%

---

# Dataset

The benchmark combines nine publicly available datasets spanning four educational domains.

| Domain | Datasets |
|---------|----------|
| Mathematics | GSM8K, JEEBench |
| Science | ARC, OpenBookQA, JEEBench |
| Reading Comprehension | SQuAD, HotpotQA, NarrativeQA, AGIEval |
| Computer Science | LiveCodeBench |

After sampling, the benchmark contains:

- Mathematics: 1,136 questions
- Science: 1,179 questions
- Reading Comprehension: 1,106 questions
- Computer Science: 1,055 questions

Total:

**4,476 questions**

---

# Installation

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Copy the environment template.

```bash
cp .env.template .env
```

Add the required API keys.

---

# Running the Pipeline

## 1. Bloom Classification

Skip this step if using the provided benchmark.

```bash
python run.py label --questions questions.csv
```

---

## 2. Query Models

```bash
python run.py query --run all
```

---

## 3. Evaluate Responses

```bash
python run.py evaluate --config all
```

---

# Output

The pipeline produces:

- Raw model responses
- Rubric scores
- BLEU scores
- ROUGE scores
- Keyword overlap metrics
- Aggregate statistics

These outputs reproduce the analyses presented in the paper.

---

# Reproducibility Notes

For reproducibility:

- Temperature = 0.0 for evaluated models
- Bloom labels hidden from evaluated models
- GPT-4.1-mini used for Bloom classification
- GPT-4.1-mini used as the evaluation judge
- Fixed random seed for sampling
- Bloom classification performed after sampling
- Human validation performed independently of automated evaluation

---

# Citation

If you use this repository, please cite:

```text
Mehta, P., Fard, F., & Mohamed, M.

Helping Educators Choose AI Tools:
A Study of Cognitive Complexity,
Task Difficulty,
and Domain Effects.
```

---

# License

This repository is released for academic and research purposes.

Please consult the accompanying LICENSE file for details.