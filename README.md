# Cognitive Complexity and AI Performance: Evaluating Language Models Across Bloom’s Taxonomy for Educational Contexts

This package contains the complete pipeline for the Honours thesis
**"Cognitive Complexity and AI Performance: Evaluating Language Models Across Bloom’s Taxonomy for Educational Contexts"**
(University of British Columbia Okanagan, 2026).

It covers three steps: labelling questions with Bloom's taxonomy levels,
querying LLMs with those questions, and scoring the responses.

## Quickstart

Everything needed to reproduce the experiment is included. From a clean directory:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API keys
cp .env.template .env        # then open .env and fill in your keys

# 3. Query all models  (questions.csv is already in the repo)
python run.py query --run all

# 4. Score all responses
python run.py evaluate --config all
```

The labeling step (Step 1 below) can be skipped — `questions.csv` already
has `bloom_level` and `bloom_name` populated for all 4,476 questions.

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
├── questions.csv     # All 4,476 benchmark questions (8 datasets, full; AGIEval redacted)
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

**4. Confirm `questions.csv` is present**

`questions.csv` is included in the repository and already has all required
columns populated, including Bloom labels. No additional setup is needed.

If you want to use your own question file instead, it must have these columns:

Required: `question_id`, `domain`, `difficulty`, `bloom_level`, `bloom_name`,
`question_text`, `ground_truth_answer`

Optional: `question_type`, `context`, `choices_json`, `dataset_name`

---

## Step 1: Label questions with Bloom's taxonomy

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

## Step 2: Query LLMs

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

## Step 3: Evaluate responses

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

- `questions.csv` is included with all 4,476 benchmark questions. Questions from 8 datasets are reproduced in full under their open licenses (MIT, Apache 2.0, CC BY-SA 4.0). AGIEval questions (206) are redacted — only the first 8 words of each question are shown — because College Board prohibits redistribution of SAT content for AI use. To reproduce the AGIEval portion, obtain those questions from the AGIEval GitHub repository and join by `question_id`.
- `OPENAI_API_KEY` is used for labeling, GPT model querying, and all judge calls.
- Token counts are word-count approximations, not exact API token counts.

---

## Dataset citations

If you use this pipeline or the questions in `questions.csv`, please cite the original datasets:

**ARC (AI2 Reasoning Challenge)**
```
@article{allenai:arc,
  author  = {Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and
             Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  title   = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  journal = {arXiv:1803.05457v1},
  year    = {2018}
}
```

**GSM8K (Grade School Math)**
```
@article{cobbe2021gsm8k,
  title   = {Training Verifiers to Solve Math Word Problems},
  author  = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and
             Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and
             Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal = {arXiv preprint arXiv:2110.14168},
  year    = {2021}
}
```

**HotpotQA**
```
@inproceedings{yang2018hotpotqa,
  title     = {{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author    = {Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and
               Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle = {Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year      = {2018}
}
```

**JEEBench**
```
@inproceedings{arora-etal-2023-llms,
  title     = {Have {LLM}s Advanced Enough? A Challenging Problem Solving Benchmark
               For Large Language Models},
  author    = {Arora, Daman and Singh, Himanshu and {Mausam}},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in
               Natural Language Processing},
  year      = {2023},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.emnlp-main.468}
}
```

**LiveCodeBench**
```
@article{jain2024livecodebench,
  title   = {LiveCodeBench: Holistic and Contamination Free Evaluation of
             Large Language Models for Code},
  author  = {Jain, Naman and Han, King and Gu, Alex and Li, Wen-Ding and
             Yan, Fanjia and Zhang, Tianjun and Wang, Sida and
             Solar-Lezama, Armando and Sen, Koushik and Stoica, Ion},
  journal = {arXiv preprint arXiv:2403.07974},
  year    = {2024}
}
```

**NarrativeQA**
```
@article{kocisky-etal-2018-narrativeqa,
  title   = {The {NarrativeQA} Reading Comprehension Challenge},
  author  = {Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}} and Schwarz, Jonathan and
             Blunsom, Phil and Dyer, Chris and Hermann, Karl Moritz and
             Melis, G{\'a}bor and Grefenstette, Edward},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {6},
  year    = {2018},
  url     = {https://aclanthology.org/Q18-1023}
}
```

**OpenBookQA**
```
@inproceedings{OpenBookQA2018,
  title     = {Can a Suit of Armor Conduct Electricity? A New Dataset for
               Open Book Question Answering},
  author    = {Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
  booktitle = {EMNLP},
  year      = {2018}
}
```

**SQuAD**
```
@inproceedings{rajpurkar-etal-2016-squad,
  title     = {{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text},
  author    = {Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and Liang, Percy},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in
               Natural Language Processing},
  year      = {2016},
  url       = {https://aclanthology.org/D16-1264}
}
```

**AGIEval** (questions not reproduced — see Notes)
```
@misc{zhong2023agieval,
  title         = {AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
  author        = {Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and
                   Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
  year          = {2023},
  eprint        = {2304.06364},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## Citation

If you use this pipeline or build on this work, please cite:

```
@thesis{shah2026bloomllm,
  author = {Mehta, Prina},
  title  = {Evaluating Large Language Models Across Bloom's Taxonomy},
  school = {University of British Columbia Okanagan},
  year   = {2026},
  type   = {Honours Thesis}
}
```

---

## Copyright

Copyright © 2026 Prina Mehta. All rights reserved.

This repository including the pipeline code, evaluation methodology,
Bloom's taxonomy labeling approach, prompt design, and compiled dataset
represents original work completed as part of an Honours thesis at the
University of British Columbia Okanagan.

**You are free to:**
- Read and reference this work with attribution
- Use the pipeline code for non-commercial academic research, provided
  you cite the thesis (see Citation above)

**You may not:**
- Reproduce, republish, or redistribute substantial portions of this
  work without written permission
- Use this work or its derivatives in commercial products or services
- Present this methodology as your own without attribution

The benchmark datasets included in `questions.csv` remain under their
original licenses (see Dataset citations above) and are subject to those
terms independently of this copyright notice.

For permissions beyond the scope above, contact the author.
