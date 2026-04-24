"""
config.py
---------
Central settings file for the EduQuestion3 pipeline.
All model definitions, prompt templates, study runs, and evaluation configs
live here so you never need to touch the logic files to change settings.
"""

# ---------------------------------------------------------------------------
# MODEL DEFINITIONS
# Each entry maps a friendly name to its API provider, exact model ID,
# temperature (0.0 = deterministic), and max output tokens.
#
# IMPORTANT: Always use versioned snapshot IDs (not aliases) to guarantee
# reproducibility. Aliases like "claude-sonnet-latest" can change over time.
# ---------------------------------------------------------------------------
MODELS_CONFIG = {
    "gpt-4o-mini": {
        "api": "openai",
        "model_id": "gpt-4o-mini",                  # stable alias → gpt-4o-mini-2024-07-18
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "gemini-2.5-flash": {
        "api": "gemini",
        "model_id": "gemini-2.5-flash",             # stable alias per Google Gemini 2.5 naming
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "claude-sonnet-4.5": {
        "api": "anthropic",
        "model_id": "claude-sonnet-4-5-20251101",   # versioned snapshot — Nov 2025
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "claude-3.5-sonnet": {
        "api": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",   # Oct 2024 snapshot (20240620 retired Oct 2025)
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "llama-3.3-70b": {
        "api": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 8000,
    },
    "llama-3.1-8b": {
        "api": "groq",
        "model_id": "llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 8000,
    },
    "deepseek-v3.2": {
        "api": "openrouter",
        "model_id": "deepseek/deepseek-v3.2",       # OpenRouter model string
        "temperature": 0.0,
        "max_tokens": 8192,
    },
}

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# Each domain gets a tailored instruction appended to the question.
# This ensures the model responds in the right format for each subject area.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    # Math: ask for step-by-step reasoning with a boxed final answer
    "math": (
        "Please solve this problem step by step and provide your final answer "
        "in \\boxed{{}}."
    ),
    # Science: ask for a clear explanation alongside the answer
    "science": "Please provide a clear answer with explanation.",
    # Reading: grounded answer based only on the provided context
    "reading": "Please provide a clear, concise answer based on the given context.",
    # CS: clean, commented code with efficiency in mind
    "computer_science": (
        "Please solve the following programming problem. "
        "Write clean, correct code with brief comments explaining your approach. "
        "If multiple solutions are possible, choose the most efficient one."
    ),
}

# Used when a question's domain doesn't match any key above
FALLBACK_TEMPLATE = "Please provide a clear, well-reasoned answer."

# ---------------------------------------------------------------------------
# STUDY RUNS
# Named groups that bundle together: which models to query, which question
# files to use, and where to save the responses.
#
# (programming problems from LiveCodeBench). Both files must be present.
#
# Run via: python run.py query --run <name>
# ---------------------------------------------------------------------------
STUDY_RUNS = {
    "gpt_gemini": {
        "models": ["gpt-4o-mini", "gemini-2.5-flash"],
        "questions": ["questions.csv"],
        "output": "responses_gpt_gemini.csv",
    },
    "claude": {
        "models": ["claude-sonnet-4.5"],
        "questions": ["questions.csv"],
        "output": "responses_claude.csv",
    },
    "llama": {
        "models": ["llama-3.3-70b", "llama-3.1-8b"],
        "questions": ["questions.csv"],
        "output": "responses_llama.csv",
    },
    "deepseek": {
        "models": ["deepseek-v3.2"],
        "questions": ["questions.csv"],
        "output": "responses_deepseek.csv",
    },
}

# ---------------------------------------------------------------------------
# EVALUATION CONFIGS
# Named configs that pair a responses CSV with the question files needed to
# join against it, the output path, and which judge model to use.
# Each eval config's questions_csv must match its corresponding study run's
# questions list so the join finds all rows.
#
# Run via: python run.py evaluate --config <name>
# ---------------------------------------------------------------------------
EVALUATION_CONFIGS = {
    "gpt_gemini": {
        "description": "GPT-4o-mini and Gemini-2.5-flash",
        "questions_csv": ["questions.csv"],
        "responses_csv": "responses_gpt_gemini.csv",
        "output_csv": "scores_gpt_gemini.csv",
        "judge_model": "gpt-4.1-mini",      # alias → gpt-4.1-mini-2025-04-14
    },
    "claude": {
        "description": "Claude Sonnet 4.5",
        "questions_csv": ["questions.csv"],
        "responses_csv": "responses_claude.csv",
        "output_csv": "scores_claude.csv",
        "judge_model": "gpt-4.1-mini",
    },
    "llama": {
        "description": "LLaMA-3.3-70B and LLaMA-3.1-8B",
        "questions_csv": ["questions.csv"],
        "responses_csv": "responses_llama.csv",
        "output_csv": "scores_llama.csv",
        "judge_model": "gpt-4.1-mini",
    },
    "deepseek": {
        "description": "DeepSeek-V3.2",
        "questions_csv": ["questions.csv"],
        "responses_csv": "responses_deepseek.csv",
        "output_csv": "scores_deepseek.csv",
        "judge_model": "gpt-4.1-mini",
    },
}

# ---------------------------------------------------------------------------
# OUTPUT SCHEMA
# All columns written to the final scores CSV, in order.
# Columns not produced by a given run are left as None/NaN.
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "score_id", "response_id", "question_id", "model_name",
    "dataset_name", "dataset_subset", "domain", "difficulty",
    "bloom_level", "bloom_name", "bloom_confidence",
    "question_type", "question_text", "ground_truth_answer",
    "context", "choices_json", "response_text", "response_time_sec",
    "token_count", "error", "error_message", "timestamp",
    "model_version", "temperature",
    # LLM-judge scores (0–10 each)
    "correctness", "completeness", "clarity", "cognitive_alignment",
    "overall_score", "justification",
    # Lexical similarity metrics
    "bleu", "rouge1", "rouge2", "rougeL",
    "keyword_precision", "keyword_recall", "keyword_f1",
    # Human validation fields (filled in separately if applicable)
    "human_validated", "human_score", "human_agreement",
    "bloom_justification", "validation_agreement",
]

# ---------------------------------------------------------------------------
# REQUIRED COLUMN CHECKS
# Used to validate input files before processing begins.
# ---------------------------------------------------------------------------
REQUIRED_QUESTION_COLUMNS = [
    "question_id", "domain", "difficulty", "bloom_level",
    "bloom_name", "question_text", "ground_truth_answer"
]

REQUIRED_RESPONSE_COLUMNS = [
    "response_id", "question_id", "model_name", "response_text"
]

# ---------------------------------------------------------------------------
# RUNTIME SETTINGS
# ---------------------------------------------------------------------------
SAVE_BATCH_SIZE = 50        # Save responses to disk every N rows during querying
SAVE_EVERY = 100            # Save scores to disk every N rows during evaluation
RETRY_DELAY = 5             # Seconds to wait between judge API retry attempts
MIN_WORD_COUNT = 5          # Minimum words for a response to be considered valid
SKIP_LEXICAL_FOR_MCQ = True # Skip BLEU/ROUGE for MCQ — single-letter answers aren't meaningful

# ---------------------------------------------------------------------------
# BLOOM'S TAXONOMY REFERENCE
# Used by the labeler (labeler.py) when assigning levels to questions.
# Levels 1–6 follow Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001).
# ---------------------------------------------------------------------------
BLOOM_LEVELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}
