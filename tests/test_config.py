"""
tests/test_config.py
--------------------
Tests for config.py — model definitions, study runs, and evaluation configs.

These tests act as a safeguard against accidental misconfigurations:
  - All model IDs are non-empty strings
  - No known-retired model IDs are present
  - Required fields exist in each model, study run, and evaluation config
  - Study run models all exist in MODELS_CONFIG
  - Evaluation config judge models are non-empty
  - Output columns list has no duplicates
  - Required column lists contain the expected column names
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class TestModelsConfig:
    def test_all_models_have_required_fields(self):
        required_fields = {"api", "model_id", "temperature", "max_tokens"}
        for name, cfg in config.MODELS_CONFIG.items():
            missing = required_fields - set(cfg.keys())
            assert not missing, f"Model '{name}' missing fields: {missing}"

    def test_all_model_ids_are_non_empty_strings(self):
        for name, cfg in config.MODELS_CONFIG.items():
            assert isinstance(cfg["model_id"], str) and cfg["model_id"].strip(), \
                f"Model '{name}' has empty or non-string model_id"

    def test_all_temperatures_are_zero(self):
        """All models should use temperature=0.0 for deterministic, reproducible outputs."""
        for name, cfg in config.MODELS_CONFIG.items():
            assert cfg["temperature"] == 0.0, \
                f"Model '{name}' has temperature={cfg['temperature']}, expected 0.0"

    def test_all_max_tokens_are_positive(self):
        for name, cfg in config.MODELS_CONFIG.items():
            assert cfg["max_tokens"] > 0, f"Model '{name}' has non-positive max_tokens"

    def test_no_retired_model_ids(self):
        """
        These specific snapshot IDs are known to be retired and will cause
        API errors if used. This test catches accidental rollbacks.
        """
        retired_ids = {
            "claude-sonnet-4-5-20250929",   # wrong date — never existed
            "claude-3-5-sonnet-20240620",   # retired Oct 2025
        }
        for name, cfg in config.MODELS_CONFIG.items():
            assert cfg["model_id"] not in retired_ids, \
                f"Model '{name}' uses retired model_id: {cfg['model_id']}"

    def test_api_values_are_known_providers(self):
        known_apis = {"openai", "gemini", "anthropic", "groq", "openrouter"}
        for name, cfg in config.MODELS_CONFIG.items():
            assert cfg["api"] in known_apis, \
                f"Model '{name}' has unknown api='{cfg['api']}'"


# ---------------------------------------------------------------------------
# Study runs
# ---------------------------------------------------------------------------

class TestStudyRuns:
    def test_all_runs_have_required_fields(self):
        required = {"models", "questions", "output"}
        for name, run in config.STUDY_RUNS.items():
            missing = required - set(run.keys())
            assert not missing, f"Study run '{name}' missing fields: {missing}"

    def test_all_run_models_exist_in_models_config(self):
        for run_name, run in config.STUDY_RUNS.items():
            for model in run["models"]:
                assert model in config.MODELS_CONFIG, \
                    f"Run '{run_name}' references unknown model '{model}'"

    def test_all_runs_have_non_empty_output(self):
        for name, run in config.STUDY_RUNS.items():
            assert run["output"] and run["output"].endswith(".csv"), \
                f"Run '{name}' has invalid output path: {run['output']}"

    def test_all_runs_have_at_least_one_question_file(self):
        for name, run in config.STUDY_RUNS.items():
            assert len(run["questions"]) >= 1, \
                f"Run '{name}' has no question files"


    def test_all_runs_use_only_questions_csv(self):
        """
        All study runs must use a single questions file: questions.csv.
        There should be no per-run question file divergence that could cause
        query/eval config mismatches.
        """
        for name, run in config.STUDY_RUNS.items():
            assert run["questions"] == ["questions.csv"], (
                f"Run '{name}' should use only questions.csv, got {run['questions']}"
            )

    def test_all_eval_configs_use_only_questions_csv(self):
        """Evaluation configs must also reference only questions.csv."""
        for name, cfg in config.EVALUATION_CONFIGS.items():
            assert cfg["questions_csv"] == ["questions.csv"], (
                f"Eval config '{name}' should use only questions.csv, got {cfg['questions_csv']}"
            )

    def test_query_and_eval_question_files_match(self):
        """
        For every named run, the query questions and eval questions_csv
        must be identical — otherwise the join will silently miss rows.
        """
        for name in config.STUDY_RUNS:
            if name not in config.EVALUATION_CONFIGS:
                continue
            run_q  = sorted(config.STUDY_RUNS[name]["questions"])
            eval_q = sorted(config.EVALUATION_CONFIGS[name]["questions_csv"])
            assert run_q == eval_q, (
                f"[{name}] query questions {run_q} != eval questions {eval_q}"
            )

# ---------------------------------------------------------------------------
# Evaluation configs
# ---------------------------------------------------------------------------

class TestEvaluationConfigs:
    def test_all_configs_have_required_fields(self):
        required = {"questions_csv", "responses_csv", "output_csv", "judge_model"}
        for name, cfg in config.EVALUATION_CONFIGS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Eval config '{name}' missing fields: {missing}"

    def test_all_judge_models_are_non_empty(self):
        for name, cfg in config.EVALUATION_CONFIGS.items():
            assert cfg["judge_model"] and isinstance(cfg["judge_model"], str), \
                f"Eval config '{name}' has invalid judge_model"

    def test_all_output_csvs_end_with_csv(self):
        for name, cfg in config.EVALUATION_CONFIGS.items():
            assert cfg["output_csv"].endswith(".csv"), \
                f"Eval config '{name}' output_csv doesn't end with .csv"


# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_no_duplicate_columns(self):
        seen = set()
        duplicates = []
        for col in config.OUTPUT_COLUMNS:
            if col in seen:
                duplicates.append(col)
            seen.add(col)
        assert not duplicates, f"Duplicate output columns: {duplicates}"

    def test_key_columns_are_present(self):
        """These columns are essential for downstream analysis."""
        essential = [
            "score_id", "response_id", "question_id", "model_name",
            "bloom_level", "bloom_name", "domain", "difficulty",
            "correctness", "completeness", "clarity", "cognitive_alignment",
            "overall_score", "bleu", "rouge1",
        ]
        for col in essential:
            assert col in config.OUTPUT_COLUMNS, f"Essential column missing: {col}"


# ---------------------------------------------------------------------------
# Required column lists
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_question_id_in_required_question_columns(self):
        assert "question_id" in config.REQUIRED_QUESTION_COLUMNS

    def test_ground_truth_in_required_question_columns(self):
        assert "ground_truth_answer" in config.REQUIRED_QUESTION_COLUMNS

    def test_bloom_level_in_required_question_columns(self):
        assert "bloom_level" in config.REQUIRED_QUESTION_COLUMNS

    def test_response_id_in_required_response_columns(self):
        assert "response_id" in config.REQUIRED_RESPONSE_COLUMNS

    def test_response_text_in_required_response_columns(self):
        assert "response_text" in config.REQUIRED_RESPONSE_COLUMNS
