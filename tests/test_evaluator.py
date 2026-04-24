"""
tests/test_evaluator.py
-----------------------
Tests for evaluator.py scoring pipeline.

Tests cover:
  - Normal run: all responses are scored and saved
  - Output CSV has the correct columns and schema
  - Resume logic: already-scored responses are skipped
  - Error rows (model failed during querying) are skipped by the judge
  - MCQ questions skip lexical metrics (BLEU/ROUGE are NaN)
  - Non-MCQ questions include lexical metrics
  - Metadata fields are correctly copied from the joined question row
  - Judge scores are correctly written to the output

All judge API calls are mocked — no real network requests are made.
"""

import sys
import os
import pytest
import pandas as pd
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_questions_df(n=2):
    return pd.DataFrame({
        "question_id":         [f"q{i:03d}" for i in range(1, n + 1)],
        "domain":              ["science"] * n,
        "difficulty":          ["easy"] * n,
        "bloom_level":         [1] * n,
        "bloom_name":          ["Remember"] * n,
        "question_text":       [f"Question {i}?" for i in range(1, n + 1)],
        "ground_truth_answer": [f"Answer {i}"    for i in range(1, n + 1)],
        "question_type":       ["open"] * n,
        "context":             [""] * n,
        "choices_json":        [None] * n,
        "dataset_name":        ["TEST"] * n,
        "dataset_subset":      [None] * n,
        "bloom_confidence":    [0.9] * n,
    })


def make_responses_df(question_ids, model_name="test-model", error=False):
    n = len(question_ids)
    return pd.DataFrame({
        "response_id":      [f"resp_{i:06d}" for i in range(1, n + 1)],
        "question_id":      question_ids,
        "model_name":       [model_name] * n,
        "response_text":    ["" if error else f"Response {i}" for i in range(1, n + 1)],
        "response_time_sec":[1.0] * n,
        "token_count":      [0 if error else 5] * n,
        "error":            [error] * n,
        "error_message":    ["API error" if error else ""] * n,
        "timestamp":        ["2025-01-01"] * n,
    })


MOCK_MODEL_CONFIG = {"test-model": {"api": "x", "model_id": "x", "temperature": 0.0}}

GOOD_JUDGE_CONTENT = (
    "CORRECTNESS: 8\nCOMPLETENESS: 7\nCLARITY: 9\n"
    "COGNITIVE_ALIGNMENT: 6\nOVERALL_SCORE: 7\nJUSTIFICATION: Solid answer."
)


def make_mock_openai(content=GOOD_JUDGE_CONTENT):
    """Return a mock OpenAI client that returns the given judge response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


def run_evaluator(questions_df, responses_df, output_path, judge_content=GOOD_JUDGE_CONTENT):
    """
    Convenience wrapper: write temp CSVs, patch OpenAI, run ResponseEvaluator.
    Returns the output DataFrame.
    """
    q_file = tempfile.mktemp(suffix=".csv")
    r_file = tempfile.mktemp(suffix=".csv")
    questions_df.to_csv(q_file, index=False)
    responses_df.to_csv(r_file, index=False)

    mock_client = make_mock_openai(judge_content)

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.object(config, "MODELS_CONFIG", MOCK_MODEL_CONFIG):
        from evaluator import ResponseEvaluator
        ev = ResponseEvaluator([q_file], r_file, output_path, "gpt-4.1-mini")
        ev.run()

    os.unlink(q_file)
    os.unlink(r_file)
    return pd.read_csv(output_path)


# ---------------------------------------------------------------------------
# Normal run
# ---------------------------------------------------------------------------

class TestNormalRun:
    def test_one_row_per_response(self):
        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(make_questions_df(2), make_responses_df(["q001", "q002"]), out)
        assert len(result) == 2
        os.unlink(out)

    def test_output_has_score_id_column(self):
        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(make_questions_df(1), make_responses_df(["q001"]), out)
        assert "score_id" in result.columns
        assert result.iloc[0]["score_id"] == "score_000001"
        os.unlink(out)

    def test_judge_scores_are_written(self):
        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(make_questions_df(1), make_responses_df(["q001"]), out)
        row = result.iloc[0]
        assert row["correctness"] == 8
        assert row["completeness"] == 7
        assert row["clarity"] == 9
        assert row["cognitive_alignment"] == 6
        assert row["overall_score"] == 7
        assert row["justification"] == "Solid answer."
        os.unlink(out)

    def test_metadata_fields_copied_from_question(self):
        """domain, difficulty, bloom_level, etc. should be copied from the question."""
        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(make_questions_df(1), make_responses_df(["q001"]), out)
        row = result.iloc[0]
        assert row["domain"] == "science"
        assert row["difficulty"] == "easy"
        assert row["bloom_level"] == 1
        assert row["bloom_name"] == "Remember"
        os.unlink(out)

    def test_model_version_looked_up_from_config(self):
        """model_version should be the model_id from config, not the friendly name."""
        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(make_questions_df(1), make_responses_df(["q001"]), out)
        assert result.iloc[0]["model_version"] == "x"   # from MOCK_MODEL_CONFIG
        os.unlink(out)

    def test_all_output_columns_present(self):
        """Every column in config.OUTPUT_COLUMNS should exist in the output."""
        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(make_questions_df(1), make_responses_df(["q001"]), out)
        for col in config.OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing output column: {col}"
        os.unlink(out)


# ---------------------------------------------------------------------------
# MCQ vs non-MCQ lexical metrics
# ---------------------------------------------------------------------------

class TestLexicalMetrics:
    def test_open_question_gets_bleu_score(self):
        """Non-MCQ responses should have a BLEU score (not NaN)."""
        questions = make_questions_df(1)
        questions.loc[0, "question_type"] = "open"
        questions.loc[0, "ground_truth_answer"] = "The mitochondria produces ATP energy"
        responses = make_responses_df(["q001"])
        responses.loc[0, "response_text"] = "Mitochondria produces ATP"

        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(questions, responses, out)
        assert not pd.isna(result.iloc[0]["bleu"])
        assert result.iloc[0]["bleu"] > 0.0
        os.unlink(out)

    def test_mcq_question_skips_lexical_metrics(self):
        """MCQ responses should have NaN for BLEU/ROUGE (single-letter answers)."""
        questions = make_questions_df(1)
        questions.loc[0, "question_type"] = "mcq"
        responses = make_responses_df(["q001"])
        responses.loc[0, "response_text"] = "B"

        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(questions, responses, out)
        assert pd.isna(result.iloc[0]["bleu"])
        assert pd.isna(result.iloc[0]["rouge1"])
        os.unlink(out)

    def test_open_question_gets_rouge_scores(self):
        """Non-MCQ responses should have ROUGE-1, ROUGE-2, and ROUGE-L scores."""
        questions = make_questions_df(1)
        questions.loc[0, "question_type"] = "open"
        responses = make_responses_df(["q001"])
        responses.loc[0, "response_text"] = "Answer 1 is the correct answer here"

        out = tempfile.mktemp(suffix=".csv")
        result = run_evaluator(questions, responses, out)
        for metric in ["rouge1", "rouge2", "rougeL"]:
            assert not pd.isna(result.iloc[0][metric]), f"{metric} should not be NaN"
        os.unlink(out)


# ---------------------------------------------------------------------------
# Error row handling
# ---------------------------------------------------------------------------

class TestErrorRowHandling:
    def test_error_rows_are_skipped_by_judge(self):
        """
        Responses where error=True (model failed during querying) should not be sent to
        the judge. The judge client should not be called at all.
        """
        questions = make_questions_df(1)
        responses = make_responses_df(["q001"], error=True)

        out = tempfile.mktemp(suffix=".csv")
        q_file = tempfile.mktemp(suffix=".csv")
        r_file = tempfile.mktemp(suffix=".csv")
        questions.to_csv(q_file, index=False)
        responses.to_csv(r_file, index=False)

        call_count = [0]
        mock_client = MagicMock()
        def count_calls(**kwargs):
            call_count[0] += 1
        mock_client.chat.completions.create.side_effect = count_calls

        with patch("openai.OpenAI", return_value=mock_client), \
             patch.object(config, "MODELS_CONFIG", MOCK_MODEL_CONFIG):
            from evaluator import ResponseEvaluator
            ResponseEvaluator([q_file], r_file, out, "gpt-4.1-mini").run()

        assert call_count[0] == 0, "Judge should not be called for error rows"
        result = pd.read_csv(out)
        assert "Skipped" in result.iloc[0]["justification"]

        for f in [q_file, r_file, out]:
            os.unlink(f)

    def test_error_row_is_still_saved_to_output(self):
        """Error rows should appear in the output CSV, just without scores."""
        questions = make_questions_df(1)
        responses = make_responses_df(["q001"], error=True)
        out = tempfile.mktemp(suffix=".csv")

        q_file = tempfile.mktemp(suffix=".csv")
        r_file = tempfile.mktemp(suffix=".csv")
        questions.to_csv(q_file, index=False)
        responses.to_csv(r_file, index=False)

        with patch("openai.OpenAI", return_value=MagicMock()), \
             patch.object(config, "MODELS_CONFIG", MOCK_MODEL_CONFIG):
            from evaluator import ResponseEvaluator
            ResponseEvaluator([q_file], r_file, out, "gpt-4.1-mini").run()

        result = pd.read_csv(out)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["overall_score"])

        for f in [q_file, r_file, out]:
            os.unlink(f)


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

class TestResumeLogic:
    def test_already_scored_responses_are_skipped(self):
        """
        If a response_id already appears in the output CSV, it should not
        be sent to the judge again.
        """
        questions = make_questions_df(2)
        responses = make_responses_df(["q001", "q002"])
        out = tempfile.mktemp(suffix=".csv")

        # Pre-populate with resp_000001 already scored
        done_row = {c: None for c in config.OUTPUT_COLUMNS}
        done_row.update({"score_id": "score_000001", "response_id": "resp_000001", "overall_score": 9})
        pd.DataFrame([done_row]).to_csv(out, index=False)

        call_count = [0]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = GOOD_JUDGE_CONTENT
        mock_client = MagicMock()
        def count_and_return(**kwargs):
            call_count[0] += 1
            return mock_resp
        mock_client.chat.completions.create.side_effect = count_and_return

        q_file = tempfile.mktemp(suffix=".csv")
        r_file = tempfile.mktemp(suffix=".csv")
        questions.to_csv(q_file, index=False)
        responses.to_csv(r_file, index=False)

        with patch("openai.OpenAI", return_value=mock_client), \
             patch.object(config, "MODELS_CONFIG", MOCK_MODEL_CONFIG):
            from evaluator import ResponseEvaluator
            ResponseEvaluator([q_file], r_file, out, "gpt-4.1-mini").run()

        assert call_count[0] == 1   # only resp_000002 was scored
        result = pd.read_csv(out)
        assert len(result) == 2     # existing + new

        for f in [q_file, r_file, out]:
            os.unlink(f)

class TestUnmatchedResponses:
    def test_unmatched_response_ids_print_warning(self, capsys):
        """
        If a response has a question_id that doesn't exist in questions.csv,
        it should be dropped from scoring and a WARNING should be printed.
        """
        questions = make_questions_df(1)   # only has q001
        responses = make_responses_df(["q001", "q999"])  # q999 has no matching question

        q_file = tempfile.mktemp(suffix=".csv")
        r_file = tempfile.mktemp(suffix=".csv")
        out    = tempfile.mktemp(suffix=".csv")
        questions.to_csv(q_file, index=False)
        responses.to_csv(r_file, index=False)

        mock_client = make_mock_openai()
        with patch("openai.OpenAI", return_value=mock_client), \
             patch.object(config, "MODELS_CONFIG", MOCK_MODEL_CONFIG):
            from evaluator import ResponseEvaluator
            ev = ResponseEvaluator([q_file], r_file, out, "gpt-4.1-mini")

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "1" in captured.out   # 1 unmatched response

        for f in [q_file, r_file, out]:
            try: os.unlink(f)
            except: pass
