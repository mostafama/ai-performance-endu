"""
tests/test_querier.py
---------------------
Tests for querier.py querying pipeline.

Tests cover:
  - Normal run: all (question, model) pairs are processed and saved
  - Output CSV has the correct columns
  - Resume logic: already-completed pairs are skipped
  - Rate limit detection: model is disabled mid-run on quota errors
  - Error responses are saved (not silently dropped)
  - Batch saving: responses are flushed periodically
  - No tasks: "all tasks already completed" exits cleanly

All LLM API calls are mocked — no real network requests are made.
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

MOCK_MODEL = "test-model"
MOCK_CONFIG = {MOCK_MODEL: {"api": "anthropic", "model_id": "test", "temperature": 0.0, "max_tokens": 100}}

def make_questions_df(n=3):
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
    })


def make_querier(questions_df, output_path, query_fn=None):
    """
    Build a LLMQuerier instance with mocked clients and a controllable
    query_model function. Default query_fn returns a successful response.
    """
    if query_fn is None:
        query_fn = lambda model, prompt, clients: ("Mock response.", 1.0, False, "")

    mock_clients = {"anthropic": MagicMock()}

    with patch("querier.init_clients", return_value=mock_clients), \
         patch("querier.query_model", side_effect=query_fn), \
         patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
        from querier import LLMQuerier
        q = LLMQuerier([MOCK_MODEL], questions_df, output_path)

    return q, mock_clients


# ---------------------------------------------------------------------------
# Normal run
# ---------------------------------------------------------------------------

class TestNormalRun:
    def test_all_pairs_produce_output_rows(self):
        """One output row per (question, model) pair."""
        df = make_questions_df(3)
        out = tempfile.mktemp(suffix=".csv")

        call_count = [0]
        def mock_query(model, prompt, clients):
            call_count[0] += 1
            return ("Response text.", 0.5, False, "")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", side_effect=mock_query), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        result = pd.read_csv(out)
        assert len(result) == 3
        assert call_count[0] == 3
        os.unlink(out)

    def test_output_has_required_columns(self):
        """Output CSV must contain all expected columns."""
        df = make_questions_df(2)
        out = tempfile.mktemp(suffix=".csv")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", return_value=("resp", 1.0, False, "")), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        result = pd.read_csv(out)
        expected_cols = [
            "response_id", "question_id", "model_name", "response_text",
            "response_time_sec", "token_count", "error", "error_message", "timestamp",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        os.unlink(out)

    def test_response_ids_are_sequential(self):
        """response_id values should be resp_000001, resp_000002, etc."""
        df = make_questions_df(3)
        out = tempfile.mktemp(suffix=".csv")
        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", return_value=("resp", 1.0, False, "")), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()
        result = pd.read_csv(out)
        assert list(result["response_id"]) == ["resp_000001", "resp_000002", "resp_000003"]
        os.unlink(out)

    def test_token_count_is_word_count_of_response(self):
        """token_count should equal the number of words in the response text."""
        df = make_questions_df(1)
        out = tempfile.mktemp(suffix=".csv")
        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", return_value=("one two three four five", 1.0, False, "")), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()
        result = pd.read_csv(out)
        assert result.iloc[0]["token_count"] == 5
        os.unlink(out)


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

class TestResumeLogic:
    def test_already_done_pairs_are_skipped(self):
        """
        If the output CSV already has a (question_id, model_name) pair,
        that pair should not be queried again.
        """
        df = make_questions_df(3)
        out = tempfile.mktemp(suffix=".csv")

        # Pre-populate with q001 already done
        existing = pd.DataFrame([{
            "response_id": "resp_000001", "question_id": "q001",
            "model_name": MOCK_MODEL, "response_text": "existing",
            "response_time_sec": 0.1, "token_count": 1,
            "error": False, "error_message": "", "timestamp": "2025-01-01",
        }])
        existing.to_csv(out, index=False)

        call_count = [0]
        def mock_query(model, prompt, clients):
            call_count[0] += 1
            return ("new response", 1.0, False, "")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", side_effect=mock_query), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        result = pd.read_csv(out)
        assert len(result) == 3       # existing + 2 new
        assert call_count[0] == 2     # only 2 API calls, not 3
        os.unlink(out)

    def test_response_counter_starts_from_existing_count(self):
        """
        If 5 responses already exist, new response_ids should start at 6.
        """
        df = make_questions_df(1)
        out = tempfile.mktemp(suffix=".csv")

        # Pre-populate with 5 existing rows (different question IDs)
        existing_rows = []
        for i in range(1, 6):
            existing_rows.append({
                "response_id": f"resp_{i:06d}", "question_id": f"existing_q{i:03d}",
                "model_name": MOCK_MODEL, "response_text": "old",
                "response_time_sec": 0.1, "token_count": 1,
                "error": False, "error_message": "", "timestamp": "2025-01-01",
            })
        pd.DataFrame(existing_rows).to_csv(out, index=False)

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", return_value=("resp", 1.0, False, "")), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        result = pd.read_csv(out)
        new_row = result[result["question_id"] == "q001"].iloc[0]
        assert new_row["response_id"] == "resp_000006"
        os.unlink(out)

    def test_all_done_exits_without_api_calls(self):
        """If all pairs are already in the output, no API calls should be made."""
        df = make_questions_df(2)
        out = tempfile.mktemp(suffix=".csv")

        existing = pd.DataFrame([
            {"response_id": f"resp_{i:06d}", "question_id": f"q{i:03d}",
             "model_name": MOCK_MODEL, "response_text": "done",
             "response_time_sec": 0.1, "token_count": 1,
             "error": False, "error_message": "", "timestamp": "2025-01-01"}
            for i in range(1, 3)
        ])
        existing.to_csv(out, index=False)

        call_count = [0]
        def mock_query(model, prompt, clients):
            call_count[0] += 1
            return ("response", 1.0, False, "")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", side_effect=mock_query), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        assert call_count[0] == 0
        os.unlink(out)


# ---------------------------------------------------------------------------
# Rate limit / error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_rate_limit_disables_model(self):
        """
        When a rate limit error is returned, the model should be disabled
        and no further API calls should be made for that model.
        """
        df = make_questions_df(3)
        out = tempfile.mktemp(suffix=".csv")

        call_count = [0]
        def mock_query(model, prompt, clients):
            call_count[0] += 1
            return ("", 0.1, True, "rate_limit exceeded 429")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", side_effect=mock_query), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            q = LLMQuerier([MOCK_MODEL], df, out)
            q.run()

        # Should stop after first call due to rate limit
        assert call_count[0] == 1
        os.unlink(out)

    def test_error_responses_are_saved(self):
        """Error rows should be saved to CSV with error=True, not silently dropped."""
        df = make_questions_df(2)
        out = tempfile.mktemp(suffix=".csv")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", return_value=("", 0.1, True, "API timeout")), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        result = pd.read_csv(out)
        assert len(result) == 2
        assert all(result["error"] == True)
        assert all(result["error_message"] == "API timeout")
        os.unlink(out)

    def test_error_response_text_is_empty_string(self):
        """
        When error=True, response_text should be empty string, not the error message.
        We read with keep_default_na=False so pandas preserves "" instead of NaN.
        """
        df = make_questions_df(1)
        out = tempfile.mktemp(suffix=".csv")

        mock_clients = {"anthropic": MagicMock()}
        with patch("querier.init_clients", return_value=mock_clients), \
             patch("querier.query_model", return_value=("", 0.1, True, "some error")), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            LLMQuerier([MOCK_MODEL], df, out).run()

        result = pd.read_csv(out, keep_default_na=False)
        assert result.iloc[0]["response_text"] == ""
        os.unlink(out)

    def test_no_active_models_error_names_missing_key(self):
        """
        When no models are available because an API key is missing,
        the RuntimeError should name the specific environment variable
        that needs to be set, not just say "check API keys".
        """
        df = make_questions_df(1)
        out = tempfile.mktemp(suffix=".csv")

        # Return empty clients dict — simulates all keys missing
        with patch("querier.init_clients", return_value={}), \
             patch.object(config, "MODELS_CONFIG", MOCK_CONFIG):
            from querier import LLMQuerier
            with pytest.raises(RuntimeError) as exc_info:
                LLMQuerier([MOCK_MODEL], df, out)

        error_msg = str(exc_info.value)
        # Should mention a specific key name, not just generic advice
        assert "API_KEY" in error_msg
        try: os.unlink(out)
        except: pass
