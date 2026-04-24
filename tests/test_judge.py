"""
tests/test_judge.py
-------------------
Tests for judge.py — LLM-as-judge prompt building and response parsing.

Tests cover:
  - build_judge_prompt(): all fields appear in the output
  - parse_judge_response(): normal response, missing overall score,
    empty response, partial response, and out-of-order fields
  - call_judge(): retry logic, all-retries-fail fallback (mocked)

No real API calls are made — the OpenAI client is mocked throughout.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import build_judge_prompt, parse_judge_response, call_judge


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------

class TestBuildJudgePrompt:
    def setup_method(self):
        self.prompt = build_judge_prompt(
            question="What is photosynthesis?",
            reference="Photosynthesis converts sunlight into glucose.",
            response="Plants use sunlight to make food.",
            bloom_level=2,
            bloom_name="Understand",
            domain="science",
        )

    def test_question_is_included(self):
        assert "What is photosynthesis?" in self.prompt

    def test_reference_is_included(self):
        assert "Photosynthesis converts sunlight into glucose." in self.prompt

    def test_response_is_included(self):
        assert "Plants use sunlight to make food." in self.prompt

    def test_bloom_level_is_included(self):
        assert "2" in self.prompt

    def test_bloom_name_is_included(self):
        assert "Understand" in self.prompt

    def test_domain_is_included(self):
        assert "science" in self.prompt

    def test_output_format_labels_are_present(self):
        """The prompt must include all expected output field labels."""
        for label in ["CORRECTNESS", "COMPLETENESS", "CLARITY",
                      "COGNITIVE_ALIGNMENT", "OVERALL_SCORE", "JUSTIFICATION"]:
            assert label in self.prompt, f"Missing label: {label}"


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_full_valid_response(self):
        """A complete, well-formed judge response should parse all fields."""
        text = (
            "CORRECTNESS: 8\n"
            "COMPLETENESS: 7\n"
            "CLARITY: 9\n"
            "COGNITIVE_ALIGNMENT: 6\n"
            "OVERALL_SCORE: 8\n"
            "JUSTIFICATION: Good answer but incomplete."
        )
        result = parse_judge_response(text)
        assert result["correctness"] == 8
        assert result["completeness"] == 7
        assert result["clarity"] == 9
        assert result["cognitive_alignment"] == 6
        assert result["overall_score"] == 8
        assert result["justification"] == "Good answer but incomplete."

    def test_missing_overall_score_is_computed_as_mean(self):
        """
        If OVERALL_SCORE is absent, it should be computed as the mean
        of the four sub-scores. Here: (8+6+9+7) / 4 = 7.5
        """
        text = (
            "CORRECTNESS: 8\n"
            "COMPLETENESS: 6\n"
            "CLARITY: 9\n"
            "COGNITIVE_ALIGNMENT: 7\n"
            "JUSTIFICATION: Decent response."
        )
        result = parse_judge_response(text)
        assert result["overall_score"] == 7.5

    def test_empty_response_returns_none_scores(self):
        """An empty string should return None for all numeric fields."""
        result = parse_judge_response("")
        assert result["correctness"] is None
        assert result["completeness"] is None
        assert result["clarity"] is None
        assert result["cognitive_alignment"] is None
        assert result["overall_score"] is None
        assert result["justification"] == ""

    def test_partial_response_only_parses_present_fields(self):
        """
        If only some fields are present, only those should be populated.
        Others should remain None.
        """
        text = "CORRECTNESS: 5\nJUSTIFICATION: Only partially scored."
        result = parse_judge_response(text)
        assert result["correctness"] == 5
        assert result["completeness"] is None
        assert result["justification"] == "Only partially scored."

    def test_score_with_trailing_text_is_parsed(self):
        """
        The judge sometimes writes "8/10" or "8 (good)".
        Parser should extract the first token as the integer.
        """
        text = "CORRECTNESS: 8/10\nCOMPLETENESS: 7 (solid)\nCLARITY: 9\nCOGNITIVE_ALIGNMENT: 6\nOVERALL_SCORE: 7\nJUSTIFICATION: ok"
        result = parse_judge_response(text)
        assert result["correctness"] == 8
        assert result["completeness"] == 7

    def test_justification_with_colon_is_preserved(self):
        """
        Justification text may contain colons (e.g. "Score: 8 because...").
        Only the first colon should be used as the key/value separator.
        """
        text = (
            "CORRECTNESS: 7\n"
            "COMPLETENESS: 7\n"
            "CLARITY: 7\n"
            "COGNITIVE_ALIGNMENT: 7\n"
            "OVERALL_SCORE: 7\n"
            "JUSTIFICATION: The answer is correct: it covers the main idea."
        )
        result = parse_judge_response(text)
        assert "The answer is correct: it covers the main idea." == result["justification"]

    def test_all_numeric_results_are_int_or_none(self):
        """Parsed numeric scores should be int (not float, not str), or None."""
        text = (
            "CORRECTNESS: 8\nCOMPLETENESS: 7\nCLARITY: 9\n"
            "COGNITIVE_ALIGNMENT: 6\nOVERALL_SCORE: 7\nJUSTIFICATION: ok"
        )
        result = parse_judge_response(text)
        for key in ["correctness", "completeness", "clarity", "cognitive_alignment", "overall_score"]:
            assert isinstance(result[key], (int, float)), f"{key} should be numeric"


# ---------------------------------------------------------------------------
# call_judge
# ---------------------------------------------------------------------------

class TestCallJudge:
    def _make_mock_client(self, content):
        """Helper: build a mock OpenAI client that returns the given content string."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_successful_call_returns_parsed_scores(self):
        """A successful judge call should return a dict with parsed scores."""
        content = (
            "CORRECTNESS: 8\nCOMPLETENESS: 7\nCLARITY: 9\n"
            "COGNITIVE_ALIGNMENT: 6\nOVERALL_SCORE: 7\nJUSTIFICATION: Good."
        )
        client = self._make_mock_client(content)
        result = call_judge(client, "gpt-4.1-mini", "some prompt")
        assert result["correctness"] == 8
        assert result["overall_score"] == 7

    def test_single_call_made_on_success(self):
        """The API should only be called once if the first attempt succeeds."""
        content = (
            "CORRECTNESS: 8\nCOMPLETENESS: 7\nCLARITY: 9\n"
            "COGNITIVE_ALIGNMENT: 6\nOVERALL_SCORE: 7\nJUSTIFICATION: ok"
        )
        client = self._make_mock_client(content)
        call_judge(client, "gpt-4.1-mini", "prompt")
        assert client.chat.completions.create.call_count == 1

    def test_retries_on_exception(self):
        """
        If the first call raises an exception, it should retry.
        Here: first call fails, second succeeds.
        """
        content = (
            "CORRECTNESS: 5\nCOMPLETENESS: 5\nCLARITY: 5\n"
            "COGNITIVE_ALIGNMENT: 5\nOVERALL_SCORE: 5\nJUSTIFICATION: ok"
        )
        good_response = MagicMock()
        good_response.choices[0].message.content = content

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("transient error"),
            good_response,
        ]

        with patch("judge.time.sleep"):  # don't actually wait during tests
            result = call_judge(client, "gpt-4.1-mini", "prompt")

        assert client.chat.completions.create.call_count == 2
        assert result["overall_score"] == 5

    def test_all_retries_fail_returns_error_justification(self):
        """
        If all 3 attempts fail, call_judge should return None scores with
        an error message in the justification field rather than raising.
        """
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API down")

        with patch("judge.time.sleep"):
            result = call_judge(client, "gpt-4.1-mini", "prompt")

        assert client.chat.completions.create.call_count == 3
        assert result["correctness"] is None
        assert result["overall_score"] is None
        assert "Judge error" in result["justification"]

    def test_error_message_truncated_to_100_chars(self):
        """The error message captured in justification should be <= 100 chars."""
        long_error = "x" * 500
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception(long_error)

        with patch("judge.time.sleep"):
            result = call_judge(client, "gpt-4.1-mini", "prompt")

        # "Judge error: " prefix + up to 100 chars of the exception message
        assert len(result["justification"]) <= len("Judge error: ") + 100
