"""
tests/test_labeler.py
---------------------
Tests for labeler.py — Bloom's taxonomy question classifier.

Tests cover:
  - build_labeling_prompt(): all required content appears in the prompt
  - parse_labeling_response(): normal response, all six levels, missing fields,
    empty response, confidence parsing, trailing text on level field
  - QuestionLabeler.run(): questions are labelled, already-labelled questions
    are skipped, --relabel overrides the skip, errors are handled gracefully,
    progress is saved to the input file

No real API calls are made — the OpenAI client is mocked throughout.
"""

import sys
import os
import pytest
import pandas as pd
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeler import build_labeling_prompt, parse_labeling_response, QuestionLabeler
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_questions_csv(rows, path=None):
    """Write a questions DataFrame to a temp CSV and return the path."""
    if path is None:
        path = tempfile.mktemp(suffix=".csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_mock_client(level=3, name="Apply", confidence=0.92, justification="Uses a procedure."):
    """Return a mock OpenAI client that returns a fixed Bloom label response."""
    content = (
        f"LEVEL: {level}\n"
        f"NAME: {name}\n"
        f"CONFIDENCE: {confidence}\n"
        f"JUSTIFICATION: {justification}"
    )
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


# ---------------------------------------------------------------------------
# build_labeling_prompt
# ---------------------------------------------------------------------------

class TestBuildLabelingPrompt:
    def test_question_text_is_included(self):
        prompt = build_labeling_prompt("What is photosynthesis?", "science")
        assert "What is photosynthesis?" in prompt

    def test_domain_is_included(self):
        prompt = build_labeling_prompt("Solve 2x+4=10", "math")
        assert "math" in prompt

    def test_all_six_bloom_levels_listed(self):
        prompt = build_labeling_prompt("Some question?", "science")
        for level, name in config.BLOOM_LEVELS.items():
            assert str(level) in prompt
            assert name in prompt

    def test_output_format_instructions_present(self):
        """Prompt must instruct the model to output LEVEL, NAME, CONFIDENCE, JUSTIFICATION."""
        prompt = build_labeling_prompt("Some question?", "science")
        for label in ["LEVEL", "NAME", "CONFIDENCE", "JUSTIFICATION"]:
            assert label in prompt

    def test_different_domains_produce_different_prompts(self):
        p1 = build_labeling_prompt("Question?", "math")
        p2 = build_labeling_prompt("Question?", "science")
        assert p1 != p2


# ---------------------------------------------------------------------------
# parse_labeling_response
# ---------------------------------------------------------------------------

class TestParseLabelingResponse:
    def test_full_valid_response(self):
        text = "LEVEL: 3\nNAME: Apply\nCONFIDENCE: 0.92\nJUSTIFICATION: Uses a procedure."
        result = parse_labeling_response(text)
        assert result["bloom_level"] == 3
        assert result["bloom_name"] == "Apply"        # canonical name from config
        assert result["bloom_confidence"] == 0.92
        assert result["bloom_justification"] == "Uses a procedure."

    def test_canonical_name_used_even_if_model_uses_wrong_case(self):
        """
        The canonical bloom_name should always come from config.BLOOM_LEVELS,
        not from whatever the model wrote. This guarantees consistent casing.
        """
        text = "LEVEL: 2\nNAME: understand\nCONFIDENCE: 0.8\nJUSTIFICATION: ok"
        result = parse_labeling_response(text)
        assert result["bloom_name"] == config.BLOOM_LEVELS[2]  # "Understand"

    def test_all_six_levels_parse_correctly(self):
        for level, name in config.BLOOM_LEVELS.items():
            text = f"LEVEL: {level}\nNAME: {name}\nCONFIDENCE: 0.9\nJUSTIFICATION: ok"
            result = parse_labeling_response(text)
            assert result["bloom_level"] == level
            assert result["bloom_name"] == name

    def test_level_with_trailing_text_is_parsed(self):
        """Model sometimes writes '3 (Apply)' — only the digit should be extracted."""
        text = "LEVEL: 3 (Apply)\nNAME: Apply\nCONFIDENCE: 0.9\nJUSTIFICATION: ok"
        result = parse_labeling_response(text)
        assert result["bloom_level"] == 3

    def test_confidence_as_integer_is_parsed(self):
        """Model might write '1' instead of '1.0' — should still parse."""
        text = "LEVEL: 1\nNAME: Remember\nCONFIDENCE: 1\nJUSTIFICATION: ok"
        result = parse_labeling_response(text)
        assert result["bloom_confidence"] == 1.0

    def test_empty_response_returns_none_fields(self):
        result = parse_labeling_response("")
        assert result["bloom_level"] is None
        assert result["bloom_name"] is None
        assert result["bloom_confidence"] is None
        assert result["bloom_justification"] == ""

    def test_invalid_level_returns_none(self):
        """Level 7 doesn't exist in Bloom's taxonomy — should return None."""
        text = "LEVEL: 7\nNAME: ???\nCONFIDENCE: 0.5\nJUSTIFICATION: ok"
        result = parse_labeling_response(text)
        assert result["bloom_level"] is None

    def test_justification_with_colon_preserved(self):
        """Justification text may itself contain colons."""
        text = "LEVEL: 4\nNAME: Analyze\nCONFIDENCE: 0.85\nJUSTIFICATION: The question asks: compare two approaches."
        result = parse_labeling_response(text)
        assert "The question asks: compare two approaches." == result["bloom_justification"]

    def test_confidence_is_rounded_to_three_decimals(self):
        text = "LEVEL: 3\nNAME: Apply\nCONFIDENCE: 0.91666\nJUSTIFICATION: ok"
        result = parse_labeling_response(text)
        assert result["bloom_confidence"] == 0.917


# ---------------------------------------------------------------------------
# QuestionLabeler
# ---------------------------------------------------------------------------

class TestQuestionLabeler:
    def _make_labeler(self, rows, mock_client, relabel=False):
        """Write rows to a temp CSV and return a QuestionLabeler with mocked client."""
        path = make_questions_csv(rows)
        with patch("labeler.OpenAI", return_value=mock_client):
            labeler = QuestionLabeler(path, relabel=relabel)
        return labeler, path

    def test_unlabelled_questions_get_labelled(self):
        rows = [
            {"question_id": "q001", "question_text": "What is ATP?",
             "domain": "science", "bloom_level": None, "bloom_name": None},
            {"question_id": "q002", "question_text": "Solve 2x=10",
             "domain": "math", "bloom_level": None, "bloom_name": None},
        ]
        client = make_mock_client(level=2, name="Understand", confidence=0.88)
        labeler, path = self._make_labeler(rows, client)

        labeler.run()

        result = pd.read_csv(path)
        assert result.iloc[0]["bloom_level"] == 2
        assert result.iloc[0]["bloom_name"] == "Understand"
        assert result.iloc[0]["bloom_confidence"] == 0.88
        os.unlink(path)

    def test_already_labelled_questions_are_skipped(self):
        """Questions with an existing bloom_level should not be re-sent to the API."""
        rows = [
            {"question_id": "q001", "question_text": "Already done",
             "domain": "science", "bloom_level": 3, "bloom_name": "Apply"},
            {"question_id": "q002", "question_text": "Needs labelling",
             "domain": "science", "bloom_level": None, "bloom_name": None},
        ]
        call_count = [0]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = (
            "LEVEL: 2\nNAME: Understand\nCONFIDENCE: 0.9\nJUSTIFICATION: ok"
        )
        mock_client = MagicMock()
        def count_and_return(**kwargs):
            call_count[0] += 1
            return mock_resp
        mock_client.chat.completions.create.side_effect = count_and_return

        labeler, path = self._make_labeler(rows, mock_client)
        labeler.run()

        assert call_count[0] == 1   # only q002 was sent to the API
        result = pd.read_csv(path)
        # q001 should still have its original label
        assert result.iloc[0]["bloom_level"] == 3
        assert result.iloc[0]["bloom_name"] == "Apply"
        os.unlink(path)

    def test_relabel_forces_reclassification_of_existing_labels(self):
        """With relabel=True, even questions with bloom_level should be re-sent."""
        rows = [
            {"question_id": "q001", "question_text": "Already done",
             "domain": "science", "bloom_level": 3, "bloom_name": "Apply"},
        ]
        call_count = [0]
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = (
            "LEVEL: 5\nNAME: Evaluate\nCONFIDENCE: 0.95\nJUSTIFICATION: re-labelled"
        )
        mock_client = MagicMock()
        def count_and_return(**kwargs):
            call_count[0] += 1
            return mock_resp
        mock_client.chat.completions.create.side_effect = count_and_return

        path = make_questions_csv(rows)
        with patch("labeler.OpenAI", return_value=mock_client):
            labeler = QuestionLabeler(path, relabel=True)
        labeler.run()

        assert call_count[0] == 1
        result = pd.read_csv(path)
        assert result.iloc[0]["bloom_level"] == 5   # updated from 3 to 5
        os.unlink(path)

    def test_results_written_back_to_input_file(self):
        """The labeler should overwrite the input CSV with updated labels."""
        rows = [{"question_id": "q001", "question_text": "Q?",
                 "domain": "science", "bloom_level": None, "bloom_name": None}]
        client = make_mock_client(level=1, name="Remember", confidence=0.99)
        labeler, path = self._make_labeler(rows, client)
        labeler.run()

        result = pd.read_csv(path)
        assert len(result) == 1
        assert result.iloc[0]["bloom_level"] == 1
        os.unlink(path)

    def test_api_error_saves_error_message_in_justification(self):
        """If all retries fail, the justification should contain the error message."""
        rows = [{"question_id": "q001", "question_text": "Q?",
                 "domain": "science", "bloom_level": None, "bloom_name": None}]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")

        path = make_questions_csv(rows)
        with patch("labeler.OpenAI", return_value=mock_client), \
             patch("labeler.time.sleep"):
            labeler = QuestionLabeler(path)
        labeler.run()

        result = pd.read_csv(path)
        assert "Labeling error" in str(result.iloc[0]["bloom_justification"])
        # bloom_level should remain None — don't write a wrong label
        assert pd.isna(result.iloc[0]["bloom_level"])
        os.unlink(path)

    def test_missing_questions_file_raises_file_not_found(self):
        with patch("labeler.OpenAI", return_value=MagicMock()):
            with pytest.raises(FileNotFoundError):
                QuestionLabeler("nonexistent.csv")

    def test_missing_question_text_column_raises_value_error(self):
        rows = [{"question_id": "q001", "domain": "science"}]  # no question_text
        path = make_questions_csv(rows)
        with patch("labeler.OpenAI", return_value=MagicMock()):
            with pytest.raises(ValueError) as exc_info:
                QuestionLabeler(path)
        assert "question_text" in str(exc_info.value)
        os.unlink(path)

    def test_bloom_columns_added_if_not_in_file(self):
        """If the CSV has no bloom columns, they should be added automatically."""
        rows = [{"question_id": "q001", "question_text": "Q?", "domain": "science"}]
        path = make_questions_csv(rows)
        with patch("labeler.OpenAI", return_value=make_mock_client()):
            labeler = QuestionLabeler(path)
        for col in ["bloom_level", "bloom_name", "bloom_confidence", "bloom_justification"]:
            assert col in labeler.df.columns
        os.unlink(path)

    def test_all_done_exits_without_api_calls(self):
        """If all questions already have bloom_level, no API calls should be made."""
        rows = [
            {"question_id": "q001", "question_text": "Q1?", "domain": "science",
             "bloom_level": 2, "bloom_name": "Understand"},
            {"question_id": "q002", "question_text": "Q2?", "domain": "math",
             "bloom_level": 4, "bloom_name": "Analyze"},
        ]
        call_count = [0]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **k: (_ for _ in ()).throw(
            AssertionError("Should not be called")
        )
        path = make_questions_csv(rows)
        with patch("labeler.OpenAI", return_value=mock_client):
            labeler = QuestionLabeler(path)
        labeler.run()  # should not raise
        os.unlink(path)
