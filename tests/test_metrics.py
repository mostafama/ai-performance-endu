"""
tests/test_metrics.py
---------------------
Tests for metrics.py — BLEU, ROUGE, and keyword overlap scoring.

Each metric function is tested for:
  - Normal inputs with known expected ranges
  - Perfect match (should score 1.0 or near 1.0)
  - Empty reference or candidate (should return 0.0, not crash)
  - Both empty (should return 0.0, not crash)
  - Return type correctness (all results should be floats)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import calculate_bleu, calculate_rouge, calculate_keyword_overlap


REF  = "The mitochondria is the powerhouse of the cell and produces ATP energy"
CAND = "Mitochondria produces ATP and acts as the powerhouse of the cell"
UNRELATED = "The stock market fell sharply on Tuesday amid rising inflation fears"


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

class TestBLEU:
    def test_similar_texts_score_above_zero(self):
        score = calculate_bleu(REF, CAND)
        assert score > 0.0

    def test_identical_texts_score_near_one(self):
        score = calculate_bleu(REF, REF)
        assert score > 0.9

    def test_unrelated_texts_score_near_zero(self):
        score = calculate_bleu(REF, UNRELATED)
        assert score < 0.1

    def test_empty_reference_returns_zero(self):
        assert calculate_bleu("", CAND) == 0.0

    def test_empty_candidate_returns_zero(self):
        assert calculate_bleu(REF, "") == 0.0

    def test_both_empty_returns_zero(self):
        assert calculate_bleu("", "") == 0.0

    def test_returns_float(self):
        assert isinstance(calculate_bleu(REF, CAND), float)

    def test_score_bounded_zero_to_one(self):
        score = calculate_bleu(REF, CAND)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

class TestROUGE:
    def test_returns_dict_with_three_keys(self):
        result = calculate_rouge(REF, CAND)
        assert set(result.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_similar_texts_score_above_zero(self):
        result = calculate_rouge(REF, CAND)
        assert result["rouge1"] > 0.0
        assert result["rouge2"] > 0.0
        assert result["rougeL"] > 0.0

    def test_identical_texts_score_near_one(self):
        result = calculate_rouge(REF, REF)
        assert result["rouge1"] > 0.95
        assert result["rougeL"] > 0.95

    def test_unrelated_texts_score_near_zero(self):
        result = calculate_rouge(REF, UNRELATED)
        # Some common stopwords may overlap, so use a generous threshold
        assert result["rouge2"] < 0.1

    def test_empty_reference_returns_zeros(self):
        result = calculate_rouge("", CAND)
        assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_empty_candidate_returns_zeros(self):
        result = calculate_rouge(REF, "")
        assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_both_empty_returns_zeros(self):
        result = calculate_rouge("", "")
        assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_all_values_are_floats(self):
        result = calculate_rouge(REF, CAND)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_all_values_bounded_zero_to_one(self):
        result = calculate_rouge(REF, CAND)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} is out of bounds"

    def test_rouge1_gte_rouge2(self):
        """
        ROUGE-1 (unigrams) should always be >= ROUGE-2 (bigrams) — more
        overlap is possible at the unigram level than the bigram level.
        """
        result = calculate_rouge(REF, CAND)
        assert result["rouge1"] >= result["rouge2"]


# ---------------------------------------------------------------------------
# Keyword Overlap
# ---------------------------------------------------------------------------

class TestKeywordOverlap:
    def test_returns_three_values(self):
        p, r, f1 = calculate_keyword_overlap(REF, CAND)
        assert isinstance(p, float)
        assert isinstance(r, float)
        assert isinstance(f1, float)

    def test_identical_texts_score_one(self):
        p, r, f1 = calculate_keyword_overlap(REF, REF)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_partial_overlap_between_zero_and_one(self):
        p, r, f1 = calculate_keyword_overlap(REF, CAND)
        assert 0.0 < p <= 1.0
        assert 0.0 < r <= 1.0
        assert 0.0 < f1 <= 1.0

    def test_unrelated_texts_low_overlap(self):
        p, r, f1 = calculate_keyword_overlap(REF, UNRELATED)
        assert f1 < 0.2

    def test_empty_reference_returns_zeros(self):
        assert calculate_keyword_overlap("", CAND) == (0.0, 0.0, 0.0)

    def test_empty_candidate_returns_zeros(self):
        assert calculate_keyword_overlap(REF, "") == (0.0, 0.0, 0.0)

    def test_both_empty_returns_zeros(self):
        assert calculate_keyword_overlap("", "") == (0.0, 0.0, 0.0)

    def test_f1_is_harmonic_mean_of_precision_recall(self):
        """F1 should equal 2*P*R / (P+R) when both are non-zero."""
        p, r, f1 = calculate_keyword_overlap(REF, CAND)
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert abs(f1 - expected_f1) < 1e-9

    def test_short_words_are_filtered(self):
        """
        Tokens of 2 characters or fewer are excluded from keyword extraction.
        So a reference of only short words should produce (0, 0, 0).
        """
        short_only = "a an is it in to of or"
        p, r, f1 = calculate_keyword_overlap(short_only, short_only)
        assert (p, r, f1) == (0.0, 0.0, 0.0)
