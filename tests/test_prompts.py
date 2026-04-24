"""
tests/test_prompts.py
---------------------
Tests for prompts.py — the domain-aware prompt builder.

Each test exercises one code path through build_prompt():
  - MCQ questions (choices appended)
  - Reading questions (context prepended)
  - CS questions with starter code (formatted as code block)
  - Math questions (plain question + boxed answer instruction)
  - Unknown domain (falls back to FALLBACK_TEMPLATE)
  - MCQ with malformed JSON choices (falls back to plain format gracefully)
  - CS question without context (no code block added)
"""

import sys
import os
import pytest
import pandas as pd

# Add parent directory to path so we can import project modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import build_prompt
import config


def make_row(**kwargs):
    """
    Helper: build a minimal question row with sensible defaults.
    Any field can be overridden via kwargs.
    """
    defaults = {
        "domain": "science",
        "question_text": "What is ATP?",
        "context": "",
        "choices_json": None,
        "question_type": "open",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# MCQ
# ---------------------------------------------------------------------------

class TestMCQPrompt:
    def test_choices_are_appended(self):
        """MCQ questions should have their answer choices listed after the question."""
        row = make_row(
            question_type="mcq",
            choices_json='{"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome"}',
        )
        prompt = build_prompt(row)
        assert "A. Nucleus" in prompt
        assert "B. Mitochondria" in prompt
        assert "C. Ribosome" in prompt

    def test_choices_are_sorted(self):
        """Choices should appear in alphabetical key order (A, B, C, D)."""
        row = make_row(
            question_type="mcq",
            choices_json='{"D": "last", "A": "first", "C": "third", "B": "second"}',
        )
        prompt = build_prompt(row)
        pos_a = prompt.index("A. first")
        pos_b = prompt.index("B. second")
        pos_c = prompt.index("C. third")
        pos_d = prompt.index("D. last")
        assert pos_a < pos_b < pos_c < pos_d

    def test_question_text_is_included(self):
        """The question text itself must always appear in the prompt."""
        row = make_row(
            question_type="mcq",
            question_text="Which organelle produces ATP?",
            choices_json='{"A": "Nucleus", "B": "Mitochondria"}',
        )
        prompt = build_prompt(row)
        assert "Which organelle produces ATP?" in prompt

    def test_malformed_json_falls_back_gracefully(self):
        """
        If choices_json is not valid JSON, build_prompt should not crash.
        It falls through to the plain question + template format.
        """
        row = make_row(
            question_type="mcq",
            choices_json="NOT VALID JSON {{",
        )
        prompt = build_prompt(row)
        assert "What is ATP?" in prompt  # question text still present
        assert "Choices:" not in prompt  # choices block was not added

    def test_nan_question_text_does_not_produce_nan_string(self):
        """
        If question_text is a float NaN (missing CSV value), the prompt must
        not contain the string 'nan'. An empty question is better than a
        corrupted one that asks the model to solve 'nan'.
        """
        import math
        row = make_row(question_text=float("nan"), domain="math")
        prompt = build_prompt(row)
        assert "nan" not in prompt.lower().split("\n")[0]  # first line should not be "nan"



# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class TestReadingPrompt:
    def test_context_is_prepended(self):
        """Reading questions should have their passage prepended as 'Context: ...'"""
        row = make_row(
            domain="reading",
            context="The author studied 100 samples and found consistent patterns.",
            question_text="What did the author conclude?",
        )
        prompt = build_prompt(row)
        assert prompt.startswith("Passage:")
        assert "The author studied 100 samples" in prompt

    def test_question_appears_after_context(self):
        """The question should come after the context block, not before."""
        row = make_row(
            domain="reading",
            context="Some passage here.",
            question_text="What is the main idea?",
        )
        prompt = build_prompt(row)
        context_pos = prompt.index("Passage:")
        question_pos = prompt.index("What is the main idea?")
        assert context_pos < question_pos

    def test_reading_template_instruction_is_included(self):
        """The reading domain instruction should be appended."""
        row = make_row(domain="reading", context="Some passage.")
        prompt = build_prompt(row)
        assert config.PROMPT_TEMPLATES["reading"] in prompt

    def test_no_context_falls_back_to_plain(self):
        """
        A reading question with no context should fall back to the plain
        'question + template' format rather than showing 'Context: '.
        """
        row = make_row(domain="reading", context="")
        prompt = build_prompt(row)
        assert "Passage:" not in prompt

    def test_reading_mcq_includes_both_passage_and_choices(self):
        """
        REGRESSION: Reading MCQs must include the passage AND the choices.
        The old code checked MCQ first, so reading MCQs got choices but no passage.
        A model answering SAT-style questions without the text is essentially guessing.
        """
        row = make_row(
            domain="reading",
            question_type="mcq",
            question_text="The author's primary purpose is to",
            context="Scientists have long debated the causes of rapid temperature change.",
            choices_json='{"A": "describe a controversy", "B": "argue for reform"}',
        )
        prompt = build_prompt(row)
        # Passage must come first
        assert "Passage:" in prompt
        assert "Scientists have long debated" in prompt
        # Choices must also be present
        assert "A. describe a controversy" in prompt
        assert "B. argue for reform" in prompt
        # Passage must come before the question
        assert prompt.index("Passage:") < prompt.index("The author's primary purpose")

    def test_reading_mcq_no_context_still_shows_choices(self):
        """A reading MCQ without a passage should still show choices (no crash)."""
        row = make_row(
            domain="reading",
            question_type="mcq",
            context="",
            choices_json='{"A": "somber", "B": "hopeful"}',
        )
        prompt = build_prompt(row)
        assert "A. somber" in prompt
        assert "Passage:" not in prompt



# ---------------------------------------------------------------------------
# Computer Science
# ---------------------------------------------------------------------------

class TestComputerSciencePrompt:
    def test_starter_code_formatted_as_code_block(self):
        """CS questions with starter code should wrap it in a python code block."""
        row = make_row(
            domain="computer_science",
            question_text="Reverse a string.",
            context="def reverse(s):\n    pass",
        )
        prompt = build_prompt(row)
        assert "```python" in prompt
        assert "def reverse(s):" in prompt
        assert "```" in prompt

    def test_instruction_comes_before_problem(self):
        """The CS instruction template should appear before the problem statement."""
        row = make_row(
            domain="computer_science",
            question_text="Sort a list.",
            context="def sort_list(lst): pass",
        )
        prompt = build_prompt(row)
        instruction_pos = prompt.index(config.PROMPT_TEMPLATES["computer_science"][:20])
        problem_pos = prompt.index("Sort a list.")
        assert instruction_pos < problem_pos

    def test_no_context_falls_back_to_plain(self):
        """CS question without starter code should not include a code block."""
        row = make_row(domain="computer_science", context="")
        prompt = build_prompt(row)
        assert "```python" not in prompt
        assert "What is ATP?" in prompt


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

class TestMathPrompt:
    def test_boxed_instruction_is_included(self):
        """Math questions should include the \\boxed{} instruction."""
        row = make_row(domain="math", question_text="Solve: 2x + 4 = 10")
        prompt = build_prompt(row)
        assert "\\boxed" in prompt

    def test_question_text_is_included(self):
        row = make_row(domain="math", question_text="Solve: 2x + 4 = 10")
        prompt = build_prompt(row)
        assert "Solve: 2x + 4 = 10" in prompt


# ---------------------------------------------------------------------------
# Fallback / unknown domain
# ---------------------------------------------------------------------------

class TestFallbackPrompt:
    def test_unknown_domain_uses_fallback_template(self):
        """Any domain not in PROMPT_TEMPLATES should use FALLBACK_TEMPLATE."""
        row = make_row(domain="history", question_text="When did WW2 end?")
        prompt = build_prompt(row)
        assert config.FALLBACK_TEMPLATE in prompt

    def test_empty_domain_uses_fallback_template(self):
        """An empty domain string should also trigger the fallback."""
        row = make_row(domain="", question_text="Some question?")
        prompt = build_prompt(row)
        assert config.FALLBACK_TEMPLATE in prompt

    def test_question_text_always_present(self):
        """Regardless of domain or type, the question text must always appear."""
        for domain in ["math", "science", "reading", "computer_science", "history", ""]:
            row = make_row(domain=domain, question_text="Unique question text XYZ")
            assert "Unique question text XYZ" in build_prompt(row), \
                f"Question text missing for domain='{domain}'"
