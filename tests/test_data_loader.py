"""
tests/test_data_loader.py
-------------------------
Tests for data_loader.py — question file loading and validation.

Tests cover:
  - Normal CSV loading with all required columns
  - Excel (.xlsx) file loading
  - Multiple files merged into one DataFrame
  - Duplicate question_id deduplication
  - skip_datasets filtering
  - Optional columns (context, choices_json, question_type) added if missing
  - FileNotFoundError when no files exist
  - ValueError when required columns are missing
  - Warning printed (not crash) when one of multiple files is missing
"""

import sys
import os
import pytest
import pandas as pd
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_questions
import config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REQUIRED_COLS = config.REQUIRED_QUESTION_COLUMNS  # the 7 required columns

def make_questions_df(n=3, dataset_names=None):
    """
    Build a minimal valid questions DataFrame with n rows.
    Optionally assign dataset_names (list of strings, length n).
    """
    data = {
        "question_id":        [f"q{i:03d}" for i in range(1, n + 1)],
        "domain":             ["science"] * n,
        "difficulty":         ["easy"] * n,
        "bloom_level":        list(range(1, n + 1)),
        "bloom_name":         ["Remember", "Understand", "Apply"][:n],
        "question_text":      [f"Question {i}?" for i in range(1, n + 1)],
        "ground_truth_answer":[f"Answer {i}"   for i in range(1, n + 1)],
    }
    if dataset_names:
        data["dataset_name"] = dataset_names
    return pd.DataFrame(data)


def write_csv(df):
    """Write a DataFrame to a temp CSV file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(f, index=False)
    f.close()
    return f.name


def write_xlsx(df):
    """Write a DataFrame to a temp Excel file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    f.close()
    df.to_excel(f.name, index=False)
    return f.name


# ---------------------------------------------------------------------------
# Normal loading
# ---------------------------------------------------------------------------

class TestNormalLoading:
    def test_loads_csv_successfully(self):
        path = write_csv(make_questions_df(3))
        result = load_questions([path])
        assert len(result) == 3
        os.unlink(path)

    def test_loads_xlsx_successfully(self):
        path = write_xlsx(make_questions_df(3))
        result = load_questions([path])
        assert len(result) == 3
        os.unlink(path)

    def test_all_required_columns_present(self):
        path = write_csv(make_questions_df(3))
        result = load_questions([path])
        for col in REQUIRED_COLS:
            assert col in result.columns, f"Missing required column: {col}"
        os.unlink(path)

    def test_optional_columns_added_if_missing(self):
        """context, choices_json, question_type must be added if not in file."""
        df = make_questions_df(2)
        # Ensure these optional columns are NOT in the input
        for col in ["context", "choices_json", "question_type"]:
            assert col not in df.columns
        path = write_csv(df)
        result = load_questions([path])
        for col in ["context", "choices_json", "question_type"]:
            assert col in result.columns, f"Optional column '{col}' was not added"
        os.unlink(path)

    def test_returns_dataframe(self):
        path = write_csv(make_questions_df(2))
        result = load_questions([path])
        assert isinstance(result, pd.DataFrame)
        os.unlink(path)


# ---------------------------------------------------------------------------
# Multiple files & deduplication
# ---------------------------------------------------------------------------

class TestMultipleFiles:
    def test_multiple_files_are_merged(self):
        """Loading two files with different question_ids should merge them."""
        df1 = make_questions_df(2)
        df2 = make_questions_df(2)
        df2["question_id"] = ["q010", "q011"]  # different IDs
        p1, p2 = write_csv(df1), write_csv(df2)
        result = load_questions([p1, p2])
        assert len(result) == 4
        os.unlink(p1); os.unlink(p2)

    def test_duplicate_question_ids_are_deduplicated(self):
        """
        If the same question_id appears in two files, only the first
        occurrence should be kept.
        """
        df = make_questions_df(3)
        p1, p2 = write_csv(df), write_csv(df)   # same file twice
        result = load_questions([p1, p2])
        assert len(result) == 3                  # not 6
        os.unlink(p1); os.unlink(p2)

    def test_missing_file_is_skipped_with_warning(self, capsys):
        """
        A missing file should print a warning but not crash.
        Other files should still be loaded.
        """
        path = write_csv(make_questions_df(2))
        result = load_questions([path, "nonexistent_file.csv"])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert len(result) == 2
        os.unlink(path)


# ---------------------------------------------------------------------------
# skip_datasets
# ---------------------------------------------------------------------------

class TestSkipDatasets:
    def test_rows_in_skip_list_are_excluded(self):
        df = make_questions_df(3, dataset_names=["MATH", "SCI", "MATH"])
        path = write_csv(df)
        result = load_questions([path], skip_datasets=["MATH"])
        assert len(result) == 1
        assert all(result["dataset_name"] == "SCI")
        os.unlink(path)

    def test_empty_skip_list_keeps_all_rows(self):
        df = make_questions_df(3, dataset_names=["MATH", "SCI", "READ"])
        path = write_csv(df)
        result = load_questions([path], skip_datasets=[])
        assert len(result) == 3
        os.unlink(path)

    def test_skip_nonexistent_dataset_keeps_all_rows(self):
        """Skipping a dataset name that doesn't exist should not remove anything."""
        df = make_questions_df(3, dataset_names=["MATH", "SCI", "READ"])
        path = write_csv(df)
        result = load_questions([path], skip_datasets=["BIOLOGY"])
        assert len(result) == 3
        os.unlink(path)

    def test_no_dataset_name_column_ignores_skip(self):
        """If dataset_name column is absent, skip_datasets should do nothing."""
        df = make_questions_df(3)               # no dataset_name column
        path = write_csv(df)
        result = load_questions([path], skip_datasets=["MATH"])
        assert len(result) == 3
        os.unlink(path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_no_valid_files_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_questions(["nonexistent1.csv", "nonexistent2.csv"])

    def test_missing_required_columns_raises_value_error(self):
        """A file missing required columns should raise ValueError with details."""
        bad_df = pd.DataFrame({"question_id": ["q001"], "domain": ["math"]})
        path = write_csv(bad_df)
        with pytest.raises(ValueError) as exc_info:
            load_questions([path])
        # Error message should name the missing columns
        assert "Missing required question columns" in str(exc_info.value)
        os.unlink(path)

    def test_error_message_lists_missing_column_names(self):
        """The ValueError should specifically name which columns are missing."""
        bad_df = pd.DataFrame({"question_id": ["q001"]})
        path = write_csv(bad_df)
        with pytest.raises(ValueError) as exc_info:
            load_questions([path])
        error_msg = str(exc_info.value)
        # At least some of the missing columns should be named in the error
        assert "domain" in error_msg or "bloom_level" in error_msg
        os.unlink(path)


class TestNullValueWarnings:
    def test_null_question_text_prints_warning(self, capsys):
        """Rows with null question_text should trigger a printed WARNING."""
        data = {
            "question_id":         ["q001", "q002"],
            "domain":              ["science", "science"],
            "difficulty":          ["easy", "easy"],
            "bloom_level":         [1, 1],
            "bloom_name":          ["Remember", "Remember"],
            "question_text":       [None, "What is ATP?"],   # one null
            "ground_truth_answer": ["A1", "A2"],
        }
        path = write_csv(pd.DataFrame(data))
        load_questions([path])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "question_text" in captured.out
        os.unlink(path)

    def test_null_ground_truth_prints_warning(self, capsys):
        """Rows with null ground_truth_answer should trigger a printed WARNING."""
        data = {
            "question_id":         ["q001"],
            "domain":              ["math"],
            "difficulty":          ["easy"],
            "bloom_level":         [1],
            "bloom_name":          ["Remember"],
            "question_text":       ["2+2=?"],
            "ground_truth_answer": [None],    # null
        }
        path = write_csv(pd.DataFrame(data))
        load_questions([path])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "ground_truth_answer" in captured.out
        os.unlink(path)

    def test_no_warning_when_all_values_present(self, capsys):
        """No warning should be printed when all required values are present."""
        path = write_csv(make_questions_df(3))
        load_questions([path])
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
        os.unlink(path)
