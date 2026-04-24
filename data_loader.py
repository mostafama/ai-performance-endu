"""
data_loader.py
--------------
Loads and validates question CSV (or Excel) files into a single DataFrame.

Supports:
  - Multiple input files merged into one (duplicates removed by question_id)
  - CSV (.csv) and Excel (.xlsx) formats
  - Optional filtering to skip specific dataset names
  - Validation that all required columns are present before processing begins
"""

from pathlib import Path
import pandas as pd
import config


def load_questions(csv_paths, skip_datasets=None):
    """
    Load one or more question files, combine them, and validate the schema.

    Args:
        csv_paths:     list of file paths (str or Path) to CSV or Excel files
        skip_datasets: optional list of dataset_name values to exclude

    Returns:
        pd.DataFrame with all questions, deduplicated by question_id

    Raises:
        FileNotFoundError: if no valid files were found
        ValueError:        if required columns are missing
    """
    dfs = []
    skip_datasets = skip_datasets or []

    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {path} not found — skipping")
            continue

        # Support both CSV and Excel input formats
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)

        # Filter out unwanted datasets before combining
        if skip_datasets and "dataset_name" in df.columns:
            df = df[~df["dataset_name"].isin(skip_datasets)]

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No question files found at: {csv_paths}")

    # Merge all files and remove any duplicate question IDs
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["question_id"])

    # Add optional columns as empty strings if they weren't in the file
    for col in ["context", "choices_json", "question_type"]:
        if col not in combined.columns:
            combined[col] = ""

    # Validate required columns exist before returning
    missing = [c for c in config.REQUIRED_QUESTION_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required question columns: {missing}")

    # Warn about rows with null question_text or ground_truth_answer.
    # These would produce malformed prompts ("nan\n\nPlease solve...") or
    # meaningless lexical scores against a null reference — flag them early.
    for col in ["question_text", "ground_truth_answer"]:
        null_count = combined[col].isna().sum()
        if null_count > 0:
            print(f"  WARNING: {null_count} row(s) have null '{col}' "
                  f"and will produce unreliable results.")

    return combined
