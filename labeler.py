"""
labeler.py
----------
Assigns Bloom's Revised Taxonomy levels to questions that don't yet have them.

Each question is sent to an LLM (GPT-4.1-mini by default) with a structured
prompt asking it to classify the question into one of the six Bloom's levels:

    1 — Remember     (recall facts, definitions, formulas)
    2 — Understand   (explain, summarise, paraphrase)
    3 — Apply        (use a procedure or method to solve a problem)
    4 — Analyze      (break down, compare, identify relationships)
    5 — Evaluate     (judge, critique, justify a position)
    6 — Create       (design, construct, produce something new)

The labeler reads questions.csv (or any CSV with a question_text column),
classifies each question, and writes the result back to the same file with
bloom_level, bloom_name, and bloom_confidence columns populated.

Questions that already have a bloom_level are skipped unless --relabel is passed.

Usage:
    python run.py label --questions questions.csv
    python run.py label --questions questions.csv --relabel
    python run.py label --questions questions.csv --model gpt-4.1-mini
"""

import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_labeling_prompt(question_text, domain):
    """
    Build the classification prompt for a single question.

    The prompt asks the judge to output a single structured block so it can
    be parsed reliably without fragile regex over free-form text.
    """
    levels_block = "\n".join(
        f"  {level} — {name}"
        for level, name in config.BLOOM_LEVELS.items()
    )
    return f"""You are an expert in educational assessment and Bloom's Revised Taxonomy.

Classify the following question into exactly one of the six Bloom's levels.

BLOOM'S LEVELS:
{levels_block}

QUESTION DOMAIN: {domain}
QUESTION: {question_text}

Respond in exactly this format — no other text:
LEVEL: [1-6]
NAME: [level name]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [one sentence explaining why]
"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_labeling_response(text):
    """
    Parse the labeler's structured response into a dict.

    Returns:
        dict with keys: bloom_level (int), bloom_name (str),
                        bloom_confidence (float), bloom_justification (str)
        Returns None for all fields if parsing fails.
    """
    result = {
        "bloom_level":         None,
        "bloom_name":          None,
        "bloom_confidence":    None,
        "bloom_justification": "",
    }

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key   = key.strip().upper()
        value = value.strip()

        if key == "LEVEL":
            # Extract leading digit in case the model writes "3 (Apply)"
            m = re.match(r"(\d)", value)
            if m:
                level = int(m.group(1))
                if level in config.BLOOM_LEVELS:
                    result["bloom_level"] = level
                    # Always use the canonical name from config, not the model's wording
                    result["bloom_name"]  = config.BLOOM_LEVELS[level]

        elif key == "CONFIDENCE":
            m = re.match(r"(\d+\.?\d*)", value)
            if m:
                result["bloom_confidence"] = round(float(m.group(1)), 3)

        elif key == "JUSTIFICATION":
            result["bloom_justification"] = value

    return result


# ---------------------------------------------------------------------------
# Labeler class
# ---------------------------------------------------------------------------

class QuestionLabeler:
    def __init__(self, questions_csv, model="gpt-4.1-mini", relabel=False):
        """
        Args:
            questions_csv: path to the questions CSV file to label
            model:         OpenAI model to use as the classifier
            relabel:       if True, re-classify questions that already have
                           a bloom_level; if False (default), skip them
        """
        self.path     = Path(questions_csv)
        self.model    = model
        self.relabel  = relabel
        self.client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.df       = self._load()

    def _load(self):
        """Load the questions CSV and add Bloom columns if missing."""
        if not self.path.exists():
            raise FileNotFoundError(f"Questions file not found: {self.path}")

        df = pd.read_csv(self.path)

        if "question_text" not in df.columns:
            raise ValueError("questions CSV must have a 'question_text' column")

        # Add Bloom columns with empty defaults if not present.
        # Use explicit dtypes so pandas doesn't infer float64 for all-None columns,
        # which would prevent writing string values like bloom_name later.
        bloom_dtypes = {
            "bloom_level":         "Int64",   # nullable integer (pandas extension type)
            "bloom_name":          "object",
            "bloom_confidence":    "float64",
            "bloom_justification": "object",
        }
        for col, dtype in bloom_dtypes.items():
            if col not in df.columns:
                df[col] = pd.array([None] * len(df), dtype=dtype)
            else:
                # Cast existing column to the correct dtype
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    df[col] = pd.array([None] * len(df), dtype=dtype)

        return df

    def _call_labeler(self, prompt):
        """
        Send a labeling prompt to the LLM with up to 3 retries.

        Returns a parsed result dict, or a dict of Nones on total failure.
        """
        for attempt in range(1, 4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,    # deterministic — same question should always get same label
                    timeout=30,
                )
                return parse_labeling_response(resp.choices[0].message.content or "")
            except Exception as exc:
                if attempt == 3:
                    print(f"    Labeler error after 3 attempts: {str(exc)[:80]}")
                    return {
                        "bloom_level":         None,
                        "bloom_name":          None,
                        "bloom_confidence":    None,
                        "bloom_justification": f"Labeling error: {str(exc)[:80]}",
                    }
                time.sleep(config.RETRY_DELAY)

    def _should_label(self, row):
        """Return True if this row needs labeling."""
        if self.relabel:
            return True
        # Skip if bloom_level is already populated with a valid integer
        level = row.get("bloom_level")
        try:
            return level is None or pd.isna(level)
        except (TypeError, ValueError):
            return True

    def run(self):
        """
        Label all questions that need labeling, saving progress every
        SAVE_BATCH_SIZE rows. Overwrites the input file in place.
        """
        to_label = [i for i, row in self.df.iterrows() if self._should_label(row)]

        if not to_label:
            print(f"All {len(self.df)} questions already labelled. "
                  f"Use --relabel to re-classify.")
            return

        print(f"Labelling {len(to_label)} of {len(self.df)} questions "
              f"using {self.model}...")

        for count, idx in enumerate(to_label, 1):
            row    = self.df.loc[idx]
            prompt = build_labeling_prompt(
                question_text=str(row.get("question_text", "")),
                domain=str(row.get("domain", "unknown")),
            )
            result = self._call_labeler(prompt)

            self.df.at[idx, "bloom_level"]         = result["bloom_level"]
            self.df.at[idx, "bloom_name"]          = result["bloom_name"]
            self.df.at[idx, "bloom_confidence"]    = result["bloom_confidence"]
            self.df.at[idx, "bloom_justification"] = result["bloom_justification"]

            if count % config.SAVE_BATCH_SIZE == 0:
                self.df.to_csv(self.path, index=False)
                print(f"  Saved {count}/{len(to_label)}")

        # Final save
        self.df.to_csv(self.path, index=False)
        labelled_ok = self.df["bloom_level"].notna().sum()
        print(f"Done. {labelled_ok}/{len(self.df)} questions now have Bloom labels.")
        print(f"Updated: {self.path}")
