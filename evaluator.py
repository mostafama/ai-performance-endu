"""
evaluator.py
-----------------------
Scores LLM responses using an LLM judge and lexical similarity metrics.

The ResponseEvaluator class:
  1. Loads the responses CSV and joins it with the questions CSV
  2. Skips rows already scored (supports resuming interrupted runs)
  3. For each response, calls the judge LLM to get 4 rubric scores (0–10)
  4. Computes BLEU, ROUGE, and keyword overlap against the reference answer
  5. Saves results in batches to the output CSV

The final output CSV contains one row per (question, model) pair with
all scores, metadata, and lexical metrics combined.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import config
from data_loader import load_questions
from judge import build_judge_prompt, call_judge
from metrics import calculate_bleu, calculate_rouge, calculate_keyword_overlap

load_dotenv()


class ResponseEvaluator:
    def __init__(self, questions_csv, responses_csv, output_csv, judge_model):
        """
        Args:
            questions_csv: list of question file paths (CSV or Excel)
            responses_csv: path to the responses CSV produced by querier.py
            output_csv:    path to write scored results
            judge_model:   model name for the judge (e.g. "gpt-4.1-mini")
        """
        # Load and join questions + responses on question_id
        self.questions_df = load_questions(questions_csv)
        self.responses_df = self._load_responses(responses_csv)
        self.data = self.responses_df.merge(self.questions_df, on="question_id", how="inner")

        # Warn if any responses could not be joined to a question.
        # This happens when question_id in the responses CSV does not exist in
        # questions.csv — those rows will be silently excluded from scoring.
        unmatched = len(self.responses_df) - len(self.data)
        if unmatched > 0:
            print(f"  WARNING: {unmatched} response(s) had no matching question_id "
                  f"in the questions file and will be skipped.")

        self.output_path = Path(output_csv)
        self.score_counter = 0

        # If output already exists, skip rows that are already scored
        if self.output_path.exists():
            done_df = pd.read_csv(self.output_path)
            self.score_counter = len(done_df)
            done_ids = set(done_df["response_id"])
            self.data = self.data[~self.data["response_id"].isin(done_ids)]

        # Initialize the OpenAI judge client
        from openai import OpenAI
        self.judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.judge_model = judge_model

        # In-memory buffer before flushing to disk
        self.scores = []

    def _load_responses(self, path):
        """Load and validate the responses CSV produced by querier.py."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Responses file not found: {path}")
        df = pd.read_csv(p)
        missing = [c for c in config.REQUIRED_RESPONSE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required response columns: {missing}")
        return df

    def _evaluate_row(self, row):
        """
        Score a single response row.

        Populates all OUTPUT_COLUMNS fields:
          - Metadata fields copied from the joined question+response row
          - Judge scores from the LLM judge
          - Lexical scores from BLEU/ROUGE/keyword overlap
          - Skips scoring if the response itself had an error
        """
        # Start with a blank record for all output columns
        base = {c: None for c in config.OUTPUT_COLUMNS}

        # Fill in metadata fields from the row
        base.update({
            "score_id":            f"score_{self.score_counter + 1:06d}",
            "response_id":         row["response_id"],
            "question_id":         row["question_id"],
            "model_name":          row["model_name"],
            "dataset_name":        row.get("dataset_name"),
            "dataset_subset":      row.get("dataset_subset"),
            "domain":              row.get("domain"),
            "difficulty":          row.get("difficulty"),
            "bloom_level":         row.get("bloom_level"),
            "bloom_name":          row.get("bloom_name"),
            "bloom_confidence":    row.get("bloom_confidence"),
            "question_type":       row.get("question_type"),
            "question_text":       row.get("question_text"),
            "ground_truth_answer": row.get("ground_truth_answer"),
            "context":             row.get("context"),
            "choices_json":        row.get("choices_json"),
            "response_text":       row.get("response_text"),
            "response_time_sec":   row.get("response_time_sec"),
            "token_count":         row.get("token_count"),
            "error":               row.get("error"),
            "error_message":       row.get("error_message"),
            "timestamp":           row.get("timestamp"),
            # Look up model version and temperature from config
            "model_version":       config.MODELS_CONFIG.get(
                                       row.get("model_name"), {}
                                   ).get("model_id"),
            "temperature":         config.MODELS_CONFIG.get(
                                       row.get("model_name"), {}
                                   ).get("temperature"),
            # Human validation fields (filled in separately if applicable)
            "human_validated":     row.get("human_validated", False),
            "human_score":         row.get("human_score"),
            "human_agreement":     row.get("human_agreement"),
            "bloom_justification": row.get("bloom_justification"),
            "validation_agreement":row.get("validation_agreement"),
            "justification":       "",
        })

        # If the model returned an error during querying, skip scoring
        if row.get("error"):
            base["justification"] = f"Skipped — model error: {row.get('error_message', '')}"
            return base

        # --- LLM Judge Scoring ---
        judge_prompt = build_judge_prompt(
            question=row["question_text"],
            reference=row["ground_truth_answer"],
            response=row["response_text"],
            bloom_level=row.get("bloom_level", "unknown"),
            bloom_name=row.get("bloom_name", "unknown"),
            domain=row.get("domain", "unknown"),
        )
        scores = call_judge(self.judge_client, self.judge_model, judge_prompt)
        base.update(scores)

        # --- Lexical Metrics ---
        # Skip lexical scoring for MCQ since answers are single letters (not meaningful)
        if row.get("question_type") != "mcq" or not config.SKIP_LEXICAL_FOR_MCQ:
            ref = str(row.get("ground_truth_answer", ""))
            resp = str(row.get("response_text", ""))

            base["bleu"] = calculate_bleu(ref, resp)

            rouge = calculate_rouge(ref, resp)
            base["rouge1"] = rouge.get("rouge1", 0)
            base["rouge2"] = rouge.get("rouge2", 0)
            base["rougeL"] = rouge.get("rougeL", 0)

            p, r, f1 = calculate_keyword_overlap(ref, resp)
            base["keyword_precision"] = p
            base["keyword_recall"]    = r
            base["keyword_f1"]        = f1

        self.score_counter += 1
        return base

    def _save(self, append=False):
        """
        Flush the in-memory scores buffer to the output CSV.
        If append=True and the file exists, concatenate rather than overwrite.
        """
        if not self.scores:
            return

        df_new = pd.DataFrame(self.scores)
        # Keep only the defined output columns, in order
        cols = [c for c in config.OUTPUT_COLUMNS if c in df_new.columns]
        df_new = df_new[cols]

        if append and self.output_path.exists():
            df_new = pd.concat([pd.read_csv(self.output_path), df_new], ignore_index=True)

        df_new.to_csv(self.output_path, index=False)
        self.scores.clear()

    def run(self):
        """
        Evaluate all unevaluated responses, saving progress every SAVE_EVERY rows.
        """
        total = len(self.data)
        if total == 0:
            print("All responses already evaluated.")
            return

        for i, (_, row) in enumerate(self.data.iterrows(), 1):
            self.scores.append(self._evaluate_row(row))

            # Periodically flush to disk so a crash doesn't lose everything
            if len(self.scores) >= config.SAVE_EVERY:
                self._save(append=True)
                print(f"  Saved {i}/{total} scores")

        # Final flush
        if self.scores:
            self._save(append=True)

        print(f"Done. Scores saved to: {self.output_path}")
