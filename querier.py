"""
querier.py
--------------------
Sends benchmark questions to LLMs and saves their raw responses.

The LLMQuerier class manages the full querying loop:
  1. Build a task list of (question, model) pairs to process
  2. Skip any pairs already in the output CSV (supports resuming interrupted runs)
  3. For each pair, build the prompt and call the model API
  4. Save responses in batches so progress isn't lost on a crash
  5. Disable a model mid-run if it hits a rate limit or quota error

Output columns per row:
  response_id, question_id, model_name, response_text,
  response_time_sec, token_count, error, error_message, timestamp
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from clients import init_clients
from prompts import build_prompt
from model_api import query_model


class LLMQuerier:
    def __init__(self, models, questions_df, output_csv):
        """
        Args:
            models:       list of model name strings (keys in config.MODELS_CONFIG)
            questions_df: DataFrame of questions from data_loader.load_questions()
            output_csv:   path to write (or append) responses CSV
        """
        self.models = models
        self.questions_df = questions_df
        self.output_path = Path(output_csv)

        # Initialize only the API clients we actually need
        self.clients = init_clients(models)

        # Only keep models whose API client was successfully initialized
        self.active_models = [
            m for m in models
            if config.MODELS_CONFIG[m]["api"] in self.clients
        ]

        if not self.active_models:
            missing = [
                m for m in models
                if config.MODELS_CONFIG[m]["api"] not in self.clients
            ]
            apis_needed = {config.MODELS_CONFIG[m]["api"] for m in missing}
            key_map = {
                "openai": "OPENAI_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "groq": "GROQ_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            keys_needed = [key_map[a] for a in apis_needed if a in key_map]
            raise RuntimeError(
                f"No active models available. "
                f"Check that these keys are set in your .env file: {keys_needed}"
            )

        # Tracks which models hit rate limits so we can skip them mid-run
        self.model_disabled = {}

        # In-memory buffer before flushing to CSV
        self.responses = []

        # Counter used to generate unique response_ids like "resp_000001"
        self.response_counter = 0

        # Build the full list of (question_row, model_name) tasks to process
        self.tasks = self._build_task_list()

    def _build_task_list(self):
        """
        Build the list of (row, model) pairs that still need to be processed.
        If the output CSV already exists, skip any (question_id, model_name)
        pairs that appear in it — this enables resuming interrupted runs.
        """
        done_set = set()
        if self.output_path.exists():
            done_df = pd.read_csv(self.output_path)
            done_set = set(zip(done_df["question_id"], done_df["model_name"]))
            # Start the counter from however many responses already exist
            self.response_counter = len(done_df)

        return [
            (row, model)
            for _, row in self.questions_df.iterrows()
            for model in self.active_models
            if (row["question_id"], model) not in done_set
        ]

    def _process(self, idx, row, model_name):
        """
        Query a single (question, model) pair and buffer the result.
        If the model is already disabled (rate limit), skips silently.
        """
        if self.model_disabled.get(model_name):
            return

        prompt = build_prompt(row)
        text, elapsed, error, error_msg = query_model(model_name, prompt, self.clients)

        # Disable this model for the rest of the run if we hit a quota/rate-limit error
        if error and any(
            k in error_msg.lower()
            for k in ("rate_limit", "quota", "429", "credit")
        ):
            print(f"  [{model_name}] disabled due to: {error_msg[:80]}")
            self.model_disabled[model_name] = True

        self.response_counter += 1
        self.responses.append({
            "response_id": f"resp_{self.response_counter:06d}",
            "question_id": row["question_id"],
            "model_name": model_name,
            "response_text": text if not error else "",
            "response_time_sec": round(elapsed, 3),
            # Rough token count — word count of the response
            "token_count": len(text.split()) if not error else 0,
            "error": error,
            "error_message": error_msg if error else "",
            "timestamp": datetime.now().isoformat(),
        })

    def _save(self, append=False):
        """
        Flush the in-memory response buffer to the output CSV.
        If append=True and the file exists, concatenate rather than overwrite.
        """
        if not self.responses:
            return

        df_new = pd.DataFrame(self.responses)
        cols = [
            "response_id", "question_id", "model_name", "response_text",
            "response_time_sec", "token_count", "error", "error_message", "timestamp",
        ]
        df_new = df_new[cols]
        # Preserve empty strings — pandas reads blank CSV cells back as NaN
        df_new["response_text"] = df_new["response_text"].fillna("")
        df_new["error_message"] = df_new["error_message"].fillna("")

        if append and self.output_path.exists():
            df_new = pd.concat([pd.read_csv(self.output_path), df_new], ignore_index=True)

        df_new.to_csv(self.output_path, index=False)
        self.responses.clear()

    def run(self):
        """
        Execute all querying tasks, saving progress every SAVE_BATCH_SIZE rows.
        """
        if not self.tasks:
            print("All tasks already completed.")
            return

        n_models    = len(self.active_models)
        n_questions = len(self.questions_df)
        print(f"Querying {n_models} model(s) across {n_questions} question(s) "
              f"— {len(self.tasks)} task(s) remaining...")

        for idx, (row, model_name) in enumerate(self.tasks, 1):
            self._process(idx, row, model_name)

            # Periodically flush to disk so a crash doesn't lose everything
            if idx % config.SAVE_BATCH_SIZE == 0:
                self._save(append=True)
                print(f"  Saved {idx}/{len(self.tasks)} responses")

        # Final flush for any remaining buffered responses
        if self.responses:
            self._save(append=True)

        print(f"Done. Responses saved to: {self.output_path}")
