"""
prompts.py
----------
Builds the prompt string that gets sent to each LLM for a given question.

Question types and how they're handled:

  Reading + passage (context)
    Always includes the passage regardless of whether the question is MCQ or open.
    Format: Passage / Question / Choices (if MCQ) / Instruction
    Without this, a model answering a SAT-style reading MCQ has no passage to
    reference and is essentially guessing — invalid for the study.

  MCQ without a passage (science, math, etc.)
    Appends the lettered choices after the question text.
    Format: Question / Choices / Instruction

  Computer Science with starter code
    Wraps the starter code in a Python code block so the model sees it clearly.
    Format: Instruction / Problem / Starter code block

  All other questions (math open, science open, fallback)
    Plain question + domain-appropriate instruction.

The domain-specific instructions are defined in config.PROMPT_TEMPLATES.
"""

import json
import pandas as pd
import config


def build_prompt(row):
    """
    Construct the full prompt string for a single question row.

    Args:
        row: a pandas Series (or dict-like) representing one question,
             with fields: domain, question_text, context, choices_json, question_type

    Returns:
        str: the complete prompt to send to the LLM
    """
    domain        = str(row.get("domain", "")).strip().lower()
    template      = config.PROMPT_TEMPLATES.get(domain, config.FALLBACK_TEMPLATE)
    # Guard against NaN values — pandas may produce float NaN for missing cells,
    # and str(NaN) produces the string "nan" which would corrupt the prompt.
    _qt = row.get("question_text", "")
    question_text = "" if not _qt or (isinstance(_qt, float) and __import__("math").isnan(_qt)) else str(_qt).strip()
    context       = str(row.get("context", "") or "").strip()
    choices_json  = row.get("choices_json")
    question_type = str(row.get("question_type", "")).strip()

    # --- Parse MCQ choices (if present) ---
    # Do this up front so every branch below can use choices_text if needed.
    choices_text = None
    if question_type == "mcq" and pd.notna(choices_json) and choices_json:
        try:
            choices = json.loads(choices_json)
            choices_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
        except Exception:
            # Malformed JSON — fall through to plain format
            pass

    # --- Reading Comprehension (with passage) ---
    # The passage MUST be included for the model to answer correctly.
    # This applies to both MCQ and open-ended reading questions.
    # BUG FIXED: the old code checked MCQ *before* checking domain=="reading",
    # so reading MCQs got their choices but NOT their passage. Models were
    # answering SAT-style questions with no text to reference.
    if domain == "reading" and context:
        if choices_text:
            # Reading MCQ: Passage → Question → Choices → Instruction
            return (
                f"Passage:\n{context}\n\n"
                f"Question: {question_text}\n\n"
                f"Choices:\n{choices_text}\n\n"
                f"{template}"
            )
        # Reading open: Passage → Question → Instruction
        return f"Passage:\n{context}\n\nQuestion: {question_text}\n\n{template}"

    # --- Non-reading MCQ ---
    # No passage needed — just question, choices, and instruction.
    if choices_text:
        return f"{question_text}\n\nChoices:\n{choices_text}\n\n{template}"

    # --- Computer Science with Starter Code ---
    # Format starter code as a Python code block so the model sees it clearly.
    if domain == "computer_science" and context:
        return (
            f"{template}\n\n"
            f"Problem:\n{question_text}\n\n"
            f"Starter code:\n```python\n{context}\n```"
        )

    # --- Default: plain question + instruction ---
    return f"{question_text}\n\n{template}"
