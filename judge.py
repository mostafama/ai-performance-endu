"""
judge.py — LLM-as-Judge
------------------------
Handles calling GPT-4.1-mini as a judge to score LLM responses.

The judge receives a structured prompt containing:
  - The original question
  - The ground truth reference answer
  - The model's response
  - Bloom's level and domain context

It returns scores (0–10) on four dimensions:
  - Correctness:         Is the answer factually right?
  - Completeness:        Does it cover all parts of the question?
  - Clarity:             Is it clearly communicated?
  - Cognitive Alignment: Does it match the expected Bloom's level of thinking?

The judge is called with up to 3 retry attempts on failure.
"""

import re
import time
import config


def build_judge_prompt(question, reference, response, bloom_level, bloom_name, domain):
    """
    Build the structured prompt sent to the judge LLM.

    The output format section instructs the judge to respond in a
    parseable key: value format, one metric per line.
    """
    return f"""You are an expert educational evaluator. Evaluate the AI response below.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

AI RESPONSE:
{response}

EVALUATION CONTEXT:
- Domain: {domain}
- Bloom level: {bloom_level} ({bloom_name})

OUTPUT FORMAT:
CORRECTNESS: [0-10]
COMPLETENESS: [0-10]
CLARITY: [0-10]
COGNITIVE_ALIGNMENT: [0-10]
OVERALL_SCORE: [0-10]
JUSTIFICATION: [2-3 sentences]
"""



def _parse_score(value):
    """
    Extract an integer score from a value string that may have trailing
    text such as "8/10", "8 (good)", or "8.5". Takes the leading digits only.
    Returns None if no integer can be extracted.
    """
    if not value:
        return None
    match = re.match(r"(\d+)", value.strip())
    return int(match.group(1)) if match else None

def parse_judge_response(text):
    """
    Parse the judge's text output into a structured scores dict.

    Expects lines like "CORRECTNESS: 8" or "JUSTIFICATION: The response..."
    If OVERALL_SCORE is missing, it's computed as the mean of the four sub-scores.

    Returns:
        dict with keys: correctness, completeness, clarity,
                        cognitive_alignment, overall_score, justification
    """
    result = {
        "correctness": None,
        "completeness": None,
        "clarity": None,
        "cognitive_alignment": None,
        "overall_score": None,
        "justification": "",
    }

    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()

        if key == "CORRECTNESS":
            result["correctness"] = _parse_score(value)
        elif key == "COMPLETENESS":
            result["completeness"] = _parse_score(value)
        elif key == "CLARITY":
            result["clarity"] = _parse_score(value)
        elif key == "COGNITIVE_ALIGNMENT":
            result["cognitive_alignment"] = _parse_score(value)
        elif key == "OVERALL_SCORE":
            result["overall_score"] = _parse_score(value)
        elif key == "JUSTIFICATION":
            result["justification"] = value

    # If judge didn't provide an overall score, compute it from sub-scores
    if result["overall_score"] is None:
        sub = [
            v for k, v in result.items()
            if k in ["correctness", "completeness", "clarity", "cognitive_alignment"]
            and v is not None
        ]
        if sub:
            result["overall_score"] = round(sum(sub) / len(sub), 2)

    return result


def call_judge(judge_client, judge_model, prompt):
    """
    Call the judge LLM with up to 3 retry attempts on failure.

    Args:
        judge_client: initialized OpenAI client
        judge_model:  model name string (e.g. "gpt-4.1-mini")
        prompt:       the full judge prompt string

    Returns:
        dict of parsed scores (same shape as parse_judge_response output)
        On all retries failing, returns None scores with an error justification.
    """
    for attempt in range(1, 4):
        try:
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,   # Slight temperature for natural language justifications
                timeout=60,
            )
            return parse_judge_response(resp.choices[0].message.content or "")
        except Exception as exc:
            if attempt == 3:
                # All retries exhausted — return empty scores with error note
                return {
                    "correctness": None,
                    "completeness": None,
                    "clarity": None,
                    "cognitive_alignment": None,
                    "overall_score": None,
                    "justification": f"Judge error: {str(exc)[:100]}",
                }
            # Wait before retrying
            time.sleep(config.RETRY_DELAY)
