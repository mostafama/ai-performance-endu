"""
metrics.py
----------
Lexical similarity metrics for comparing LLM responses to reference answers.

Three metric families are used:

1. BLEU  — measures n-gram precision between candidate and reference.
           A score of 1.0 means the response matches the reference exactly;
           0.0 means no overlap. Used widely in NLP evaluation.

2. ROUGE — recall-oriented metrics that measure how much of the reference
           is covered by the candidate. We compute ROUGE-1 (unigrams),
           ROUGE-2 (bigrams), and ROUGE-L (longest common subsequence).

3. Keyword Overlap — a simpler measure: extract the top content words
           (alphabetic tokens > 2 chars) from both texts and compute
           precision, recall, and F1 on the overlap set.

All functions return 0.0 on empty input or import errors.
"""


def calculate_bleu(reference, candidate):
    """
    Compute sentence-level BLEU score using sacrebleu.

    Returns a float in [0, 1]. Returns 0.0 if inputs are empty
    or if sacrebleu is not installed.
    """
    try:
        from sacrebleu import sentence_bleu
        if not reference or not candidate:
            return 0.0
        # sacrebleu returns scores in [0, 100], so divide by 100 to normalize
        return sentence_bleu(candidate, [reference]).score / 100.0
    except Exception:
        return 0.0


def calculate_rouge(reference, candidate):
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns a dict: {"rouge1": float, "rouge2": float, "rougeL": float}
    All values in [0, 1]. Returns zeros on empty input or import error.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        if not reference or not candidate:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        scores = scorer.score(reference, candidate)
        # Extract F1 from each metric's Precision/Recall/F1 namedtuple
        return {k: scores[k].fmeasure for k in scores}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def _extract_keywords(text):
    """
    Extract up to 20 content words from text.
    Filters to alphabetic tokens longer than 2 characters,
    preserving insertion order (deduplication via dict).
    """
    if not text:
        return []
    tokens = [
        token for token in text.lower().split()
        if token.isalnum() and len(token) > 2
    ]
    return list(dict.fromkeys(tokens))[:20]


def calculate_keyword_overlap(reference, candidate):
    """
    Compute keyword-level precision, recall, and F1.

    Extracts content keywords from both texts and measures set overlap.

    Returns:
        (precision, recall, f1) — all floats in [0, 1]
        Returns (0.0, 0.0, 0.0) if either input is empty.
    """
    ref_kw = set(_extract_keywords(reference))
    cand_kw = set(_extract_keywords(candidate))

    if not ref_kw or not cand_kw:
        return 0.0, 0.0, 0.0

    common = ref_kw.intersection(cand_kw)
    precision = len(common) / len(cand_kw)   # how much of the response is relevant
    recall = len(common) / len(ref_kw)        # how much of the reference is covered
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) else 0.0
    )
    return precision, recall, f1
