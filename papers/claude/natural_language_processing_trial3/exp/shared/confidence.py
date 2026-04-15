import re
import math
import numpy as np
from .evaluation import normalize_answer


def token_probability(logprobs_list):
    """Mean log-prob of generated answer tokens."""
    if not logprobs_list:
        return -10.0  # very low confidence
    return float(np.mean(logprobs_list))


def token_entropy(logprobs_list):
    """Mean entropy across token positions (negate for confidence)."""
    # With only top-1 logprob, approximate entropy as -logprob
    if not logprobs_list:
        return 10.0
    return float(-np.mean(logprobs_list))


def verbalized_confidence(response_text):
    """Parse confidence 0-100 from response, default 50 on failure."""
    match = re.search(r'[Cc]onfidence[:\s]*(\d+)', response_text)
    if match:
        val = int(match.group(1))
        return min(max(val, 0), 100) / 100.0
    return 0.5


def self_consistency(answers_list, reference_answer):
    """Fraction of sampled answers matching reference after normalization."""
    if not answers_list:
        return 0.0
    ref_norm = normalize_answer(reference_answer)
    matches = sum(1 for a in answers_list if normalize_answer(a) == ref_norm)
    return matches / len(answers_list)


def token_prob_delta(parametric_logprob, rag_logprob):
    """Difference in mean log-prob: RAG - parametric."""
    return rag_logprob - parametric_logprob
