"""
ConsistBench metrics: CFA, CPA, FCAG, CAR, PFI, DCS.
All metrics operate on per-question agreement data.
"""
import numpy as np
from scipy import stats


def cross_format_agreement(format_answers: dict, correct_answers: dict, equivalence_fn=None):
    """
    Compute Cross-Format Agreement (CFA) for a single question.

    Args:
        format_answers: dict mapping format_type -> extracted_answer
        correct_answers: dict mapping format_type -> ground_truth
        equivalence_fn: function(a, b, question) -> bool for answer equivalence

    Returns:
        cfa: float, pairwise agreement rate across formats
        pairwise: dict of (fmt1, fmt2) -> bool agreement
    """
    formats = sorted(format_answers.keys())
    pairs = []
    pairwise = {}

    for i in range(len(formats)):
        for j in range(i + 1, len(formats)):
            f1, f2 = formats[i], formats[j]
            a1 = format_answers[f1]
            a2 = format_answers[f2]

            # Check if both answers are correct (format-aware)
            c1 = correct_answers[f1]
            c2 = correct_answers[f2]

            # Two answers are "consistent" if they both lead to the same
            # correctness verdict (both correct or both incorrect in a
            # compatible way)
            if equivalence_fn:
                correct_1 = equivalence_fn(a1, c1)
                correct_2 = equivalence_fn(a2, c2)
                agree = (correct_1 == correct_2)
            else:
                agree = (normalize_answer(a1) == normalize_answer(a2))

            pairs.append(agree)
            pairwise[(f1, f2)] = agree

    cfa = np.mean(pairs) if pairs else 0.0
    return cfa, pairwise


def cross_phrasing_agreement(original_answer: str, paraphrase_answers: dict, equivalence_fn=None):
    """
    Compute Cross-Phrasing Agreement (CPA) per paraphrase type.

    Args:
        original_answer: answer to the original question
        paraphrase_answers: dict mapping paraphrase_type -> answer
        equivalence_fn: function(a, b) -> bool

    Returns:
        cpa_per_type: dict mapping paraphrase_type -> bool (agrees with original)
    """
    cpa = {}
    for ptype, answer in paraphrase_answers.items():
        if equivalence_fn:
            cpa[ptype] = equivalence_fn(original_answer, answer)
        else:
            cpa[ptype] = (normalize_answer(original_answer) == normalize_answer(answer))
    return cpa


def paraphrase_fragility_index(agreements_per_type: dict):
    """
    Compute PFI per paraphrase type.
    PFI = fraction of questions where answer changes (1 - agreement rate).

    Args:
        agreements_per_type: dict mapping ptype -> list of bool (per question)

    Returns:
        pfi: dict mapping ptype -> float
    """
    pfi = {}
    for ptype, agreements in agreements_per_type.items():
        pfi[ptype] = 1.0 - np.mean(agreements)
    return pfi


def format_conditional_accuracy_gap(accuracies_per_format: dict):
    """
    FCAG = accuracy of best format - accuracy of worst format.

    Args:
        accuracies_per_format: dict mapping format -> float accuracy

    Returns:
        fcag: float
        best_format: str
        worst_format: str
    """
    best_fmt = max(accuracies_per_format, key=accuracies_per_format.get)
    worst_fmt = min(accuracies_per_format, key=accuracies_per_format.get)
    fcag = accuracies_per_format[best_fmt] - accuracies_per_format[worst_fmt]
    return fcag, best_fmt, worst_fmt


def consistency_accuracy_ratio(cfa: float, accuracy: float):
    """CAR = CFA / accuracy. Measures how consistent a model is relative to its accuracy."""
    if accuracy == 0:
        return 0.0
    return cfa / accuracy


def domain_consistency_score(cfa_per_domain: dict, cpa_per_domain: dict = None):
    """
    DCS = mean(CFA, CPA) per domain. If CPA not available, DCS = CFA.
    """
    dcs = {}
    for domain in cfa_per_domain:
        if cpa_per_domain and domain in cpa_per_domain:
            dcs[domain] = (cfa_per_domain[domain] + cpa_per_domain[domain]) / 2
        else:
            dcs[domain] = cfa_per_domain[domain]
    return dcs


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    lower = means[int((1 - ci) / 2 * n_bootstrap)]
    upper = means[int((1 + ci) / 2 * n_bootstrap)]
    return float(np.mean(values)), lower, upper


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove trailing punctuation
    answer = answer.rstrip('.,;:!?')
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer
