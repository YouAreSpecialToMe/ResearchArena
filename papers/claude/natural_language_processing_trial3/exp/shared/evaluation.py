import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text, remove articles, punctuation, extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(pred, gold_answers):
    """Check if any normalized gold answer is contained in normalized prediction."""
    pred_norm = normalize_answer(pred)
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in pred_norm:
            return True
    return False


def token_f1(pred, gold_answers):
    """Compute max token F1 across gold answers."""
    pred_tokens = normalize_answer(pred).split()
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1
