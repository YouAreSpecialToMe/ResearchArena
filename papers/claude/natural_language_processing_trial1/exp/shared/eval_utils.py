"""Evaluation utilities for open-domain QA."""
import re
import string
from collections import Counter


def normalize_answer(s):
    """Normalize answer string (SQuAD-style)."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def extract_short_answer(text):
    """Extract a short answer from a potentially long model response."""
    text = text.strip()
    # Remove common prefixes
    for prefix in ["Answer:", "The answer is", "Based on the", "According to"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Take first sentence
    for sep in ['.', '\n', '!', ';']:
        idx = text.find(sep)
        if 0 < idx < 200:
            text = text[:idx]
            break
    # If still long, take first clause
    if len(text.split()) > 15:
        for sep in [',', ' - ', ' but ', ' however', ' although']:
            idx = text.lower().find(sep)
            if 0 < idx < 150:
                text = text[:idx]
                break
    return text.strip()


def exact_match(prediction, gold_answers):
    """Check if prediction exactly matches any gold answer."""
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(g) == norm_pred for g in gold_answers)


def substring_match(prediction, gold_answers):
    """Check if any gold answer appears as substring in prediction."""
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(g) in norm_pred for g in gold_answers)


def token_f1(prediction, gold_answers):
    """Compute max token-level F1 between prediction and gold answers."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def is_correct(prediction, gold_answers, threshold=0.5):
    """Binary correctness: EM, substring match, or token F1 >= threshold.
    Also tries with extracted short answer."""
    if exact_match(prediction, gold_answers):
        return True
    if substring_match(prediction, gold_answers):
        return True
    # Try with extracted short answer
    short_pred = extract_short_answer(prediction)
    if exact_match(short_pred, gold_answers):
        return True
    if substring_match(short_pred, gold_answers):
        return True
    if token_f1(short_pred, gold_answers) >= threshold:
        return True
    return token_f1(prediction, gold_answers) >= threshold
