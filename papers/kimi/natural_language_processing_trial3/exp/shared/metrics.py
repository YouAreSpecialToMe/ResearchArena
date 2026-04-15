"""Evaluation metrics for ESR experiments."""

import re
import numpy as np
from typing import List, Dict, Any, Tuple


def normalize_answer(answer: Any) -> float:
    """Normalize answer to float for comparison."""
    if answer is None:
        return None
    try:
        if isinstance(answer, str):
            # Remove common formatting
            answer = answer.strip()
            answer = answer.replace(",", "")
            answer = answer.replace("$", "")
            answer = answer.replace("%", "")
            # Check for fraction
            if "/" in answer:
                parts = answer.split("/")
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            return float(answer)
        return float(answer)
    except:
        return None


def compare_answers(pred: Any, gold: Any, tolerance: float = 1e-3) -> bool:
    """Compare two answers with tolerance."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm is None or gold_norm is None:
        # Fallback to string comparison
        return str(pred).strip() == str(gold).strip()
    
    # Compare with tolerance
    return abs(pred_norm - gold_norm) < tolerance


def extract_answer_from_text(text: str) -> Any:
    """Extract answer from generated text."""
    if not text:
        return None
    
    # Look for #### pattern
    if "####" in text:
        match = re.search(r"####\s*([-\d.,]+)", text)
        if match:
            return match.group(1)
    
    # Look for boxed answer
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for "answer is" pattern
    answer_match = re.search(r"(?:final answer|answer is|answer:)\s*([-\d.,]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    
    # Look for last number in text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    
    return text.strip()


def compute_accuracy(predictions: List[Any], references: List[Any]) -> float:
    """Compute accuracy."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    correct = sum(1 for p, r in zip(predictions, references) if compare_answers(p, r))
    return correct / len(predictions) if predictions else 0.0


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean and std of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }


def categorize_revision_outcome(
    initial_correct: bool,
    final_correct: bool,
    revision_triggered: bool
) -> str:
    """Categorize revision outcome."""
    if revision_triggered:
        if initial_correct and final_correct:
            return "true_negative"  # No revision needed, none performed effectively
        elif not initial_correct and final_correct:
            return "true_positive"  # Successful correction
        elif initial_correct and not final_correct:
            return "false_positive"  # Harm - correct became wrong
        else:
            return "false_negative"  # Revision attempted but failed
    else:
        if initial_correct:
            return "true_negative"  # Correct, no revision
        else:
            return "false_negative"  # Wrong, no revision triggered


def compute_revision_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute revision-related metrics."""
    outcomes = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0
    }
    
    for r in results:
        cat = categorize_revision_outcome(
            r.get("initial_correct", False),
            r.get("final_correct", False),
            r.get("revision_triggered", False)
        )
        outcomes[cat] += 1
    
    total = sum(outcomes.values())
    if total == 0:
        return {
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "harm_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "outcomes": outcomes
        }
    
    # Harm rate = false positives / total
    harm_rate = outcomes["false_positive"] / total
    
    # Precision = TP / (TP + FP)
    tp_fp = outcomes["true_positive"] + outcomes["false_positive"]
    precision = outcomes["true_positive"] / tp_fp if tp_fp > 0 else 0.0
    
    # Recall = TP / (TP + FN)  where FN includes both failed revisions and no revision when needed
    tp_fn = outcomes["true_positive"] + outcomes["false_negative"]
    recall = outcomes["true_positive"] / tp_fn if tp_fn > 0 else 0.0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positive_rate": outcomes["true_positive"] / total,
        "false_positive_rate": outcomes["false_positive"] / total,
        "harm_rate": harm_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "outcomes": outcomes,
        "correction_rate": outcomes["true_positive"] / (outcomes["true_positive"] + outcomes["false_negative"])
                          if (outcomes["true_positive"] + outcomes["false_negative"]) > 0 else 0.0
    }


def aggregate_results_by_seed(results_by_seed: List[List[Dict]]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    # Flatten all results
    all_results = []
    for seed_results in results_by_seed:
        all_results.extend(seed_results)
    
    # Compute accuracy per seed
    accuracies = []
    for seed_results in results_by_seed:
        if seed_results:
            preds = [r.get("final_answer") for r in seed_results]
            refs = [r.get("reference_answer") for r in seed_results]
            acc = compute_accuracy(preds, refs)
            accuracies.append(acc)
    
    # Compute token statistics
    token_counts = [r.get("total_tokens", 0) for r in all_results]
    
    # Compute revision statistics
    revision_triggered = sum(1 for r in all_results if r.get("revision_triggered", False))
    revision_count = [r.get("revision_count", 0) for r in all_results]
    
    return {
        "accuracy": compute_statistics(accuracies),
        "tokens_per_problem": compute_statistics(token_counts),
        "revision_rate": revision_triggered / len(all_results) if all_results else 0.0,
        "revisions_per_problem": compute_statistics(revision_count),
        "total_problems": len(all_results),
        "num_seeds": len(results_by_seed)
    }
