"""Evaluation metrics for FlipBench."""

import numpy as np
from collections import defaultdict


def compute_metrics(results, dataset):
    """Compute all FlipBench metrics from evaluation results.

    Args:
        results: list of dicts with 'id', 'parsed_answer', 'correct' fields
        dataset: list of original dataset instances

    Returns:
        dict with all metrics
    """
    # Index dataset by id
    ds_by_id = {d['id']: d for d in dataset}

    # Build results lookup
    res_by_id = {r['id']: r for r in results}

    metrics = {}

    # Per-domain metrics
    domains = ['propositional_logic', 'arithmetic_reasoning',
               'relational_reasoning', 'function_computation']

    for domain in domains:
        domain_instances = [d for d in dataset if d['domain'] == domain]
        fwd = [d for d in domain_instances if d['direction'] == 'forward']
        bwd = [d for d in domain_instances if d['direction'] == 'backward']

        fwd_correct = sum(1 for d in fwd if res_by_id.get(d['id'], {}).get('correct', False))
        bwd_correct = sum(1 for d in bwd if res_by_id.get(d['id'], {}).get('correct', False))

        fa = fwd_correct / len(fwd) if fwd else 0
        ba = bwd_correct / len(bwd) if bwd else 0
        drg = fa - ba

        # Consistency rate: both directions correct for matched pairs
        pair_ids = set(d['matched_pair_id'] for d in domain_instances)
        both_correct = 0
        fwd_only = 0
        bwd_only = 0
        both_wrong = 0

        for pid in pair_ids:
            pair_insts = [d for d in domain_instances if d['matched_pair_id'] == pid]
            fwd_inst = [d for d in pair_insts if d['direction'] == 'forward']
            bwd_inst = [d for d in pair_insts if d['direction'] == 'backward']

            if not fwd_inst or not bwd_inst:
                continue

            f_correct = res_by_id.get(fwd_inst[0]['id'], {}).get('correct', False)
            b_correct = res_by_id.get(bwd_inst[0]['id'], {}).get('correct', False)

            if f_correct and b_correct:
                both_correct += 1
            elif f_correct:
                fwd_only += 1
            elif b_correct:
                bwd_only += 1
            else:
                both_wrong += 1

        total_pairs = both_correct + fwd_only + bwd_only + both_wrong
        cr = both_correct / total_pairs if total_pairs else 0

        metrics[domain] = {
            'forward_accuracy': round(fa, 4),
            'backward_accuracy': round(ba, 4),
            'drg': round(drg, 4),
            'consistency_rate': round(cr, 4),
            'n_forward': len(fwd),
            'n_backward': len(bwd),
            'n_pairs': total_pairs,
            'both_correct': both_correct,
            'fwd_only_correct': fwd_only,
            'bwd_only_correct': bwd_only,
            'both_wrong': both_wrong
        }

        # By difficulty
        for diff in [1, 2, 3]:
            diff_fwd = [d for d in fwd if d['difficulty'] == diff]
            diff_bwd = [d for d in bwd if d['difficulty'] == diff]

            diff_fwd_correct = sum(1 for d in diff_fwd
                                   if res_by_id.get(d['id'], {}).get('correct', False))
            diff_bwd_correct = sum(1 for d in diff_bwd
                                   if res_by_id.get(d['id'], {}).get('correct', False))

            diff_fa = diff_fwd_correct / len(diff_fwd) if diff_fwd else 0
            diff_ba = diff_bwd_correct / len(diff_bwd) if diff_bwd else 0

            metrics[domain][f'difficulty_{diff}'] = {
                'forward_accuracy': round(diff_fa, 4),
                'backward_accuracy': round(diff_ba, 4),
                'drg': round(diff_fa - diff_ba, 4),
                'n_forward': len(diff_fwd),
                'n_backward': len(diff_bwd)
            }

    # Overall metrics
    all_fwd = [d for d in dataset if d['direction'] == 'forward']
    all_bwd = [d for d in dataset if d['direction'] == 'backward']
    all_fwd_correct = sum(1 for d in all_fwd if res_by_id.get(d['id'], {}).get('correct', False))
    all_bwd_correct = sum(1 for d in all_bwd if res_by_id.get(d['id'], {}).get('correct', False))

    overall_fa = all_fwd_correct / len(all_fwd) if all_fwd else 0
    overall_ba = all_bwd_correct / len(all_bwd) if all_bwd else 0

    metrics['overall'] = {
        'forward_accuracy': round(overall_fa, 4),
        'backward_accuracy': round(overall_ba, 4),
        'drg': round(overall_fa - overall_ba, 4),
        'total_instances': len(dataset),
        'total_correct': all_fwd_correct + all_bwd_correct
    }

    return metrics
