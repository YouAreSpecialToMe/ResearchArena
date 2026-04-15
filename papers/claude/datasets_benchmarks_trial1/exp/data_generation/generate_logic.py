"""Propositional Logic Generator for FlipBench.

Generates matched forward-backward reasoning pairs over propositional logic rules.
Both directions use True/False answers under closed-world assumption.

Forward: Given rules + fact, can we derive that target proposition is true?
Backward: Given rules + conclusion is true, must a specific proposition have been true?

- Difficulty 1: 1-hop (1 rule, 1 fact)
- Difficulty 2: 2-hop (2 chained rules)
- Difficulty 3: 3-hop (3 chained rules + distractors)
"""

import random
import string

PROP_NAMES = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A"]


def generate_chain(rng, num_hops, all_props):
    """Generate a chain of implications: p0 -> p1 -> ... -> p_{num_hops}."""
    chain_props = rng.sample(all_props, num_hops + 1)
    rules = []
    for i in range(num_hops):
        rules.append((chain_props[i], chain_props[i + 1]))
    return chain_props, rules


def generate_distractors(rng, chain_props, all_props, num_distractors):
    """Generate distractor rules that don't connect to the chain."""
    remaining = [p for p in all_props if p not in chain_props]
    distractors = []
    for _ in range(num_distractors):
        if len(remaining) < 2:
            break
        pair = rng.sample(remaining, 2)
        distractors.append((pair[0], pair[1]))
    return distractors


def format_rule(antecedent, consequent):
    return f"If {antecedent} is true, then {consequent} is true."


CWA_NOTE = "Assume only the rules listed above can make propositions true (closed-world assumption)."


def generate_pair(rng, difficulty, pair_id, all_props):
    """Generate one matched forward-backward pair for propositional logic.

    Both forward and backward are True/False questions.
    50% of pairs are 'True' cases, 50% are 'False' cases.
    """
    if difficulty == 1:
        num_hops = 1
        num_distractors = 0
    elif difficulty == 2:
        num_hops = 2
        num_distractors = 0
    else:
        num_hops = 3
        num_distractors = rng.randint(2, 3)

    chain_props, rules = generate_chain(rng, num_hops, all_props)
    distractors = generate_distractors(rng, chain_props, all_props, num_distractors)

    fact_prop = chain_props[0]
    conclusion_prop = chain_props[-1]

    all_rules = rules + distractors
    rng.shuffle(all_rules)
    rules_text = "\n".join([format_rule(a, c) for a, c in all_rules])

    # Get non-chain propositions for False cases
    non_chain = [p for p in all_props if p not in chain_props]
    # Also get distractor props
    distractor_props = set()
    for a, c in distractors:
        distractor_props.add(a)
        distractor_props.add(c)

    is_true_pair = pair_id % 2 == 0

    if is_true_pair:
        # Forward TRUE: Given fact, is conclusion derivable? → Yes
        forward_problem = (
            f"Given the following rules:\n{rules_text}\n\n"
            f"{CWA_NOTE}\n"
            f"Given that {fact_prop} is true, is {conclusion_prop} true or false?"
        )
        forward_answer = "True"

        # Backward TRUE: Given conclusion is true, must starting fact have been true? → Yes
        backward_problem = (
            f"Given the following rules:\n{rules_text}\n\n"
            f"{CWA_NOTE}\n"
            f"Given that {conclusion_prop} is true, must {fact_prop} have been initially true? "
            f"Answer True or False."
        )
        backward_answer = "True"
    else:
        # Forward FALSE: Given fact, is an unreachable proposition true? → No
        if non_chain:
            false_target_fwd = rng.choice(non_chain)
        else:
            # Fallback: use a distractor consequent (which is not derivable from fact)
            false_target_fwd = rng.choice(list(distractor_props)) if distractor_props else chain_props[-2]

        forward_problem = (
            f"Given the following rules:\n{rules_text}\n\n"
            f"{CWA_NOTE}\n"
            f"Given that {fact_prop} is true, is {false_target_fwd} true or false?"
        )
        forward_answer = "False"

        # Backward FALSE: Given conclusion is true, must an unrelated prop have been true? → No
        # Pick a prop not in the chain
        non_chain_for_bwd = [p for p in non_chain if p != false_target_fwd]
        if non_chain_for_bwd:
            false_target_bwd = rng.choice(non_chain_for_bwd)
        elif non_chain:
            false_target_bwd = non_chain[0]
        else:
            false_target_bwd = rng.choice(list(distractor_props)) if distractor_props else chain_props[1]

        backward_problem = (
            f"Given the following rules:\n{rules_text}\n\n"
            f"{CWA_NOTE}\n"
            f"Given that {conclusion_prop} is true, must {false_target_bwd} have been initially true? "
            f"Answer True or False."
        )
        backward_answer = "False"

    forward_instance = {
        "id": f"logic_d{difficulty}_fwd_{pair_id}",
        "domain": "propositional_logic",
        "difficulty": difficulty,
        "direction": "forward",
        "problem_text": forward_problem,
        "answer": forward_answer,
        "matched_pair_id": f"logic_d{difficulty}_pair_{pair_id}",
        "metadata": {
            "rules": [(a, c) for a, c in rules],
            "distractors": [(a, c) for a, c in distractors],
            "chain": chain_props,
            "fact": fact_prop,
            "conclusion": conclusion_prop,
            "is_true_case": is_true_pair
        }
    }

    backward_instance = {
        "id": f"logic_d{difficulty}_bwd_{pair_id}",
        "domain": "propositional_logic",
        "difficulty": difficulty,
        "direction": "backward",
        "problem_text": backward_problem,
        "answer": backward_answer,
        "matched_pair_id": f"logic_d{difficulty}_pair_{pair_id}",
        "metadata": {
            "rules": [(a, c) for a, c in rules],
            "distractors": [(a, c) for a, c in distractors],
            "chain": chain_props,
            "fact": fact_prop,
            "conclusion": conclusion_prop,
            "is_true_case": is_true_pair
        }
    }

    return forward_instance, backward_instance


def generate_logic_dataset(seed, num_per_difficulty=100):
    """Generate the full propositional logic dataset."""
    rng = random.Random(seed)
    all_props = PROP_NAMES.copy()
    instances = []

    for difficulty in [1, 2, 3]:
        for i in range(num_per_difficulty):
            fwd, bwd = generate_pair(rng, difficulty, i, all_props)
            instances.extend([fwd, bwd])

    return instances


if __name__ == "__main__":
    import json
    dataset = generate_logic_dataset(42)
    print(f"Generated {len(dataset)} logic instances")
    # Show balance
    fwd_true = sum(1 for d in dataset if d['direction'] == 'forward' and d['answer'] == 'True')
    fwd_false = sum(1 for d in dataset if d['direction'] == 'forward' and d['answer'] == 'False')
    bwd_true = sum(1 for d in dataset if d['direction'] == 'backward' and d['answer'] == 'True')
    bwd_false = sum(1 for d in dataset if d['direction'] == 'backward' and d['answer'] == 'False')
    print(f"Forward: {fwd_true} True, {fwd_false} False")
    print(f"Backward: {bwd_true} True, {bwd_false} False")
    print(json.dumps(dataset[0], indent=2))
    print(json.dumps(dataset[1], indent=2))
