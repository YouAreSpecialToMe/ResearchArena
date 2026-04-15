"""Relational/Family Reasoning Generator for FlipBench.

Generates matched forward-backward reasoning pairs over family relationships.
Both directions ask for relationship types (no entity identification / multiple choice).

Forward: compose relationships to derive a new one
Backward: decompose a compound relationship given partial information

- Difficulty 1: single-hop (direct vs inverse relationship)
- Difficulty 2: two-hop (compose vs decompose)
- Difficulty 3: three-hop with distractor entities (compose vs decompose)
"""

import random

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
         "Quinn", "Rose", "Sam", "Tina"]

# Inverse relationships
INVERSE = {
    "parent": "child",
    "child": "parent",
    "sibling": "sibling",
    "grandparent": "grandchild",
    "grandchild": "grandparent",
    "great-grandparent": "great-grandchild",
    "great-grandchild": "great-grandparent",
}


def generate_single_hop(rng, pair_id, all_names):
    """Difficulty 1: direct relationship.

    Forward: stated relationship → answer directly
    Backward: stated relationship → derive inverse
    """
    names = rng.sample(all_names, 2)
    a, b = names[0], names[1]

    rel = rng.choice(["parent", "child", "sibling"])
    inverse = INVERSE[rel]

    facts = f"{a} is {b}'s {rel}."

    # Forward: What is A to B? (directly stated)
    fwd_text = f"Given: {facts}\nWhat is {a} to {b}?"
    fwd_answer = rel

    # Backward: What is B to A? (requires inverse reasoning)
    bwd_text = f"Given: {facts}\nWhat is {b} to {a}?"
    bwd_answer = inverse

    return fwd_text, fwd_answer, bwd_text, bwd_answer


def generate_two_hop(rng, pair_id, all_names):
    """Difficulty 2: two-hop transitive relationship.

    Forward: compose two relationships to derive compound relationship
    Backward: given compound + one hop, derive the other hop
    """
    names = rng.sample(all_names, 3)
    a, b, c = names

    # Chain types and their derived relations
    chain_options = [
        (("parent", "parent"), "grandparent"),
        (("child", "child"), "grandchild"),
    ]
    chain_type, derived = rng.choice(chain_options)

    facts_list = [
        f"{a} is {b}'s {chain_type[0]}.",
        f"{b} is {c}'s {chain_type[1]}."
    ]
    rng.shuffle(facts_list)
    facts = "\n".join(facts_list)

    # Forward: What is A to C? (compose two hops)
    fwd_text = f"Given:\n{facts}\nWhat family relationship is {a} to {c}?"
    fwd_answer = derived

    # Backward: Given the compound relationship and one hop, find the other.
    # We tell: A is C's derived, and A is B's chain_type[0]. What is B to C?
    bwd_text = (
        f"Given:\n"
        f"{a} is {c}'s {derived}.\n"
        f"{a} is {b}'s {chain_type[0]}.\n"
        f"What family relationship is {b} to {c}?"
    )
    bwd_answer = chain_type[1]

    return fwd_text, fwd_answer, bwd_text, bwd_answer


def generate_three_hop(rng, pair_id, all_names):
    """Difficulty 3: three-hop with distractors.

    Forward: compose three hops to derive compound relationship
    Backward: given compound + two hops, derive the missing one
    """
    names = rng.sample(all_names, rng.randint(5, 7))
    a, b, c, d = names[0], names[1], names[2], names[3]
    distractors = names[4:]

    # Vary the chain type
    chain_options = [
        (["parent", "parent", "parent"], "great-grandparent"),
        (["child", "child", "child"], "great-grandchild"),
    ]
    hops, derived = rng.choice(chain_options)

    facts_list = [
        f"{a} is {b}'s {hops[0]}.",
        f"{b} is {c}'s {hops[1]}.",
        f"{c} is {d}'s {hops[2]}."
    ]

    # Add distractor relationships
    distractor_facts = []
    for dist in distractors:
        rel_target = rng.choice([a, b, c, d])
        rel_type = rng.choice(["friend", "neighbor", "colleague"])
        distractor_facts.append(f"{dist} is {rel_target}'s {rel_type}.")

    all_facts = facts_list + distractor_facts
    rng.shuffle(all_facts)
    facts = "\n".join(all_facts)

    # Forward: What is A to D? (compose three hops)
    fwd_text = (
        f"Given:\n{facts}\n\n"
        f"What family relationship is {a} to {d}?\n"
        f"(Consider only family relationships like parent, child, grandparent, "
        f"great-grandparent, sibling, etc.)"
    )
    fwd_answer = derived

    # Backward: Given compound + two of three hops, derive the missing hop.
    # Give: A is D's derived, A is B's hop[0], C is D's hop[2].
    # Missing: B is C's hop[1].
    bwd_facts = [
        f"{a} is {d}'s {derived}.",
        f"{a} is {b}'s {hops[0]}.",
        f"{c} is {d}'s {hops[2]}."
    ]
    # Add distractors to backward too
    bwd_all = bwd_facts + distractor_facts
    rng.shuffle(bwd_all)
    bwd_facts_text = "\n".join(bwd_all)

    bwd_text = (
        f"Given:\n{bwd_facts_text}\n\n"
        f"What family relationship is {b} to {c}?\n"
        f"(Consider only family relationships like parent, child, grandparent, "
        f"great-grandparent, sibling, etc.)"
    )
    bwd_answer = hops[1]

    return fwd_text, fwd_answer, bwd_text, bwd_answer


def generate_pair(rng, difficulty, pair_id, all_names):
    if difficulty == 1:
        fwd_text, fwd_ans, bwd_text, bwd_ans = generate_single_hop(rng, pair_id, all_names)
    elif difficulty == 2:
        fwd_text, fwd_ans, bwd_text, bwd_ans = generate_two_hop(rng, pair_id, all_names)
    else:
        fwd_text, fwd_ans, bwd_text, bwd_ans = generate_three_hop(rng, pair_id, all_names)

    forward_instance = {
        "id": f"rel_d{difficulty}_fwd_{pair_id}",
        "domain": "relational_reasoning",
        "difficulty": difficulty,
        "direction": "forward",
        "problem_text": fwd_text,
        "answer": fwd_ans,
        "matched_pair_id": f"rel_d{difficulty}_pair_{pair_id}",
    }

    backward_instance = {
        "id": f"rel_d{difficulty}_bwd_{pair_id}",
        "domain": "relational_reasoning",
        "difficulty": difficulty,
        "direction": "backward",
        "problem_text": bwd_text,
        "answer": bwd_ans,
        "matched_pair_id": f"rel_d{difficulty}_pair_{pair_id}",
    }

    return forward_instance, backward_instance


def generate_relational_dataset(seed, num_per_difficulty=100):
    rng = random.Random(seed)
    instances = []
    for difficulty in [1, 2, 3]:
        for i in range(num_per_difficulty):
            fwd, bwd = generate_pair(rng, difficulty, i, NAMES)
            instances.extend([fwd, bwd])
    return instances


if __name__ == "__main__":
    import json
    dataset = generate_relational_dataset(42)
    print(f"Generated {len(dataset)} relational instances")
    print(json.dumps(dataset[0], indent=2))
    print(json.dumps(dataset[1], indent=2))
    # Check answer distribution
    from collections import Counter
    fwd = [d['answer'] for d in dataset if d['direction'] == 'forward']
    bwd = [d['answer'] for d in dataset if d['direction'] == 'backward']
    print(f"\nForward answers: {Counter(fwd)}")
    print(f"Backward answers: {Counter(bwd)}")
