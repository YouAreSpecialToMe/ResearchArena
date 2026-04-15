"""Arithmetic Reasoning Generator for FlipBench.

Generates matched forward-backward arithmetic word problems.
- Difficulty 1: single operation
- Difficulty 2: two chained operations
- Difficulty 3: three chained operations with mixed operators
"""

import random

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul"]

ITEMS = ["apples", "books", "cookies", "dollars", "eggs", "flowers",
         "grapes", "hats", "items", "jars", "keys", "lemons"]


def generate_single_op(rng):
    """Difficulty 1: single operation."""
    op = rng.choice(['+', '-', '*'])
    if op == '+':
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        result = a + b
        op_word = "receives"
        fwd_text = (f"{rng.choice(NAMES)} has {a} {rng.choice(ITEMS)} and "
                    f"{op_word} {b} more. How many does he/she have now?")
        bwd_text = (f"{rng.choice(NAMES)} had some {rng.choice(ITEMS)} and "
                    f"received {b} more, ending up with {result}. "
                    f"How many did he/she start with?")
        bwd_answer = a
    elif op == '-':
        a = rng.randint(20, 80)
        b = rng.randint(1, a - 1)
        result = a - b
        fwd_text = (f"{rng.choice(NAMES)} has {a} {rng.choice(ITEMS)} and "
                    f"gives away {b}. How many remain?")
        bwd_text = (f"{rng.choice(NAMES)} had some {rng.choice(ITEMS)}, "
                    f"gave away {b}, and had {result} left. "
                    f"How many did he/she start with?")
        bwd_answer = a
    else:  # *
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        result = a * b
        item = rng.choice(ITEMS)
        fwd_text = (f"{rng.choice(NAMES)} buys {a} boxes of {item}, "
                    f"each containing {b} pieces. How many total pieces?")
        bwd_text = (f"{rng.choice(NAMES)} bought some boxes of {item}, "
                    f"each containing {b} pieces, for a total of {result} pieces. "
                    f"How many boxes were bought?")
        bwd_answer = a

    return result, bwd_answer, fwd_text, bwd_text


def generate_two_ops(rng):
    """Difficulty 2: two chained operations."""
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)

    a = rng.randint(5, 30)
    b = rng.randint(1, 15)
    c = rng.randint(2, 5)

    after_add = a + b
    result = after_add * c

    fwd_text = (f"{name} starts with {a} {item}, receives {b} more, "
                f"then triples each batch into {c} groups of equal size. "
                f"Wait - actually, the total is multiplied by {c}. "
                f"How many {item} are there in the end?")
    # Cleaner version
    fwd_text = (f"{name} starts with {a} {item}, receives {b} more, "
                f"then the total is multiplied by {c}. "
                f"How many {item} are there in the end?")

    bwd_text = (f"{name} started with some {item}, received {b} more, "
                f"then the total was multiplied by {c}, resulting in {result} {item}. "
                f"How many {item} did {name} start with?")
    bwd_answer = a

    return result, bwd_answer, fwd_text, bwd_text


def generate_three_ops(rng):
    """Difficulty 3: three chained operations."""
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)

    a = rng.randint(5, 20)
    b = rng.randint(2, 8)
    c = rng.randint(1, 10)
    d = rng.randint(2, 4)

    step1 = a * b
    step2 = step1 + c
    result = step2 * d

    fwd_text = (f"{name} has {a} bags of {item} with {b} in each bag. "
                f"Then {name} finds {c} more loose {item}. "
                f"Finally, everything is packed into {d} equal shipments "
                f"(multiply total by {d}). How many {item} total?")

    bwd_text = (f"{name} had some bags of {item} with {b} in each bag, "
                f"found {c} more loose {item}, then packed everything "
                f"into {d} equal shipments (multiplied total by {d}), "
                f"ending up with {result} {item}. "
                f"How many bags did {name} originally have?")
    bwd_answer = a

    return result, bwd_answer, fwd_text, bwd_text


def generate_pair(rng, difficulty, pair_id):
    if difficulty == 1:
        result, bwd_answer, fwd_text, bwd_text = generate_single_op(rng)
    elif difficulty == 2:
        result, bwd_answer, fwd_text, bwd_text = generate_two_ops(rng)
    else:
        result, bwd_answer, fwd_text, bwd_text = generate_three_ops(rng)

    forward_instance = {
        "id": f"arith_d{difficulty}_fwd_{pair_id}",
        "domain": "arithmetic_reasoning",
        "difficulty": difficulty,
        "direction": "forward",
        "problem_text": fwd_text,
        "answer": str(result),
        "matched_pair_id": f"arith_d{difficulty}_pair_{pair_id}",
    }

    backward_instance = {
        "id": f"arith_d{difficulty}_bwd_{pair_id}",
        "domain": "arithmetic_reasoning",
        "difficulty": difficulty,
        "direction": "backward",
        "problem_text": bwd_text,
        "answer": str(bwd_answer),
        "matched_pair_id": f"arith_d{difficulty}_pair_{pair_id}",
    }

    return forward_instance, backward_instance


def generate_arithmetic_dataset(seed, num_per_difficulty=100):
    rng = random.Random(seed)
    instances = []
    for difficulty in [1, 2, 3]:
        for i in range(num_per_difficulty):
            fwd, bwd = generate_pair(rng, difficulty, i)
            instances.extend([fwd, bwd])
    return instances


if __name__ == "__main__":
    import json
    dataset = generate_arithmetic_dataset(42)
    print(f"Generated {len(dataset)} arithmetic instances")
    print(json.dumps(dataset[0], indent=2))
    print(json.dumps(dataset[1], indent=2))
