"""Function Computation Generator for FlipBench.

Generates matched forward-backward function computation pairs.
- Difficulty 1: f(x) = a*x + b (linear)
- Difficulty 2: f(x) = a*(x+b) + c (two composed operations)
- Difficulty 3: f(x) = a*(b*x + c) + d (three composed operations)

All functions are bijective on the integer domain to ensure unique inverses.
"""

import random


def generate_linear(rng):
    """Difficulty 1: f(x) = a*x + b."""
    a = rng.choice([2, 3, 4, 5, -2, -3])
    b = rng.randint(-20, 20)
    x = rng.randint(1, 30)
    result = a * x + b

    func_desc = f"f(x) = {a}*x + {b}" if b >= 0 else f"f(x) = {a}*x - {abs(b)}"

    fwd_text = f"Given the function {func_desc}, compute f({x})."
    bwd_text = f"Given the function {func_desc}, find x such that f(x) = {result}."

    return str(result), str(x), fwd_text, bwd_text, func_desc


def generate_two_composed(rng):
    """Difficulty 2: f(x) = a*(x+b) + c."""
    a = rng.choice([2, 3, 4, 5])
    b = rng.randint(-10, 10)
    c = rng.randint(-15, 15)
    x = rng.randint(1, 20)

    result = a * (x + b) + c

    func_desc = f"f(x) = {a} * (x + {b}) + {c}"

    fwd_text = f"Given the function {func_desc}, compute f({x})."
    bwd_text = f"Given the function {func_desc}, find the integer x such that f(x) = {result}."

    return str(result), str(x), fwd_text, bwd_text, func_desc


def generate_three_composed(rng):
    """Difficulty 3: f(x) = a*(b*x + c) + d."""
    a = rng.choice([2, 3])
    b = rng.choice([2, 3, 4])
    c = rng.randint(-5, 5)
    d = rng.randint(-10, 10)
    x = rng.randint(1, 15)

    result = a * (b * x + c) + d

    func_desc = f"f(x) = {a} * ({b}*x + {c}) + {d}"

    fwd_text = f"Given the function {func_desc}, compute f({x})."
    bwd_text = f"Given the function {func_desc}, find the integer x such that f(x) = {result}."

    return str(result), str(x), fwd_text, bwd_text, func_desc


def generate_pair(rng, difficulty, pair_id):
    if difficulty == 1:
        fwd_ans, bwd_ans, fwd_text, bwd_text, func_desc = generate_linear(rng)
    elif difficulty == 2:
        fwd_ans, bwd_ans, fwd_text, bwd_text, func_desc = generate_two_composed(rng)
    else:
        fwd_ans, bwd_ans, fwd_text, bwd_text, func_desc = generate_three_composed(rng)

    forward_instance = {
        "id": f"func_d{difficulty}_fwd_{pair_id}",
        "domain": "function_computation",
        "difficulty": difficulty,
        "direction": "forward",
        "problem_text": fwd_text,
        "answer": fwd_ans,
        "matched_pair_id": f"func_d{difficulty}_pair_{pair_id}",
    }

    backward_instance = {
        "id": f"func_d{difficulty}_bwd_{pair_id}",
        "domain": "function_computation",
        "difficulty": difficulty,
        "direction": "backward",
        "problem_text": bwd_text,
        "answer": bwd_ans,
        "matched_pair_id": f"func_d{difficulty}_pair_{pair_id}",
    }

    return forward_instance, backward_instance


def generate_function_dataset(seed, num_per_difficulty=100):
    rng = random.Random(seed)
    instances = []
    for difficulty in [1, 2, 3]:
        for i in range(num_per_difficulty):
            fwd, bwd = generate_pair(rng, difficulty, i)
            instances.extend([fwd, bwd])
    return instances


if __name__ == "__main__":
    import json
    dataset = generate_function_dataset(42)
    print(f"Generated {len(dataset)} function instances")
    print(json.dumps(dataset[0], indent=2))
    print(json.dumps(dataset[1], indent=2))
