"""
SkillStack Composition Engine

Combines atomic skills into pairwise and triple compositions.
Each composition creates a question that genuinely requires both/all skills.
Two composition styles:
  - Sequential: output of skill A feeds into skill B
  - Parallel: both skills needed on different parts of the same problem
"""
import random
from itertools import combinations
from .skills import (
    SKILL_GENERATORS, SKILL_CODES, SKILL_NAMES,
    _make_fictional_names, _random_word, _make_set, _set_str, _ordinal
)


def compose_pairwise(skill_a: str, skill_b: str, rng: random.Random, difficulty="medium"):
    """Generate a pairwise composition question requiring both skills."""
    key = tuple(sorted([skill_a, skill_b]))
    if key in PAIRWISE_COMPOSERS:
        composers = PAIRWISE_COMPOSERS[key]
        fn = rng.choice(composers)
        return fn(rng, difficulty, skill_a, skill_b)
    else:
        return _generic_pairwise(rng, difficulty, skill_a, skill_b)


def compose_triple(skill_a: str, skill_b: str, skill_c: str, rng: random.Random, difficulty="medium"):
    """Generate a triple composition by chaining two pairwise compositions."""
    key = tuple(sorted([skill_a, skill_b, skill_c]))
    # Strategy: compose (a,b) then feed into c, or parallel (a,b) + c
    skills = [skill_a, skill_b, skill_c]
    rng.shuffle(skills)
    s1, s2, s3 = skills

    # Generate a question requiring all three skills
    return _generic_triple(rng, difficulty, s1, s2, s3)


# ============================================================
# Specific pairwise composers (for common/important pairs)
# ============================================================

def _ar_co(rng, difficulty, a, b):
    """Arithmetic + Comparison: compute two expressions, compare results."""
    lo, hi = (5, 30) if difficulty == "medium" else (20, 80)
    a1, a2, a3, a4 = [rng.randint(lo, hi) for _ in range(4)]
    val1 = a1 * a2
    val2 = a3 + a4
    q = f"Which is larger: {a1} × {a2} or {a3} + {a4}? Give the numerical value of the larger one."
    ans = str(max(val1, val2))
    return q, ans, "integer", {"template": "ar_co_compute_compare", "dependency": "sequential"}


def _ar_cn(rng, difficulty, a, b):
    """Arithmetic + Counting: count items matching a computed threshold."""
    lo, hi = (5, 25) if difficulty == "medium" else (15, 50)
    nums = [rng.randint(1, 50) for _ in range(rng.randint(8, 12))]
    a1, a2 = rng.randint(lo, hi), rng.randint(2, 8)
    threshold = a1 + a2
    q = (f"Consider the list [{', '.join(map(str, nums))}]. "
         f"First compute {a1} + {a2}. Then count how many numbers in the list are greater than that sum.")
    ans = str(sum(1 for x in nums if x > threshold))
    return q, ans, "integer", {"template": "ar_cn_threshold", "dependency": "sequential"}


def _ar_se(rng, difficulty, a, b):
    """Arithmetic + Set Operations: compute set operation, then sum."""
    s1 = sorted(rng.sample(range(1, 20), rng.randint(4, 7)))
    s2 = sorted(rng.sample(range(1, 20), rng.randint(4, 7)))
    op = rng.choice(["union", "intersection"])
    if op == "union":
        result = sorted(set(s1) | set(s2))
        q = f"Find the union of {_set_str(s1)} and {_set_str(s2)}, then compute the sum of all elements in the result."
    else:
        result = sorted(set(s1) & set(s2))
        q = f"Find the intersection of {_set_str(s1)} and {_set_str(s2)}, then compute the sum of all elements in the result."
    ans = str(sum(result))
    return q, ans, "integer", {"template": "ar_se_sum", "dependency": "sequential"}


def _co_cn(rng, difficulty, a, b):
    """Comparison + Counting: compare items and count which group is larger."""
    names = ["Alice", "Bob", "Charlie", "Diana"][:rng.randint(3, 4)]
    scores = {n: [rng.randint(40, 100) for _ in range(3)] for n in names}
    avgs = {n: sum(s) / len(s) for n, s in scores.items()}
    threshold = rng.randint(55, 80)
    desc = "; ".join([f"{n} scored {', '.join(map(str, s))}" for n, s in scores.items()])
    q = (f"Students took 3 tests: {desc}. "
         f"How many students have an average score above {threshold}?")
    ans = str(sum(1 for avg in avgs.values() if avg > threshold))
    return q, ans, "integer", {"template": "co_cn_avg_count", "dependency": "parallel"}


def _cn_se(rng, difficulty, a, b):
    """Counting + Set Operations: count elements in a set operation result."""
    s1 = sorted(rng.sample(range(1, 25), rng.randint(5, 8)))
    s2 = sorted(rng.sample(range(1, 25), rng.randint(5, 8)))
    op = rng.choice(["union", "intersection", "difference"])
    if op == "union":
        result = set(s1) | set(s2)
        op_str = "union"
    elif op == "intersection":
        result = set(s1) & set(s2)
        op_str = "intersection"
    else:
        result = set(s1) - set(s2)
        op_str = f"{_set_str(s1)} minus {_set_str(s2)}"
    threshold = rng.randint(5, 15)
    above = sum(1 for x in result if x > threshold)
    if op == "difference":
        q = f"Compute the set difference {op_str}. How many elements in the result are greater than {threshold}?"
    else:
        q = f"Compute the {op_str} of {_set_str(s1)} and {_set_str(s2)}. How many elements in the result are greater than {threshold}?"
    return q, str(above), "integer", {"template": "cn_se_count_set", "dependency": "sequential"}


def _ld_co(rng, difficulty, a, b):
    """Logical Deduction + Comparison: deduce values then compare."""
    names = _make_fictional_names(rng, 3)
    base = rng.randint(10, 30)
    offset1 = rng.randint(1, 10)
    offset2 = rng.randint(1, 10)
    q = (f"{names[0]} has {base} coins. {names[1]} has {offset1} more coins than {names[0]}. "
         f"{names[2]} has {offset2} fewer coins than {names[1]}. "
         f"Who has the most coins?")
    vals = {names[0]: base, names[1]: base + offset1, names[2]: base + offset1 - offset2}
    ans = max(vals, key=vals.get)
    return q, ans, "name", {"template": "ld_co_deduce_compare", "dependency": "sequential"}


def _te_ar(rng, difficulty, a, b):
    """Temporal + Arithmetic: compute time differences, do arithmetic on them."""
    meetings = []
    for i in range(3):
        sh = rng.randint(8, 16)
        sm = rng.randint(0, 3) * 15
        dur = rng.randint(15, 90)
        meetings.append((sh, sm, dur))
    total = sum(d for _, _, d in meetings)
    descs = []
    for i, (sh, sm, dur) in enumerate(meetings):
        descs.append(f"Meeting {i+1} starts at {sh}:{sm:02d} and lasts {dur} minutes")
    q = f"{'. '.join(descs)}. What is the total duration of all meetings in minutes?"
    return q, str(total), "integer", {"template": "te_ar_total_duration", "dependency": "parallel"}


def _te_co(rng, difficulty, a, b):
    """Temporal + Comparison: compute end times and compare."""
    names = ["Event A", "Event B"]
    s1_h, s1_m = rng.randint(8, 14), rng.randint(0, 3) * 15
    d1 = rng.randint(30, 150)
    s2_h, s2_m = rng.randint(8, 14), rng.randint(0, 3) * 15
    d2 = rng.randint(30, 150)
    e1 = s1_h * 60 + s1_m + d1
    e2 = s2_h * 60 + s2_m + d2
    q = (f"{names[0]} starts at {s1_h}:{s1_m:02d} and lasts {d1} minutes. "
         f"{names[1]} starts at {s2_h}:{s2_m:02d} and lasts {d2} minutes. "
         f"Which event ends later?")
    ans = names[0] if e1 > e2 else names[1]
    return q, ans, "name", {"template": "te_co_end_compare", "dependency": "parallel"}


def _st_cn(rng, difficulty, a, b):
    """String + Counting: manipulate string, then count characters."""
    word = _random_word(rng, length=rng.randint(8, 12))
    old_char = rng.choice(list(set(word)))
    new_char = rng.choice("xyz")
    while new_char == old_char:
        new_char = rng.choice("xyz")
    new_word = word.replace(old_char, new_char)
    target = rng.choice(list(set(new_word)))
    q = (f"Take the string '{word}' and replace all '{old_char}' with '{new_char}'. "
         f"In the resulting string, how many times does '{target}' appear?")
    ans = str(new_word.count(target))
    return q, ans, "integer", {"template": "st_cn_replace_count", "dependency": "sequential"}


def _st_co(rng, difficulty, a, b):
    """String + Comparison: compare lengths after manipulation."""
    w1 = _random_word(rng)
    w2 = _random_word(rng)
    q = f"Which string is longer: the reverse of '{w1}' or '{w2}'?"
    # Reverse doesn't change length, so just compare lengths
    if len(w1) > len(w2):
        ans = f"the reverse of '{w1}'"
    elif len(w2) > len(w1):
        ans = f"'{w2}'"
    else:
        ans = "they are the same length"
    # Make it more interesting - use substring
    start = rng.randint(0, max(0, len(w1) - 4))
    end = start + rng.randint(2, min(4, len(w1) - start))
    sub = w1[start:end]
    q = f"Consider the substring of '{w1}' from position {start+1} to {end} (1-indexed, inclusive): '{sub}'. Is this substring longer than the string '{w2[:3]}'?"
    ans = "yes" if len(sub) > 3 else ("no" if len(sub) < 3 else "they are the same length")
    return q, ans, "yesno_or_equal", {"template": "st_co_length", "dependency": "sequential"}


def _sp_ld(rng, difficulty, a, b):
    """Spatial + Logical Deduction: determine positions then deduce."""
    names = _make_fictional_names(rng, 4)
    q = (f"{names[0]} is north of {names[1]}. {names[1]} is north of {names[2]}. "
         f"If only someone who is north of {names[2]} can enter the castle, "
         f"can {names[0]} enter the castle?")
    # names[0] is north of names[1] which is north of names[2], so names[0] is north of names[2]
    return q, "yes", "yesno", {"template": "sp_ld_position_deduce", "dependency": "sequential"}


def _sp_ar(rng, difficulty, a, b):
    """Spatial + Arithmetic: compute distances from positions."""
    names = _make_fictional_names(rng, 3)
    d1 = rng.randint(2, 15)
    d2 = rng.randint(2, 15)
    q = (f"{names[0]} is {d1} km east of {names[1]}. {names[2]} is {d2} km west of {names[1]}. "
         f"What is the total distance from {names[0]} to {names[2]}?")
    return q, f"{d1 + d2} km", "distance", {"template": "sp_ar_distance", "dependency": "sequential"}


def _ld_se(rng, difficulty, a, b):
    """Logical Deduction + Set: determine set membership through logic."""
    s1 = sorted(rng.sample(range(1, 20), rng.randint(4, 7)))
    s2 = sorted(rng.sample(range(1, 20), rng.randint(4, 7)))
    inter = sorted(set(s1) & set(s2))
    q = (f"Set A = {_set_str(s1)} and Set B = {_set_str(s2)}. "
         f"If a number must be in both Set A and Set B to be considered 'valid', "
         f"how many valid numbers are there?")
    return q, str(len(inter)), "integer", {"template": "ld_se_membership", "dependency": "parallel"}


# ============================================================
# Generic composers (fallback for pairs without specific templates)
# ============================================================

def _generic_pairwise(rng, difficulty, skill_a, skill_b):
    """Generic pairwise: generate two single-skill sub-questions and combine."""
    from .skills import SKILL_GENERATORS
    q1, a1, t1, m1 = SKILL_GENERATORS[skill_a](rng, difficulty)
    q2, a2, t2, m2 = SKILL_GENERATORS[skill_b](rng, difficulty)

    # Parallel composition: answer both parts
    q = f"Answer both parts:\nPart 1: {q1}\nPart 2: {q2}\nGive your answers as 'Part 1: [answer1], Part 2: [answer2]'."
    ans = f"Part 1: {a1}, Part 2: {a2}"
    return q, ans, "multi_part", {"template": "generic_parallel", "dependency": "parallel",
                                   "sub_answers": [a1, a2], "sub_types": [t1, t2]}


def _generic_triple(rng, difficulty, s1, s2, s3):
    """Generic triple: combine three skills."""
    from .skills import SKILL_GENERATORS

    # Try to chain: s1 feeds s2, s2 feeds s3
    # For simplicity: parallel three-part question
    q1, a1, t1, m1 = SKILL_GENERATORS[s1](rng, difficulty)
    q2, a2, t2, m2 = SKILL_GENERATORS[s2](rng, difficulty)
    q3, a3, t3, m3 = SKILL_GENERATORS[s3](rng, difficulty)

    q = (f"Answer all three parts:\n"
         f"Part 1: {q1}\n"
         f"Part 2: {q2}\n"
         f"Part 3: {q3}\n"
         f"Give your answers as 'Part 1: [answer1], Part 2: [answer2], Part 3: [answer3]'.")
    ans = f"Part 1: {a1}, Part 2: {a2}, Part 3: {a3}"
    return q, ans, "multi_part", {"template": "generic_triple_parallel", "dependency": "parallel",
                                   "sub_answers": [a1, a2, a3], "sub_types": [t1, t2, t3]}


# ============================================================
# Registry of specific pairwise composers
# ============================================================
PAIRWISE_COMPOSERS = {
    tuple(sorted(["AR", "CO"])): [_ar_co],
    tuple(sorted(["AR", "CN"])): [_ar_cn],
    tuple(sorted(["AR", "SE"])): [_ar_se],
    tuple(sorted(["CO", "CN"])): [_co_cn],
    tuple(sorted(["CN", "SE"])): [_cn_se],
    tuple(sorted(["LD", "CO"])): [_ld_co],
    tuple(sorted(["TE", "AR"])): [_te_ar],
    tuple(sorted(["TE", "CO"])): [_te_co],
    tuple(sorted(["ST", "CN"])): [_st_cn],
    tuple(sorted(["ST", "CO"])): [_st_co],
    tuple(sorted(["SP", "LD"])): [_sp_ld],
    tuple(sorted(["SP", "AR"])): [_sp_ar],
    tuple(sorted(["LD", "SE"])): [_ld_se],
}

# Classify dependency types
DEPENDENCY_TYPES = {}
for key, composers in PAIRWISE_COMPOSERS.items():
    # Check first composer's metadata
    test_rng = random.Random(0)
    _, _, _, meta = composers[0](test_rng, "medium", key[0], key[1])
    DEPENDENCY_TYPES[key] = meta.get("dependency", "parallel")
