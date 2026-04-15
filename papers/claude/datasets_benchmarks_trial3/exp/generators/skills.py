"""
SkillStack: 8 Atomic Cognitive Skill Generators

Each generator produces (question, answer, answer_type, metadata) tuples.
All answers are deterministic given the random state.
"""
import random
import string
from itertools import combinations


# ============================================================
# Skill 1: Arithmetic (AR)
# ============================================================
def generate_arithmetic(rng: random.Random, difficulty="medium"):
    """Multi-step integer arithmetic."""
    templates = [
        _ar_chain_addition,
        _ar_mixed_operations,
        _ar_word_problem,
        _ar_nested_computation,
        _ar_remainder_problem,
    ]
    template_fn = rng.choice(templates)
    return template_fn(rng, difficulty)


def _ar_chain_addition(rng, difficulty):
    lo, hi = (10, 99) if difficulty == "medium" else (100, 999)
    nums = [rng.randint(lo, hi) for _ in range(rng.randint(3, 5))]
    q = f"What is {' + '.join(map(str, nums))}?"
    return q, str(sum(nums)), "integer", {"template": "ar_chain_add"}


def _ar_mixed_operations(rng, difficulty):
    lo, hi = (10, 50) if difficulty == "medium" else (50, 200)
    a, b, c = [rng.randint(lo, hi) for _ in range(3)]
    op = rng.choice(["add_mul", "sub_mul", "mul_add"])
    if op == "add_mul":
        q = f"What is ({a} + {b}) × {c}?"
        ans = (a + b) * c
    elif op == "sub_mul":
        q = f"What is ({a} - {b}) × {c}?"
        ans = (a - b) * c
    else:
        q = f"What is {a} × {b} + {c}?"
        ans = a * b + c
    return q, str(ans), "integer", {"template": "ar_mixed_ops"}


def _ar_word_problem(rng, difficulty):
    lo, hi = (5, 30) if difficulty == "medium" else (20, 100)
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    n1, n2 = rng.sample(names, 2)
    items = rng.choice(["apples", "books", "marbles", "coins", "stickers"])
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    gave = rng.randint(1, min(a, 10))
    q = f"{n1} has {a} {items} and {n2} has {b} {items}. {n1} gives {gave} {items} to {n2}. How many {items} do they have in total?"
    return q, str(a + b), "integer", {"template": "ar_word"}


def _ar_nested_computation(rng, difficulty):
    lo, hi = (2, 15) if difficulty == "medium" else (10, 50)
    a, b, c, d = [rng.randint(lo, hi) for _ in range(4)]
    q = f"Compute ({a} + {b}) × ({c} - {d})."
    ans = (a + b) * (c - d)
    return q, str(ans), "integer", {"template": "ar_nested"}


def _ar_remainder_problem(rng, difficulty):
    lo, hi = (20, 100) if difficulty == "medium" else (100, 500)
    a = rng.randint(lo, hi)
    b = rng.randint(2, 12)
    q = f"What is the remainder when {a} is divided by {b}?"
    return q, str(a % b), "integer", {"template": "ar_remainder"}


# ============================================================
# Skill 2: Comparison (CO)
# ============================================================
def generate_comparison(rng: random.Random, difficulty="medium"):
    templates = [
        _co_number_compare,
        _co_fraction_compare,
        _co_expression_compare,
        _co_word_compare,
        _co_multi_compare,
    ]
    return rng.choice(templates)(rng, difficulty)


def _co_number_compare(rng, difficulty):
    lo, hi = (10, 999) if difficulty == "medium" else (1000, 9999)
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    while a == b:
        b = rng.randint(lo, hi)
    q = f"Which is larger: {a} or {b}?"
    return q, str(max(a, b)), "integer", {"template": "co_number"}


def _co_fraction_compare(rng, difficulty):
    denoms = [3, 4, 5, 6, 8, 10]
    d1, d2 = rng.choice(denoms), rng.choice(denoms)
    n1 = rng.randint(1, d1 - 1)
    n2 = rng.randint(1, d2 - 1)
    attempts = 0
    while n1 / d1 == n2 / d2 and attempts < 20:
        d2 = rng.choice(denoms)
        n2 = rng.randint(1, d2 - 1)
        attempts += 1
    q = f"Which fraction is larger: {n1}/{d1} or {n2}/{d2}?"
    if n1 / d1 > n2 / d2:
        ans = f"{n1}/{d1}"
    else:
        ans = f"{n2}/{d2}"
    return q, ans, "fraction", {"template": "co_fraction"}


def _co_expression_compare(rng, difficulty):
    lo, hi = (2, 20) if difficulty == "medium" else (10, 50)
    a, b, c, d = [rng.randint(lo, hi) for _ in range(4)]
    val1, val2 = a * b, c + d
    q = f"Which is greater: {a} × {b} or {c} + {d}?"
    if val1 > val2:
        ans = f"{a} × {b}"
    elif val2 > val1:
        ans = f"{c} + {d}"
    else:
        ans = "they are equal"
    return q, ans, "expression", {"template": "co_expression"}


def _co_word_compare(rng, difficulty):
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    n1, n2 = rng.sample(names, 2)
    item = rng.choice(["apples", "points", "coins", "books"])
    a, b = rng.randint(5, 50), rng.randint(5, 50)
    while a == b:
        b = rng.randint(5, 50)
    q = f"{n1} has {a} {item} and {n2} has {b} {item}. Who has more {item}?"
    ans = n1 if a > b else n2
    return q, ans, "name", {"template": "co_word"}


def _co_multi_compare(rng, difficulty):
    lo, hi = (10, 100) if difficulty == "medium" else (50, 500)
    nums = [rng.randint(lo, hi) for _ in range(rng.randint(3, 5))]
    q = f"What is the smallest number among {', '.join(map(str, nums))}?"
    return q, str(min(nums)), "integer", {"template": "co_multi"}


# ============================================================
# Skill 3: Counting (CN)
# ============================================================
def generate_counting(rng: random.Random, difficulty="medium"):
    templates = [
        _cn_count_predicate,
        _cn_count_even_odd,
        _cn_count_greater,
        _cn_count_letters,
        _cn_count_items,
    ]
    return rng.choice(templates)(rng, difficulty)


def _cn_count_predicate(rng, difficulty):
    n = rng.randint(8, 12) if difficulty == "medium" else rng.randint(12, 18)
    nums = [rng.randint(1, 50) for _ in range(n)]
    threshold = rng.randint(15, 35)
    q = f"How many numbers in the list [{', '.join(map(str, nums))}] are greater than {threshold}?"
    ans = sum(1 for x in nums if x > threshold)
    return q, str(ans), "integer", {"template": "cn_predicate"}


def _cn_count_even_odd(rng, difficulty):
    n = rng.randint(8, 12) if difficulty == "medium" else rng.randint(12, 18)
    nums = [rng.randint(1, 100) for _ in range(n)]
    parity = rng.choice(["even", "odd"])
    q = f"How many {parity} numbers are in the list [{', '.join(map(str, nums))}]?"
    if parity == "even":
        ans = sum(1 for x in nums if x % 2 == 0)
    else:
        ans = sum(1 for x in nums if x % 2 == 1)
    return q, str(ans), "integer", {"template": "cn_even_odd"}


def _cn_count_greater(rng, difficulty):
    n = rng.randint(8, 12) if difficulty == "medium" else rng.randint(12, 18)
    nums = [rng.randint(1, 100) for _ in range(n)]
    mean_val = sum(nums) / len(nums)
    q = f"In the list [{', '.join(map(str, nums))}], how many numbers are above the average? (The average is {mean_val:.1f})"
    ans = sum(1 for x in nums if x > mean_val)
    return q, str(ans), "integer", {"template": "cn_above_avg"}


def _cn_count_letters(rng, difficulty):
    words = ["banana", "mississippi", "programming", "education", "communication",
             "elephant", "butterfly", "strawberry", "independence", "celebration"]
    word = rng.choice(words)
    letter = rng.choice(list(set(word)))
    q = f"How many times does the letter '{letter}' appear in the word '{word}'?"
    ans = word.count(letter)
    return q, str(ans), "integer", {"template": "cn_letters"}


def _cn_count_items(rng, difficulty):
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    items = ["ball", "cube", "star", "ring", "disk"]
    n = rng.randint(8, 14)
    objects = [(rng.choice(colors), rng.choice(items)) for _ in range(n)]
    target_color = rng.choice(colors)
    obj_strs = [f"a {c} {i}" for c, i in objects]
    q = f"In a collection of {', '.join(obj_strs)}, how many objects are {target_color}?"
    ans = sum(1 for c, i in objects if c == target_color)
    return q, str(ans), "integer", {"template": "cn_items"}


# ============================================================
# Skill 4: Logical Deduction (LD)
# ============================================================
def generate_logical_deduction(rng: random.Random, difficulty="medium"):
    templates = [
        _ld_syllogism,
        _ld_ordering,
        _ld_conditional,
        _ld_negation,
        _ld_elimination,
    ]
    return rng.choice(templates)(rng, difficulty)


def _make_fictional_names(rng, n=3):
    syllables = ["Zor", "Blix", "Kren", "Mof", "Tev", "Pla", "Dru", "Vex", "Nol", "Gri"]
    endings = ["ax", "ik", "on", "ul", "em", "is", "ar"]
    all_names = [s + e for s in syllables for e in endings]
    rng.shuffle(all_names)
    return all_names[:n]


def _ld_syllogism(rng, difficulty):
    names = _make_fictional_names(rng, 3)
    props = ["tall", "happy", "wise", "brave", "fast"]
    p1, p2 = rng.sample(props, 2)
    q = (f"All {names[0]}s are {p1}. All {p1} things are {p2}. "
         f"Is a {names[0]} {p2}?")
    return q, "yes", "yesno", {"template": "ld_syllogism"}


def _ld_ordering(rng, difficulty):
    names = _make_fictional_names(rng, 4)
    # A > B, B > C, C > D
    q = (f"{names[0]} is taller than {names[1]}. {names[1]} is taller than {names[2]}. "
         f"{names[2]} is taller than {names[3]}. Who is the tallest?")
    return q, names[0], "name", {"template": "ld_ordering"}


def _ld_conditional(rng, difficulty):
    names = _make_fictional_names(rng, 2)
    conds = [("raining", "carries an umbrella"), ("sunny", "wears sunglasses"),
             ("cold", "wears a coat"), ("windy", "stays inside")]
    cond, result = rng.choice(conds)
    is_true = rng.choice([True, False])
    if is_true:
        q = f"If it is {cond}, then {names[0]} {result}. It is {cond}. Does {names[0]} {result.split()[0]} {' '.join(result.split()[1:])}?"
        ans = "yes"
    else:
        other_cond = rng.choice([c for c, _ in conds if c != cond])
        q = f"If it is {cond}, then {names[0]} {result}. It is {other_cond}. Does {names[0]} necessarily {result.split()[0]} {' '.join(result.split()[1:])}?"
        ans = "no"
    return q, ans, "yesno", {"template": "ld_conditional"}


def _ld_negation(rng, difficulty):
    names = _make_fictional_names(rng, 2)
    prop = rng.choice(["a swimmer", "a runner", "a painter", "a singer"])
    is_neg = rng.choice([True, False])
    if is_neg:
        q = f"No {names[0]} is {prop}. {names[1]} is a {names[0]}. Is {names[1]} {prop}?"
        ans = "no"
    else:
        q = f"Every {names[0]} is {prop}. {names[1]} is a {names[0]}. Is {names[1]} {prop}?"
        ans = "yes"
    return q, ans, "yesno", {"template": "ld_negation"}


def _ld_elimination(rng, difficulty):
    names = _make_fictional_names(rng, 3)
    roles = rng.sample(["the doctor", "the teacher", "the chef", "the artist"], 3)
    # names[0] is roles[0], names[1] is roles[1], names[2] is roles[2]
    q = (f"Among {names[0]}, {names[1]}, and {names[2]}, one is {roles[0]}, one is {roles[1]}, "
         f"and one is {roles[2]}. {names[0]} is not {roles[1]} and not {roles[2]}. "
         f"What is {names[0]}'s role?")
    return q, roles[0], "role", {"template": "ld_elimination"}


# ============================================================
# Skill 5: Spatial Reasoning (SP)
# ============================================================
def generate_spatial(rng: random.Random, difficulty="medium"):
    templates = [
        _sp_direction_chain,
        _sp_relative_position,
        _sp_grid_position,
        _sp_distance_compare,
        _sp_left_right,
    ]
    return rng.choice(templates)(rng, difficulty)


def _sp_direction_chain(rng, difficulty):
    names = _make_fictional_names(rng, 4)
    dirs = ["north", "south", "east", "west"]
    opposites = {"north": "south", "south": "north", "east": "west", "west": "east"}
    d1 = rng.choice(dirs)
    d2 = rng.choice(dirs)
    d3 = rng.choice(dirs)
    q = (f"{names[0]} is {d1} of {names[1]}. {names[1]} is {d2} of {names[2]}. "
         f"{names[2]} is {d3} of {names[3]}. "
         f"If you travel from {names[3]} to {names[0]}, which direction do you primarily go?")
    # Compute net displacement
    dx, dy = 0, 0
    for d in [d3, d2, d1]:  # from names[3] toward names[0]: reverse chain
        # Actually: names[0] is d1 of names[1] means names[0] is in direction d1 from names[1]
        pass
    # Simplify: just track from names[3] -> names[2] -> names[1] -> names[0]
    # names[2] is d3 of names[3], so from names[3] go d3 to reach names[2]
    # names[1] is: names[1] has names[2] to its d2-opposite. Since names[1] is d2 of names[2] means...
    # Actually "A is north of B" means A is to the north of B, so to go from B to A you go north
    # From names[3] to names[2]: go opposite(d3) since "names[2] is d3 of names[3]" means names[2] is in d3 direction from names[3]
    # Wait: "names[2] is d3 of names[3]" means names[2] is to the d3 of names[3]
    # So from names[3], to get to names[2], you go d3
    # From names[2] to names[1]: "names[1] is d2 of names[2]" so go d2
    # From names[1] to names[0]: "names[0] is d1 of names[1]" so go d1
    dir_map = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
    dx, dy = 0, 0
    for d in [d3, d2, d1]:
        ddx, ddy = dir_map[d]
        dx += ddx
        dy += ddy
    if abs(dy) >= abs(dx):
        ans = "north" if dy > 0 else "south"
    else:
        ans = "east" if dx > 0 else "west"
    if dx == 0 and dy == 0:
        ans = "nowhere (same position)"
    return q, ans, "direction", {"template": "sp_direction_chain"}


def _sp_relative_position(rng, difficulty):
    names = _make_fictional_names(rng, 3)
    d1 = rng.choice(["left", "right"])
    d2 = rng.choice(["left", "right"])
    q = (f"{names[0]} is to the {d1} of {names[1]}. {names[1]} is to the {d2} of {names[2]}. "
         f"Is {names[0]} to the left or right of {names[2]}?")
    # positions: names[2] at 0. names[1] is d2 of names[2], so names[1] at +1 if right, -1 if left
    pos1 = 1 if d2 == "right" else -1
    # names[0] is d1 of names[1]
    pos0 = pos1 + (1 if d1 == "right" else -1)
    ans = "right" if pos0 > 0 else "left"
    if pos0 == 0:
        ans = "same position"
    return q, ans, "direction", {"template": "sp_relative"}


def _sp_grid_position(rng, difficulty):
    name = _make_fictional_names(rng, 1)[0]
    moves = []
    x, y = 0, 0
    n_moves = rng.randint(3, 5)
    dir_map = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
    dirs = list(dir_map.keys())
    for _ in range(n_moves):
        d = rng.choice(dirs)
        steps = rng.randint(1, 4)
        moves.append(f"{steps} step{'s' if steps > 1 else ''} {d}")
        dx, dy = dir_map[d]
        x += dx * steps
        y += dy * steps
    q = f"{name} starts at position (0, 0) on a grid and moves {', then '.join(moves)}. What is {name}'s final position (x, y)? (East is +x, North is +y)"
    ans = f"({x}, {y})"
    return q, ans, "coordinate", {"template": "sp_grid"}


def _sp_distance_compare(rng, difficulty):
    names = _make_fictional_names(rng, 3)
    d1 = rng.randint(2, 15)
    d2 = rng.randint(2, 15)
    while d1 == d2:
        d2 = rng.randint(2, 15)
    q = (f"{names[0]} is {d1} km north of {names[1]}. {names[2]} is {d2} km south of {names[1]}. "
         f"Who is closer to {names[1]}?")
    ans = names[0] if d1 < d2 else names[2]
    return q, ans, "name", {"template": "sp_distance"}


def _sp_left_right(rng, difficulty):
    names = _make_fictional_names(rng, 5)
    # Create a line ordering
    order = names[:]
    rng.shuffle(order)
    q_parts = []
    for i in range(len(order) - 1):
        q_parts.append(f"{order[i]} is to the left of {order[i+1]}")
    q = ". ".join(q_parts) + f". Who is in the middle position?"
    ans = order[2]
    return q, ans, "name", {"template": "sp_left_right"}


# ============================================================
# Skill 6: String Manipulation (ST)
# ============================================================
def generate_string(rng: random.Random, difficulty="medium"):
    templates = [
        _st_reverse,
        _st_char_at,
        _st_count_char,
        _st_substring,
        _st_replace,
    ]
    return rng.choice(templates)(rng, difficulty)


def _random_word(rng, length=None):
    if length is None:
        length = rng.randint(6, 10)
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += rng.choice(consonants)
        else:
            word += rng.choice(vowels)
    return word


def _st_reverse(rng, difficulty):
    word = _random_word(rng)
    q = f"What is the reverse of the string '{word}'?"
    return q, word[::-1], "string", {"template": "st_reverse"}


def _st_char_at(rng, difficulty):
    word = _random_word(rng)
    pos = rng.randint(1, len(word))
    q = f"What is the {_ordinal(pos)} character of the string '{word}'?"
    return q, word[pos - 1], "character", {"template": "st_char_at"}


def _st_count_char(rng, difficulty):
    word = _random_word(rng, length=rng.randint(8, 12))
    char = rng.choice(list(set(word)))
    q = f"How many times does '{char}' appear in '{word}'?"
    return q, str(word.count(char)), "integer", {"template": "st_count_char"}


def _st_substring(rng, difficulty):
    word = _random_word(rng, length=rng.randint(8, 12))
    start = rng.randint(0, len(word) - 4)
    end = start + rng.randint(2, 4)
    q = f"What is the substring of '{word}' from position {start + 1} to {end} (inclusive, 1-indexed)?"
    return q, word[start:end], "string", {"template": "st_substring"}


def _st_replace(rng, difficulty):
    word = _random_word(rng)
    old_char = rng.choice(list(set(word)))
    new_char = rng.choice("xyz")
    while new_char == old_char:
        new_char = rng.choice("xyz")
    q = f"Replace all occurrences of '{old_char}' with '{new_char}' in the string '{word}'. What is the result?"
    return q, word.replace(old_char, new_char), "string", {"template": "st_replace"}


def _ordinal(n):
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][min(n % 10, 4) if n % 10 < 4 else 0]}"


# ============================================================
# Skill 7: Temporal Reasoning (TE)
# ============================================================
def generate_temporal(rng: random.Random, difficulty="medium"):
    templates = [
        _te_duration,
        _te_overlap,
        _te_sequence,
        _te_time_diff,
        _te_day_calc,
    ]
    return rng.choice(templates)(rng, difficulty)


def _te_duration(rng, difficulty):
    h = rng.randint(1, 5)
    m = rng.randint(0, 59)
    start_h = rng.randint(6, 18)
    start_m = rng.randint(0, 59)
    end_total = start_h * 60 + start_m + h * 60 + m
    end_h, end_m = end_total // 60, end_total % 60
    q = (f"A meeting starts at {start_h}:{start_m:02d} and lasts {h} hour{'s' if h > 1 else ''} "
         f"and {m} minutes. What time does it end? (Use 24-hour format HH:MM)")
    return q, f"{end_h}:{end_m:02d}", "time", {"template": "te_duration"}


def _te_overlap(rng, difficulty):
    names = ["Meeting A", "Meeting B"]
    s1_h = rng.randint(8, 14)
    s1_m = rng.randint(0, 3) * 15
    d1 = rng.randint(30, 120)
    e1 = s1_h * 60 + s1_m + d1

    s2_h = rng.randint(s1_h, s1_h + 2)
    s2_m = rng.randint(0, 3) * 15
    d2 = rng.randint(30, 120)
    e2 = s2_h * 60 + s2_m + d2
    s2_total = s2_h * 60 + s2_m
    s1_total = s1_h * 60 + s1_m

    overlap_start = max(s1_total, s2_total)
    overlap_end = min(e1, e2)
    overlap = max(0, overlap_end - overlap_start)

    q = (f"{names[0]} runs from {s1_h}:{s1_m:02d} to {e1 // 60}:{e1 % 60:02d}. "
         f"{names[1]} runs from {s2_h}:{s2_m:02d} to {e2 // 60}:{e2 % 60:02d}. "
         f"How many minutes do they overlap?")
    return q, str(overlap), "integer", {"template": "te_overlap"}


def _te_sequence(rng, difficulty):
    events = ["breakfast", "meeting", "lunch", "presentation", "review"]
    n = rng.randint(3, 4)
    sel = rng.sample(events, n)
    durations = [rng.randint(15, 90) for _ in range(n)]
    start_h = rng.randint(7, 10)
    start_m = 0
    times = []
    cur = start_h * 60 + start_m
    for d in durations:
        times.append(cur)
        cur += d
    desc = []
    for i, (ev, dur) in enumerate(zip(sel, durations)):
        desc.append(f"{ev} takes {dur} minutes")
    q = (f"Events happen in sequence starting at {start_h}:00: {', '.join(desc)}. "
         f"At what time does the last event end? (24-hour format HH:MM)")
    ans = f"{cur // 60}:{cur % 60:02d}"
    return q, ans, "time", {"template": "te_sequence"}


def _te_time_diff(rng, difficulty):
    h1 = rng.randint(6, 12)
    m1 = rng.randint(0, 59)
    h2 = rng.randint(h1, h1 + rng.randint(1, 8))
    m2 = rng.randint(0, 59)
    diff = (h2 * 60 + m2) - (h1 * 60 + m1)
    if diff < 0:
        diff += 24 * 60
    q = f"How many minutes are there between {h1}:{m1:02d} and {h2}:{m2:02d}?"
    return q, str(diff), "integer", {"template": "te_time_diff"}


def _te_day_calc(rng, difficulty):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    start_day = rng.choice(days)
    offset = rng.randint(2, 20)
    end_idx = (days.index(start_day) + offset) % 7
    q = f"If today is {start_day}, what day of the week will it be in {offset} days?"
    return q, days[end_idx], "day", {"template": "te_day_calc"}


# ============================================================
# Skill 8: Set Operations (SE)
# ============================================================
def generate_set_ops(rng: random.Random, difficulty="medium"):
    templates = [
        _se_union,
        _se_intersection,
        _se_difference,
        _se_symmetric_diff,
        _se_cardinality,
    ]
    return rng.choice(templates)(rng, difficulty)


def _make_set(rng, size=None, lo=1, hi=20):
    if size is None:
        size = rng.randint(4, 8)
    return sorted(rng.sample(range(lo, hi + 1), min(size, hi - lo + 1)))


def _set_str(s):
    return "{" + ", ".join(map(str, s)) + "}"


def _se_union(rng, difficulty):
    a = _make_set(rng)
    b = _make_set(rng)
    result = sorted(set(a) | set(b))
    q = f"What is the union of {_set_str(a)} and {_set_str(b)}?"
    return q, _set_str(result), "set", {"template": "se_union"}


def _se_intersection(rng, difficulty):
    # Ensure some overlap
    base = _make_set(rng, size=6)
    a = sorted(rng.sample(base, rng.randint(3, len(base))) + _make_set(rng, size=2))
    b = sorted(rng.sample(base, rng.randint(3, len(base))) + _make_set(rng, size=2))
    a = sorted(set(a))
    b = sorted(set(b))
    result = sorted(set(a) & set(b))
    q = f"What is the intersection of {_set_str(a)} and {_set_str(b)}?"
    if not result:
        return q, "empty set", "set", {"template": "se_intersection"}
    return q, _set_str(result), "set", {"template": "se_intersection"}


def _se_difference(rng, difficulty):
    a = _make_set(rng)
    b = _make_set(rng)
    result = sorted(set(a) - set(b))
    q = f"What is {_set_str(a)} minus {_set_str(b)} (set difference)?"
    if not result:
        return q, "empty set", "set", {"template": "se_difference"}
    return q, _set_str(result), "set", {"template": "se_difference"}


def _se_symmetric_diff(rng, difficulty):
    a = _make_set(rng)
    b = _make_set(rng)
    result = sorted(set(a) ^ set(b))
    q = f"What is the symmetric difference of {_set_str(a)} and {_set_str(b)} (elements in one but not both)?"
    if not result:
        return q, "empty set", "set", {"template": "se_symmetric_diff"}
    return q, _set_str(result), "set", {"template": "se_symmetric_diff"}


def _se_cardinality(rng, difficulty):
    a = _make_set(rng)
    b = _make_set(rng)
    op = rng.choice(["union", "intersection"])
    if op == "union":
        result = set(a) | set(b)
        q = f"How many elements are in the union of {_set_str(a)} and {_set_str(b)}?"
    else:
        result = set(a) & set(b)
        q = f"How many elements are in the intersection of {_set_str(a)} and {_set_str(b)}?"
    return q, str(len(result)), "integer", {"template": "se_cardinality"}


# ============================================================
# Registry
# ============================================================
SKILL_GENERATORS = {
    "AR": generate_arithmetic,
    "CO": generate_comparison,
    "CN": generate_counting,
    "LD": generate_logical_deduction,
    "SP": generate_spatial,
    "ST": generate_string,
    "TE": generate_temporal,
    "SE": generate_set_ops,
}

SKILL_NAMES = {
    "AR": "Arithmetic",
    "CO": "Comparison",
    "CN": "Counting",
    "LD": "Logical Deduction",
    "SP": "Spatial Reasoning",
    "ST": "String Manipulation",
    "TE": "Temporal Reasoning",
    "SE": "Set Operations",
}

SKILL_CODES = list(SKILL_GENERATORS.keys())
