import random
from collections import deque


def canonicalize_state(state):
    return tuple(sorted(state))


def parse_fact(fact):
    fact = fact.strip()
    assert fact.startswith("(") and fact.endswith(")")
    toks = fact[1:-1].split()
    return toks[0], toks[1:]


def blocks_from_state(state):
    blocks = set()
    for pred in state:
        _, args = parse_fact(pred)
        blocks.update(args)
    return sorted(blocks)


def state_to_supports(state, blocks):
    support = {b: "table" for b in blocks}
    holding = None
    for pred in state:
        name, args = parse_fact(pred)
        if name == "on":
            support[args[0]] = args[1]
        elif name == "holding":
            holding = args[0]
    return support, holding


def supports_to_state(support, holding, blocks):
    occupied = {v for v in support.values() if v != "table"}
    state = set()
    for b in blocks:
        if b == holding:
            continue
        loc = support[b]
        state.add(f"(ontable {b})" if loc == "table" else f"(on {b} {loc})")
    state.add("(handempty)" if holding is None else f"(holding {holding})")
    for b in blocks:
        if b == holding:
            continue
        if b not in occupied:
            state.add(f"(clear {b})")
    return canonicalize_state(state)


def applicable_actions(state):
    blocks = blocks_from_state(state)
    support, holding = state_to_supports(state, blocks)
    occupied = {v for v in support.values() if v != "table"}
    clear = {b for b in blocks if b not in occupied and b != holding}
    actions = []
    if holding is None:
        for b in blocks:
            if b not in clear:
                continue
            if support[b] == "table":
                actions.append(("pick-up", b))
            else:
                actions.append(("unstack", b, support[b]))
    else:
        b = holding
        actions.append(("put-down", b))
        for dest in sorted(clear):
            if dest != b:
                actions.append(("stack", b, dest))
    return actions


def transition(state, action):
    blocks = blocks_from_state(state)
    support, holding = state_to_supports(state, blocks)
    kind = action[0]
    if kind == "pick-up":
        support.pop(action[1], None)
        holding = action[1]
    elif kind == "put-down":
        support[action[1]] = "table"
        holding = None
    elif kind == "unstack":
        b, src = action[1], action[2]
        if support.get(b) != src:
            return None
        support.pop(b, None)
        holding = b
    elif kind == "stack":
        support[action[1]] = action[2]
        holding = None
    else:
        return None
    return supports_to_state(support, holding, blocks)


def action_preconditions(action):
    kind = action[0]
    if kind == "pick-up":
        b = action[1]
        return {f"(clear {b})", f"(ontable {b})", "(handempty)"}
    if kind == "unstack":
        b, src = action[1], action[2]
        return {f"(clear {b})", f"(on {b} {src})", "(handempty)"}
    if kind == "put-down":
        return {f"(holding {action[1]})"}
    if kind == "stack":
        b, dest = action[1], action[2]
        return {f"(holding {b})", f"(clear {dest})"}
    return set()


def goal_satisfied(state, goal_facts):
    visible = set(state)
    return all(goal in visible for goal in goal_facts)


def bfs_solve(initial_state, goal_facts, max_depth=8):
    init = canonicalize_state(initial_state)
    if goal_satisfied(init, goal_facts):
        return True, []
    queue = deque([(init, [])])
    seen = {init}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in applicable_actions(state):
            nxt = transition(state, action)
            if nxt is None or nxt in seen:
                continue
            new_path = path + [action]
            if goal_satisfied(nxt, goal_facts):
                return True, new_path
            seen.add(nxt)
            queue.append((nxt, new_path))
    return False, []


def random_initial_state(blocks, rng):
    perm = list(blocks)
    rng.shuffle(perm)
    support = {}
    stacks = []
    for b in perm:
        if not stacks or rng.random() < 0.45:
            stacks.append([b])
        else:
            stacks[rng.randrange(len(stacks))].append(b)
    for stack in stacks:
        support[stack[0]] = "table"
        for idx in range(1, len(stack)):
            support[stack[idx]] = stack[idx - 1]
    return supports_to_state(support, None, blocks)


def random_goal(blocks, rng):
    order = list(blocks)
    rng.shuffle(order)
    length = min(len(blocks), max(2, rng.randint(2, len(blocks))))
    chain = order[:length]
    facts = [f"(ontable {chain[0]})"]
    for idx in range(1, len(chain)):
        facts.append(f"(on {chain[idx]} {chain[idx - 1]})")
    return tuple(sorted(facts))


def render_fact(fact, family):
    if family == "A":
        return fact
    if family == "B":
        return fact.replace("(", "").replace(")", "")
    return fact.replace("ontable", "on_table")


def render_state(state, family="A"):
    prefix = {"A": "Initial facts:", "B": "World state:", "C": "Current arrangement:"}[family]
    joiner = {"A": " ", "B": "; ", "C": " | "}[family]
    return f"{prefix} {joiner.join(render_fact(fact, family) for fact in sorted(state))}"


def render_goal(goal_facts, family="A"):
    prefix = {"A": "Goal facts:", "B": "Target configuration:", "C": "Wanted end state:"}[family]
    joiner = {"A": " ", "B": "; ", "C": " | "}[family]
    return f"{prefix} {joiner.join(render_fact(fact, family) for fact in sorted(goal_facts))}"


def choose_clear_complement(state, hidden_fact):
    blocks = blocks_from_state(state)
    support, holding = state_to_supports(state, blocks)
    if holding is not None:
        return None
    _, args = parse_fact(hidden_fact)
    target = args[0]
    if support.get(target) is None:
        return None
    occupied = {v for v in support.values() if v != "table"}
    candidate_blocks = [
        b for b in blocks if b not in occupied and support.get(b) == "table" and b != target
    ]
    if not candidate_blocks:
        return None
    moved = candidate_blocks[0]
    support[moved] = target
    return supports_to_state(support, None, blocks)


def choose_handempty_complement(state):
    blocks = blocks_from_state(state)
    support, holding = state_to_supports(state, blocks)
    if holding is not None:
        return None
    occupied = {v for v in support.values() if v != "table"}
    candidate_blocks = [b for b in blocks if b not in occupied]
    if not candidate_blocks:
        return None
    chosen = candidate_blocks[0]
    support.pop(chosen, None)
    return supports_to_state(support, chosen, blocks)


def candidate_hidden_facts(state):
    return sorted([fact for fact in state if fact == "(handempty)" or fact.startswith("(clear ")])


def complement_from_hidden_fact(state, hidden_fact):
    if hidden_fact == "(handempty)":
        return choose_handempty_complement(state)
    return choose_clear_complement(state, hidden_fact)


def clean_candidate_metadata(state, goal, fact, max_depth):
    complement_state = complement_from_hidden_fact(state, fact)
    if complement_state is None:
        return None
    visible = tuple(sorted(set(state) - {fact}))
    full_solvable, full_plan = bfs_solve(state, goal, max_depth=max_depth)
    visible_plus_fact_solvable, _ = bfs_solve(tuple(sorted(set(visible) | {fact})), goal, max_depth=max_depth)
    complement_solvable, _ = bfs_solve(complement_state, goal, max_depth=max_depth)
    return {
        "hidden_fact": fact,
        "visible_state": list(visible),
        "full_solvable": full_solvable,
        "full_plan_length": len(full_plan),
        "visible_plus_fact_solvable": visible_plus_fact_solvable,
        "complement_state": list(complement_state),
        "complement_solvable": complement_solvable,
    }


def planning_prompt(state, goal, family, underspecified=False):
    instruction = (
        "Task: decide whether to ask for one missing fact or answer now.\n"
        if underspecified
        else "Task: answer whether the goal is reachable within the horizon.\n"
    )
    suffix = (
        "If one fact is missing, output ASK: <fact>. Otherwise output ANSWER: yes or ANSWER: no."
        if underspecified
        else "Output exactly ANSWER: yes or ANSWER: no."
    )
    return f"{instruction}{render_state(state, family)}\n{render_goal(goal, family)}\n{suffix}"


def clarified_planning_prompt(visible_state, hidden_fact, goal, family):
    return (
        f"{planning_prompt(visible_state, goal, family, underspecified=True)}\n"
        f"Provided missing fact: {render_fact(hidden_fact, family)}\n"
        "Now answer whether the goal is reachable within the horizon.\n"
        "Output exactly ANSWER: yes or ANSWER: no."
    )


def build_planning_example(state, goal, family, max_depth):
    solvable, plan = bfs_solve(state, goal, max_depth=max_depth)
    return {
        "domain": "planning",
        "task_type": "full",
        "input_text": planning_prompt(state, goal, family, underspecified=False),
        "target_text": f"ANSWER: {'yes' if solvable else 'no'}",
        "family": family,
        "state": list(state),
        "goal": list(goal),
        "plan_length": len(plan) if solvable else None,
    }


def rerender_underspecified(item, family):
    return {
        **item,
        "family": family,
        "input_text": planning_prompt(tuple(item["visible_state"]), tuple(item["goal"]), family, underspecified=True),
        "clarified_input_text": clarified_planning_prompt(
            tuple(item["visible_state"]), item["hidden_fact"], tuple(item["goal"]), family
        ),
        "full_input_text": planning_prompt(tuple(item["state"]), tuple(item["goal"]), family, underspecified=False),
    }


def build_planning_pool(train_full=120, train_underspecified=120, eval_full=40, eval_underspecified=40, max_depth=8, seed=101):
    rng = random.Random(seed)
    family_to_split = {"A": "train", "B": "validation", "C": "test"}
    quotas = {
        "A": {"full_yes": train_full // 2, "full_no": train_full - (train_full // 2), "clean": train_underspecified, "noisy": train_underspecified},
        "B": {"full_yes": eval_full // 2, "full_no": eval_full - (eval_full // 2), "clean": eval_underspecified, "noisy": eval_underspecified},
        "C": {"full_yes": eval_full // 2, "full_no": eval_full - (eval_full // 2), "clean": eval_underspecified, "noisy": eval_underspecified},
    }
    pool = {
        "train": {"full": [], "clean": [], "noisy": [], "aux_non_unique": []},
        "validation": {"full": [], "clean": [], "noisy": [], "aux_non_unique": []},
        "test": {"full": [], "clean": [], "noisy": [], "aux_non_unique": []},
        "robustness": {"A": [], "B": [], "C": []},
    }
    stats = {"attempts": 0, "clean_candidates": 0, "clean_accepts": 0, "noisy_accepts": 0, "non_unique_candidates": 0}

    def pending():
        return any(any(value > 0 for value in quota.values()) for quota in quotas.values())

    while pending() and stats["attempts"] < 50000:
        stats["attempts"] += 1
        family = min(
            quotas,
            key=lambda fam: (
                sum(quotas[fam].values()),
                fam,
            ),
        )
        if sum(quotas[family].values()) == 0:
            family = rng.choice([fam for fam, quota in quotas.items() if sum(quota.values()) > 0])
        split = family_to_split[family]
        num_blocks = rng.randint(4, 5)
        blocks = [chr(ord("a") + idx) for idx in range(num_blocks)]
        state = random_initial_state(blocks, rng)
        goal = random_goal(blocks, rng)
        solvable, _ = bfs_solve(state, goal, max_depth=max_depth)
        label = "full_yes" if solvable else "full_no"
        if quotas[family][label] > 0:
            pool[split]["full"].append(build_planning_example(state, goal, family, max_depth))
            quotas[family][label] -= 1

        if not solvable:
            continue

        metas = []
        for fact in candidate_hidden_facts(state):
            meta = clean_candidate_metadata(state, goal, fact, max_depth)
            if meta is not None:
                metas.append(meta)
        if not metas:
            continue
        stats["clean_candidates"] += len(metas)
        clean_metas = [
            meta for meta in metas if meta["full_solvable"] and meta["visible_plus_fact_solvable"] and not meta["complement_solvable"]
        ]
        if len(clean_metas) > 1:
            stats["non_unique_candidates"] += 1
            if split == "test":
                chosen = clean_metas[0]
                base = {
                    "domain": "planning",
                    "task_type": "underspecified",
                    "hidden_fact": chosen["hidden_fact"],
                    "state": list(state),
                    "visible_state": chosen["visible_state"],
                    "goal": list(goal),
                    "full_input_text": planning_prompt(state, goal, family, underspecified=False),
                    "full_target_text": "ANSWER: yes",
                    "clarified_input_text": clarified_planning_prompt(
                        chosen["visible_state"], chosen["hidden_fact"], goal, family
                    ),
                    "valid_asks": [chosen["hidden_fact"]],
                    "complement_state": chosen["complement_state"],
                    "num_clean_candidates": len(clean_metas),
                    "unique_clean_candidate": False,
                    "family": family,
                    "input_text": planning_prompt(chosen["visible_state"], goal, family, underspecified=True),
                    "target_text": f"ASK: {chosen['hidden_fact']}",
                }
                pool[split]["aux_non_unique"].append(base)
        if len(clean_metas) != 1:
            continue
        chosen = clean_metas[0]
        noisy_choices = [meta for meta in metas if meta["hidden_fact"] != chosen["hidden_fact"] and meta not in clean_metas]
        if quotas[family]["clean"] > 0:
            item = {
                "domain": "planning",
                "task_type": "underspecified",
                "hidden_fact": chosen["hidden_fact"],
                "state": list(state),
                "visible_state": chosen["visible_state"],
                "goal": list(goal),
                "full_input_text": planning_prompt(state, goal, family, underspecified=False),
                "full_target_text": "ANSWER: yes",
                "clarified_input_text": clarified_planning_prompt(
                    chosen["visible_state"], chosen["hidden_fact"], goal, family
                ),
                "valid_asks": [chosen["hidden_fact"]],
                "complement_state": chosen["complement_state"],
                "num_clean_candidates": 1,
                "unique_clean_candidate": True,
                "family": family,
                "input_text": planning_prompt(chosen["visible_state"], goal, family, underspecified=True),
                "target_text": f"ASK: {chosen['hidden_fact']}",
            }
            pool[split]["clean"].append(item)
            quotas[family]["clean"] -= 1
            stats["clean_accepts"] += 1
            if split == "test":
                for eval_family in ["A", "B", "C"]:
                    pool["robustness"][eval_family].append(rerender_underspecified(item, eval_family))
        if noisy_choices and quotas[family]["noisy"] > 0:
            noisy = rng.choice(noisy_choices)
            pool[split]["noisy"].append(
                {
                    "domain": "planning",
                    "task_type": "underspecified",
                    "hidden_fact": noisy["hidden_fact"],
                    "state": list(state),
                    "visible_state": noisy["visible_state"],
                    "goal": list(goal),
                    "full_input_text": planning_prompt(state, goal, family, underspecified=False),
                    "full_target_text": "ANSWER: yes",
                    "clarified_input_text": clarified_planning_prompt(
                        noisy["visible_state"], noisy["hidden_fact"], goal, family
                    ),
                    "valid_asks": [noisy["hidden_fact"]],
                    "complement_state": noisy["complement_state"],
                    "num_clean_candidates": len(clean_metas),
                    "unique_clean_candidate": False,
                    "family": family,
                    "input_text": planning_prompt(noisy["visible_state"], goal, family, underspecified=True),
                    "target_text": f"ASK: {noisy['hidden_fact']}",
                }
            )
            quotas[family]["noisy"] -= 1
            stats["noisy_accepts"] += 1

    stats["acceptance_rate_clean"] = stats["clean_accepts"] / max(1, stats["attempts"])
    return {"train": pool["train"], "validation": pool["validation"], "test": pool["test"], "robustness": pool["robustness"], "stats": stats}
