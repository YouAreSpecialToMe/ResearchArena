import ast
import random
import re

from datasets import load_dataset


TRIPLE_RE = re.compile(r"triple(\d+)")


def split_theory(theory):
    return [s.strip().rstrip(".") + "." for s in theory.split(".") if s.strip()]


def normalize_sentence(text):
    return re.sub(r"\s+", " ", text.strip().rstrip(".").lower())


def normalize_answer(answer):
    answer = answer.strip().lower()
    mapping = {"entails": "true", "contradicts": "false", "unknown": "unknown"}
    return mapping.get(answer, answer)


def parse_allproofs(text):
    if not text:
        return []
    items = []
    for part in text.split("]"):
        if ".[" not in part:
            continue
        sent, proof = part.split(".[", 1)
        sent = sent.split(":", 1)[-1].strip().rstrip(".") + "."
        proof = proof.strip().lstrip("(").rstrip(")")
        items.append((sent, proof))
    return items


def proof_candidates_for_question(ex):
    question = ex["question"].strip().rstrip(".") + "."
    candidates = []
    for sent, proof in parse_allproofs(ex["allProofs"]):
        if sent != question:
            continue
        triples = [int(idx) for idx in TRIPLE_RE.findall(proof)]
        if not triples:
            continue
        candidates.append(
            {
                "question": question,
                "proof_text": proof,
                "triples": triples,
                "proof_len": len(triples),
            }
        )
    return sorted(candidates, key=lambda item: (item["proof_len"], item["proof_text"]))


def logic_prompt(context_sents, question, underspecified=False):
    instruction = (
        "Task: decide whether one fact is missing before answering.\n"
        if underspecified
        else "Task: answer the reasoning question.\n"
    )
    suffix = (
        "If one fact is missing, output ASK: <fact>. Otherwise output ANSWER: true, ANSWER: false, or ANSWER: unknown."
        if underspecified
        else "Output exactly ANSWER: true, ANSWER: false, or ANSWER: unknown."
    )
    return f"{instruction}Context: {' '.join(context_sents)}\nQuestion: {question}\n{suffix}"


def clarified_logic_prompt(visible_context, hidden_fact, question):
    base = logic_prompt(visible_context, question, underspecified=True)
    return (
        f"{base}\nProvided missing fact: {hidden_fact}\n"
        "Now answer the original question.\n"
        "Output exactly ANSWER: true, ANSWER: false, or ANSWER: unknown."
    )


def select_clean_leaf(theory_sents, question, candidates):
    if not candidates:
        return None, []
    canonical = candidates[0]
    if len(canonical["triples"]) != 1:
        return None, []
    triple_id = canonical["triples"][0]
    if triple_id < 1 or triple_id > len(theory_sents):
        return None, []
    hidden_fact = theory_sents[triple_id - 1]
    if normalize_sentence(hidden_fact) == normalize_sentence(question):
        return None, []
    return (triple_id, hidden_fact), [(triple_id, hidden_fact)]


def build_logic_example(theory_sents, question, answer, qid):
    return {
        "domain": "logic",
        "task_type": "full",
        "input_text": logic_prompt(theory_sents, question, underspecified=False),
        "target_text": f"ANSWER: {answer}",
        "qid": qid,
        "answer": answer,
        "question": question,
        "context": theory_sents,
    }


def label_quotas(limit_full):
    base = limit_full // 3
    rem = limit_full - (3 * base)
    quotas = {"true": base, "false": base, "unknown": base}
    for label in ["true", "false", "unknown"][:rem]:
        quotas[label] += 1
    return quotas


def build_logic_pool(n_full=120, n_underspecified=120, seed=101):
    rng = random.Random(seed)

    def collect(split_name, limit_full, limit_ud):
        ds = load_dataset("tasksource/proofwriter", split=split_name)
        full = []
        clean = []
        noisy = []
        full_counts = {key: 0 for key in ["true", "false", "unknown"]}
        quotas = label_quotas(limit_full)
        validator_counts = {
            "eligible_positive_examples": 0,
            "proof_backed_examples": 0,
            "clean_accepts": 0,
            "noisy_accepts": 0,
        }
        for ex in ds:
            answer = normalize_answer(ex["answer"])
            theory_sents = split_theory(ex["theory"])
            question = ex["question"].strip().rstrip(".") + "."
            if answer in quotas and full_counts[answer] < quotas[answer]:
                full.append(build_logic_example(theory_sents, question, answer, ex["id"]))
                full_counts[answer] += 1

            if answer == "true" and (len(clean) < limit_ud or len(noisy) < limit_ud):
                validator_counts["eligible_positive_examples"] += 1
                candidates = proof_candidates_for_question(ex)
                if candidates:
                    validator_counts["proof_backed_examples"] += 1
                selected, eligible = select_clean_leaf(theory_sents, question, candidates)
                if selected is not None:
                    triple_id, hidden_fact = selected
                    visible = [s for idx, s in enumerate(theory_sents, start=1) if idx != triple_id]
                    base_meta = {
                        "domain": "logic",
                        "task_type": "underspecified",
                        "hidden_fact": hidden_fact,
                        "question": question,
                        "qid": ex["id"],
                        "answer": answer,
                        "canonical_proof": candidates[0]["proof_text"],
                        "canonical_triples": candidates[0]["triples"],
                        "all_candidate_proofs": candidates,
                        "num_clean_candidates": len(eligible),
                        "full_input_text": logic_prompt(theory_sents, question, underspecified=False),
                        "full_target_text": "ANSWER: true",
                        "clarified_input_text": clarified_logic_prompt(visible, hidden_fact, question),
                        "valid_asks": [hidden_fact],
                    }
                    if len(clean) < limit_ud:
                        clean.append(
                            {
                                **base_meta,
                                "input_text": logic_prompt(visible, question, underspecified=True),
                                "target_text": f"ASK: {hidden_fact}",
                                "visible_context": visible,
                                "canonical_triple_id": triple_id,
                            }
                        )
                        validator_counts["clean_accepts"] += 1
                    distractor_ids = [
                        idx
                        for idx in range(1, len(theory_sents) + 1)
                        if idx != triple_id and normalize_sentence(theory_sents[idx - 1]) != normalize_sentence(question)
                    ]
                    if distractor_ids and len(noisy) < limit_ud:
                        noise_id = rng.choice(distractor_ids)
                        noisy_hidden = theory_sents[noise_id - 1]
                        noisy_visible = [s for idx, s in enumerate(theory_sents, start=1) if idx != noise_id]
                        noisy.append(
                            {
                                **base_meta,
                                "input_text": logic_prompt(noisy_visible, question, underspecified=True),
                                "target_text": f"ASK: {noisy_hidden}",
                                "hidden_fact": noisy_hidden,
                                "visible_context": noisy_visible,
                                "removed_triple_id": noise_id,
                                "clean_triple_id": triple_id,
                                "clarified_input_text": clarified_logic_prompt(noisy_visible, noisy_hidden, question),
                                "valid_asks": [noisy_hidden],
                            }
                        )
                        validator_counts["noisy_accepts"] += 1
            if len(full) >= limit_full and len(clean) >= limit_ud and len(noisy) >= limit_ud:
                break
        return {"full": full[:limit_full], "clean": clean[:limit_ud], "noisy": noisy[:limit_ud], "stats": validator_counts}

    train = collect("train", n_full, n_underspecified)
    validation = collect("validation", max(40, n_full // 3), max(40, n_underspecified // 3))
    test = collect("test", max(40, n_full // 3), max(40, n_underspecified // 3))
    return {"train": train, "validation": validation, "test": test, "stats": train["stats"]}


def parse_questbench_gt(gt_qs_raw):
    if isinstance(gt_qs_raw, (list, tuple, set, frozenset)):
        return list(gt_qs_raw)
    try:
        value = ast.literal_eval(gt_qs_raw)
        if isinstance(value, frozenset):
            return list(value)
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]
    except Exception:
        try:
            value = eval(gt_qs_raw, {"frozenset": frozenset})
            if isinstance(value, (list, tuple, set, frozenset)):
                return list(value)
            return [value]
        except Exception:
            return []


def parse_pythonish(value):
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return eval(value, {"frozenset": frozenset})


def render_logic_rule(rule):
    antecedents = list(rule[:-1])
    conclusion = rule[-1]
    if not antecedents:
        return conclusion
    return f"{' and '.join(antecedents)} -> {conclusion}"


def questbench_logic_prompt(example):
    known_true = parse_questbench_gt(example["known_facts"])
    known_false = parse_questbench_gt(example["known_untrue_facts"])
    rules = parse_pythonish(example["rules"])
    lines = [
        "Task: ask for one missing fact if needed to determine the goal.",
        f"Known true facts: {', '.join(known_true) if known_true else 'none'}",
        f"Known false facts: {', '.join(known_false) if known_false else 'none'}",
        "Rules:",
    ]
    lines.extend(f"- {render_logic_rule(rule)}" for rule in rules)
    lines.append(f"Goal: {example['goal']}")
    lines.append("Output ASK: <fact> or ANSWER: no_question_needed.")
    return "\n".join(lines)


def infer_logic_goal(known_facts, known_untrue_facts, rules, goal, max_depth):
    derived = set(known_facts) | {f"not {fact}" for fact in known_untrue_facts}
    parsed_rules = [list(rule) for rule in rules]
    for _ in range(max_depth):
        changed = False
        for rule in parsed_rules:
            antecedents = rule[:-1]
            conclusion = rule[-1]
            if all(ant in derived for ant in antecedents) and conclusion not in derived:
                derived.add(conclusion)
                changed = True
        if not changed:
            break
    if goal in derived:
        return "ANSWER: true"
    if f"not {goal}" in derived:
        return "ANSWER: false"
    return "ANSWER: unknown"
