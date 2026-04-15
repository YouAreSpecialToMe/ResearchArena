"""
Two-stage answer equivalence system for ConsistBench.
Stage 1: Rule-based normalization + exact/fuzzy matching.
Stage 2: LLM-based semantic equivalence judge (Qwen2.5-7B-Instruct).
"""
import re
import numpy as np


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # Remove articles
    for article in ['a ', 'an ', 'the ']:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove trailing punctuation
    answer = answer.rstrip('.,;:!?')
    # Normalize whitespace
    answer = ' '.join(answer.split())
    # Normalize common patterns
    answer = answer.replace("'s", "s").replace("'", "")
    # Normalize numbers written as words
    num_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12'
    }
    if answer in num_words:
        answer = num_words[answer]
    return answer


def rule_based_match(answer1: str, answer2: str) -> bool:
    """Stage 1: Rule-based matching with normalization and fuzzy matching."""
    from fuzzywuzzy import fuzz

    n1 = normalize_answer(answer1)
    n2 = normalize_answer(answer2)

    if not n1 or not n2:
        return False

    # Exact match after normalization
    if n1 == n2:
        return True

    # One contains the other
    if n1 in n2 or n2 in n1:
        return True

    # Fuzzy match
    if fuzz.ratio(n1, n2) > 85:
        return True

    # Partial ratio for substring matching
    if fuzz.partial_ratio(n1, n2) > 90:
        return True

    return False


def build_judge_prompts(pairs, tokenizer=None):
    """
    Build judge prompts for LLM-based equivalence checking.
    pairs: list of (question_text, answer1, answer2) tuples
    Returns: list of formatted prompt strings
    """
    prompts = []
    for question, ans1, ans2 in pairs:
        prompt = (
            f"You are an answer equivalence judge. Given a question and two answers, "
            f"determine if they are semantically equivalent (convey the same core information).\n\n"
            f"Question: {question}\n"
            f"Answer A: {ans1}\n"
            f"Answer B: {ans2}\n\n"
            f"Are these two answers semantically equivalent? Consider them equivalent if they "
            f"convey the same core fact or value, even if phrased differently (e.g., 'Paris' "
            f"and 'The capital is Paris' are equivalent). Respond with only EQUIVALENT or NOT_EQUIVALENT."
        )
        if tokenizer is not None:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(formatted)
            except Exception:
                prompts.append(prompt)
        else:
            prompts.append(prompt)
    return prompts


def parse_judge_output(output: str) -> bool:
    """Parse LLM judge output into equivalence verdict."""
    output = output.strip().upper()
    if "NOT_EQUIVALENT" in output or "NOT EQUIVALENT" in output:
        return False
    if "EQUIVALENT" in output:
        return True
    # Fallback: check for yes/no
    if output.startswith("YES") or output.startswith("EQUIV"):
        return True
    return False


class AnswerEquivalenceJudge:
    """
    Two-stage answer equivalence judge.
    Uses rule-based matching first, then LLM judge for uncertain cases.
    """

    def __init__(self, llm=None, tokenizer=None):
        """
        llm: vLLM LLM instance (for batch generation)
        tokenizer: tokenizer for formatting prompts
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.judge_cache = {}
        self.stats = {
            'rule_based_matches': 0,
            'llm_judge_calls': 0,
            'llm_equivalent': 0,
            'llm_not_equivalent': 0,
        }

    def check_equivalence_batch(self, items, use_llm=True):
        """
        Check equivalence for a batch of items.
        items: list of (question_text, answer1, answer2, format_type) tuples
        Returns: list of bool (equivalent or not)
        """
        from vllm import SamplingParams

        results = [None] * len(items)
        llm_needed = []

        # Stage 1: Rule-based matching
        for i, (question, ans1, ans2, fmt) in enumerate(items):
            # For constrained formats (mcq, yesno, truefalse), use exact comparison
            if fmt in ('mcq', 'yesno', 'truefalse'):
                results[i] = normalize_answer(ans1) == normalize_answer(ans2)
                self.stats['rule_based_matches'] += 1
            elif rule_based_match(ans1, ans2):
                results[i] = True
                self.stats['rule_based_matches'] += 1
            elif use_llm and self.llm is not None:
                # Need LLM judge
                cache_key = (normalize_answer(ans1), normalize_answer(ans2))
                if cache_key in self.judge_cache:
                    results[i] = self.judge_cache[cache_key]
                    self.stats['rule_based_matches'] += 1
                else:
                    llm_needed.append((i, question, ans1, ans2))
            else:
                results[i] = False

        # Stage 2: LLM judge for remaining items
        if llm_needed and self.llm is not None:
            pairs = [(q, a1, a2) for _, q, a1, a2 in llm_needed]
            judge_prompts = build_judge_prompts(pairs, self.tokenizer)

            sampling_params = SamplingParams(
                temperature=0.0, max_tokens=20, top_p=1.0
            )
            outputs = self.llm.generate(judge_prompts, sampling_params)

            for j, (idx, question, ans1, ans2) in enumerate(llm_needed):
                verdict = parse_judge_output(outputs[j].outputs[0].text)
                results[idx] = verdict
                self.stats['llm_judge_calls'] += 1
                if verdict:
                    self.stats['llm_equivalent'] += 1
                else:
                    self.stats['llm_not_equivalent'] += 1
                # Cache
                cache_key = (normalize_answer(ans1), normalize_answer(ans2))
                self.judge_cache[cache_key] = verdict

        return results

    def check_correctness_batch(self, items):
        """
        Check correctness for a batch of items.
        items: list of (question_text, extracted_answer, correct_answer, format_type) tuples
        Returns: list of bool
        """
        return self.check_equivalence_batch(items, use_llm=True)

    def get_stats(self):
        return dict(self.stats)
