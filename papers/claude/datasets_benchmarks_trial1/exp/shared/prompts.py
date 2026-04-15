"""Prompt templates for FlipBench evaluation."""

SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. Answer the question below. "
    "Give your final answer on the last line in the format: ANSWER: <your answer>"
)

SYSTEM_PROMPT_COT = (
    "You are a precise reasoning assistant. Answer the question below. "
    "Think step by step before giving your final answer. "
    "Give your final answer on the last line in the format: ANSWER: <your answer>"
)


def format_prompt(instance, use_cot=False):
    """Format a FlipBench instance into a prompt."""
    system = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT
    user_msg = instance['problem_text']
    return system, user_msg


def format_chat_messages(instance, use_cot=False):
    """Format as chat messages for vLLM."""
    system, user = format_prompt(instance, use_cot)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
