"""
Shared utilities for ConsistBench experiments.
Model loading, prompt formatting, answer extraction.
"""
import re
import json
import os

# Format-specific instruction suffixes
FORMAT_INSTRUCTIONS = {
    'mcq': 'Respond with only the letter (A, B, C, or D).',
    'open': 'Respond with only the answer, no explanation.',
    'yesno': 'Respond with only Yes or No.',
    'truefalse': 'Respond with only True or False.',
    'fitb': 'Respond with only the word or phrase that fills the blank.',
}

SYSTEM_PROMPT = 'You are a helpful assistant. Answer the question directly and concisely. Follow the specified answer format exactly.'

# Model configurations
MODEL_CONFIGS = [
    {
        'name': 'Phi-3.5-mini',
        'model_id': 'microsoft/Phi-3.5-mini-instruct',
        'size_b': 3.8,
        'family': 'Phi',
        'gpu_mem_gb': 8,
        'quantization': None,
    },
    {
        'name': 'Mistral-7B',
        'model_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'size_b': 7.0,
        'family': 'Mistral',
        'gpu_mem_gb': 15,
        'quantization': None,
    },
    {
        'name': 'Qwen2.5-7B',
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'size_b': 7.0,
        'family': 'Qwen',
        'gpu_mem_gb': 15,
        'quantization': None,
    },
    {
        'name': 'Llama-3.1-8B',
        'model_id': 'meta-llama/Llama-3.1-8B-Instruct',
        'size_b': 8.0,
        'family': 'Llama',
        'gpu_mem_gb': 17,
        'quantization': None,
    },
    {
        'name': 'Qwen2.5-14B',
        'model_id': 'Qwen/Qwen2.5-14B-Instruct',
        'size_b': 14.0,
        'family': 'Qwen',
        'gpu_mem_gb': 30,
        'quantization': None,
    },
    {
        'name': 'Llama-3.1-70B-AWQ',
        'model_id': 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
        'size_b': 70.0,
        'family': 'Llama',
        'gpu_mem_gb': 40,
        'quantization': 'awq',
    },
]


def format_prompt(question_text: str, format_type: str, choices=None):
    """Format a question with format-specific instructions."""
    instruction = FORMAT_INSTRUCTIONS[format_type]
    if format_type == 'mcq' and choices:
        options_text = '\n'.join([f'{chr(65+i)}) {c}' for i, c in enumerate(choices)])
        prompt = f"Question: {question_text}\n{options_text}\n{instruction}"
    else:
        prompt = f"{question_text}\n{instruction}"
    return prompt


def extract_answer(raw_output: str, format_type: str) -> str:
    """Extract answer from model output based on format type."""
    raw = raw_output.strip()
    if not raw:
        return ""

    if format_type == 'mcq':
        # Extract first letter A-D
        match = re.search(r'\b([A-D])\b', raw.upper())
        if match:
            return match.group(1)
        # Try first character
        if raw[0].upper() in 'ABCD':
            return raw[0].upper()
        return raw[:1].upper()

    elif format_type == 'yesno':
        raw_lower = raw.lower()
        if 'yes' in raw_lower:
            return 'yes'
        elif 'no' in raw_lower:
            return 'no'
        return raw_lower.split()[0] if raw_lower.split() else ''

    elif format_type == 'truefalse':
        raw_lower = raw.lower()
        if 'true' in raw_lower:
            return 'true'
        elif 'false' in raw_lower:
            return 'false'
        return raw_lower.split()[0] if raw_lower.split() else ''

    elif format_type in ('open', 'fitb'):
        # Return full response stripped and lowered
        answer = raw.strip().lower()
        # Take first sentence/line if multi-line
        answer = answer.split('\n')[0].strip()
        # Limit length
        if len(answer) > 200:
            answer = answer[:200]
        return answer

    return raw.strip().lower()


def check_correctness(extracted_answer: str, correct_answer: str, format_type: str) -> bool:
    """Check if extracted answer is correct for the given format."""
    if format_type == 'mcq':
        return extracted_answer.upper() == correct_answer.upper()
    elif format_type in ('yesno', 'truefalse'):
        return extracted_answer.lower() == correct_answer.lower()
    else:
        # For open and fitb, use normalized comparison
        from .metrics import normalize_answer
        return normalize_answer(extracted_answer) == normalize_answer(correct_answer)


def check_correctness_fuzzy(extracted_answer: str, correct_answer: str, format_type: str) -> bool:
    """Check correctness with fuzzy matching for open-ended formats."""
    if format_type in ('mcq', 'yesno', 'truefalse'):
        return check_correctness(extracted_answer, correct_answer, format_type)

    from .metrics import normalize_answer
    from fuzzywuzzy import fuzz

    norm_ext = normalize_answer(extracted_answer)
    norm_cor = normalize_answer(correct_answer)

    # Exact match after normalization
    if norm_ext == norm_cor:
        return True

    # Check if correct answer is contained in the response
    if norm_cor in norm_ext:
        return True

    # Fuzzy match
    if fuzz.ratio(norm_ext, norm_cor) > 85:
        return True

    # Partial ratio for when answer is a substring
    if fuzz.partial_ratio(norm_ext, norm_cor) > 90:
        return True

    return False


def save_json(data, path):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    """Load JSON data."""
    with open(path, 'r') as f:
        return json.load(f)
