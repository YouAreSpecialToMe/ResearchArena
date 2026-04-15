"""Answer parser for FlipBench evaluation outputs."""

import re


def parse_answer(raw_output, domain, direction=None):
    """Extract the final answer from model output.

    Returns (parsed_answer, parse_success).
    """
    if not raw_output or not raw_output.strip():
        return None, False

    text = raw_output.strip()

    # Try to find "ANSWER: ..." pattern
    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # Fallback: try <answer>...</answer> tags (for R1 models)
        tag_match = re.search(r'<answer>\s*(.+?)\s*</answer>', text, re.IGNORECASE)
        if tag_match:
            answer_text = tag_match.group(1).strip()
        else:
            # For R1/reasoning models: try conclusion patterns
            conclusion_patterns = [
                r'(?:the answer is|therefore,?\s*the answer is|so the answer is)\s+["\']?([^"\'\n,]+)',
                r'(?:so|therefore|thus|hence),?\s+(?:the (?:answer|result) is\s+)?["\']?(\w+)',
            ]
            answer_text = None
            for pat in conclusion_patterns:
                matches = re.findall(pat, text, re.IGNORECASE)
                if matches:
                    answer_text = matches[-1].strip()
                    break

            if not answer_text:
                # Final fallback: use the last non-empty line
                lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
                if lines:
                    answer_text = lines[-1]
                    for prefix in ['The answer is ', 'Answer: ', 'Result: ', 'So, ', 'Therefore, ']:
                        if answer_text.lower().startswith(prefix.lower()):
                            answer_text = answer_text[len(prefix):].strip()
                else:
                    return None, False

    # Domain-specific parsing
    if domain == 'propositional_logic':
        return parse_logic_answer(answer_text)
    elif domain == 'arithmetic_reasoning':
        return parse_numeric_answer(answer_text)
    elif domain == 'relational_reasoning':
        return parse_relational_answer(answer_text)
    elif domain == 'function_computation':
        return parse_numeric_answer(answer_text)
    else:
        return answer_text, True


def parse_logic_answer(text):
    """Parse True/False answer for logic domain.

    Both forward and backward questions now use True/False answers.
    """
    text_clean = text.strip().lower().rstrip('.').strip()

    # Direct True/False
    if text_clean in ('true', 'yes'):
        return 'True', True
    if text_clean in ('false', 'no'):
        return 'False', True

    # Check for patterns in longer text
    # "is true" / "must be true" / "the answer is true"
    if re.search(r'\b(?:is true|must (?:be|have been) true|yes,?\s)', text_clean):
        return 'True', True
    if re.search(r'\b(?:is false|not (?:necessarily |)true|cannot|no,?\s|must not)', text_clean):
        return 'False', True

    # Check if the text ends with true/false
    if text_clean.endswith('true'):
        return 'True', True
    if text_clean.endswith('false'):
        return 'False', True

    # Last resort: look for true/false anywhere
    true_count = len(re.findall(r'\btrue\b', text_clean))
    false_count = len(re.findall(r'\bfalse\b', text_clean))
    if true_count > 0 and false_count == 0:
        return 'True', True
    if false_count > 0 and true_count == 0:
        return 'False', True

    # If we still can't determine, check for "not" as indicator of False
    if 'not' in text_clean or 'cannot' in text_clean or "doesn't" in text_clean:
        return 'False', True

    return text.strip(), False


def parse_numeric_answer(text):
    """Parse numeric answer."""
    text = text.strip().rstrip('.')
    # Extract number (possibly negative)
    num_match = re.search(r'(-?\d+)', text)
    if num_match:
        return num_match.group(1), True
    return text, False


def parse_relational_answer(text):
    """Parse relational reasoning answer (relationship name)."""
    text = text.strip().rstrip('.')

    # Check for known relationships (order matters: check compound first)
    relationships = [
        'great-grandparent', 'great-grandchild',
        'grandparent', 'grandchild',
        'uncle/aunt', 'uncle', 'aunt',
        'cousin',
        'parent', 'child', 'sibling',
    ]
    text_lower = text.lower()
    for rel in relationships:
        if rel in text_lower:
            return rel, True

    # Check for names (capitalized words) — only for legacy compatibility
    name_match = re.search(r'\b([A-Z][a-z]+)\b', text)
    if name_match:
        return name_match.group(1), True

    return text, True


def check_answer(parsed, gold, domain, direction):
    """Check if parsed answer matches gold answer."""
    if parsed is None:
        return False

    parsed_str = str(parsed).strip().lower()
    gold_str = str(gold).strip().lower()

    if domain == 'propositional_logic':
        # Both directions are now True/False
        parsed_bool = parsed_str in ('true', 'yes')
        gold_bool = gold_str in ('true', 'yes')
        if parsed_str in ('true', 'yes', 'false', 'no'):
            return parsed_bool == gold_bool
        return parsed_str == gold_str

    elif domain in ('arithmetic_reasoning', 'function_computation'):
        try:
            return int(parsed_str) == int(gold_str)
        except ValueError:
            return parsed_str == gold_str

    elif domain == 'relational_reasoning':
        # Fuzzy match for relationships
        if gold_str in parsed_str or parsed_str in gold_str:
            return True
        # Handle uncle/aunt
        if gold_str == 'uncle/aunt' and parsed_str in ('uncle', 'aunt', 'uncle/aunt'):
            return True
        return parsed_str == gold_str

    return parsed_str == gold_str
