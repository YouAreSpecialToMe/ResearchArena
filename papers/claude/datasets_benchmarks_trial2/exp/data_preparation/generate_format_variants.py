"""
Step 2: Generate cross-format variants for all 1000 base questions.
5 formats: MCQ, open-ended, yes/no, true/false, fill-in-the-blank
Total: 5000 instances
"""
import json
import os
import random
import re

SEED = 42
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DATA_DIR, 'base_questions.json'), 'r') as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} base questions")

format_variants = []
yesno_mapping = {}  # Track yes/no ground truth
truefalse_mapping = {}  # Track true/false ground truth


def generate_distractors(question, correct_answer, existing_choices=None):
    """Generate 3 plausible distractors for MCQ format."""
    if existing_choices and len(existing_choices) >= 4:
        # Already has choices, return them
        return existing_choices[:4]

    # Simple distractor generation based on answer type
    distractors = []

    # Try numeric distractors for numbers
    try:
        num = float(correct_answer.replace(',', ''))
        offsets = [num * 0.5, num * 1.5, num + 10, num - 10, num * 2, num + 1, num - 1]
        random.shuffle(offsets)
        for off in offsets:
            if off != num and len(distractors) < 3:
                if num == int(num):
                    distractors.append(str(int(off)))
                else:
                    distractors.append(f"{off:.2f}")
    except (ValueError, TypeError):
        pass

    # Generic distractors if we don't have enough
    generic_pools = {
        'science': ['oxygen', 'nitrogen', 'carbon', 'hydrogen', 'helium', 'iron',
                     'gravity', 'friction', 'inertia', 'momentum', 'acceleration',
                     'mitosis', 'meiosis', 'osmosis', 'diffusion', 'photosynthesis'],
        'history': ['1776', '1492', '1066', '1945', '1812', '1914',
                    'Rome', 'Athens', 'Constantinople', 'Babylon', 'Alexandria',
                    'Napoleon', 'Caesar', 'Alexander', 'Lincoln', 'Churchill'],
        'commonsense': ['kitchen', 'bedroom', 'office', 'garden', 'school',
                       'happy', 'sad', 'angry', 'tired', 'excited',
                       'running', 'walking', 'sleeping', 'eating', 'reading'],
        'world_knowledge': ['London', 'Paris', 'Tokyo', 'Berlin', 'Moscow',
                           'Einstein', 'Newton', 'Darwin', 'Curie', 'Pasteur'],
        'logic': ['All of the above', 'None of the above', 'Cannot be determined',
                  'Insufficient information', 'Both A and B'],
        'math': ['0', '1', '2', '10', '100', '42', '7', '12', '24', '36'],
    }

    while len(distractors) < 3:
        # Pick from generic pool, avoiding correct answer
        pool = []
        for domain_pool in generic_pools.values():
            pool.extend(domain_pool)
        random.shuffle(pool)
        for item in pool:
            if item.lower() != correct_answer.lower() and item not in distractors:
                distractors.append(item)
                break
        else:
            distractors.append(f"Option {len(distractors) + 1}")

    # Combine correct + distractors and shuffle
    all_choices = [correct_answer] + distractors[:3]
    random.shuffle(all_choices)
    return all_choices


def make_statement_from_question(question_text, answer):
    """Convert a question + answer into a declarative statement."""
    q = question_text.strip().rstrip('?')

    # Simple patterns
    patterns = [
        (r'^What is (.+)', f'{answer} is \\1'),
        (r'^What are (.+)', f'{answer} are \\1'),
        (r'^Who is (.+)', f'{answer} is \\1'),
        (r'^Who was (.+)', f'{answer} was \\1'),
        (r'^Where is (.+)', f'{answer} is \\1'),
        (r'^When did (.+)', f'{answer} is when \\1'),
        (r'^How many (.+)', f'The number of \\1 is {answer}'),
    ]

    for pattern, replacement in patterns:
        match = re.match(pattern, q, re.IGNORECASE)
        if match:
            try:
                result = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
                return result + '.'
            except:
                pass

    # Fallback: simple concatenation
    return f"The answer to \"{question_text}\" is {answer}."


def make_cloze(question_text, answer):
    """Convert question + answer into fill-in-the-blank."""
    statement = make_statement_from_question(question_text, answer)
    # Replace the answer in the statement with a blank
    if answer.lower() in statement.lower():
        idx = statement.lower().index(answer.lower())
        cloze = statement[:idx] + '____' + statement[idx + len(answer):]
        return cloze
    return f"The answer to \"{question_text}\" is ____."


for q in questions:
    qid = q['question_id']
    qt = q['question_text']
    ca = q['correct_answer']
    domain = q['domain']
    choices = q.get('original_choices', None)

    # 1. MCQ format
    mcq_choices = generate_distractors(qt, ca, choices)
    correct_letter = chr(65 + mcq_choices.index(ca)) if ca in mcq_choices else 'A'
    options_text = '\n'.join([f'{chr(65+i)}) {c}' for i, c in enumerate(mcq_choices)])
    format_variants.append({
        'question_id': qid,
        'format_type': 'mcq',
        'prompt_text': f"Question: {qt}\n{options_text}\nRespond with only the letter (A, B, C, or D).",
        'correct_answer': correct_letter,
        'format_metadata': {'choices': mcq_choices, 'correct_letter': correct_letter},
    })

    # 2. Open-ended format
    format_variants.append({
        'question_id': qid,
        'format_type': 'open',
        'prompt_text': f"Question: {qt}\nRespond with only the answer, no explanation.",
        'correct_answer': ca,
        'format_metadata': {},
    })

    # 3. Yes/No format - 50% use correct answer (yes), 50% use wrong answer (no)
    use_correct = random.random() < 0.5
    if use_correct:
        statement = make_statement_from_question(qt, ca)
        yn_answer = 'yes'
    else:
        # Generate a wrong answer for the no case
        wrong_answers = ['incorrect_placeholder', 'wrong_answer']
        if choices:
            wrong_opts = [c for c in choices if c.lower() != ca.lower()]
            if wrong_opts:
                wrong = random.choice(wrong_opts)
            else:
                wrong = 'something else'
        else:
            wrong = 'something else'
        statement = make_statement_from_question(qt, wrong)
        yn_answer = 'no'

    format_variants.append({
        'question_id': qid,
        'format_type': 'yesno',
        'prompt_text': f"Is it true that {statement.rstrip('.')}? Answer yes or no.\nRespond with only Yes or No.",
        'correct_answer': yn_answer,
        'format_metadata': {'uses_correct': use_correct},
    })

    # 4. True/False format
    use_true = random.random() < 0.5
    if use_true:
        tf_statement = make_statement_from_question(qt, ca)
        tf_answer = 'true'
    else:
        if choices:
            wrong_opts = [c for c in choices if c.lower() != ca.lower()]
            if wrong_opts:
                wrong = random.choice(wrong_opts)
            else:
                wrong = 'something incorrect'
        else:
            wrong = 'something incorrect'
        tf_statement = make_statement_from_question(qt, wrong)
        tf_answer = 'false'

    format_variants.append({
        'question_id': qid,
        'format_type': 'truefalse',
        'prompt_text': f"True or False: {tf_statement}\nRespond with only True or False.",
        'correct_answer': tf_answer,
        'format_metadata': {'uses_correct': use_true},
    })

    # 5. Fill-in-the-blank format
    cloze = make_cloze(qt, ca)
    format_variants.append({
        'question_id': qid,
        'format_type': 'fitb',
        'prompt_text': f"Complete the following: {cloze}\nRespond with only the word or phrase that fills the blank.",
        'correct_answer': ca,
        'format_metadata': {},
    })

# Validate
format_counts = {}
for v in format_variants:
    ft = v['format_type']
    format_counts[ft] = format_counts.get(ft, 0) + 1

print(f"\n=== Format Variant Statistics ===")
print(f"Total instances: {len(format_variants)}")
print(f"Per format: {json.dumps(format_counts, indent=2)}")
assert len(format_variants) == 5000, f"Expected 5000, got {len(format_variants)}"

# Save
out_path = os.path.join(DATA_DIR, 'format_variants.json')
with open(out_path, 'w') as f:
    json.dump(format_variants, f, indent=2)
print(f"Saved to {out_path}")
