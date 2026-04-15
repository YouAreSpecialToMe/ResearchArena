"""
Step 1: Curate ~1000 base questions from 6 domains using open QA datasets.
Domains: Science, History, Math, Commonsense, World Knowledge, Logic
"""
import json
import os
import random
import re
from datasets import load_dataset

SEED = 42
random.seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(OUTPUT_DIR))

questions = []
qid = 0


def add_questions(items, domain, source, target_count):
    global qid
    random.shuffle(items)
    added = 0
    for item in items:
        if added >= target_count:
            break
        q = item.get('question', '').strip()
        a = item.get('answer', '').strip()
        if not q or not a:
            continue
        if len(a) > 100:  # Skip very long answers
            continue
        qid += 1
        questions.append({
            'question_id': f'Q{qid:04d}',
            'domain': domain,
            'source_dataset': source,
            'question_text': q,
            'correct_answer': a,
            'difficulty': item.get('difficulty', 'medium'),
            'original_choices': item.get('choices', None),
        })
        added += 1
    print(f"  {domain}/{source}: added {added} questions")
    return added


print("=== Curating base questions ===\n")

# 1. Science from ARC-Challenge
print("Loading ARC-Challenge...")
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
arc_items = []
for row in arc:
    choices = row['choices']['text']
    labels = row['choices']['label']
    answer_key = row['answerKey']
    correct_idx = labels.index(answer_key) if answer_key in labels else 0
    correct_answer = choices[correct_idx]
    arc_items.append({
        'question': row['question'],
        'answer': correct_answer,
        'choices': choices,
        'difficulty': 'medium',
    })
random.seed(SEED)
add_questions(arc_items, 'science', 'arc_challenge', 165)

# 2. History/Social Science from MMLU
print("Loading MMLU...")
try:
    mmlu = load_dataset("cais/mmlu", "all", split="test")
except:
    mmlu = load_dataset("lukaemon/mmlu", "all", split="test")

history_subjects = ['high_school_us_history', 'high_school_world_history', 'high_school_european_history',
                    'prehistory', 'world_religions', 'philosophy', 'moral_scenarios', 'moral_disputes',
                    'us_foreign_policy', 'sociology', 'high_school_government_and_politics',
                    'public_relations', 'human_sexuality', 'global_facts']

mmlu_history = []
for row in mmlu:
    subj = row.get('subject', '')
    if subj in history_subjects:
        choices = [row['choices'][i] for i in range(4)] if isinstance(row['choices'], list) else [row['A'], row['B'], row['C'], row['D']]
        answer_idx = row['answer'] if isinstance(row['answer'], int) else ord(row['answer']) - ord('A')
        if 0 <= answer_idx < len(choices):
            mmlu_history.append({
                'question': row['question'],
                'answer': choices[answer_idx],
                'choices': choices,
                'difficulty': 'medium',
            })
random.seed(SEED)
add_questions(mmlu_history, 'history', 'mmlu', 165)

# 3. Math from GSM8K
print("Loading GSM8K...")
gsm = load_dataset("openai/gsm8k", "main", split="test")
gsm_items = []
for row in gsm:
    # Extract numeric answer from "#### NUMBER"
    answer_text = row['answer']
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        answer = match.group(1).strip()
        # Determine difficulty by number of steps
        steps = answer_text.count('\n')
        if steps <= 2:
            diff = 'easy'
        elif steps <= 4:
            diff = 'medium'
        else:
            diff = 'hard'
        gsm_items.append({
            'question': row['question'],
            'answer': answer,
            'difficulty': diff,
        })
random.seed(SEED)
add_questions(gsm_items, 'math', 'gsm8k', 170)

# 4. Commonsense from CommonsenseQA
print("Loading CommonsenseQA...")
csqa = load_dataset("tau/commonsense_qa", split="validation")
csqa_items = []
for row in csqa:
    choices = row['choices']['text']
    labels = row['choices']['label']
    answer_key = row['answerKey']
    if answer_key in labels:
        correct_idx = labels.index(answer_key)
        csqa_items.append({
            'question': row['question'],
            'answer': choices[correct_idx],
            'choices': choices,
            'difficulty': 'medium',
        })
random.seed(SEED)
add_questions(csqa_items, 'commonsense', 'commonsense_qa', 165)

# 5. World Knowledge from TriviaQA
print("Loading TriviaQA...")
tqa = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
tqa_items = []
for row in tqa:
    answer = row['answer']['value'] if isinstance(row['answer'], dict) else str(row['answer'])
    if len(answer) <= 50:
        tqa_items.append({
            'question': row['question'],
            'answer': answer,
            'difficulty': 'medium',
        })
random.seed(SEED)
add_questions(tqa_items, 'world_knowledge', 'trivia_qa', 170)

# 6. Logic from LogiQA
print("Loading LogiQA...")
try:
    logiqa = load_dataset("lucasmccabe/logiqa", split="test")
    logiqa_items = []
    for row in logiqa:
        options = row['options']
        answer_idx = row['correct_option']
        if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
            context = row.get('context', '')
            question = row.get('query', '')
            if context:
                question = f"{context}\n{question}"
            if len(question) < 500:
                logiqa_items.append({
                    'question': question,
                    'answer': options[answer_idx],
                    'choices': options,
                    'difficulty': 'hard',
                })
    random.seed(SEED)
    add_questions(logiqa_items, 'logic', 'logiqa', 165)
except Exception as e:
    print(f"LogiQA loading failed ({e}), using MMLU logic subjects instead")
    logic_subjects = ['logical_fallacies', 'formal_logic', 'abstract_algebra',
                      'machine_learning', 'computer_science', 'high_school_mathematics',
                      'elementary_mathematics', 'college_mathematics']
    mmlu_logic = []
    for row in mmlu:
        subj = row.get('subject', '')
        if subj in logic_subjects:
            choices = [row['choices'][i] for i in range(4)] if isinstance(row['choices'], list) else [row['A'], row['B'], row['C'], row['D']]
            answer_idx = row['answer'] if isinstance(row['answer'], int) else ord(row['answer']) - ord('A')
            if 0 <= answer_idx < len(choices):
                mmlu_logic.append({
                    'question': row['question'],
                    'answer': choices[answer_idx],
                    'choices': choices,
                    'difficulty': 'medium',
                })
    random.seed(SEED)
    add_questions(mmlu_logic, 'logic', 'mmlu_logic', 165)

# Assign difficulty more carefully if not already assigned
difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
for q in questions:
    difficulty_counts[q['difficulty']] += 1

# Reassign to balance if needed
if difficulty_counts['easy'] < 200 or difficulty_counts['hard'] < 200:
    random.seed(SEED)
    n = len(questions)
    third = n // 3
    indices = list(range(n))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        if i < third:
            questions[idx]['difficulty'] = 'easy'
        elif i < 2 * third:
            questions[idx]['difficulty'] = 'medium'
        else:
            questions[idx]['difficulty'] = 'hard'

# Stats
print(f"\n=== Dataset Statistics ===")
print(f"Total questions: {len(questions)}")
domain_counts = {}
diff_counts = {'easy': 0, 'medium': 0, 'hard': 0}
for q in questions:
    domain_counts[q['domain']] = domain_counts.get(q['domain'], 0) + 1
    diff_counts[q['difficulty']] = diff_counts.get(q['difficulty'], 0) + 1

print(f"Per domain: {json.dumps(domain_counts, indent=2)}")
print(f"Per difficulty: {json.dumps(diff_counts, indent=2)}")
avg_len = sum(len(q['question_text']) for q in questions) / len(questions)
print(f"Average question length: {avg_len:.1f} chars")

# Save
out_path = os.path.join(OUTPUT_DIR, 'base_questions.json')
with open(out_path, 'w') as f:
    json.dump(questions, f, indent=2)
print(f"\nSaved {len(questions)} questions to {out_path}")

stats = {
    'total_questions': len(questions),
    'domain_counts': domain_counts,
    'difficulty_counts': diff_counts,
    'avg_question_length': avg_len,
}
stats_path = os.path.join(OUTPUT_DIR, 'dataset_stats.json')
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Saved stats to {stats_path}")
