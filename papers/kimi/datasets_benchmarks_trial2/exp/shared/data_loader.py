"""
Data loading utilities for EVOLVE experiments.
"""

import json
import os
from typing import List, Dict
import numpy as np


def load_questions(filepath: str) -> List[Dict]:
    """Load questions from JSON file."""
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        return []


def create_synthetic_questions(n_mmlu: int = 1500, n_gsm8k: int = 700, seed: int = 42) -> tuple:
    """
    Create synthetic question datasets.
    Returns: (mmlu_questions, gsm8k_questions)
    """
    np.random.seed(seed)
    
    # MMLU-style questions (multiple choice)
    mmlu_subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
        'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
        'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]
    
    mmlu_questions = []
    for i in range(n_mmlu):
        subject = np.random.choice(mmlu_subjects)
        mmlu_questions.append({
            'id': f'mmlu_{i}',
            'subject': subject,
            'type': 'multiple_choice'
        })
    
    # GSM8K-style questions (math word problems)
    gsm8k_questions = []
    for i in range(n_gsm8k):
        gsm8k_questions.append({
            'id': f'gsm8k_{i}',
            'subject': 'grade_school_math',
            'type': 'math'
        })
    
    return mmlu_questions, gsm8k_questions


def save_questions(questions: List[Dict], filepath: str):
    """Save questions to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(questions, f, indent=2)


def ensure_data_exists(data_dir: str = 'data'):
    """Ensure data files exist, creating synthetic data if needed."""
    os.makedirs(data_dir, exist_ok=True)
    
    mmlu_path = os.path.join(data_dir, 'mmlu_test.json')
    gsm8k_path = os.path.join(data_dir, 'gsm8k_test.json')
    
    if not os.path.exists(mmlu_path) or not os.path.exists(gsm8k_path):
        print("Creating synthetic question datasets...")
        mmlu_qs, gsm8k_qs = create_synthetic_questions()
        
        if not os.path.exists(mmlu_path):
            save_questions(mmlu_qs, mmlu_path)
            print(f"  Created {mmlu_path} with {len(mmlu_qs)} questions")
        
        if not os.path.exists(gsm8k_path):
            save_questions(gsm8k_qs, gsm8k_path)
            print(f"  Created {gsm8k_path} with {len(gsm8k_qs)} questions")
