"""
Data loading and preprocessing utilities for multi-hop QA experiments.
"""
import json
import random
import re
from typing import List, Dict, Tuple, Any
from collections import Counter
import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def load_hotpotqa_data(filepath: str, max_samples: int = None) -> List[Dict]:
    """Load HotpotQA data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        # Sample stratified by type if possible
        bridge = [d for d in data if d.get('type') == 'bridge'][:max_samples//2]
        comparison = [d for d in data if d.get('type') == 'comparison'][:max_samples//2]
        data = bridge + comparison
        data = data[:max_samples]
    
    return data


def load_2wikimultihopqa_data(filepath: str, max_samples: int = None) -> List[Dict]:
    """Load 2WikiMultiHopQA data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def prepare_qa_samples(data: List[Dict], dataset_type: str = 'hotpotqa') -> List[Dict]:
    """Prepare QA samples in a standard format."""
    samples = []
    
    for item in data:
        if dataset_type == 'hotpotqa':
            sample = {
                'id': item['_id'],
                'question': item['question'],
                'answer': item['answer'],
                'type': item.get('type', 'unknown'),
                'supporting_facts': item.get('supporting_facts', []),
                'context': item.get('context', [])
            }
        elif dataset_type == '2wikimultihopqa':
            sample = {
                'id': item.get('_id', item.get('id')),
                'question': item['question'],
                'answer': item['answer'],
                'type': item.get('type', 'unknown'),
                'supporting_facts': item.get('supporting_facts', []),
                'context': item.get('context', [])
            }
        else:
            sample = item
        
        samples.append(sample)
    
    return samples


def split_data(data: List[Dict], train_ratio=0.6, val_ratio=0.2, seed=42) -> Tuple[List, List, List]:
    """Split data into train/val/test."""
    random.seed(seed)
    np.random.seed(seed)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def create_synthetic_multihop_data(num_samples: int = 800, seed: int = 42) -> Tuple[List, List, List]:
    """
    Create synthetic multi-hop QA data for experiments.
    Used when real datasets are not available.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Templates for 2-hop questions
    templates = [
        {
            'template': "What award did the director of {movie} win?",
            'entities': [["Titanic", "Avatar"], ["The Godfather", "Apocalypse Now"], ["Jaws", "E.T."]],
            'answers': ["Academy Award", "Golden Globe", "Oscar"]
        },
        {
            'template': "Who is the spouse of the CEO of {company}?",
            'entities': [["Apple", "Microsoft"], ["Google", "Amazon"], ["Tesla", "Meta"]],
            'answers': ["Person A", "Person B", "Person C"]
        },
        {
            'template': "What is the capital of the country where {city} is located?",
            'entities': [["Paris", "Berlin"], ["Tokyo", "Beijing"], ["London", "Madrid"]],
            'answers': ["France", "Germany", "Japan"]
        },
        {
            'template': "When did the author of {book} die?",
            'entities': [["1984", "Animal Farm"], ["Hamlet", "Macbeth"], ["Pride and Prejudice", "Emma"]],
            'answers': ["1950", "1616", "1817"]
        },
        {
            'template': "What university did the inventor of {invention} attend?",
            'entities': [["light bulb", "phonograph"], ["telephone", "airplane"], ["computer", "internet"]],
            'answers': ["MIT", "Harvard", "Stanford"]
        }
    ]
    
    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        entity_pair = random.choice(template['entities'])
        entity = random.choice(entity_pair)
        answer = random.choice(template['answers'])
        
        question = template['template'].format(movie=entity, company=entity, city=entity, 
                                                book=entity, invention=entity)
        
        # Create synthetic context passages
        context = [
            [f"{entity} is a famous work/product created by Person X."],
            [f"Person X studied at {answer} and made significant contributions."],
            [f"The {entity} was developed over many years with great effort."],
            [f"Person X is well known for their expertise in the field."],
            [f"The institution {answer} is located in a major city."],
            [f"Many students attend {answer} each year from around the world."]
        ]
        
        # Bridge entity (for 2-hop reasoning)
        bridge_entity = f"Person X"
        
        data.append({
            '_id': f'syn_{i}',
            'question': question,
            'answer': answer,
            'type': random.choice(['bridge', 'comparison']),
            'context': context,
            'bridge_entity': bridge_entity,
            'supporting_facts': [[context[0][0], 0], [context[1][0], 1]]
        })
    
    return split_data(data, train_ratio=0.6, val_ratio=0.2, seed=seed)


def extract_entities_from_text(text: str) -> List[str]:
    """Simple heuristic entity extraction (placeholder for NER)."""
    # Simple extraction of capitalized phrases as entities
    words = text.split()
    entities = []
    current_entity = []
    
    for word in words:
        if word and word[0].isupper():
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
    
    if current_entity:
        entities.append(' '.join(current_entity))
    
    return entities[:5]  # Limit to top 5 entities


if __name__ == '__main__':
    # Test the data loader
    train, val, test = create_synthetic_multihop_data(100)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Sample question: {train[0]['question']}")
    print(f"Sample answer: {train[0]['answer']}")
