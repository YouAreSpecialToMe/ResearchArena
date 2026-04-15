"""
Data loading and preprocessing utilities for PopBench experiments.
CRITICAL: Fixed seed propagation throughout the pipeline.
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class ModelMetadata:
    """Metadata for a language model."""
    name: str
    family: str  # llama2, llama3, qwen2, qwen3, gemma, mistral, phi, other
    params: float  # parameter count in billions
    is_instruct: bool
    architecture: str  # base, instruct, chat
    
    def to_features(self) -> np.ndarray:
        """Convert to feature vector for metadata network.
        
        Returns 11-dim vector:
        - log(params + 1): 1 dim
        - is_instruct: 1 dim  
        - arch_code (normalized): 1 dim
        - family one-hot: 8 dims
        TOTAL: 11 dimensions
        """
        # Family one-hot encoding (8 families)
        families = ['llama2', 'llama3', 'qwen2', 'qwen3', 'gemma', 'mistral', 'phi', 'other']
        family_onehot = np.array([1.0 if self.family == f else 0.0 for f in families])
        
        # Architecture code: base=0, instruct=1, chat=2
        arch_map = {'base': 0, 'instruct': 1, 'chat': 2}
        arch_code = arch_map.get(self.architecture, 0)
        
        features = np.concatenate([
            [np.log(self.params + 1)],  # log parameter count (1 dim)
            [1.0 if self.is_instruct else 0.0],  # is_instruct (1 dim)
            [arch_code / 2.0],  # normalized architecture code (1 dim)
            family_onehot  # 8-dim family encoding
        ])
        return features.astype(np.float32)
    
    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return 11


@dataclass
class SubjectData:
    """Data for a single MMLU subject."""
    name: str
    category: str  # STEM, Humanities, Applied
    n_items: int
    item_params: Optional[Dict] = None  # a, b parameters for items


class MMLUDataset:
    """MMLU dataset with simulated model responses."""
    
    # Subject categories based on MMLU structure
    SUBJECT_CATEGORIES = {
        'STEM': [
            'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_physics',
            'computer_security', 'conceptual_physics', 'electrical_engineering',
            'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
            'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
            'high_school_statistics', 'machine_learning'
        ],
        'Humanities': [
            'formal_logic', 'high_school_european_history', 'high_school_us_history',
            'high_school_world_history', 'international_law', 'jurisprudence',
            'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
            'prehistory', 'professional_law', 'world_religions', 'high_school_government_and_politics',
            'human_aging', 'politics', 'security_studies', 'sociology'
        ],
        'Applied': [
            'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
            'econometrics', 'global_facts', 'high_school_geography', 'high_school_macroeconomics',
            'high_school_microeconomics', 'high_school_psychology', 'human_sexuality',
            'management', 'marketing', 'medical_genetics', 'miscellaneous',
            'nutrition', 'professional_accounting', 'professional_medicine', 'public_relations',
            'us_foreign_policy', 'virology', 'marketing', 'business_ethics'
        ]
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.models: Dict[str, ModelMetadata] = {}
        self.subjects: Dict[str, SubjectData] = {}
        self.responses: Dict[str, np.ndarray] = {}  # model_name -> (n_items,)
        self.true_abilities: Dict[str, np.ndarray] = {}  # model_name -> 3D ability vector
        self.generation_seed = None  # Track which seed was used
        
    def generate_synthetic_data(self, n_models: int = 80, n_items_per_subject: int = 50, seed: int = 42):
        """
        Generate synthetic MMLU-like data based on realistic model performance patterns.
        
        CRITICAL: Uses the provided seed for ALL random operations.
        """
        # FIXED: Store and use seed consistently
        self.generation_seed = seed
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        
        # Define model families with realistic scaling patterns
        families_config = {
            'llama2': {'sizes': [7, 13, 70], 'base_acc': 0.45, 'scale_rate': 0.15},
            'llama3': {'sizes': [8, 70], 'base_acc': 0.50, 'scale_rate': 0.18},
            'qwen2': {'sizes': [0.5, 1.5, 7, 14, 32, 72], 'base_acc': 0.48, 'scale_rate': 0.16},
            'qwen3': {'sizes': [0.6, 1.7, 4, 8, 14, 32, 72, 235], 'base_acc': 0.52, 'scale_rate': 0.17},
            'gemma': {'sizes': [2, 4, 9, 27], 'base_acc': 0.46, 'scale_rate': 0.15},
            'mistral': {'sizes': [7, 8, 24], 'base_acc': 0.49, 'scale_rate': 0.16},
            'phi': {'sizes': [2, 3], 'base_acc': 0.47, 'scale_rate': 0.14},
        }
        
        # Generate models
        model_list = []
        model_idx = 0
        
        for family, config in families_config.items():
            for size in config['sizes']:
                if model_idx >= n_models:
                    break
                    
                # Base model
                base_name = f"{family}-{size}b"
                base_acc = config['base_acc'] + config['scale_rate'] * np.log(size + 1) / np.log(100)
                
                model_list.append({
                    'name': base_name,
                    'family': family,
                    'params': size,
                    'is_instruct': False,
                    'architecture': 'base',
                    'base_acc': min(base_acc, 0.95)
                })
                model_idx += 1
                
                # Instruct variant (50% chance for larger models)
                if model_idx < n_models and size >= 7 and rng.rand() > 0.3:
                    instruct_acc = base_acc + 0.05 + rng.randn() * 0.02
                    model_list.append({
                        'name': f"{base_name}-instruct",
                        'family': family,
                        'params': size,
                        'is_instruct': True,
                        'architecture': 'instruct',
                        'base_acc': min(instruct_acc, 0.95)
                    })
                    model_idx += 1
        
        # Fill remaining slots with random models
        other_families = ['falcon', 'bloom', 'gpt2', 'stablelm']
        while model_idx < n_models:
            family = rng.choice(other_families)
            size = rng.choice([3, 7, 13, 30, 70])
            base_acc = 0.40 + 0.12 * np.log(size + 1) / np.log(100) + rng.randn() * 0.03
            
            model_list.append({
                'name': f"{family}-{model_idx}-{size}b",
                'family': 'other',
                'params': size,
                'is_instruct': rng.rand() > 0.5,
                'architecture': 'base' if rng.rand() > 0.5 else 'instruct',
                'base_acc': min(max(base_acc, 0.25), 0.90)
            })
            model_idx += 1
        
        # Generate subjects
        all_subjects = []
        for category, subjects in self.SUBJECT_CATEGORIES.items():
            for subject in subjects:
                all_subjects.append({
                    'name': subject,
                    'category': category,
                    'n_items': n_items_per_subject
                })
        
        self.subjects = {s['name']: SubjectData(**s) for s in all_subjects}
        
        # Generate true 3D abilities (STEM, Humanities, Applied) for each model
        # Use family-level structure with correlation across dimensions
        family_means = {}
        for family in families_config.keys():
            family_means[family] = rng.randn(3) * 0.5 + np.array([0.5, 0.5, 0.5])
        family_means['other'] = rng.randn(3) * 0.5 + np.array([0.4, 0.4, 0.4])
        
        # Family covariance structure (models in same family are correlated)
        family_cov = np.array([
            [0.3, 0.2, 0.15],
            [0.2, 0.25, 0.1],
            [0.15, 0.1, 0.2]
        ])
        
        for m in model_list:
            family = m['family']
            base_ability = family_means[family] + rng.multivariate_normal([0, 0, 0], family_cov * 0.3)
            
            # Add size-based scaling
            size_boost = np.log(m['params'] + 1) / np.log(100) * 0.8
            instruct_boost = 0.3 if m['is_instruct'] else 0.0
            
            ability = base_ability + size_boost + instruct_boost + rng.randn(3) * 0.2
            self.true_abilities[m['name']] = ability
            
            self.models[m['name']] = ModelMetadata(
                name=m['name'],
                family=m['family'],
                params=m['params'],
                is_instruct=m['is_instruct'],
                architecture=m['architecture']
            )
        
        # Generate item parameters and responses
        for model_name, metadata in self.models.items():
            ability = self.true_abilities[model_name]
            responses = []
            
            for subject_name, subject_data in self.subjects.items():
                # Determine which dimension this subject belongs to
                if subject_data.category == 'STEM':
                    dim = 0
                elif subject_data.category == 'Humanities':
                    dim = 1
                else:
                    dim = 2
                
                # Generate item discrimination and difficulty using the seeded RNG
                a = rng.gamma(2, 0.5, subject_data.n_items)  # Discrimination
                b = rng.randn(subject_data.n_items) * 0.8  # Difficulty
                
                # 2PL IRT model: P(correct) = sigmoid(a * (theta - b))
                theta = ability[dim]
                probs = 1 / (1 + np.exp(-a * (theta - b)))
                
                # Add some noise and item-specific variation
                probs = np.clip(probs + rng.randn(subject_data.n_items) * 0.02, 0.05, 0.95)
                
                # Sample responses using the seeded RNG
                subject_responses = (rng.rand(subject_data.n_items) < probs).astype(np.float32)
                responses.extend(subject_responses)
            
            self.responses[model_name] = np.array(responses)
        
        # Store item parameters (use average for simplicity)
        n_total_items = len(responses)
        self.item_params = {
            'discrimination': rng.gamma(2, 0.5, n_total_items),
            'difficulty': rng.randn(n_total_items) * 0.8
        }
        
        return self
    
    def get_train_test_split(self, n_train: int = 60, n_test: int = 20, seed: int = 42) -> Tuple[List, List]:
        """Split models into train and test sets using provided seed."""
        rng = np.random.RandomState(seed)
        model_names = list(self.models.keys())
        rng.shuffle(model_names)
        
        train_models = model_names[:n_train]
        test_models = model_names[n_train:n_train+n_test]
        
        return train_models, test_models
    
    def save(self, path_prefix: str):
        """Save dataset to disk."""
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        data = {
            'models': {k: {
                'name': v.name,
                'family': v.family,
                'params': float(v.params),
                'is_instruct': bool(v.is_instruct),
                'architecture': v.architecture
            } for k, v in self.models.items()},
            'subjects': {k: {
                'name': v.name,
                'category': v.category,
                'n_items': int(v.n_items)
            } for k, v in self.subjects.items()},
            'responses': {k: convert_to_serializable(v) for k, v in self.responses.items()},
            'true_abilities': {k: convert_to_serializable(v) for k, v in self.true_abilities.items()},
            'item_params': {k: convert_to_serializable(v) for k, v in self.item_params.items()},
            'generation_seed': self.generation_seed
        }
        
        with open(f"{path_prefix}_full.json", 'w') as f:
            json.dump(data, f)
    
    def load(self, path_prefix: str):
        """Load dataset from disk."""
        with open(f"{path_prefix}_full.json", 'r') as f:
            data = json.load(f)
        
        self.models = {k: ModelMetadata(**v) for k, v in data['models'].items()}
        self.subjects = {k: SubjectData(**v) for k, v in data['subjects'].items()}
        self.responses = {k: np.array(v) for k, v in data['responses'].items()}
        self.true_abilities = {k: np.array(v) for k, v in data['true_abilities'].items()}
        self.item_params = {k: np.array(v) for k, v in data['item_params'].items()}
        self.generation_seed = data.get('generation_seed', 42)
        
        return self


def create_dimension_mapping(subjects: Dict[str, SubjectData]) -> Dict[str, int]:
    """Map subjects to dimensions (STEM=0, Humanities=1, Applied=2)."""
    mapping = {}
    for name, subject in subjects.items():
        if subject.category == 'STEM':
            mapping[name] = 0
        elif subject.category == 'Humanities':
            mapping[name] = 1
        else:
            mapping[name] = 2
    return mapping
