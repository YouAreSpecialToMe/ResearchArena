"""
Data generation utilities for DynaScale experiments.
Includes template-based item generation and response simulation.
"""

import numpy as np
import json
import random
from typing import List, Dict, Tuple


# ==================== Math Problem Templates ====================

MATH_TEMPLATES = [
    # Template 1: Simple arithmetic
    {
        'template': "What is {a} + {b}?",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(1, 1000), 'b': random.randint(1, 1000)},
        'solver': lambda params: params['a'] + params['b'],
        'difficulty_proxy': lambda params: 1 if max(params['a'], params['b']) < 100 else 2
    },
    # Template 2: Subtraction
    {
        'template': "Calculate {a} - {b}.",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(50, 500), 'b': random.randint(1, 50)},
        'solver': lambda params: params['a'] - params['b'],
        'difficulty_proxy': lambda params: 1
    },
    # Template 3: Multiplication
    {
        'template': "What is {a} × {b}?",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(2, 50), 'b': random.randint(2, 50)},
        'solver': lambda params: params['a'] * params['b'],
        'difficulty_proxy': lambda params: 2 if params['a'] > 20 or params['b'] > 20 else 1
    },
    # Template 4: Division
    {
        'template': "If {a} items are divided equally among {b} people, how many items does each person get?",
        'answer_template': "{answer}",
        'generator': lambda: (lambda b: {'a': b * random.randint(2, 20), 'b': b})(random.randint(2, 10)),
        'solver': lambda params: params['a'] // params['b'],
        'difficulty_proxy': lambda params: 2
    },
    # Template 5: Multi-step arithmetic
    {
        'template': "Calculate ({a} + {b}) × {c}.",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(1, 50), 'b': random.randint(1, 50), 'c': random.randint(2, 10)},
        'solver': lambda params: (params['a'] + params['b']) * params['c'],
        'difficulty_proxy': lambda params: 3
    },
    # Template 6: Word problem - simple
    {
        'template': "Sarah has {a} apples. She buys {b} more. How many apples does she have now?",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(1, 100), 'b': random.randint(1, 50)},
        'solver': lambda params: params['a'] + params['b'],
        'difficulty_proxy': lambda params: 1
    },
    # Template 7: Word problem - multi-step
    {
        'template': "A store has {a} boxes with {b} items each. If they sell {c} items, how many remain?",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(5, 20), 'b': random.randint(5, 20), 
                            'c': random.randint(10, 50)},
        'solver': lambda params: params['a'] * params['b'] - params['c'],
        'difficulty_proxy': lambda params: 3
    },
    # Template 8: Percentage
    {
        'template': "What is {p}% of {n}?",
        'answer_template': "{answer}",
        'generator': lambda: {'p': random.choice([10, 20, 25, 50, 75]), 
                            'n': random.randint(20, 200)},
        'solver': lambda params: int(params['p'] * params['n'] / 100),
        'difficulty_proxy': lambda params: 2
    },
    # Template 9: Larger numbers addition
    {
        'template': "Calculate {a} + {b} + {c}.",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(100, 999), 'b': random.randint(100, 999), 
                            'c': random.randint(100, 999)},
        'solver': lambda params: params['a'] + params['b'] + params['c'],
        'difficulty_proxy': lambda params: 3 if max(params.values()) > 500 else 2
    },
    # Template 10: Ratio/proportion
    {
        'template': "If {a} workers can complete a job in {b} days, how many days will it take {c} workers?",
        'answer_template': "{answer}",
        'generator': lambda: {'a': random.randint(2, 10), 'b': random.randint(5, 30), 
                            'c': random.randint(2, 10)},
        'solver': lambda params: (params['a'] * params['b']) // params['c'] if params['c'] > 0 else 0,
        'difficulty_proxy': lambda params: 4
    },
]

# Additional templates for higher difficulty
def generate_advanced_math_templates():
    """Generate additional math templates with variable complexity."""
    templates = []
    
    # Algebra templates
    for i in range(10):
        templates.append({
            'template': f"Solve for x: x + {random.randint(5, 50)} = {random.randint(60, 150)}",
            'answer_template': "{answer}",
            'generator': lambda: {},
            'solver': lambda params, i=i: random.randint(60, 150) - (random.randint(5, 50)),
            'difficulty_proxy': lambda params: 3 + random.randint(0, 2),
            'is_static': True,
            'id': i
        })
    
    # More complex arithmetic
    for i in range(10):
        a, b, c = random.randint(10, 100), random.randint(10, 100), random.randint(2, 9)
        templates.append({
            'template': f"Calculate {a} × {b} - {c} × {random.randint(10, 50)}",
            'answer_template': "{answer}",
            'generator': lambda a=a, b=b, c=c: {'a': a, 'b': b, 'c': c, 'd': random.randint(10, 50)},
            'solver': lambda params: params['a'] * params['b'] - params['c'] * params['d'],
            'difficulty_proxy': lambda params: 4 + random.randint(0, 2),
        })
    
    return templates


# ==================== Code Problem Templates ====================

CODE_TEMPLATES = [
    # Template 1: Simple function
    {
        'template': "Write a function that returns the sum of two numbers.",
        'test_cases': [(1, 2, 3), (0, 0, 0), (-1, 1, 0), (100, 200, 300)],
        'solution': "def sum_two(a, b):\\n    return a + b",
        'difficulty_proxy': 1
    },
    # Template 2: List reversal
    {
        'template': "Write a function that reverses a list.",
        'test_cases': [([1, 2, 3], [3, 2, 1]), (['a', 'b'], ['b', 'a']), ([], [])],
        'solution': "def reverse_list(lst):\\n    return lst[::-1]",
        'difficulty_proxy': 1
    },
    # Template 3: String manipulation
    {
        'template': "Write a function to check if a string is a palindrome.",
        'test_cases': [('racecar', True), ('hello', False), ('', True), ('a', True)],
        'solution': "def is_palindrome(s):\\n    return s == s[::-1]",
        'difficulty_proxy': 2
    },
    # Template 4: Factorial
    {
        'template': "Write a function to compute factorial of n.",
        'test_cases': [(0, 1), (1, 1), (5, 120), (3, 6)],
        'solution': "def factorial(n):\\n    return 1 if n <= 1 else n * factorial(n-1)",
        'difficulty_proxy': 2
    },
    # Template 5: Fibonacci
    {
        'template': "Write a function to return the nth Fibonacci number.",
        'test_cases': [(0, 0), (1, 1), (5, 5), (10, 55)],
        'solution': "def fibonacci(n):\\n    if n <= 1: return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
        'difficulty_proxy': 3
    },
    # Template 6: Prime check
    {
        'template': "Write a function to check if a number is prime.",
        'test_cases': [(2, True), (3, True), (4, False), (17, True), (1, False)],
        'solution': "def is_prime(n):\\n    if n < 2: return False\\n    for i in range(2, int(n**0.5)+1):\\n        if n % i == 0: return False\\n    return True",
        'difficulty_proxy': 3
    },
    # Template 7: Find maximum
    {
        'template': "Write a function to find the maximum element in a list without using max().",
        'test_cases': [([1, 2, 3], 3), ([-1, -5, -3], -1), ([5], 5)],
        'solution': "def find_max(lst):\\n    max_val = lst[0]\\n    for x in lst[1:]:\\n        if x > max_val: max_val = x\\n    return max_val",
        'difficulty_proxy': 2
    },
    # Template 8: Two sum
    {
        'template': "Write a function that finds two numbers in a list that sum to a target.",
        'test_cases': [([2, 7, 11, 15], 9, [2, 7]), ([3, 2, 4], 6, [2, 4])],
        'solution': "def two_sum(nums, target):\\n    seen = {}\\n    for i, n in enumerate(nums):\\n        if target - n in seen: return [seen[target-n], n]\\n        seen[n] = n",
        'difficulty_proxy': 4
    },
]


# ==================== Science Problem Templates ====================

SCIENCE_TEMPLATES = [
    # Template 1: Simple physics
    {
        'template': "If an object travels at {speed} m/s for {time} seconds, how far does it travel?",
        'answer_template': "{answer} meters",
        'generator': lambda: {'speed': random.randint(5, 50), 'time': random.randint(2, 20)},
        'solver': lambda params: params['speed'] * params['time'],
        'difficulty_proxy': 1,
        'domain': 'physics'
    },
    # Template 2: Density
    {
        'template': "What is the density of an object with mass {mass}g and volume {volume}cm³?",
        'answer_template': "{answer} g/cm³",
        'generator': lambda: {'mass': random.randint(10, 100), 'volume': random.randint(5, 20)},
        'solver': lambda params: round(params['mass'] / params['volume'], 2),
        'difficulty_proxy': 2,
        'domain': 'physics'
    },
    # Template 3: Chemistry - moles
    {
        'template': "How many moles are in {mass}g of water (H₂O, molar mass 18g/mol)?",
        'answer_template': "{answer} mol",
        'generator': lambda: {'mass': random.choice([18, 36, 54, 90, 180])},
        'solver': lambda params: params['mass'] / 18,
        'difficulty_proxy': 2,
        'domain': 'chemistry'
    },
    # Template 4: Biology - cell division
    {
        'template': "Starting with 1 cell, how many cells will there be after {generations} generations of division?",
        'answer_template': "{answer}",
        'generator': lambda: {'generations': random.randint(3, 8)},
        'solver': lambda params: 2 ** params['generations'],
        'difficulty_proxy': 3,
        'domain': 'biology'
    },
    # Template 5: Physics - force
    {
        'template': "What is the force on a {mass}kg object accelerating at {accel}m/s²?",
        'answer_template': "{answer} N",
        'generator': lambda: {'mass': random.randint(1, 20), 'accel': random.randint(1, 10)},
        'solver': lambda params: params['mass'] * params['accel'],
        'difficulty_proxy': 2,
        'domain': 'physics'
    },
    # Template 6: Chemistry - concentration
    {
        'template': "What is the molarity of a solution with {moles} moles of solute in {volume} L of solution?",
        'answer_template': "{answer} M",
        'generator': lambda: {'moles': random.randint(1, 10), 'volume': random.choice([1, 2, 5, 10])},
        'solver': lambda params: round(params['moles'] / params['volume'], 2),
        'difficulty_proxy': 2,
        'domain': 'chemistry'
    },
    # Template 7: Physics - work
    {
        'template': "How much work is done when a force of {force}N moves an object {distance}m?",
        'answer_template': "{answer} J",
        'generator': lambda: {'force': random.randint(10, 100), 'distance': random.randint(1, 20)},
        'solver': lambda params: params['force'] * params['distance'],
        'difficulty_proxy': 2,
        'domain': 'physics'
    },
    # Template 8: Energy
    {
        'template': "What is the kinetic energy of a {mass}kg object moving at {speed}m/s? (KE = ½mv²)",
        'answer_template': "{answer} J",
        'generator': lambda: {'mass': random.randint(1, 10), 'speed': random.randint(2, 20)},
        'solver': lambda params: round(0.5 * params['mass'] * params['speed'] ** 2, 1),
        'difficulty_proxy': 3,
        'domain': 'physics'
    },
]


def generate_math_problems(n_problems, difficulty_range=(1, 5), seed=None):
    """Generate math word problems using templates.
    
    Args:
        n_problems: Number of problems to generate
        difficulty_range: (min, max) difficulty levels
        seed: Random seed
        
    Returns:
        problems: List of problem dictionaries
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    templates = MATH_TEMPLATES + generate_advanced_math_templates()
    problems = []
    
    for i in range(n_problems):
        # Select template based on difficulty distribution
        def get_difficulty(t):
            dp = t.get('difficulty_proxy', 1)
            if callable(dp):
                try:
                    return dp({})
                except:
                    return 1
            return dp
        
        valid_templates = [t for t in templates 
                          if difficulty_range[0] <= get_difficulty(t) <= difficulty_range[1]]
        if not valid_templates:
            valid_templates = templates
            
        template = random.choice(valid_templates)
        
        # Generate parameters
        params = template['generator']() if not template.get('is_static') else {}
        
        # Compute answer
        if template.get('is_static'):
            # For static templates, extract numbers from template string
            import re
            text = template['template']
            if 'x +' in text:
                match = re.search(r'x \+ (\d+) = (\d+)', text)
                if match:
                    answer = int(match.group(2)) - int(match.group(1))
                else:
                    answer = 0
            else:
                answer = template['solver']({})
        else:
            answer = template['solver'](params)
        
        # Format problem
        problem_text = template['template'].format(**params, answer=answer)
        
        problems.append({
            'id': f'math_{i}',
            'question': problem_text,
            'answer': str(answer),
            'domain': 'math',
            'difficulty_proxy': template['difficulty_proxy'](params) if callable(template['difficulty_proxy']) else template['difficulty_proxy'],
            'template_id': template.get('id', 0),
            'parameters': params
        })
    
    return problems


def generate_code_problems(n_problems, difficulty_range=(1, 5), seed=None):
    """Generate code problems.
    
    Args:
        n_problems: Number of problems to generate
        difficulty_range: (min, max) difficulty levels
        seed: Random seed
        
    Returns:
        problems: List of problem dictionaries
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    problems = []
    templates = [t for t in CODE_TEMPLATES 
                 if difficulty_range[0] <= t['difficulty_proxy'] <= difficulty_range[1]]
    
    if not templates:
        templates = CODE_TEMPLATES
    
    for i in range(n_problems):
        template = random.choice(templates)
        
        problems.append({
            'id': f'code_{i}',
            'question': template['template'],
            'answer': template['solution'],
            'test_cases': template['test_cases'],
            'domain': 'code',
            'difficulty_proxy': template['difficulty_proxy']
        })
    
    return problems


def generate_science_problems(n_problems, difficulty_range=(1, 5), seed=None):
    """Generate science problems using templates.
    
    Args:
        n_problems: Number of problems to generate
        difficulty_range: (min, max) difficulty levels
        seed: Random seed
        
    Returns:
        problems: List of problem dictionaries
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    problems = []
    templates = [t for t in SCIENCE_TEMPLATES 
                 if difficulty_range[0] <= t['difficulty_proxy'] <= difficulty_range[1]]
    
    if not templates:
        templates = SCIENCE_TEMPLATES
    
    for i in range(n_problems):
        template = random.choice(templates)
        
        # Generate parameters
        params = template['generator']()
        answer = template['solver'](params)
        
        # Format problem
        problem_text = template['template'].format(**params)
        answer_text = template['answer_template'].format(answer=answer)
        
        problems.append({
            'id': f'science_{i}',
            'question': problem_text,
            'answer': str(answer),
            'domain': template['domain'],
            'difficulty_proxy': template['difficulty_proxy'],
            'parameters': params
        })
    
    return problems


def simulate_responses(abilities, difficulties, discriminations, seed=None, noise_rate=0.05):
    """Simulate model responses using 2PL IRT model.
    
    Args:
        abilities: (n_models,) array of model abilities
        difficulties: (n_items,) array of item difficulties
        discriminations: (n_items,) array of item discriminations
        seed: Random seed
        noise_rate: Probability of random response (for realism)
        
    Returns:
        responses: (n_models, n_items) binary response matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_models = len(abilities)
    n_items = len(difficulties)
    
    # Compute probabilities using 2PL model
    theta = abilities[:, np.newaxis]  # (n_models, 1)
    b = difficulties[np.newaxis, :]   # (1, n_items)
    a = discriminations[np.newaxis, :]  # (1, n_items)
    
    z = a * (theta - b)
    probs = 1 / (1 + np.exp(-z))
    
    # Generate responses
    responses = (np.random.random((n_models, n_items)) < probs).astype(int)
    
    # Add noise
    noise_mask = np.random.random((n_models, n_items)) < noise_rate
    responses[noise_mask] = 1 - responses[noise_mask]
    
    return responses


def create_model_population(n_models=28, ability_range=(-2, 3), frontier_progression=True, seed=None):
    """Create a simulated model population with abilities.
    
    Args:
        n_models: Number of models
        ability_range: (min, max) ability values
        frontier_progression: If True, models are arranged to show frontier progression
        seed: Random seed
        
    Returns:
        population: List of model dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    if frontier_progression:
        # Create models that span the ability range with some clustering at frontier
        base_abilities = np.linspace(ability_range[0], ability_range[1], n_models)
        # Add some noise
        abilities = base_abilities + np.random.normal(0, 0.2, n_models)
    else:
        abilities = np.random.uniform(ability_range[0], ability_range[1], n_models)
    
    # Sort by ability
    abilities = np.sort(abilities)
    
    model_names = [f'model_{i:02d}' for i in range(n_models)]
    
    population = []
    for i, (name, ability) in enumerate(zip(model_names, abilities)):
        population.append({
            'id': i,
            'name': name,
            'ability': float(ability),
            'family': ['small', 'medium', 'large'][min(i // 10, 2)],
            'is_frontier': ability > (ability_range[1] - 0.5)
        })
    
    return population


def simulate_temporal_evolution(base_population, n_timepoints=5, ability_growth=0.4, seed=None):
    """Simulate how model abilities evolve over time.
    
    Args:
        base_population: Base model population
        n_timepoints: Number of time points to simulate
        ability_growth: How much frontier models improve per time step
        seed: Random seed
        
    Returns:
        evolution: Dict mapping time -> list of model abilities
    """
    if seed is not None:
        np.random.seed(seed)
    
    evolution = {}
    
    for t in range(n_timepoints):
        time_population = []
        for model in base_population:
            # Frontier models improve over time
            if model['is_frontier']:
                growth = ability_growth * t + np.random.normal(0, 0.1)
            else:
                growth = ability_growth * 0.5 * t + np.random.normal(0, 0.1)
            
            new_ability = model['ability'] + growth
            time_population.append({
                **model,
                'ability': float(new_ability),
                'time': t
            })
        
        evolution[t] = time_population
    
    return evolution
