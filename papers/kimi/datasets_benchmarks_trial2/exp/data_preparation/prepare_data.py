"""Prepare IntrospectBench dataset with controlled error injection."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
import re
import numpy as np
from typing import Dict, List, Tuple
from shared.utils import set_seed, save_jsonl, save_json


# Synthetic problem templates for each domain
MATH_PROBLEMS = [
    {
        "question": "If a train travels 120 miles in 2 hours, and then travels another 150 miles in 3 hours, what is the average speed for the entire journey?",
        "correct_steps": [
            "First, calculate the total distance traveled: 120 + 150 = 270 miles.",
            "Next, calculate the total time taken: 2 + 3 = 5 hours.",
            "Then, use the formula: Average Speed = Total Distance / Total Time.",
            "Substitute the values: Average Speed = 270 / 5 = 54 miles per hour.",
            "Therefore, the average speed for the entire journey is 54 mph."
        ],
        "answer": "54"
    },
    {
        "question": "A rectangle has a length of 15 cm and a width of 8 cm. What is the area of the rectangle?",
        "correct_steps": [
            "Recall the formula for the area of a rectangle: Area = length × width.",
            "Identify the given values: length = 15 cm, width = 8 cm.",
            "Substitute into the formula: Area = 15 × 8.",
            "Calculate the product: 15 × 8 = 120.",
            "Therefore, the area of the rectangle is 120 square cm."
        ],
        "answer": "120"
    },
    {
        "question": "John has $250 in his savings account. He deposits $150 and then withdraws $80. How much money does he have now?",
        "correct_steps": [
            "Start with the initial amount: $250.",
            "Add the deposit: $250 + $150 = $400.",
            "Subtract the withdrawal: $400 - $80 = $320.",
            "Therefore, John has $320 in his account."
        ],
        "answer": "320"
    },
    {
        "question": "A baker made 48 cookies and wants to pack them into boxes of 6. How many boxes does she need?",
        "correct_steps": [
            "Identify the total number of cookies: 48.",
            "Identify the number of cookies per box: 6.",
            "Divide the total by the box size: 48 ÷ 6 = 8.",
            "Therefore, the baker needs 8 boxes."
        ],
        "answer": "8"
    },
    {
        "question": "The sum of three consecutive integers is 72. What is the largest integer?",
        "correct_steps": [
            "Let the three consecutive integers be n, n+1, and n+2.",
            "Set up the equation: n + (n+1) + (n+2) = 72.",
            "Simplify: 3n + 3 = 72.",
            "Solve for n: 3n = 69, so n = 23.",
            "The largest integer is n+2 = 25."
        ],
        "answer": "25"
    }
]

LOGIC_PROBLEMS = [
    {
        "question": "All cats are mammals. All mammals are warm-blooded. Is a cat warm-blooded?",
        "correct_steps": [
            "Premise 1: All cats are mammals.",
            "Premise 2: All mammals are warm-blooded.",
            "By syllogistic reasoning, if all cats are mammals and all mammals are warm-blooded, then all cats must be warm-blooded.",
            "Therefore, a cat is warm-blooded."
        ],
        "answer": "Yes"
    },
    {
        "question": "If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?",
        "correct_steps": [
            "Given: If it rains, then the ground gets wet (rain → wet ground).",
            "Given: The ground is wet.",
            "This is the logical fallacy of affirming the consequent.",
            "The ground could be wet from other sources: sprinklers, someone watering plants, etc.",
            "Therefore, it is not necessarily true that it rained."
        ],
        "answer": "No"
    },
    {
        "question": "No reptiles are mammals. All snakes are reptiles. Are any snakes mammals?",
        "correct_steps": [
            "Premise 1: No reptiles are mammals.",
            "Premise 2: All snakes are reptiles.",
            "If no reptiles are mammals, and all snakes are reptiles, then no snakes can be mammals.",
            "This is valid syllogistic reasoning (EAE-2).",
            "Therefore, no snakes are mammals."
        ],
        "answer": "No"
    },
    {
        "question": "Either the meeting is at 2 PM or at 3 PM. The meeting is not at 2 PM. When is the meeting?",
        "correct_steps": [
            "Premise 1: The meeting is at 2 PM OR at 3 PM (exclusive or inclusive disjunction).",
            "Premise 2: The meeting is NOT at 2 PM.",
            "By disjunctive syllogism, if one disjunct is false, the other must be true.",
            "Therefore, the meeting is at 3 PM."
        ],
        "answer": "3 PM"
    }
]

COMMONSENSE_PROBLEMS = [
    {
        "question": "If you leave ice cream outside on a hot day, what will happen to it?",
        "correct_steps": [
            "Ice cream is frozen and needs cold temperatures to maintain its solid state.",
            "Hot weather provides heat energy to the ice cream.",
            "Heat causes the ice cream to melt.",
            "Therefore, the ice cream will melt."
        ],
        "answer": "It will melt"
    },
    {
        "question": "Why do people use umbrellas when it's raining?",
        "correct_steps": [
            "Rain consists of water droplets falling from the sky.",
            "People generally want to stay dry.",
            "Umbrellas have a waterproof canopy that blocks falling water.",
            "Therefore, people use umbrellas to stay dry during rain."
        ],
        "answer": "To stay dry"
    },
    {
        "question": "A person wants to cut a piece of wood. What tool should they use?",
        "correct_steps": [
            "Wood is a solid material that requires a cutting tool.",
            "A saw is specifically designed for cutting wood.",
            "Other tools like hammers or screwdrivers are not designed for cutting.",
            "Therefore, a saw is the appropriate tool."
        ],
        "answer": "A saw"
    },
    {
        "question": "Why do we put food in the refrigerator?",
        "correct_steps": [
            "Bacteria grow rapidly at room temperature, causing food spoilage.",
            "Refrigerators maintain low temperatures.",
            "Low temperatures slow down bacterial growth.",
            "Therefore, we refrigerate food to keep it fresh longer."
        ],
        "answer": "To keep it fresh"
    }
]

CODE_PROBLEMS = [
    {
        "question": "Write a function to calculate the factorial of a number n.",
        "correct_steps": [
            "Define the function factorial(n) that takes an integer n.",
            "Check if n is 0 or 1, return 1 (base case).",
            "Otherwise, return n multiplied by factorial(n-1).",
            "This uses recursion to compute n! = n × (n-1) × ... × 1."
        ],
        "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    },
    {
        "question": "Write a function to check if a string is a palindrome.",
        "correct_steps": [
            "Define the function is_palindrome(s) that takes a string s.",
            "Remove any non-alphanumeric characters and convert to lowercase.",
            "Compare the string with its reverse.",
            "Return True if they are equal, False otherwise."
        ],
        "code": "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]"
    },
    {
        "question": "Write a function to find the maximum element in a list.",
        "correct_steps": [
            "Define the function find_max(lst) that takes a list.",
            "Initialize max_val with the first element.",
            "Iterate through the list comparing each element with max_val.",
            "Update max_val if a larger element is found.",
            "Return max_val."
        ],
        "code": "def find_max(lst):\n    max_val = lst[0]\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val"
    },
    {
        "question": "Write a function to reverse a list in-place.",
        "correct_steps": [
            "Define the function reverse_list(lst) that takes a list.",
            "Use two pointers: one at start (left), one at end (right).",
            "Swap elements at left and right pointers.",
            "Move left pointer forward and right pointer backward.",
            "Continue until pointers meet."
        ],
        "code": "def reverse_list(lst):\n    left, right = 0, len(lst) - 1\n    while left < right:\n        lst[left], lst[right] = lst[right], lst[left]\n        left += 1\n        right -= 1\n    return lst"
    }
]


def generate_dataset(seed: int = 42) -> List[Dict]:
    """Generate full dataset with controlled error injection."""
    set_seed(seed)
    
    # Create multiple variations of each problem
    all_problems = []
    
    # Generate Math problems
    for i in range(100):  # 100 base math problems
        template = random.choice(MATH_PROBLEMS)
        problem = create_problem_variant(template, "math", i, seed)
        all_problems.extend(problem)
    
    # Generate Logic problems
    for i in range(75):  # 75 base logic problems
        template = random.choice(LOGIC_PROBLEMS)
        problem = create_problem_variant(template, "logic", i + 100, seed)
        all_problems.extend(problem)
    
    # Generate Commonsense problems
    for i in range(75):  # 75 base commonsense problems
        template = random.choice(COMMONSENSE_PROBLEMS)
        problem = create_problem_variant(template, "commonsense", i + 175, seed)
        all_problems.extend(problem)
    
    # Generate Code problems
    for i in range(50):  # 50 base code problems
        template = random.choice(CODE_PROBLEMS)
        problem = create_problem_variant(template, "code", i + 250, seed)
        all_problems.extend(problem)
    
    return all_problems


def create_problem_variant(template: Dict, domain: str, base_id: int, seed: int) -> List[Dict]:
    """Create problem variants with and without errors."""
    variants = []
    
    # Create correct version
    correct_problem = {
        "problem_id": f"{domain}_{base_id}_correct",
        "domain": domain,
        "question": template["question"],
        "correct_steps": template["correct_steps"],
        "has_error": False,
        "error_type": None,
        "error_step": None,
        "error_position": None,
        "answer": template.get("answer", template.get("code", "")),
        "code": template.get("code", "")
    }
    variants.append(correct_problem)
    
    # Create error versions (2 per problem with different error types/positions)
    error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
    positions = ["early", "middle", "late"]
    
    num_steps = len(template["correct_steps"])
    
    for err_idx in range(2):
        error_type = random.choice(error_types)
        position = positions[err_idx % 3]
        
        # Determine error step based on position
        if position == "early":
            error_step = random.randint(1, min(2, num_steps))
        elif position == "middle":
            error_step = random.randint(max(1, num_steps//2 - 1), min(num_steps//2 + 1, num_steps))
        else:  # late
            error_step = random.randint(max(1, num_steps - 2), num_steps)
        
        # Create corrupted steps
        corrupted_steps, actual_error_type = inject_error(
            template["correct_steps"].copy(), 
            error_step - 1,  # 0-indexed
            error_type,
            domain
        )
        
        error_problem = {
            "problem_id": f"{domain}_{base_id}_error{err_idx}",
            "domain": domain,
            "question": template["question"],
            "correct_steps": template["correct_steps"],
            "corrupted_steps": corrupted_steps,
            "has_error": True,
            "error_type": actual_error_type,
            "error_step": error_step,
            "error_position": position,
            "num_steps": num_steps,
            "answer": template.get("answer", template.get("code", "")),
            "code": template.get("code", "")
        }
        variants.append(error_problem)
    
    return variants


def inject_error(steps: List[str], error_idx: int, error_type: str, domain: str) -> Tuple[List[str], str]:
    """Inject a controlled error into the reasoning steps."""
    corrupted = steps.copy()
    
    if error_idx >= len(steps):
        error_idx = len(steps) - 1
    
    original_step = steps[error_idx]
    
    if error_type == "calculation":
        # Inject arithmetic error
        corrupted[error_idx] = inject_calculation_error(original_step)
    elif error_type == "logic":
        # Inject logical fallacy
        corrupted[error_idx] = inject_logic_error(original_step, domain)
    elif error_type == "factuality":
        # Inject incorrect fact
        corrupted[error_idx] = inject_factuality_error(original_step, domain)
    elif error_type == "omission":
        # Remove this step and merge with next
        corrupted = inject_omission_error(corrupted, error_idx)
    elif error_type == "misinterpretation":
        # Misinterpret constraints
        corrupted[error_idx] = inject_misinterpretation_error(original_step, domain)
    elif error_type == "premature":
        # Add conclusion too early
        corrupted[error_idx] = inject_premature_error(original_step)
    
    return corrupted, error_type


def inject_calculation_error(step: str) -> str:
    """Inject calculation error by modifying numbers."""
    # Find numbers and slightly modify them
    numbers = re.findall(r'\d+', step)
    if numbers:
        num = random.choice(numbers)
        # Add or subtract 1
        wrong_num = str(int(num) + random.choice([-1, 1]))
        step = step.replace(num, wrong_num, 1)
    else:
        # Add an incorrect arithmetic statement
        step += " This equals approximately 42."
    return step


def inject_logic_error(step: str, domain: str) -> str:
    """Inject logical error."""
    logic_errors = [
        "Therefore, by the transitive property, the opposite must also be true.",
        "This implies that all similar cases behave identically.",
        "Since A causes B, and B happened, A must have occurred.",
        "This is always true without exception."
    ]
    return step + " " + random.choice(logic_errors)


def inject_factuality_error(step: str, domain: str) -> str:
    """Inject factuality error."""
    if domain == "math":
        return step.replace("×", "+").replace("*", "+") if ("×" in step or "*" in step) else step + " Remember that division by zero equals one."
    elif domain == "code":
        return step.replace("return", "print") if "return" in step else step + " Note that Python uses static typing by default."
    else:
        return step + " This is a well-known fact established in 1850."


def inject_omission_error(steps: List[str], error_idx: int) -> List[str]:
    """Remove a critical step."""
    if len(steps) > 2 and error_idx < len(steps) - 1:
        steps.pop(error_idx)
    return steps


def inject_misinterpretation_error(step: str, domain: str) -> str:
    """Inject misinterpretation of constraints."""
    misinterp = [
        " (assuming the opposite of what was stated)",
        " ignoring the constraints mentioned earlier",
        " which applies to all cases, not just this one"
    ]
    return step + random.choice(misinterp)


def inject_premature_error(step: str) -> str:
    """Add premature conclusion."""
    return step + " Therefore, we can conclude the final answer now without further analysis."


def split_dataset(problems: List[Dict], seed: int = 42) -> Dict[str, List[Dict]]:
    """Split dataset into train/val/test."""
    set_seed(seed)
    
    # Separate correct and error problems
    correct = [p for p in problems if not p.get("has_error", False)]
    errors = [p for p in problems if p.get("has_error", False)]
    
    random.shuffle(correct)
    random.shuffle(errors)
    
    # Split 80/10/10
    def split_list(lst):
        n = len(lst)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]
    
    c_train, c_val, c_test = split_list(correct)
    e_train, e_val, e_test = split_list(errors)
    
    splits = {
        "train": c_train + e_train,
        "val": c_val + e_val,
        "test": c_test + e_test
    }
    
    # Shuffle each split
    for key in splits:
        random.shuffle(splits[key])
    
    return splits


def main():
    print("Generating IntrospectBench dataset...")
    
    # Generate dataset
    problems = generate_dataset(seed=42)
    print(f"Generated {len(problems)} total problem variants")
    
    # Split dataset
    splits = split_dataset(problems, seed=42)
    
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} examples")
    
    # Save splits
    os.makedirs("data/processed", exist_ok=True)
    for split_name, split_data in splits.items():
        save_jsonl(split_data, f"data/processed/{split_name}.jsonl")
    
    # Create statistics
    stats = {
        "total": len(problems),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "domains": {},
        "error_types": {},
        "error_positions": {}
    }
    
    for p in problems:
        domain = p["domain"]
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
        
        if p.get("has_error"):
            err_type = p.get("error_type", "unknown")
            stats["error_types"][err_type] = stats["error_types"].get(err_type, 0) + 1
            
            pos = p.get("error_position", "unknown")
            stats["error_positions"][pos] = stats["error_positions"].get(pos, 0) + 1
    
    save_json(stats, "data/processed/statistics.json")
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
