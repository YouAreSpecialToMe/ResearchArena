"""
Generate structured contrast pairs for component vector extraction.
Uses manually crafted templates to ensure quality without API dependency.
"""
import sys
sys.path.append('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01')

import json
import os

# Manually crafted contrast pairs for math reasoning components
# Each pair contrasts presence vs absence of specific reasoning component

PLANNING_PAIRS = [
    {
        "positive": "I'll solve this step-by-step. First, I need to identify what information is given and what I need to find. Let me break this down systematically.",
        "negative": "I'll solve this directly."
    },
    {
        "positive": "My strategy will be: (1) identify the variables, (2) set up the equation, (3) solve for the unknown.",
        "negative": "Let me just calculate the answer."
    },
    {
        "positive": "To approach this problem, I should first understand what is being asked and plan my method before computing.",
        "negative": "I can see the numbers so I'll just multiply them."
    },
    {
        "positive": "Let me outline a plan: I'll identify the key quantities and determine the operations needed.",
        "negative": "The answer should be 5 times 6."
    },
    {
        "positive": "First, I'll determine what type of problem this is and select the appropriate approach.",
        "negative": "This is just basic arithmetic."
    },
    {
        "positive": "I need to establish a clear plan: understand the problem, extract relevant numbers, and execute calculations in order.",
        "negative": "I see the problem, the answer is 42."
    },
    {
        "positive": "Strategy formulation: identify knowns, unknowns, and the logical path from one to the other.",
        "negative": "I'll add these numbers together."
    },
    {
        "positive": "Before calculating, let me think about the structure of this problem and how to organize my solution.",
        "negative": "Calculation is straightforward here."
    },
    {
        "positive": "My approach will involve systematic analysis: parse the problem, identify relationships, then compute.",
        "negative": "Just need to do the math."
    },
    {
        "positive": "Let me devise a methodical approach to ensure I address all aspects of this problem correctly.",
        "negative": "The solution is obvious."
    },
    {
        "positive": "I'll begin by formulating a clear strategy to tackle this problem efficiently.",
        "negative": "Let me work this out."
    },
    {
        "positive": "Step one is planning: I need to map out the logical steps before performing any calculations.",
        "negative": "I can solve this quickly."
    },
    {
        "positive": "Planning phase: identify goal, given data, constraints, and the algorithm to connect them.",
        "negative": "The numbers are clear."
    },
    {
        "positive": "I'll develop a structured approach: analyze requirements, design solution steps, then implement.",
        "negative": "Simple computation."
    },
    {
        "positive": "Before diving into numbers, let me establish the conceptual framework for solving this.",
        "negative": "Ready to calculate."
    }
]

COMPUTATION_PAIRS = [
    {
        "positive": "Step 1: 5 + 3 = 8. Step 2: 8 × 2 = 16. Step 3: Therefore, the answer is 16.",
        "negative": "The answer is 16."
    },
    {
        "positive": "First, I'll add 12 and 8 to get 20. Then I'll divide by 4 to get 5. So the answer is 5.",
        "negative": "5"
    },
    {
        "positive": "Let me show my work: 7 × 6 = 42, then 42 + 10 = 52, then 52 - 2 = 50. Final answer: 50.",
        "negative": "50"
    },
    {
        "positive": "Calculation steps: (1) 15 - 7 = 8, (2) 8 × 3 = 24, (3) 24 ÷ 2 = 12. Answer: 12.",
        "negative": "The result is 12."
    },
    {
        "positive": "I'll compute step by step: starting with 100, subtract 25 to get 75, then add 10 to get 85.",
        "negative": "85"
    },
    {
        "positive": "Working through: 9 + 6 = 15, carry the 1. 1 + 2 + 3 = 6. Final result is 65.",
        "negative": "65"
    },
    {
        "positive": "Step-by-step: multiply 14 × 3 = 42. Then divide by 7 = 6. Add 4 = 10. Answer is 10.",
        "negative": "Answer: 10"
    },
    {
        "positive": "Let me calculate: 20% of 150 is 0.20 × 150 = 30. Then 150 - 30 = 120.",
        "negative": "120"
    },
    {
        "positive": "Computing: 3² + 4² = 9 + 16 = 25. √25 = 5. The length is 5 units.",
        "negative": "5 units"
    },
    {
        "positive": "Breaking it down: 48 ÷ 6 = 8. Then 8 × 7 = 56. Finally, 56 - 6 = 50.",
        "negative": "50"
    },
    {
        "positive": "Detailed calculation: 1/4 + 1/2 = 1/4 + 2/4 = 3/4 = 0.75 or 75%.",
        "negative": "0.75"
    },
    {
        "positive": "Step 1: Find LCM of 4 and 6, which is 12. Step 2: Convert both fractions. Step 3: Add to get 5/12.",
        "negative": "5/12"
    },
    {
        "positive": "Working it through: If x = 5, then 2x + 3 = 2(5) + 3 = 10 + 3 = 13.",
        "negative": "13"
    },
    {
        "positive": "Calculation: Area = length × width = 8 × 5 = 40 square meters.",
        "negative": "40 square meters"
    },
    {
        "positive": "Stepwise: 3.14 × 5² = 3.14 × 25 = 78.5. The area is 78.5 square units.",
        "negative": "78.5"
    }
]

VERIFICATION_PAIRS = [
    {
        "positive": "My answer is 42. Let me verify: 6 × 7 = 42 ✓. This checks out.",
        "negative": "My answer is 42."
    },
    {
        "positive": "I got 15. Verification: if I reverse the operation, 15 - 7 = 8, which matches my starting number. ✓",
        "negative": "I got 15."
    },
    {
        "positive": "The answer is 28. Checking: 28 ÷ 4 = 7, which is correct. Confirmed. ✓",
        "negative": "The answer is 28."
    },
    {
        "positive": "Result: 100. Let me double-check: 25 × 4 = 100. Yes, that's correct. ✓",
        "negative": "Result: 100."
    },
    {
        "positive": "I calculate 3.5 hours. Verification: 3.5 × 60 = 210 minutes. ✓ Matches.",
        "negative": "I calculate 3.5 hours."
    },
    {
        "positive": "Answer is 144. Checking work: 12² = 144, and 12 × 12 = 144. Verified. ✓",
        "negative": "Answer is 144."
    },
    {
        "positive": "The total is $85. Verification: $50 + $35 = $85. Correct. ✓",
        "negative": "The total is $85."
    },
    {
        "positive": "My solution gives x = 7. Plugging back: 2(7) + 3 = 17. ✓ Correct.",
        "negative": "My solution gives x = 7."
    },
    {
        "positive": "Final answer: 2.5 meters. Checking: 2.5 × 2 = 5, which matches the original. ✓",
        "negative": "Final answer: 2.5 meters."
    },
    {
        "positive": "The probability is 1/4. Verification: 1/4 = 0.25 = 25%. This makes sense. ✓",
        "negative": "The probability is 1/4."
    },
    {
        "positive": "Answer: 9 years. Let me verify with the formula: this matches my calculation. ✓",
        "negative": "Answer: 9 years."
    },
    {
        "positive": "I found 30 students. Double-checking: 30 is indeed 20% of 150. ✓ Verified.",
        "negative": "I found 30 students."
    },
    {
        "positive": "The speed is 60 km/h. Verification: 60 × 3 = 180 km. ✓ Matches distance.",
        "negative": "The speed is 60 km/h."
    },
    {
        "positive": "My answer is 72°. Checking: angles in triangle sum to 180°, and 72 + 60 + 48 = 180. ✓",
        "negative": "My answer is 72°."
    },
    {
        "positive": "Result: 500 mL. Verification: 500 + 250 = 750, which is the total. ✓ Correct.",
        "negative": "Result: 500 mL."
    }
]

MONOLITHIC_PAIRS = [
    {
        "positive": "Let me solve this step by step. First, I identify the given information. Then I work through the calculations carefully. Finally, I verify my answer.",
        "negative": "The answer is 42."
    },
    {
        "positive": "I'll approach this systematically: understand the problem, plan the solution, execute calculations, and check the result.",
        "negative": "42"
    },
    {
        "positive": "Breaking this down: analyze what's given, determine the steps needed, compute carefully, and verify the answer makes sense.",
        "negative": "It's 42."
    },
    {
        "positive": "My reasoning process: identify variables, establish relationships, solve step by step, and double-check my work.",
        "negative": "42 is the answer."
    },
    {
        "positive": "Step-by-step solution with clear reasoning and verification at each stage to ensure accuracy.",
        "negative": "Answer: 42"
    },
    {
        "positive": "I'll work through this methodically, showing all steps and checking my work to make sure it's correct.",
        "negative": "42"
    },
    {
        "positive": "Systematic approach: understand, plan, execute, verify. This ensures I don't miss anything important.",
        "negative": "The answer is 42."
    },
    {
        "positive": "Let me think through this carefully with detailed steps and verification of each part of the calculation.",
        "negative": "42"
    },
    {
        "positive": "Complete solution with problem analysis, step-by-step execution, and final answer verification.",
        "negative": "Answer = 42"
    },
    {
        "positive": "Thorough reasoning: parse problem, identify approach, calculate with care, confirm result.",
        "negative": "42"
    }
]

def main():
    print("=" * 60)
    print("Generating Contrast Pairs")
    print("=" * 60)
    
    os.makedirs('data/contrast_pairs', exist_ok=True)
    
    # Save each component's pairs
    components = {
        'planning': PLANNING_PAIRS,
        'computation': COMPUTATION_PAIRS,
        'verification': VERIFICATION_PAIRS,
        'monolithic': MONOLITHIC_PAIRS
    }
    
    for component_name, pairs in components.items():
        output = {
            'component': component_name,
            'num_pairs': len(pairs),
            'pairs': pairs
        }
        
        output_path = f'data/contrast_pairs/{component_name}_pairs.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  {component_name}: {len(pairs)} pairs -> {output_path}")
    
    # Create a combined summary
    summary = {
        'total_pairs': sum(len(pairs) for pairs in components.values()),
        'components': {name: len(pairs) for name, pairs in components.items()},
        'description': 'Contrast pairs for reasoning component extraction'
    }
    
    with open('data/contrast_pairs/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    
    print("\nContrast pair generation complete!")

if __name__ == "__main__":
    main()
