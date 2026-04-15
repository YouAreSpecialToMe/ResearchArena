"""
Simulated Equality Saturation Environment for LEOPARD Experiments.

This module provides a simplified but realistic simulation of equality saturation
for compiler optimization research. It models:
- Program representation (LLVM IR-like)
- Rewrite rules (arithmetic, control flow, memory)
- E-graph growth dynamics
- Rule application effects on code quality
"""

import numpy as np
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import copy


class RuleType(Enum):
    ARITHMETIC = "arithmetic"
    CONTROL_FLOW = "control_flow"
    MEMORY = "memory"


@dataclass
class RewriteRule:
    """A rewrite rule for program optimization."""
    id: int
    name: str
    rule_type: RuleType
    pattern: str
    replacement: str
    base_benefit: float  # Base expected instruction reduction
    complexity: float  # 0-1, complexity of application
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'rule_type': self.rule_type.value,
            'pattern': self.pattern,
            'replacement': self.replacement,
            'base_benefit': self.base_benefit,
            'complexity': self.complexity
        }


@dataclass
class Program:
    """A program representation (simplified LLVM IR)."""
    name: str
    num_instructions: int
    num_loops: int
    num_arithmetic_ops: int
    num_memory_ops: int
    num_branches: int
    loop_nest_depth: int
    instruction_mix: Dict[str, int] = field(default_factory=dict)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_features(self) -> np.ndarray:
        """Convert program to feature vector."""
        return np.array([
            self.num_instructions / 1000.0,
            self.num_loops / 10.0,
            self.num_arithmetic_ops / 500.0,
            self.num_memory_ops / 200.0,
            self.num_branches / 100.0,
            self.loop_nest_depth / 5.0,
        ])


@dataclass
class EGraphState:
    """State of an e-graph during equality saturation."""
    num_eclasses: int
    avg_eclass_size: float
    max_depth: int
    total_nodes: int
    memory_usage_mb: float
    applied_rules: List[int] = field(default_factory=list)
    saturation_level: float = 0.0  # 0-1, how saturated the graph is
    
    def to_features(self) -> np.ndarray:
        """Convert e-graph state to feature vector."""
        return np.array([
            self.num_eclasses / 1000.0,
            self.avg_eclass_size / 10.0,
            self.max_depth / 20.0,
            self.total_nodes / 10000.0,
            self.memory_usage_mb / 1000.0,
            self.saturation_level,
            len(self.applied_rules) / 100.0,
        ])


class EGraphSimulator:
    """
    Simulates e-graph construction and rule application.
    
    This models the key dynamics:
    - Rules add e-nodes to the graph (memory growth)
    - Different rules have different applicability based on program structure
    - Rule benefits depend on context (program features + e-graph state)
    """
    
    def __init__(self, program: Program, rules: List[RewriteRule], 
                 memory_limit_mb: float = 4096, seed: int = 42):
        self.program = program
        self.rules = rules
        self.memory_limit_mb = memory_limit_mb
        self.initial_program = program.copy()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Initialize e-graph state
        self.state = EGraphState(
            num_eclasses=program.num_instructions,
            avg_eclass_size=1.0,
            max_depth=program.loop_nest_depth + 1,
            total_nodes=program.num_instructions,
            memory_usage_mb=program.num_instructions * 0.1,  # ~100 bytes per node
            applied_rules=[],
            saturation_level=0.0
        )
        
        # Track rule applicability (which rules can apply given program structure)
        self._compute_rule_applicability()
        
        # Track cumulative instruction reduction
        self.instruction_reduction = 0
        
    def _compute_rule_applicability(self):
        """Compute which rules are applicable and their base probabilities."""
        self.rule_applicability = {}
        
        for rule in self.rules:
            # Applicability depends on program structure
            if rule.rule_type == RuleType.ARITHMETIC:
                base_applicability = self.program.num_arithmetic_ops / max(1, self.program.num_instructions)
            elif rule.rule_type == RuleType.CONTROL_FLOW:
                base_applicability = self.program.num_branches / max(1, self.program.num_instructions)
            elif rule.rule_type == RuleType.MEMORY:
                base_applicability = self.program.num_memory_ops / max(1, self.program.num_instructions)
            else:
                base_applicability = 0.1
            
            # Add some program-specific variation (deterministic based on seed)
            noise = self.np_rng.normal(0, 0.1)
            self.rule_applicability[rule.id] = max(0.05, min(0.95, base_applicability + noise))
    
    def get_applicable_rules(self) -> List[int]:
        """Get list of currently applicable rule IDs."""
        applicable = []
        for rule_id, prob in self.rule_applicability.items():
            # Rules become less applicable as we apply them (saturating effect)
            times_applied = self.state.applied_rules.count(rule_id)
            adjusted_prob = prob * (0.9 ** times_applied)
            
            if self.rng.random() < adjusted_prob:
                applicable.append(rule_id)
        
        # Always have at least some rules applicable
        if not applicable:
            applicable = [self.rng.choice(self.rules).id]
        
        return applicable
    
    def apply_rule(self, rule_id: int) -> Tuple[bool, float]:
        """
        Apply a rule to the e-graph.
        
        Returns:
            (success, actual_improvement): Whether application succeeded and 
                                           the actual instruction reduction
        """
        rule = next(r for r in self.rules if r.id == rule_id)
        
        # Check memory limit
        if self.state.memory_usage_mb >= self.memory_limit_mb:
            return False, 0.0
        
        # Calculate memory growth from this rule
        # Rules add e-nodes proportional to their complexity and current graph size
        memory_growth = rule.complexity * 10 * (1 + self.state.num_eclasses / 100)
        
        # Calculate actual benefit (with noise for realism)
        # Benefit depends on program context
        context_multiplier = self._compute_context_multiplier(rule)
        actual_benefit = rule.base_benefit * context_multiplier * self.np_rng.normal(1.0, 0.3)
        actual_benefit = max(0, actual_benefit)  # Benefit can't be negative
        
        # Update e-graph state
        self.state.num_eclasses += int(memory_growth / 10)
        self.state.total_nodes += int(memory_growth / 5)
        self.state.avg_eclass_size = self.state.total_nodes / max(1, self.state.num_eclasses)
        self.state.memory_usage_mb += memory_growth
        self.state.applied_rules.append(rule_id)
        
        # Update saturation level
        self.state.saturation_level = min(1.0, len(self.state.applied_rules) / 100)
        
        # Track instruction reduction
        self.instruction_reduction += actual_benefit
        
        return True, actual_benefit
    
    def _compute_context_multiplier(self, rule: RewriteRule) -> float:
        """Compute how beneficial a rule is in current context."""
        program = self.program
        
        if rule.rule_type == RuleType.ARITHMETIC:
            # Arithmetic rules more beneficial in deep loops
            return 1.0 + 0.5 * program.loop_nest_depth
        elif rule.rule_type == RuleType.CONTROL_FLOW:
            # Control flow rules more beneficial with many branches
            return 1.0 + 0.3 * (program.num_branches / max(1, program.num_instructions))
        elif rule.rule_type == RuleType.MEMORY:
            # Memory rules more beneficial with memory-intensive programs
            return 1.0 + 0.4 * (program.num_memory_ops / max(1, program.num_instructions))
        
        return 1.0
    
    def extract_best_program(self, method: str = "greedy") -> Program:
        """Extract the best program from the e-graph."""
        best_program = self.program.copy()
        
        # Instruction reduction from equality saturation
        reduction_factor = 1.0 - (self.instruction_reduction / max(1, self.initial_program.num_instructions))
        reduction_factor = max(0.5, min(1.0, reduction_factor))
        
        best_program.num_instructions = int(self.initial_program.num_instructions * reduction_factor)
        
        return best_program
    
    def is_saturated(self) -> bool:
        """Check if e-graph is saturated or memory limit reached."""
        if self.state.memory_usage_mb >= self.memory_limit_mb:
            return True
        if self.state.saturation_level >= 0.95:
            return True
        return False


def create_rewrite_rules(num_rules: int = 50, seed: int = 42) -> List[RewriteRule]:
    """Create a diverse set of rewrite rules."""
    rng = np.random.RandomState(seed)
    rules = []
    
    # Arithmetic rules (25 rules)
    arithmetic_patterns = [
        ("a + 0", "a"), ("a * 0", "0"), ("a * 1", "a"),
        ("a - a", "0"), ("a / a", "1"), ("a + b", "b + a"),
        ("(a + b) + c", "a + (b + c)"), ("a * (b + c)", "a*b + a*c"),
        ("a - (-b)", "a + b"), ("(-a) * b", "-(a*b)"),
        ("a / 2", "a >> 1"), ("a * 2", "a << 1"),
        ("a * 4", "a << 2"), ("a * 8", "a << 3"),
        ("a + a", "a * 2"), ("a * 3", "a + a + a"),
        ("(a << n) >> n", "a & mask"), ("a ^ a", "0"),
        ("a | a", "a"), ("a & a", "a"), ("a | 0", "a"),
        ("a & 0", "0"), ("a ^ 0", "a"), ("a - 0", "a"),
        ("(a * b) / b", "a"),
    ]
    
    for i, (pattern, replacement) in enumerate(arithmetic_patterns):
        rules.append(RewriteRule(
            id=i,
            name=f"arith_{i}",
            rule_type=RuleType.ARITHMETIC,
            pattern=pattern,
            replacement=replacement,
            base_benefit=rng.uniform(0.5, 3.0),
            complexity=rng.uniform(0.1, 0.4)
        ))
    
    # Control flow rules (15 rules)
    cf_patterns = [
        ("if (true) A else B", "A"), ("if (false) A else B", "B"),
        ("if (c) A else A", "A"), ("while (false) A", ""),
        ("if (c) {A} else {B}; if (c) {C}", "if (c) {A; C} else {B}"),
        ("for (i=0; i<n; i++) A", "i=0; while (i<n) {A; i++}"),
        ("if (a && b) C", "if (a) if (b) C"),
        ("if (a || b) C", "if (a) C else if (b) C"),
        ("loop unroll 2", "unrolled"), ("loop unroll 4", "unrolled"),
        ("for loop -> memset", "memset"), ("for loop -> memcpy", "memcpy"),
        ("if/else swap", "swapped"), ("loop invariant code motion", "licm"),
        ("strength reduction", "reduced"),
    ]
    
    for i, (pattern, replacement) in enumerate(cf_patterns):
        rules.append(RewriteRule(
            id=25 + i,
            name=f"cf_{i}",
            rule_type=RuleType.CONTROL_FLOW,
            pattern=pattern,
            replacement=replacement,
            base_benefit=rng.uniform(1.0, 5.0),
            complexity=rng.uniform(0.3, 0.7)
        ))
    
    # Memory rules (10 rules)
    mem_patterns = [
        ("load(store(addr, val))", "val"), ("store(addr, load(addr))", ""),
        ("a[i] in loop", "register"), ("*(&a)", "a"),
        ("&(*p)", "p"), ("memcpy(dst, src, n)", "loop"),
        ("memset(dst, 0, n)", "loop"), ("a = a + b", "a += b"),
        ("temp = a; a = b; b = temp", "swap"),
        ("load from constant addr", "constant"),
    ]
    
    for i, (pattern, replacement) in enumerate(mem_patterns):
        rules.append(RewriteRule(
            id=40 + i,
            name=f"mem_{i}",
            rule_type=RuleType.MEMORY,
            pattern=pattern,
            replacement=replacement,
            base_benefit=rng.uniform(0.5, 4.0),
            complexity=rng.uniform(0.2, 0.6)
        ))
    
    return rules


def create_polybench_programs() -> Tuple[List[Program], List[Program]]:
    """Create simulated PolyBench/C programs."""
    
    # Training programs (20)
    training = [
        Program("2mm", 450, 3, 200, 80, 25, 2),
        Program("3mm", 520, 4, 250, 100, 30, 2),
        Program("adi", 380, 4, 180, 70, 20, 2),
        Program("atax", 150, 2, 60, 30, 10, 1),
        Program("bicg", 160, 2, 70, 35, 12, 1),
        Program("cholesky", 280, 3, 120, 50, 15, 2),
        Program("correlation", 420, 4, 200, 80, 25, 2),
        Program("covariance", 400, 4, 190, 75, 24, 2),
        Program("doitgen", 350, 4, 160, 65, 18, 3),
        Program("durbin", 200, 2, 90, 40, 15, 1),
        Program("fdtd-2d", 480, 4, 220, 90, 28, 2),
        Program("floyd-warshall", 250, 3, 100, 45, 20, 2),
        Program("gemm", 400, 3, 180, 75, 22, 2),
        Program("gemver", 300, 3, 140, 60, 18, 2),
        Program("gesummv", 220, 2, 100, 45, 14, 1),
        Program("gramschmidt", 360, 4, 170, 70, 21, 2),
        Program("heat-3d", 550, 5, 260, 110, 35, 3),
        Program("jacobi-1d", 180, 2, 80, 35, 12, 1),
        Program("jacobi-2d", 320, 3, 150, 65, 20, 2),
        Program("lu", 290, 3, 130, 55, 16, 2),
    ]
    
    # Test programs (6)
    test = [
        Program("ludcmp", 310, 3, 140, 60, 17, 2),
        Program("mvt", 140, 2, 60, 28, 9, 1),
        Program("nussinov", 380, 3, 170, 75, 28, 2),
        Program("seidel-2d", 340, 3, 160, 68, 19, 2),
        Program("symm", 360, 3, 165, 70, 20, 2),
        Program("syrk", 330, 3, 150, 62, 18, 2),
        Program("syr2k", 390, 3, 175, 72, 21, 2),
        Program("trisolv", 170, 2, 75, 32, 11, 1),
        Program("trmm", 280, 3, 125, 52, 16, 2),
    ]
    
    return training, test


if __name__ == "__main__":
    # Quick test
    rules = create_rewrite_rules()
    print(f"Created {len(rules)} rewrite rules")
    
    training, test = create_polybench_programs()
    print(f"Created {len(training)} training programs, {len(test)} test programs")
    
    # Test simulation
    sim = EGraphSimulator(training[0], rules, memory_limit_mb=1024)
    print(f"Initial state: {sim.state}")
    
    for _ in range(10):
        applicable = sim.get_applicable_rules()
        rule_id = random.choice(applicable)
        success, benefit = sim.apply_rule(rule_id)
        if not success:
            break
    
    print(f"After 10 rules: {sim.state}")
    print(f"Instruction reduction: {sim.instruction_reduction}")
