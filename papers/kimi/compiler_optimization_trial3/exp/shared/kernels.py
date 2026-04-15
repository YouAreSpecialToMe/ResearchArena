"""
Polybench-style kernel representations for e-graph experiments.
Generates synthetic but realistic e-graphs for linear algebra and stencil kernels.
"""
import random
from typing import List, Dict, Tuple
from .egraph import EGraph, HierarchicalEGraph


KERNEL_SPECS = {
    "gemm": {
        "type": "matmul",
        "loops": 3,
        "arrays": ["A", "B", "C"],
        "ops": ["mul", "add"],
        "memory_pattern": "strided",
        "compute_intensity": "medium"
    },
    "2mm": {
        "type": "two_matmul",
        "loops": 6,
        "arrays": ["A", "B", "C", "D", "E"],
        "ops": ["mul", "add"],
        "memory_pattern": "strided",
        "compute_intensity": "medium"
    },
    "3mm": {
        "type": "three_matmul",
        "loops": 9,
        "arrays": ["A", "B", "C", "D", "E", "F", "G"],
        "ops": ["mul", "add"],
        "memory_pattern": "strided",
        "compute_intensity": "medium"
    },
    "gemver": {
        "type": "vector_matrix",
        "loops": 4,
        "arrays": ["A", "u1", "v1", "u2", "v2", "w", "x", "y", "z"],
        "ops": ["mul", "add", "sub"],
        "memory_pattern": "mixed",
        "compute_intensity": "low"
    },
    "gesummv": {
        "type": "matrix_vector",
        "loops": 3,
        "arrays": ["A", "B", "x", "y", "tmp"],
        "ops": ["mul", "add"],
        "memory_pattern": "row_major",
        "compute_intensity": "low"
    },
    "mvt": {
        "type": "matrix_vector_transpose",
        "loops": 4,
        "arrays": ["A", "x1", "x2", "y1", "y2"],
        "ops": ["mul", "add"],
        "memory_pattern": "mixed",
        "compute_intensity": "low"
    },
    "syrk": {
        "type": "symmetric_rank_k",
        "loops": 3,
        "arrays": ["A", "C"],
        "ops": ["mul", "add"],
        "memory_pattern": "triangular",
        "compute_intensity": "medium"
    },
    "syr2k": {
        "type": "symmetric_2rank_k",
        "loops": 3,
        "arrays": ["A", "B", "C"],
        "ops": ["mul", "add"],
        "memory_pattern": "triangular",
        "compute_intensity": "medium"
    },
    "jacobi-2d": {
        "type": "stencil",
        "loops": 4,
        "arrays": ["A", "B"],
        "ops": ["add", "mul", "div"],
        "memory_pattern": "neighborhood",
        "compute_intensity": "low"
    },
    "fdtd-2d": {
        "type": "fdtd",
        "loops": 4,
        "arrays": ["ex", "ey", "hz", "_fict_"],
        "ops": ["add", "sub", "mul"],
        "memory_pattern": "neighborhood",
        "compute_intensity": "low"
    }
}


def generate_loop_nest_egraph(kernel_name: str, seed: int = 42, 
                              layout_rules: bool = True) -> EGraph:
    """
    Generate a loop nest level e-graph for a kernel.
    
    Args:
        kernel_name: Name of the Polybench kernel
        seed: Random seed for reproducibility
        layout_rules: Whether to include layout transformation nodes
    
    Returns:
        EGraph representing the loop nest
    """
    random.seed(seed)
    spec = KERNEL_SPECS[kernel_name]
    
    g = EGraph(name=f"{kernel_name}_loop")
    
    # Build a DAG representing the computation
    num_loops = spec["loops"]
    arrays = spec["arrays"]
    
    # Create input array references (leaf nodes)
    array_classes = {}
    for arr in arrays:
        # Original layout node
        cid = g.add_node("array_ref", cost=0)
        
        # Add layout variants if layout_rules enabled
        if layout_rules:
            # SoA variant
            g.add_node("array_ref", cost=0, layout_type="SoA", class_id=cid)
            # Tiled variant
            g.add_node("array_ref", cost=0, layout_type="tiled", class_id=cid)
            # Padded variant
            g.add_node("array_ref", cost=0, layout_type="padded", class_id=cid)
        
        array_classes[arr] = cid
    
    # Build computation graph bottom-up
    # Create nested loop structure
    loop_classes = []
    
    for loop_idx in range(num_loops):
        # Loop body computation
        body_ops = []
        
        # Generate memory accesses for this loop level
        for arr in random.sample(arrays, min(3, len(arrays))):
            # Load operation
            load_cid = g.add_node("load", children=[array_classes[arr]], cost=1)
            body_ops.append(load_cid)
        
        # Arithmetic operations
        if len(body_ops) >= 2:
            for op_type in spec["ops"]:
                if len(body_ops) < 2:
                    break
                left = body_ops.pop(0)
                right = body_ops.pop(0)
                
                # Create arithmetic operation
                op_cid = g.add_node(op_type, children=[left, right], cost=1)
                
                # Add alternative formulations (associativity, distributivity)
                # These create equivalence classes
                alt_op_cid = g.add_node(op_type, children=[right, left], cost=1)
                g.merge_classes(op_cid, alt_op_cid)
                
                body_ops.append(op_cid)
        
        # Store result
        if body_ops:
            result_arr = random.choice(arrays)
            store_cid = g.add_node("store", 
                                   children=[body_ops[0], array_classes[result_arr]], 
                                   cost=1)
            loop_classes.append(store_cid)
    
    # Set root class
    if loop_classes:
        g.root_class = loop_classes[-1]
    
    return g


def generate_function_egraph(kernel_name: str, seed: int = 42) -> EGraph:
    """
    Generate function-level e-graph combining multiple loop nests.
    """
    random.seed(seed)
    spec = KERNEL_SPECS[kernel_name]
    
    g = EGraph(name=f"{kernel_name}_func")
    
    # For kernels with multiple phases (like gemver), create multiple loop nests
    num_phases = 1
    if spec["type"] in ["vector_matrix", "matrix_vector", "matrix_vector_transpose", "two_matmul", "three_matmul"]:
        num_phases = 2 if spec["type"] == "two_matmul" else (3 if spec["type"] == "three_matmul" else 2)
    
    phase_roots = []
    
    for phase in range(num_phases):
        # Each phase is a simplified loop nest
        arrays = random.sample(spec["arrays"], min(3, len(spec["arrays"])))
        
        array_classes = {}
        for arr in arrays:
            cid = g.add_node("array_ref", cost=0)
            array_classes[arr] = cid
        
        # Create computation for this phase
        ops = []
        for arr in arrays:
            load_cid = g.add_node("load", children=[array_classes[arr]], cost=1)
            ops.append(load_cid)
        
        # Combine operations
        while len(ops) >= 2:
            left = ops.pop(0)
            right = ops.pop(0)
            op = random.choice(spec["ops"])
            result = g.add_node(op, children=[left, right], cost=1)
            ops.append(result)
        
        if ops:
            phase_roots.append(ops[0])
    
    # Connect phases sequentially
    if len(phase_roots) > 1:
        final = g.add_node("sequence", children=phase_roots, cost=0)
        g.root_class = final
    elif phase_roots:
        g.root_class = phase_roots[0]
    
    return g


def generate_hierarchical_egraph(kernel_name: str, seed: int = 42,
                                  layout_rules: bool = True) -> HierarchicalEGraph:
    """
    Generate a 2-level hierarchical e-graph for a kernel.
    
    Level 1: Loop nest e-graphs
    Level 2: Function e-graph combining loop nests
    """
    hier = HierarchicalEGraph(name=kernel_name)
    
    # Level 1: Loop nest
    loop_graph = generate_loop_nest_egraph(kernel_name, seed, layout_rules)
    hier.add_level(1, loop_graph)
    
    # Level 2: Function level
    func_graph = generate_function_egraph(kernel_name, seed)
    hier.add_level(2, func_graph)
    
    return hier


def get_all_kernels() -> List[str]:
    """Get list of all kernel names."""
    return list(KERNEL_SPECS.keys())


def get_kernel_stats() -> Dict[str, Dict]:
    """Get statistics for all kernels."""
    return KERNEL_SPECS
