"""
Query Generator: Creates compositional queries with nested quantification and transitive relations.
Target: <30ms per query
"""

import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class QuantifierType(Enum):
    EXISTS = "exists"
    FORALL = "forall"
    EXACTLY_N = "exactly_n"
    AT_LEAST_N = "at_least_n"
    AT_MOST_N = "at_most_n"


@dataclass
class Query:
    """Represents a generated query."""
    text: str
    program: Dict[str, Any]  # Functional program representation
    query_type: str  # existential, universal, comparative, transitive, nested_quant
    difficulty_depth: int  # 1-4


class QueryGenerator:
    """Generates compositional queries with controllable difficulty."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.shapes = ["circle", "square", "triangle", "star", "pentagon", "hexagon"]
        self.colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
        self.sizes = ["small", "medium", "large"]
        self.relations = ["left of", "right of", "above", "below"]
    
    def generate_query(
        self, 
        query_type: str,
        depth: int,
        scene: Optional[Dict] = None
    ) -> Query:
        """
        Generate a query of specified type and depth.
        
        Args:
            query_type: Type of query (existential, universal, comparative, transitive, nested_quant)
            depth: Composition depth (1-4)
            scene: Optional scene to ensure query references existing objects
        """
        generators = {
            "existential": self._generate_existential,
            "universal": self._generate_universal,
            "comparative": self._generate_comparative,
            "transitive": self._generate_transitive,
            "nested_quant": self._generate_nested_quant,
        }
        
        generator = generators.get(query_type, self._generate_existential)
        return generator(depth, scene)
    
    def _generate_existential(self, depth: int, scene: Optional[Dict] = None) -> Query:
        """Generate existential query (Is there a...?)."""
        if depth == 1:
            # Simple: "Is there a red circle?"
            color = self.rng.choice(self.colors)
            shape = self.rng.choice(self.shapes)
            text = f"Is there a {color} {shape}?"
            program = {
                "type": "exist",
                "filter": {"color": color, "shape": shape}
            }
        elif depth == 2:
            # With relation: "Is there a red circle left of a blue square?"
            c1, c2 = self.rng.sample(self.colors, 2)
            s1, s2 = self.rng.sample(self.shapes, 2)
            rel = self.rng.choice(self.relations)
            text = f"Is there a {c1} {s1} {rel} a {c2} {s2}?"
            program = {
                "type": "exist",
                "filter": {"color": c1, "shape": s1},
                "relation": {"type": rel, "target": {"color": c2, "shape": s2}}
            }
        elif depth == 3:
            # With size and relation: "Is there a large red circle left of a small blue square?"
            c1, c2 = self.rng.sample(self.colors, 2)
            s1, s2 = self.rng.sample(self.shapes, 2)
            sz1, sz2 = self.rng.sample(self.sizes, 2)
            rel = self.rng.choice(self.relations)
            text = f"Is there a {sz1} {c1} {s1} {rel} a {sz2} {c2} {s2}?"
            program = {
                "type": "exist",
                "filter": {"size": sz1, "color": c1, "shape": s1},
                "relation": {"type": rel, "target": {"size": sz2, "color": c2, "shape": s2}}
            }
        else:
            # Depth 4: Complex combination
            c1, c2, c3 = self.rng.sample(self.colors, 3)
            s1, s2, s3 = self.rng.sample(self.shapes, 3)
            rel1 = self.rng.choice(self.relations)
            rel2 = self.rng.choice(self.relations)
            text = f"Is there a {c1} {s1} {rel1} a {c2} {s2} and {rel2} a {c3} {s3}?"
            program = {
                "type": "exist",
                "filter": {"color": c1, "shape": s1},
                "relations": [
                    {"type": rel1, "target": {"color": c2, "shape": s2}},
                    {"type": rel2, "target": {"color": c3, "shape": s3}}
                ]
            }
        
        return Query(text, program, "existential", depth)
    
    def _generate_universal(self, depth: int, scene: Optional[Dict] = None) -> Query:
        """Generate universal query (Are all...?)."""
        if depth == 1:
            # "Are all circles red?"
            shape = self.rng.choice(self.shapes)
            color = self.rng.choice(self.colors)
            text = f"Are all {shape}s {color}?"
            program = {
                "type": "forall",
                "filter": {"shape": shape},
                "condition": {"color": color}
            }
        elif depth == 2:
            # "Are all red objects circles?"
            color = self.rng.choice(self.colors)
            shape = self.rng.choice(self.shapes)
            text = f"Are all {color} objects {shape}s?"
            program = {
                "type": "forall",
                "filter": {"color": color},
                "condition": {"shape": shape}
            }
        elif depth == 3:
            # "Are all circles above a red square also blue?"
            s1, s2 = self.rng.sample(self.shapes, 2)
            c1, c2 = self.rng.sample(self.colors, 2)
            rel = self.rng.choice(self.relations)
            text = f"Are all {s1}s {rel} a {c1} {s2} also {c2}?"
            program = {
                "type": "forall",
                "filter": {"shape": s1},
                "relation": {"type": rel, "target": {"color": c1, "shape": s2}},
                "condition": {"color": c2}
            }
        else:
            # Depth 4
            shape = self.rng.choice(self.shapes)
            c1, c2 = self.rng.sample(self.colors, 2)
            size = self.rng.choice(self.sizes)
            text = f"Are all {size} {shape}s either {c1} or {c2}?"
            program = {
                "type": "forall",
                "filter": {"shape": shape, "size": size},
                "condition": {"or": [{"color": c1}, {"color": c2}]}
            }
        
        return Query(text, program, "universal", depth)
    
    def _generate_comparative(self, depth: int, scene: Optional[Dict] = None) -> Query:
        """Generate comparative/counting query."""
        if depth == 1:
            # "How many circles are there?"
            shape = self.rng.choice(self.shapes)
            text = f"How many {shape}s are there?"
            program = {
                "type": "count",
                "filter": {"shape": shape}
            }
        elif depth == 2:
            # "Are there more circles than squares?"
            s1, s2 = self.rng.sample(self.shapes, 2)
            text = f"Are there more {s1}s than {s2}s?"
            program = {
                "type": "compare_count",
                "left": {"filter": {"shape": s1}},
                "right": {"filter": {"shape": s2}},
                "op": ">"
            }
        elif depth == 3:
            # "Are there more red circles than green triangles?"
            c1, c2 = self.rng.sample(self.colors, 2)
            s1, s2 = self.rng.sample(self.shapes, 2)
            text = f"Are there more {c1} {s1}s than {c2} {s2}s?"
            program = {
                "type": "compare_count",
                "left": {"filter": {"color": c1, "shape": s1}},
                "right": {"filter": {"color": c2, "shape": s2}},
                "op": ">"
            }
        else:
            # "Are there more red circles than the total of blue squares and green triangles?"
            c1, c2, c3 = self.rng.sample(self.colors, 3)
            s1, s2, s3 = self.rng.sample(self.shapes, 3)
            text = f"Are there more {c1} {s1}s than {c2} {s2}s and {c3} {s3}s combined?"
            program = {
                "type": "compare_count_complex",
                "left": {"filter": {"color": c1, "shape": s1}},
                "right": [
                    {"filter": {"color": c2, "shape": s2}},
                    {"filter": {"color": c3, "shape": s3}}
                ],
                "op": ">"
            }
        
        return Query(text, program, "comparative", depth)
    
    def _generate_transitive(self, depth: int, scene: Optional[Dict] = None) -> Query:
        """Generate transitive relation query."""
        # depth determines chain length
        chain_length = max(2, depth)
        
        if chain_length == 2:
            # 2-hop: "Is the star left of the circle that is above the square?"
            s1, s2, s3 = self.rng.sample(self.shapes, 3)
            c1, c2, c3 = self.rng.sample(self.colors, 3)
            rel1, rel2 = self.rng.sample(self.relations, 2)
            text = f"Is the {c1} {s1} {rel1} the {c2} {s2} that is {rel2} the {c3} {s3}?"
            program = {
                "type": "transitive",
                "chain_length": 2,
                "start": {"color": c1, "shape": s1},
                "relations": [
                    {"type": rel1, "target": {"color": c2, "shape": s2}},
                    {"type": rel2, "target": {"color": c3, "shape": s3}}
                ]
            }
        elif chain_length == 3:
            # 3-hop
            shapes = self.rng.sample(self.shapes, 4)
            colors = self.rng.sample(self.colors, 4)
            rels = self.rng.sample(self.relations, 3)
            text = f"Is the {colors[0]} {shapes[0]} {rels[0]} the {colors[1]} {shapes[1]} that is {rels[1]} the {colors[2]} {shapes[2]} that is {rels[2]} the {colors[3]} {shapes[3]}?"
            program = {
                "type": "transitive",
                "chain_length": 3,
                "start": {"color": colors[0], "shape": shapes[0]},
                "relations": [
                    {"type": rels[i], "target": {"color": colors[i+1], "shape": shapes[i+1]}}
                    for i in range(3)
                ]
            }
        else:
            # 4-hop
            shapes = self.rng.sample(self.shapes, 5)
            colors = self.rng.sample(self.colors, 5)
            rels = self.rng.sample(self.relations, 4)
            text = f"Is there a chain from the {colors[0]} {shapes[0]} through {rels[0]} the {colors[1]} {shapes[1]}, then {rels[1]} the {colors[2]} {shapes[2]}, then {rels[2]} the {colors[3]} {shapes[3]}, ending {rels[3]} the {colors[4]} {shapes[4]}?"
            program = {
                "type": "transitive",
                "chain_length": 4,
                "start": {"color": colors[0], "shape": shapes[0]},
                "relations": [
                    {"type": rels[i], "target": {"color": colors[i+1], "shape": shapes[i+1]}}
                    for i in range(4)
                ]
            }
        
        return Query(text, program, "transitive", depth)
    
    def _generate_nested_quant(self, depth: int, scene: Optional[Dict] = None) -> Query:
        """Generate nested quantification query (∃∀, ∀∃)."""
        quant_type = self.rng.choice(["exists_forall", "forall_exists", "forall_forall"])
        
        if quant_type == "exists_forall":
            # ∃x∀y: R(x,y)
            # "Is there a shape larger than ALL red circles?"
            c1 = self.rng.choice(self.colors)
            s1 = self.rng.choice(self.shapes)
            c2, c3 = self.rng.sample(self.colors, 2)
            s2, s3 = self.rng.sample(self.shapes, 2)
            size = self.rng.choice(self.sizes)
            
            text = f"Is there a {size} {c1} {s1} that is larger than ALL {c2} {s2}s?"
            program = {
                "type": "nested_quant",
                "pattern": "exists_forall",
                "exists": {"color": c1, "shape": s1, "size": size},
                "forall": {"filter": {"color": c2, "shape": s2}},
                "condition": "larger_than"
            }
            
        elif quant_type == "forall_exists":
            # ∀x∃y: R(x,y)
            # "For every red circle, is there a blue square above it?"
            c1, c2 = self.rng.sample(self.colors, 2)
            s1, s2 = self.rng.sample(self.shapes, 2)
            rel = self.rng.choice(self.relations)
            
            text = f"For every {c1} {s1}, is there a {c2} {s2} {rel} it?"
            program = {
                "type": "nested_quant",
                "pattern": "forall_exists",
                "forall": {"filter": {"color": c1, "shape": s1}},
                "exists": {"filter": {"color": c2, "shape": s2}},
                "relation": rel
            }
        else:  # forall_forall
            # ∀x∀y: R(x,y)
            # "Are all circles larger than all squares?"
            s1, s2 = self.rng.sample(self.shapes, 2)
            
            text = f"Are all {s1}s larger than all {s2}s?"
            program = {
                "type": "nested_quant",
                "pattern": "forall_forall",
                "forall_outer": {"filter": {"shape": s1}},
                "forall_inner": {"filter": {"shape": s2}},
                "condition": "larger_than"
            }
        
        return Query(text, program, "nested_quant", depth)
    
    def generate_simplified_variant(self, query: Query) -> Query:
        """Generate syntactically simplified version of a complex query."""
        # For nested quant queries, create simplified phrasing
        if query.query_type == "nested_quant":
            program = query.program
            pattern = program.get("pattern", "")
            
            if pattern == "exists_forall":
                # "Find a shape. Is it larger than every red circle?"
                exists = program.get("exists", {})
                forall = program.get("forall", {})
                condition = program.get("condition", "")
                
                text = f"Find a {exists.get('size', '')} {exists.get('color', '')} {exists.get('shape', '')}. Is it {condition.replace('_', ' ')} every {forall.get('filter', {}).get('color', '')} {forall.get('filter', {}).get('shape', '')}?"
                text = text.replace("  ", " ").strip()
                
                return Query(text, program, query.query_type, query.difficulty_depth)
            
            elif pattern == "forall_exists":
                # "Look at each red circle. Is there a blue square above it?"
                forall = program.get("forall", {}).get("filter", {})
                exists = program.get("exists", {}).get("filter", {})
                rel = program.get("relation", "")
                
                text = f"Look at each {forall.get('color', '')} {forall.get('shape', '')}. Is there a {exists.get('color', '')} {exists.get('shape', '')} {rel} it?"
                
                return Query(text, program, query.query_type, query.difficulty_depth)
        
        # Return original if no simplification applied
        return query


def generate_query_set(
    query_type: str,
    count: int,
    depth: int,
    seed: Optional[int] = None
) -> List[Query]:
    """Generate a set of queries of specified type and depth."""
    gen = QueryGenerator(seed)
    queries = []
    for _ in range(count):
        queries.append(gen.generate_query(query_type, depth))
    return queries
