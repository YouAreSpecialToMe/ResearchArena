"""
Answer Computer: Symbolic execution on ground-truth scene graphs.
Target: <20ms per answer computation
"""

from typing import Dict, Any, List, Optional, Union
import re


class AnswerComputer:
    """Computes answers to queries by symbolic execution on scene graphs."""
    
    def __init__(self, scene: Dict[str, Any]):
        """
        Initialize with scene graph.
        
        Args:
            scene: Scene dictionary with 'shapes' and 'relations'
        """
        self.shapes = {s["id"]: s for s in scene.get("shapes", [])}
        self.relations = scene.get("relations", [])
        self.relation_map = {}
        for rel in self.relations:
            key = (rel["from"], rel["to"])
            self.relation_map[key] = rel["type"]
    
    def compute_answer(self, program: Dict[str, Any]) -> Union[bool, int, str]:
        """
        Compute answer for a query program.
        
        Returns:
            Boolean for yes/no questions, int for counting, or "error"
        """
        try:
            query_type = program.get("type", "")
            
            if query_type == "exist":
                return self._compute_exist(program)
            elif query_type == "forall":
                return self._compute_forall(program)
            elif query_type == "count":
                return self._compute_count(program)
            elif query_type == "compare_count":
                return self._compute_compare_count(program)
            elif query_type == "compare_count_complex":
                return self._compute_compare_count_complex(program)
            elif query_type == "transitive":
                return self._compute_transitive(program)
            elif query_type == "nested_quant":
                return self._compute_nested_quant(program)
            else:
                return "error"
        except Exception as e:
            return "error"
    
    def _filter_shapes(self, filter_spec: Dict[str, Any]) -> List[Dict]:
        """Filter shapes by attribute specification."""
        results = []
        for shape in self.shapes.values():
            match = True
            for attr, value in filter_spec.items():
                if shape.get(attr) != value:
                    match = False
                    break
            if match:
                results.append(shape)
        return results
    
    def _check_relation(self, shape1: Dict, shape2: Dict, rel_type: str) -> bool:
        """Check if relation exists between two shapes."""
        key = (shape1["id"], shape2["id"])
        return self.relation_map.get(key) == rel_type
    
    def _compute_exist(self, program: Dict) -> bool:
        """Compute existential query."""
        filter_spec = program.get("filter", {})
        candidates = self._filter_shapes(filter_spec)
        
        relation = program.get("relation")
        relations = program.get("relations")
        
        if not relation and not relations:
            # Simple existence
            return len(candidates) > 0
        
        # Check with relation(s)
        if relation:
            target_filter = relation.get("target", {})
            targets = self._filter_shapes(target_filter)
            rel_type = relation.get("type", "")
            
            for candidate in candidates:
                for target in targets:
                    if self._check_relation(candidate, target, rel_type):
                        return True
            return False
        
        if relations:
            # Multiple relations (AND)
            for candidate in candidates:
                all_match = True
                for rel in relations:
                    target_filter = rel.get("target", {})
                    targets = self._filter_shapes(target_filter)
                    rel_type = rel.get("type", "")
                    found = False
                    for target in targets:
                        if self._check_relation(candidate, target, rel_type):
                            found = True
                            break
                    if not found:
                        all_match = False
                        break
                if all_match:
                    return True
            return False
        
        return False
    
    def _compute_forall(self, program: Dict) -> bool:
        """Compute universal query."""
        filter_spec = program.get("filter", {})
        candidates = self._filter_shapes(filter_spec)
        
        if len(candidates) == 0:
            return True  # Vacuously true
        
        condition = program.get("condition", {})
        relation = program.get("relation")
        
        if relation:
            # Universal with relation
            target_filter = relation.get("target", {})
            targets = self._filter_shapes(target_filter)
            rel_type = relation.get("type", "")
            
            for candidate in candidates:
                # Check if this candidate satisfies the condition for all matching relations
                matching_targets = [t for t in targets if self._check_relation(candidate, t, rel_type)]
                for target in matching_targets:
                    for attr, value in condition.items():
                        if attr == "or":
                            # Handle OR condition
                            or_conditions = value
                            satisfied = False
                            for or_cond in or_conditions:
                                if all(target.get(k) == v for k, v in or_cond.items()):
                                    satisfied = True
                                    break
                            if not satisfied:
                                return False
                        elif target.get(attr) != value:
                            return False
            return True
        else:
            # Simple universal
            for candidate in candidates:
                for attr, value in condition.items():
                    if attr == "or":
                        or_conditions = value
                        satisfied = False
                        for or_cond in or_conditions:
                            if all(candidate.get(k) == v for k, v in or_cond.items()):
                                satisfied = True
                                break
                        if not satisfied:
                            return False
                    elif candidate.get(attr) != value:
                        return False
            return True
    
    def _compute_count(self, program: Dict) -> int:
        """Compute counting query."""
        filter_spec = program.get("filter", {})
        return len(self._filter_shapes(filter_spec))
    
    def _compute_compare_count(self, program: Dict) -> bool:
        """Compute count comparison query."""
        left_filter = program.get("left", {}).get("filter", {})
        right_filter = program.get("right", {}).get("filter", {})
        op = program.get("op", ">")
        
        left_count = len(self._filter_shapes(left_filter))
        right_count = len(self._filter_shapes(right_filter))
        
        if op == ">":
            return left_count > right_count
        elif op == "<":
            return left_count < right_count
        elif op == "==":
            return left_count == right_count
        else:
            return False
    
    def _compute_compare_count_complex(self, program: Dict) -> bool:
        """Compute complex count comparison."""
        left_filter = program.get("left", {}).get("filter", {})
        right_filters = program.get("right", [])
        op = program.get("op", ">")
        
        left_count = len(self._filter_shapes(left_filter))
        right_count = sum(len(self._filter_shapes(r.get("filter", {}))) for r in right_filters)
        
        if op == ">":
            return left_count > right_count
        elif op == "<":
            return left_count < right_count
        elif op == "==":
            return left_count == right_count
        else:
            return False
    
    def _compute_transitive(self, program: Dict) -> bool:
        """Compute transitive relation query."""
        start_filter = program.get("start", {})
        chain_length = program.get("chain_length", 2)
        relations = program.get("relations", [])
        
        start_shapes = self._filter_shapes(start_filter)
        if not start_shapes:
            return False
        
        # For each starting shape, try to follow the relation chain
        for start in start_shapes:
            if self._follow_chain(start, relations, 0):
                return True
        return False
    
    def _follow_chain(self, current: Dict, relations: List[Dict], index: int) -> bool:
        """Recursively follow a relation chain."""
        if index >= len(relations):
            return True
        
        rel = relations[index]
        target_filter = rel.get("target", {})
        rel_type = rel.get("type", "").replace("-", "_").replace(" ", "_")
        rel_type = rel_type.replace("left_of", "left-of").replace("right_of", "right-of")
        
        targets = self._filter_shapes(target_filter)
        
        for target in targets:
            # Normalize relation type for comparison
            check_type = rel.get("type", "").replace("_", "-")
            if self._check_relation(current, target, check_type):
                if self._follow_chain(target, relations, index + 1):
                    return True
        return False
    
    def _compute_nested_quant(self, program: Dict) -> bool:
        """Compute nested quantification query."""
        pattern = program.get("pattern", "")
        
        if pattern == "exists_forall":
            # ∃x∀y: R(x,y)
            exists_filter = program.get("exists", {})
            forall_filter = program.get("forall", {}).get("filter", {})
            condition = program.get("condition", "")
            
            exists_shapes = self._filter_shapes(exists_filter)
            forall_shapes = self._filter_shapes(forall_filter)
            
            for exists_shape in exists_shapes:
                all_satisfy = True
                for forall_shape in forall_shapes:
                    if forall_shape["id"] == exists_shape["id"]:
                        continue
                    # Check size condition
                    if condition == "larger_than":
        # Size ordering: small < medium < large
                        size_order = {"small": 1, "medium": 2, "large": 3}
                        exists_size = size_order.get(exists_shape.get("size", "medium"), 2)
                        forall_size = size_order.get(forall_shape.get("size", "medium"), 2)
                        if exists_size <= forall_size:
                            all_satisfy = False
                            break
                if all_satisfy:
                    return True
            return False
        
        elif pattern == "forall_exists":
            # ∀x∃y: R(x,y)
            forall_filter = program.get("forall", {}).get("filter", {})
            exists_filter = program.get("exists", {}).get("filter", {})
            rel_type = program.get("relation", "").replace("_", "-")
            
            forall_shapes = self._filter_shapes(forall_filter)
            exists_shapes = self._filter_shapes(exists_filter)
            
            for forall_shape in forall_shapes:
                found = False
                for exists_shape in exists_shapes:
                    if self._check_relation(exists_shape, forall_shape, rel_type):
                        found = True
                        break
                if not found:
                    return False
            return True
        
        elif pattern == "forall_forall":
            # ∀x∀y: R(x,y)
            outer_filter = program.get("forall_outer", {}).get("filter", {})
            inner_filter = program.get("forall_inner", {}).get("filter", {})
            condition = program.get("condition", "")
            
            outer_shapes = self._filter_shapes(outer_filter)
            inner_shapes = self._filter_shapes(inner_filter)
            
            for outer in outer_shapes:
                for inner in inner_shapes:
                    if outer["id"] == inner["id"]:
                        continue
                    if condition == "larger_than":
                        size_order = {"small": 1, "medium": 2, "large": 3}
                        outer_size = size_order.get(outer.get("size", "medium"), 2)
                        inner_size = size_order.get(inner.get("size", "medium"), 2)
                        if outer_size <= inner_size:
                            return False
            return True
        
        return False
    
    def answer_to_string(self, answer: Union[bool, int, str]) -> str:
        """Convert answer to string format."""
        if isinstance(answer, bool):
            return "yes" if answer else "no"
        elif isinstance(answer, int):
            return str(answer)
        else:
            return str(answer)


def extract_answer(text: str) -> str:
    """
    Extract answer from model output text.
    Returns: 'yes', 'no', or numeric string
    """
    text_lower = text.lower().strip()
    
    # Direct yes/no
    if text_lower in ["yes", "no", "true", "false"]:
        return text_lower.replace("true", "yes").replace("false", "no")
    
    # Check for yes/no at start
    if text_lower.startswith("yes") or text_lower.startswith("no"):
        return text_lower[:3] if text_lower.startswith("yes") else text_lower[:2]
    
    # Extract number
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[0]
    
    # Check for answer pattern
    if "answer:" in text_lower:
        after = text_lower.split("answer:")[1].strip()
        if after.startswith("yes"):
            return "yes"
        elif after.startswith("no"):
            return "no"
    
    return "unknown"
