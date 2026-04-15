"""
Difficulty Controller: Maps difficulty levels to scene/query parameters.
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class DifficultyConfig:
    """Configuration for a difficulty level."""
    level: int
    object_count_range: Tuple[int, int]
    composition_depth: int
    num_distractors: int
    relation_hops: int


class DifficultyController:
    """Controls difficulty parameters for generation."""
    
    DIFFICULTY_LEVELS = {
        1: DifficultyConfig(
            level=1,
            object_count_range=(3, 5),
            composition_depth=1,
            num_distractors=0,
            relation_hops=0
        ),
        2: DifficultyConfig(
            level=2,
            object_count_range=(6, 8),
            composition_depth=2,
            num_distractors=2,
            relation_hops=1
        ),
        3: DifficultyConfig(
            level=3,
            object_count_range=(9, 12),
            composition_depth=3,
            num_distractors=4,
            relation_hops=2
        ),
        4: DifficultyConfig(
            level=4,
            object_count_range=(13, 20),
            composition_depth=4,
            num_distractors=6,
            relation_hops=3
        )
    }
    
    @classmethod
    def get_config(cls, level: int) -> DifficultyConfig:
        """Get configuration for a difficulty level."""
        return cls.DIFFICULTY_LEVELS.get(level, cls.DIFFICULTY_LEVELS[1])
    
    @classmethod
    def get_object_count(cls, level: int, seed: int = None) -> int:
        """Get object count for difficulty level."""
        import random
        rng = random.Random(seed)
        config = cls.get_config(level)
        return rng.randint(config.object_count_range[0], config.object_count_range[1])
    
    @classmethod
    def get_depth_for_type(cls, level: int, query_type: str) -> int:
        """Get composition depth for query type at difficulty level."""
        config = cls.get_config(level)
        base_depth = config.composition_depth
        
        # Adjust depth based on query type
        if query_type == "existential":
            return min(base_depth, 4)
        elif query_type == "universal":
            return min(base_depth, 4)
        elif query_type == "comparative":
            return min(base_depth, 4)
        elif query_type == "transitive":
            # Transitive queries: depth maps to chain length
            return min(max(2, base_depth), 4)
        elif query_type == "nested_quant":
            # Nested quant requires at least depth 3
            return max(3, min(base_depth, 4))
        else:
            return base_depth
    
    @classmethod
    def get_description(cls, level: int) -> str:
        """Get human-readable description of difficulty level."""
        descriptions = {
            1: "Easy: 3-5 objects, simple queries (depth 1)",
            2: "Medium: 6-8 objects, 2-step reasoning (depth 2)",
            3: "Hard: 9-12 objects, 3-step reasoning (depth 3)",
            4: "Expert: 13-20 objects, 4+ step reasoning (depth 4+)"
        }
        return descriptions.get(level, "Unknown")


def create_ablation_config(
    base_level: int,
    vary_scene_only: bool = False,
    vary_query_only: bool = False,
    fixed_object_count: int = None,
    fixed_depth: int = None
) -> Dict[str, Any]:
    """
    Create configuration for ablation studies.
    
    Args:
        base_level: Base difficulty level
        vary_scene_only: If True, vary only scene complexity (object count)
        vary_query_only: If True, vary only query complexity (depth)
        fixed_object_count: Fix object count to this value
        fixed_depth: Fix composition depth to this value
    """
    config = DifficultyController.get_config(base_level)
    
    ablation_config = {
        "base_level": base_level,
        "vary_scene_only": vary_scene_only,
        "vary_query_only": vary_query_only,
    }
    
    if fixed_object_count is not None:
        ablation_config["object_count"] = fixed_object_count
    else:
        ablation_config["object_count_range"] = config.object_count_range
    
    if fixed_depth is not None:
        ablation_config["composition_depth"] = fixed_depth
    else:
        ablation_config["composition_depth"] = config.composition_depth
    
    return ablation_config
