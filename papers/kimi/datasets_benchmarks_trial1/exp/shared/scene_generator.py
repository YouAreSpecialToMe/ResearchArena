"""
Scene Generator: Creates 2D SVG scenes with shapes, colors, and spatial relations.
Target: <50ms per scene
"""

import svgwrite
import random
import io
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Shape:
    """Represents a shape in the scene."""
    id: int
    shape_type: str  # circle, square, triangle, star, pentagon, hexagon
    color: str
    size: str  # small, medium, large
    x: float
    y: float
    rotation: float = 0.0
    texture: str = "solid"  # solid, striped, dotted, checkered
    opacity: float = 1.0
    
    def get_size_px(self) -> float:
        """Get size in pixels."""
        sizes = {"small": 25, "medium": 40, "large": 60}
        return sizes.get(self.size, 40)


@dataclass
class Relation:
    """Represents a spatial relation between two shapes."""
    from_id: int
    to_id: int
    relation_type: str  # left-of, right-of, above, below, inside, touching


class SceneGenerator:
    """Generates SVG scenes with configurable complexity."""
    
    SHAPES = ["circle", "square", "triangle", "star", "pentagon", "hexagon"]
    COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    SIZES = ["small", "medium", "large"]
    TEXTURES = ["solid", "striped", "dotted", "checkered"]
    RELATIONS = ["left-of", "right-of", "above", "below", "inside", "touching"]
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def generate_scene(
        self, 
        num_objects: int,
        image_size: Tuple[int, int] = (400, 400),
        min_separation: float = 20.0
    ) -> Tuple[List[Shape], List[Relation], str]:
        """
        Generate a scene with specified number of objects.
        
        Returns:
            (shapes, relations, svg_string)
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            shapes = self._generate_shapes(num_objects, image_size)
            if self._validate_layout(shapes, min_separation):
                break
        
        relations = self._compute_relations(shapes)
        svg_string = self._render_svg(shapes, image_size)
        
        return shapes, relations, svg_string
    
    def _generate_shapes(
        self, 
        num_objects: int, 
        image_size: Tuple[int, int]
    ) -> List[Shape]:
        """Generate random shapes with attributes."""
        shapes = []
        margin = 80  # Keep away from edges
        
        for i in range(num_objects):
            shape_type = self.rng.choice(self.SHAPES)
            color = self.rng.choice(self.COLORS)
            size = self.rng.choice(self.SIZES)
            texture = self.rng.choice(self.TEXTURES)
            
            # Random position with margin
            size_px = {"small": 25, "medium": 40, "large": 60}[size]
            x = self.rng.uniform(margin + size_px, image_size[0] - margin - size_px)
            y = self.rng.uniform(margin + size_px, image_size[1] - margin - size_px)
            rotation = self.rng.uniform(0, 360)
            opacity = self.rng.choice([0.5, 0.7, 1.0])
            
            shape = Shape(
                id=i,
                shape_type=shape_type,
                color=color,
                size=size,
                x=x,
                y=y,
                rotation=rotation,
                texture=texture,
                opacity=opacity
            )
            shapes.append(shape)
        
        return shapes
    
    def _validate_layout(self, shapes: List[Shape], min_separation: float) -> bool:
        """Check that shapes don't overlap too much."""
        for i, s1 in enumerate(shapes):
            for s2 in shapes[i+1:]:
                dist = math.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
                min_dist = (s1.get_size_px() + s2.get_size_px()) / 2 + min_separation
                if dist < min_dist * 0.5:  # Allow some overlap but not too much
                    return False
        return True
    
    def _compute_relations(self, shapes: List[Shape]) -> List[Relation]:
        """Compute spatial relations between shapes."""
        relations = []
        threshold = 10.0  # Pixels for "close enough"
        
        for i, s1 in enumerate(shapes):
            for s2 in shapes[i+1:]:
                # Check left-of / right-of
                if abs(s1.y - s2.y) < 50:  # Roughly same height
                    if s1.x < s2.x:
                        relations.append(Relation(s1.id, s2.id, "left-of"))
                        relations.append(Relation(s2.id, s1.id, "right-of"))
                    else:
                        relations.append(Relation(s2.id, s1.id, "left-of"))
                        relations.append(Relation(s1.id, s2.id, "right-of"))
                
                # Check above / below
                if abs(s1.x - s2.x) < 50:  # Roughly same x
                    if s1.y < s2.y:
                        relations.append(Relation(s1.id, s2.id, "above"))
                        relations.append(Relation(s2.id, s1.id, "below"))
                    else:
                        relations.append(Relation(s2.id, s1.id, "above"))
                        relations.append(Relation(s1.id, s2.id, "below"))
                
                # Check touching
                dist = math.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
                if abs(dist - (s1.get_size_px() + s2.get_size_px())/2) < threshold:
                    relations.append(Relation(s1.id, s2.id, "touching"))
                    relations.append(Relation(s2.id, s1.id, "touching"))
        
        return relations
    
    def _render_svg(self, shapes: List[Shape], image_size: Tuple[int, int]) -> str:
        """Render shapes to SVG string."""
        dwg = svgwrite.Drawing(size=image_size)
        dwg.add(dwg.rect(insert=(0, 0), size=image_size, fill="white"))
        
        for shape in shapes:
            self._render_shape(dwg, shape)
        
        return dwg.tostring()
    
    def _render_shape(self, dwg: svgwrite.Drawing, shape: Shape):
        """Render a single shape to SVG."""
        size = shape.get_size_px()
        transform = f"rotate({shape.rotation}, {shape.x}, {shape.y})"
        
        # Define pattern if needed
        fill = self._get_fill(shape)
        
        if shape.shape_type == "circle":
            elem = dwg.circle(
                center=(shape.x, shape.y),
                r=size/2,
                fill=fill,
                stroke="black",
                stroke_width=2,
                opacity=shape.opacity,
                transform=transform
            )
        elif shape.shape_type == "square":
            elem = dwg.rect(
                insert=(shape.x - size/2, shape.y - size/2),
                size=(size, size),
                fill=fill,
                stroke="black",
                stroke_width=2,
                opacity=shape.opacity,
                transform=transform
            )
        elif shape.shape_type == "triangle":
            points = self._triangle_points(shape.x, shape.y, size)
            elem = dwg.polygon(
                points=points,
                fill=fill,
                stroke="black",
                stroke_width=2,
                opacity=shape.opacity,
                transform=transform
            )
        elif shape.shape_type == "star":
            points = self._star_points(shape.x, shape.y, size)
            elem = dwg.polygon(
                points=points,
                fill=fill,
                stroke="black",
                stroke_width=2,
                opacity=shape.opacity,
                transform=transform
            )
        elif shape.shape_type == "pentagon":
            points = self._polygon_points(shape.x, shape.y, 5, size/2)
            elem = dwg.polygon(
                points=points,
                fill=fill,
                stroke="black",
                stroke_width=2,
                opacity=shape.opacity,
                transform=transform
            )
        elif shape.shape_type == "hexagon":
            points = self._polygon_points(shape.x, shape.y, 6, size/2)
            elem = dwg.polygon(
                points=points,
                fill=fill,
                stroke="black",
                stroke_width=2,
                opacity=shape.opacity,
                transform=transform
            )
        
        dwg.add(elem)
    
    def _get_fill(self, shape: Shape) -> str:
        """Get fill color/pattern for shape."""
        if shape.texture == "solid":
            return shape.color
        # For simplicity, return color with opacity for textures
        # Full pattern support would require SVG patterns
        return shape.color
    
    def _triangle_points(self, x: float, y: float, size: float) -> List[Tuple[float, float]]:
        """Generate triangle points."""
        r = size / 2
        return [
            (x, y - r),
            (x - r * 0.866, y + r * 0.5),
            (x + r * 0.866, y + r * 0.5)
        ]
    
    def _star_points(self, x: float, y: float, size: float) -> List[Tuple[float, float]]:
        """Generate star points."""
        r_outer = size / 2
        r_inner = r_outer * 0.4
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = r_outer if i % 2 == 0 else r_inner
            px = x + r * math.cos(angle)
            py = y - r * math.sin(angle)
            points.append((px, py))
        return points
    
    def _polygon_points(self, x: float, y: float, n: int, r: float) -> List[Tuple[float, float]]:
        """Generate regular polygon points."""
        points = []
        for i in range(n):
            angle = -math.pi / 2 + i * 2 * math.pi / n
            px = x + r * math.cos(angle)
            py = y + r * math.sin(angle)
            points.append((px, py))
        return points


def scene_to_dict(shapes: List[Shape], relations: List[Relation]) -> Dict[str, Any]:
    """Convert scene to dictionary representation."""
    return {
        "shapes": [
            {
                "id": s.id,
                "shape_type": s.shape_type,
                "color": s.color,
                "size": s.size,
                "x": s.x,
                "y": s.y,
                "rotation": s.rotation,
                "texture": s.texture,
                "opacity": s.opacity
            }
            for s in shapes
        ],
        "relations": [
            {
                "from": r.from_id,
                "to": r.to_id,
                "type": r.relation_type
            }
            for r in relations
        ]
    }
