"""
Core E-Graph Data Structures for MemSat
"""
import json
import random
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx


@dataclass
class ENode:
    """An e-node represents a program operation or value."""
    id: int
    op: str  # Operation type: 'add', 'mul', 'load', 'store', 'layout_aos', etc.
    children: List[int] = field(default_factory=list)  # E-class IDs of children
    cost: float = 1.0  # Local cost
    layout_type: Optional[str] = None  # 'AoS', 'SoA', 'tiled', 'padded', or None
    
    def __hash__(self):
        return self.id


@dataclass  
class EClass:
    """An e-class contains equivalent e-nodes (program variants)."""
    id: int
    nodes: List[ENode] = field(default_factory=list)
    
    def __hash__(self):
        return self.id


class EGraph:
    """
    Equality Graph representation for program optimization.
    Supports both flat and hierarchical representations.
    """
    
    def __init__(self, name: str = "egraph"):
        self.name = name
        self.classes: Dict[int, EClass] = {}
        self.next_class_id = 0
        self.next_node_id = 0
        self.root_class: Optional[int] = None
        
    def add_class(self) -> int:
        """Add a new empty e-class."""
        cid = self.next_class_id
        self.classes[cid] = EClass(id=cid)
        self.next_class_id += 1
        return cid
    
    def add_node(self, op: str, children: List[int] = None, 
                 cost: float = 1.0, layout_type: Optional[str] = None,
                 class_id: Optional[int] = None) -> int:
        """Add an e-node to the graph. Returns the e-class ID."""
        if children is None:
            children = []
            
        node = ENode(
            id=self.next_node_id,
            op=op,
            children=children,
            cost=cost,
            layout_type=layout_type
        )
        self.next_node_id += 1
        
        # Add to existing class or create new one
        if class_id is None or class_id not in self.classes:
            class_id = self.add_class()
        
        self.classes[class_id].nodes.append(node)
        return class_id
    
    def merge_classes(self, cid1: int, cid2: int) -> int:
        """Merge two e-classes (union operation)."""
        if cid1 == cid2:
            return cid1
        
        # Merge smaller into larger
        if len(self.classes[cid1].nodes) < len(self.classes[cid2].nodes):
            cid1, cid2 = cid2, cid1
            
        # Move all nodes from cid2 to cid1
        self.classes[cid1].nodes.extend(self.classes[cid2].nodes)
        del self.classes[cid2]
        
        # Update all references to cid2 -> cid1
        for ec in self.classes.values():
            for node in ec.nodes:
                for i, child in enumerate(node.children):
                    if child == cid2:
                        node.children[i] = cid1
        
        return cid1
    
    def to_conflict_graph(self) -> nx.Graph:
        """
        Convert e-graph to conflict graph for treewidth computation.
        Vertices = e-classes, edges = data dependencies between classes.
        """
        G = nx.Graph()
        
        # Add all e-classes as nodes
        for cid in self.classes:
            G.add_node(cid)
        
        # Add edges for data dependencies
        for cid, eclass in self.classes.items():
            for node in eclass.nodes:
                for child_cid in node.children:
                    if child_cid in self.classes and child_cid != cid:
                        G.add_edge(cid, child_cid)
        
        return G
    
    def compute_treewidth(self) -> Tuple[int, Any]:
        """
        Compute treewidth using min-fill-in heuristic.
        Returns (treewidth, tree_decomposition).
        """
        G = self.to_conflict_graph()
        
        if len(G.nodes) == 0:
            return 0, None
        
        # Use NetworkX's treewidth approximation
        try:
            tw, tree_decomp = nx.algorithms.approximation.treewidth_min_fill_in(G)
        except:
            # Fallback: compute using simpler method
            tw = self._estimate_treewidth_brute(G)
            tree_decomp = None
            
        return tw, tree_decomp
    
    def _estimate_treewidth_brute(self, G: nx.Graph) -> int:
        """Simple treewidth upper bound using min-degree heuristic."""
        if len(G.nodes) <= 1:
            return 0
        
        H = G.copy()
        max_clique_size = 0
        
        while len(H.nodes) > 0:
            # Find node with minimum degree
            min_node = min(H.nodes, key=lambda n: H.degree(n))
            degree = H.degree(min_node)
            max_clique_size = max(max_clique_size, degree)
            
            # Connect all neighbors (fill-in)
            neighbors = list(H.neighbors(min_node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    H.add_edge(neighbors[i], neighbors[j])
            
            # Remove the node
            H.remove_node(min_node)
        
        return max_clique_size
    
    def size(self) -> Dict[str, int]:
        """Return statistics about e-graph size."""
        num_classes = len(self.classes)
        num_nodes = sum(len(ec.nodes) for ec in self.classes.values())
        num_edges = sum(
            len(node.children) 
            for ec in self.classes.values() 
            for node in ec.nodes
        )
        return {
            "eclasses": num_classes,
            "enodes": num_nodes,
            "edges": num_edges
        }
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "classes": {
                str(cid): {
                    "nodes": [
                        {
                            "id": n.id,
                            "op": n.op,
                            "children": n.children,
                            "cost": n.cost,
                            "layout_type": n.layout_type
                        }
                        for n in ec.nodes
                    ]
                }
                for cid, ec in self.classes.items()
            },
            "root_class": self.root_class
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'EGraph':
        """Deserialize from dictionary."""
        g = EGraph(name=data.get("name", "egraph"))
        
        for cid_str, ec_data in data["classes"].items():
            cid = int(cid_str)
            g.classes[cid] = EClass(id=cid)
            for node_data in ec_data["nodes"]:
                node = ENode(
                    id=node_data["id"],
                    op=node_data["op"],
                    children=node_data["children"],
                    cost=node_data["cost"],
                    layout_type=node_data.get("layout_type")
                )
                g.classes[cid].nodes.append(node)
                g.next_node_id = max(g.next_node_id, node_data["id"] + 1)
            g.next_class_id = max(g.next_class_id, cid + 1)
        
        g.root_class = data.get("root_class")
        return g


class HierarchicalEGraph:
    """
    Multi-level e-graph: loop nest -> function -> module
    """
    
    def __init__(self, name: str = "hierarchical"):
        self.name = name
        self.levels: Dict[int, EGraph] = {}  # level -> egraph
        
    def add_level(self, level: int, egraph: EGraph):
        """Add an e-graph at a hierarchy level."""
        self.levels[level] = egraph
        
    def get_treewidth_stats(self) -> Dict[int, int]:
        """Get treewidth for each level."""
        return {
            level: egraph.compute_treewidth()[0]
            for level, egraph in self.levels.items()
        }
