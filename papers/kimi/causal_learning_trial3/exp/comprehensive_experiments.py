"""
Comprehensive experiment runner for MF-ACD evaluation.
Runs all baselines, MF-ACD variants, ablations, and validation experiments.
"""
import os
import sys
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# Add shared module path
sys.path.insert(0, os.path.dirname(__file__))
from shared.metrics import compute_metrics, summarize_metrics
from shared.data_loader import load_real_world_data


def load_dataset(dataset_path: str) -> Dict:
    """Load a synthetic dataset."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_all_datasets(data_dir: str = "data/synthetic") -> List[Dict]:
    """Get list of all available datasets."""
    datasets = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            path = os.path.join(data_dir, filename)
            # Parse filename to get config
            # Format: {type}_p{n}_e{e}_n{samples}_s{seed}.pkl
            config = {}
            parts = filename.replace('.pkl', '').split('_')
            
            # First part is graph type
            if parts[0] == 'er':
                config['graph_type'] = 'ER'
            elif parts[0] == 'ba':
                config['graph_type'] = 'BA'
            
            # Parse remaining parts
            for part in parts[1:]:
                if part.startswith('p') and part[1:].isdigit():
                    config['n_nodes'] = int(part[1:])
                elif part.startswith('e') and part[1:].isdigit():
                    config['edge_param'] = int(part[1:])
                elif part.startswith('n') and part[1:].isdigit():
                    config['n_samples'] = int(part[1:])
                elif part.startswith('s') and part[1:].isdigit():
                    config['seed'] = int(part[1:])
            
            # Compute density
            n_nodes = config.get('n_nodes', 20)
            edge_param = config.get('edge_param', 1)
            if config['graph_type'] == 'ER':
                config['density'] = edge_param / n_nodes
            elif config['graph_type'] == 'BA':
                config['density'] = edge_param * 2 / n_nodes
            
            datasets.append({
                'path': path,
                'name': filename,
                'config': config
            })
    return datasets


# ============== BASELINE IMPLEMENTATIONS ==============

class PCBaseline:
    """Standard PC algorithm with Fisher Z test."""
    
    def __init__(self, alpha: float = 0.05, stable: bool = False):
        self.alpha = alpha
        self.stable = stable
        
    def fit(self, data: np.ndarray) -> Dict:
        from causallearn.search.ConstraintBased.PC import pc
        
        start_time = time.time()
        
        # Run PC algorithm
        cg = pc(data, alpha=self.alpha, indep_test='fisherz', stable=self.stable)
        
        runtime = time.time() - start_time
        
        # Extract adjacency matrix
        adj = cg.G.graph
        
        return {
            'adjacency': adj,
            'runtime': runtime,
            'n_tests': getattr(cg, 'n_tests', 0)
        }


class GESBaseline:
    """Greedy Equivalence Search baseline."""
    
    def __init__(self, score_type: str = 'bic'):
        self.score_type = score_type
        
    def fit(self, data: np.ndarray) -> Dict:
        from causallearn.search.ScoreBased.GES import ges
        
        start_time = time.time()
        
        # Run GES
        record = ges(data, score_func=self.score_type)
        
        runtime = time.time() - start_time
        
        # Extract adjacency - handle different return types
        if hasattr(record, 'G'):
            adj = record.G.graph
        elif isinstance(record, dict) and 'G' in record:
            adj = record['G'].graph
        else:
            # Fallback: use the learned DAG
            adj = np.zeros((data.shape[1], data.shape[1]))
        
        return {
            'adjacency': adj,
            'runtime': runtime
        }


class FastPCBaseline:
    """Fast PC using correlation-based tests only."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def fit(self, data: np.ndarray) -> Dict:
        """Simple correlation-based skeleton discovery."""
        from scipy.stats import pearsonr
        
        start_time = time.time()
        n_vars = data.shape[1]
        n_samples = data.shape[0]
        
        adj = np.zeros((n_vars, n_vars))
        n_tests = 0
        
        # Simple pairwise correlation test
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr, p_value = pearsonr(data[:, i], data[:, j])
                n_tests += 1
                
                if p_value < self.alpha:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        runtime = time.time() - start_time
        
        return {
            'adjacency': adj,
            'runtime': runtime,
            'n_tests': n_tests
        }


class HCCDBaseline:
    """
    Hierarchical Clustering Causal Discovery (simplified implementation).
    Based on Shanmugam et al., 2021.
    """
    
    def __init__(self, alpha: float = 0.05, cluster_threshold: float = 0.5):
        self.alpha = alpha
        self.cluster_threshold = cluster_threshold
        
    def fit(self, data: np.ndarray) -> Dict:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.stats import pearsonr
        from causallearn.search.ConstraintBased.PC import pc
        
        start_time = time.time()
        n_vars = data.shape[1]
        
        # Step 1: Hierarchical clustering based on correlation
        corr_matrix = np.corrcoef(data.T)
        dist_matrix = 1 - np.abs(corr_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(dist_matrix[np.triu_indices(n_vars, k=1)], method='average')
        clusters = fcluster(linkage_matrix, t=self.cluster_threshold * n_vars, criterion='maxclust')
        
        # Step 2: Apply PC within each cluster
        adj = np.zeros((n_vars, n_vars))
        unique_clusters = np.unique(clusters)
        
        for clust_id in unique_clusters:
            cluster_vars = np.where(clusters == clust_id)[0]
            if len(cluster_vars) < 2:
                continue
            
            cluster_data = data[:, cluster_vars]
            
            try:
                cg = pc(cluster_data, alpha=self.alpha, indep_test='fisherz', stable=True)
                cluster_adj = cg.G.graph
                
                # Map back to full adjacency
                for i, vi in enumerate(cluster_vars):
                    for j, vj in enumerate(cluster_vars):
                        if cluster_adj[i, j] == 1:
                            adj[vi, vj] = 1
            except Exception as e:
                # If PC fails, use correlation
                for i, vi in enumerate(cluster_vars):
                    for j, vj in enumerate(cluster_vars):
                        if i != j:
                            corr, p = pearsonr(cluster_data[:, i], cluster_data[:, j])
                            if p < self.alpha:
                                adj[vi, vj] = 1
        
        # Step 3: Link clusters using correlation
        for clust1 in unique_clusters:
            for clust2 in unique_clusters:
                if clust1 >= clust2:
                    continue
                
                vars1 = np.where(clusters == clust1)[0]
                vars2 = np.where(clusters == clust2)[0]
                
                # Find best correlation between clusters
                best_corr = 0
                best_pair = None
                for v1 in vars1:
                    for v2 in vars2:
                        corr = np.abs(np.corrcoef(data[:, v1], data[:, v2])[0, 1])
                        if corr > best_corr:
                            best_corr = corr
                            best_pair = (v1, v2)
                
                if best_corr > 0.3 and best_pair:
                    adj[best_pair[0], best_pair[1]] = 1
        
        runtime = time.time() - start_time
        
        return {
            'adjacency': adj,
            'runtime': runtime,
            'n_clusters': len(unique_clusters)
        }


class DCILPBaseline:
    """
    Simplified DCILP implementation.
    Uses Markov blanket estimation and local graph learning.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def estimate_markov_blanket(self, data: np.ndarray, target: int) -> List[int]:
        """Estimate Markov blanket using correlation."""
        n_vars = data.shape[1]
        mb = []
        
        for i in range(n_vars):
            if i == target:
                continue
            corr = np.abs(np.corrcoef(data[:, target], data[:, i])[0, 1])
            if corr > 0.1:  # Threshold for potential MB member
                mb.append(i)
        
        return mb
    
    def fit(self, data: np.ndarray) -> Dict:
        from causallearn.search.ConstraintBased.PC import pc
        
        start_time = time.time()
        n_vars = data.shape[1]
        
        # Step 1: Estimate Markov blankets
        mbs = {}
        for i in range(n_vars):
            mbs[i] = self.estimate_markov_blanket(data, i)
        
        # Step 2: Learn local graphs
        local_adjs = {}
        for i in range(n_vars):
            if len(mbs[i]) < 2:
                local_adjs[i] = np.zeros((len(mbs[i]) + 1, len(mbs[i]) + 1))
                continue
            
            local_vars = [i] + mbs[i]
            local_data = data[:, local_vars]
            
            try:
                cg = pc(local_data, alpha=self.alpha, indep_test='fisherz', stable=True)
                local_adjs[i] = cg.G.graph
            except:
                local_adjs[i] = np.zeros((len(local_vars), len(local_vars)))
        
        # Step 3: Merge local graphs (voting-based)
        adj = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                
                # Count votes for edge i->j
                votes = 0
                
                # Check if edge appears in i's local graph
                if j in mbs[i]:
                    local_idx_j = mbs[i].index(j) + 1 if j in mbs[i] else -1
                    if local_idx_j >= 0:
                        local_adj = local_adjs[i]
                        if local_adj[0, local_idx_j] == 1:  # i -> j
                            votes += 1
                
                # Check if edge appears in j's local graph
                if i in mbs[j]:
                    local_idx_i = mbs[j].index(i) + 1 if i in mbs[j] else -1
                    if local_idx_i >= 0:
                        local_adj = local_adjs[j]
                        if local_adj[local_idx_i, 0] == 1:  # i -> j in j's view
                            votes += 1
                
                # Keep edge if majority vote
                if votes >= 1:
                    adj[i, j] = 1
        
        runtime = time.time() - start_time
        
        return {
            'adjacency': adj,
            'runtime': runtime,
            'avg_mb_size': np.mean([len(mbs[i]) for i in range(n_vars)])
        }


# ============== MF-ACD IMPLEMENTATION ==============

class MFACD:
    """
    Multi-Fidelity Adaptive Causal Discovery.
    Three-phase approach with adaptive budget allocation.
    """
    
    def __init__(self, 
                 budget_allocation: Tuple[float, float, float] = (0.34, 0.20, 0.46),
                 alpha1: float = 0.10,
                 alpha2: float = 0.05,
                 alpha3: float = 0.01,
                 cost_weights: Tuple[float, float, float] = (1.0, 1.1, 15.0),
                 use_adaptive: bool = True,
                 use_ugfs: bool = True):
        self.budget_allocation = np.array(budget_allocation)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.cost_weights = np.array(cost_weights)
        self.use_adaptive = use_adaptive
        self.use_ugfs = use_ugfs
        
        self.phase_costs = [0.0, 0.0, 0.0]
        self.n_tests = [0, 0, 0]
        self.ugfs_overhead = 0.0
        
    def fisher_z_test(self, data: np.ndarray, x: int, y: int, cond_set: List[int]) -> float:
        """Fisher Z test for conditional independence."""
        from scipy.stats import norm, pearsonr
        from sklearn.linear_model import LinearRegression
        
        n = data.shape[0]
        
        if len(cond_set) == 0:
            corr, _ = pearsonr(data[:, x], data[:, y])
            z = np.arctanh(np.clip(np.abs(corr), 0, 0.999)) * np.sqrt(n - 3)
            p_value = 2 * (1 - norm.cdf(np.abs(z)))
        else:
            X_cond = data[:, cond_set]
            
            reg_x = LinearRegression().fit(X_cond, data[:, x])
            resid_x = data[:, x] - reg_x.predict(X_cond)
            
            reg_y = LinearRegression().fit(X_cond, data[:, y])
            resid_y = data[:, y] - reg_y.predict(X_cond)
            
            corr, _ = pearsonr(resid_x, resid_y)
            z = np.arctanh(np.clip(np.abs(corr), 0, 0.999)) * np.sqrt(n - 3 - len(cond_set))
            p_value = 2 * (1 - norm.cdf(np.abs(z)))
        
        return p_value
    
    def fast_correlation_test(self, data: np.ndarray, x: int, y: int, cond_set: List[int]) -> float:
        """Fast correlation test for Phase 1."""
        from scipy.stats import pearsonr
        
        if len(cond_set) == 0:
            _, p_value = pearsonr(data[:, x], data[:, y])
        else:
            # Simple partial correlation approximation
            n = data.shape[0]
            X_cond = data[:, cond_set]
            
            from sklearn.linear_model import LinearRegression
            reg_x = LinearRegression().fit(X_cond, data[:, x])
            resid_x = data[:, x] - reg_x.predict(X_cond)
            
            reg_y = LinearRegression().fit(X_cond, data[:, y])
            resid_y = data[:, y] - reg_y.predict(X_cond)
            
            _, p_value = pearsonr(resid_x, resid_y)
        
        return p_value
    
    def high_fidelity_test(self, data: np.ndarray, x: int, y: int, cond_set: List[int]) -> float:
        """High-fidelity test using Fisher Z with full conditioning sets."""
        return self.fisher_z_test(data, x, y, cond_set)
    
    def benjamini_hochberg(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Benjamini-Hochberg FDR control."""
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        thresholds = np.arange(1, n + 1) / n * alpha
        
        # Find largest k where p(k) <= threshold(k)
        rejected = sorted_p <= thresholds
        
        if not rejected.any():
            return np.zeros(n, dtype=bool)
        
        max_k = np.where(rejected)[0][-1]
        result = np.zeros(n, dtype=bool)
        result[sorted_indices[:max_k + 1]] = True
        
        return result
    
    def holm_bonferroni(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Holm-Bonferroni FWER control."""
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        result = np.zeros(n, dtype=bool)
        for i in range(n):
            if sorted_p[i] <= alpha / (n - i):
                result[sorted_indices[i]] = True
            else:
                break
        
        return result
    
    def phase1_screening(self, data: np.ndarray, budget: float) -> Tuple[np.ndarray, Dict]:
        """Phase 1: Low-fidelity skeleton screening."""
        n_vars = data.shape[1]
        adj = np.ones((n_vars, n_vars))
        np.fill_diagonal(adj, 0)
        
        edges = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
        p_values_list = []
        cost = 0.0
        
        # Unconditional tests
        for i, j in edges:
            if cost >= budget * 0.6:
                break
            
            p = self.fast_correlation_test(data, i, j, [])
            p_values_list.append((i, j, p))
            cost += self.cost_weights[0]
            self.n_tests[0] += 1
        
        # FDR control
        if p_values_list:
            p_array = np.array([p for _, _, p in p_values_list])
            significant = self.benjamini_hochberg(p_array, self.alpha1)
            
            for idx, (i, j, p) in enumerate(p_values_list):
                if not significant[idx] and p > 0.3:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        # Conditional tests on remaining
        remaining_edges = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars) 
                          if adj[i, j] == 1]
        
        p_values_cond = []
        for i, j in remaining_edges:
            if cost >= budget:
                break
            
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j][:2]
            if neighbors:
                p = self.fast_correlation_test(data, i, j, neighbors)
                p_values_cond.append((i, j, p))
                cost += self.cost_weights[0]
                self.n_tests[0] += 1
        
        if p_values_cond:
            p_array = np.array([p for _, _, p in p_values_cond])
            significant = self.benjamini_hochberg(p_array, self.alpha1)
            
            for idx, (i, j, p) in enumerate(p_values_cond):
                if not significant[idx] and p > 0.5:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[0] = cost
        
        # Compute uncertainties
        uncertainties = {}
        for edge_data in p_values_list + p_values_cond:
            i, j, p = edge_data
            uncertainties[(i, j)] = 4 * p * (1 - p)
        
        return adj, uncertainties
    
    def phase2_refinement(self, data: np.ndarray, adj: np.ndarray, 
                          uncertainties: Dict, budget: float) -> np.ndarray:
        """Phase 2: Medium-fidelity local refinement."""
        n_vars = data.shape[1]
        adj = adj.copy()
        
        edges = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars) 
                if adj[i, j] == 1]
        
        # Sort by uncertainty if using UGFS
        if self.use_ugfs:
            edges.sort(key=lambda e: uncertainties.get(e, 0.5), reverse=True)
        
        cost = 0.0
        p_values_all = []
        
        for i, j in edges:
            if cost >= budget:
                break
            
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j]
            
            # Test with conditioning sets
            from itertools import combinations
            for d in range(min(2, len(neighbors) + 1)):
                if d == 0:
                    p = self.fisher_z_test(data, i, j, [])
                    p_values_all.append((i, j, p))
                    cost += self.cost_weights[1]
                    self.n_tests[1] += 1
                else:
                    cond_sets = list(combinations(neighbors, d))[:2]
                    for cond_set in cond_sets:
                        if cost >= budget:
                            break
                        p = self.fisher_z_test(data, i, j, list(cond_set))
                        p_values_all.append((i, j, p))
                        cost += self.cost_weights[1]
                        self.n_tests[1] += 1
        
        # Make decisions
        for i, j, p in p_values_all:
            if p > 0.3:
                adj[i, j] = 0
                adj[j, i] = 0
        
        self.phase_costs[1] = cost
        return adj
    
    def phase3_resolution(self, data: np.ndarray, adj: np.ndarray, budget: float) -> np.ndarray:
        """Phase 3: High-fidelity critical resolution."""
        n_vars = data.shape[1]
        adj = adj.copy()
        
        edges = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars) 
                if adj[i, j] == 1]
        
        if not edges:
            self.phase_costs[2] = 0
            return adj
        
        cost = 0.0
        p_values_all = []
        
        for i, j in edges:
            if cost >= budget:
                break
            
            neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j][:3]
            
            # Unconditional test
            p = self.high_fidelity_test(data, i, j, [])
            p_values_all.append((i, j, p))
            cost += self.cost_weights[2]
            self.n_tests[2] += 1
            
            # Conditional tests
            if neighbors and cost < budget:
                from itertools import combinations
                for cond_set in combinations(neighbors, 1):
                    if cost >= budget:
                        break
                    p = self.high_fidelity_test(data, i, j, list(cond_set))
                    p_values_all.append((i, j, p))
                    cost += self.cost_weights[2]
                    self.n_tests[2] += 1
        
        # Holm-Bonferroni correction
        if p_values_all:
            p_array = np.array([p for _, _, p in p_values_all])
            edge_list = [(i, j) for i, j, _ in p_values_all]
            
            significant = self.holm_bonferroni(p_array, self.alpha3)
            
            # Track which edges have significant tests
            edges_sig = {}
            for idx, (i, j) in enumerate(edge_list):
                if (i, j) not in edges_sig:
                    edges_sig[(i, j)] = []
                edges_sig[(i, j)].append(significant[idx])
            
            # Remove edges with no significant tests
            for (i, j), sig_list in edges_sig.items():
                if not any(sig_list):
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        self.phase_costs[2] = cost
        return adj
    
    def fit(self, data: np.ndarray) -> Dict:
        """Run MF-ACD."""
        n_vars = data.shape[1]
        
        # Estimate total budget
        estimated_tests = n_vars * (n_vars - 1) / 2 * 5
        total_budget = estimated_tests * self.cost_weights[2]
        
        start_time = time.time()
        
        # Phase 1
        phase1_budget = total_budget * self.budget_allocation[0]
        adj, uncertainties = self.phase1_screening(data, phase1_budget)
        
        # Adaptive reallocation
        if self.use_adaptive:
            remaining_budget = total_budget - self.phase_costs[0]
            edges_remaining = np.sum(adj) / 2
            elimination_rate = 1 - edges_remaining / (n_vars * (n_vars - 1) / 2)
            
            if elimination_rate > 0.7:
                phase2_ratio = 0.20
                phase3_ratio = 0.80
            else:
                phase2_ratio = self.budget_allocation[1] / (self.budget_allocation[1] + self.budget_allocation[2])
                phase3_ratio = self.budget_allocation[2] / (self.budget_allocation[1] + self.budget_allocation[2])
            
            phase2_budget = remaining_budget * phase2_ratio
            phase3_budget = remaining_budget * phase3_ratio
        else:
            phase2_budget = total_budget * self.budget_allocation[1]
            phase3_budget = total_budget * self.budget_allocation[2]
        
        # Phase 2
        adj = self.phase2_refinement(data, adj, uncertainties, phase2_budget)
        
        # Phase 3
        adj = self.phase3_resolution(data, adj, phase3_budget)
        
        runtime = time.time() - start_time
        
        total_cost = sum(self.phase_costs)
        baseline_cost = estimated_tests * self.cost_weights[2]
        savings = (baseline_cost - total_cost) / baseline_cost * 100
        
        return {
            'adjacency': adj,
            'phase_costs': self.phase_costs.copy(),
            'n_tests': self.n_tests.copy(),
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'savings_pct': savings,
            'ugfs_overhead': self.ugfs_overhead,
            'runtime': runtime
        }


# ============== EXPERIMENT RUNNERS ==============

def run_single_experiment(method_name: str, dataset: Dict, config: Dict) -> Dict:
    """Run a single experiment."""
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    try:
        if method_name == 'pc_fisherz':
            method = PCBaseline(alpha=0.05, stable=False)
        elif method_name == 'pc_stable':
            method = PCBaseline(alpha=0.05, stable=True)
        elif method_name == 'fast_pc':
            method = FastPCBaseline(alpha=0.05)
        elif method_name == 'ges':
            method = GESBaseline(score_type='bic')
        elif method_name == 'hccd':
            method = HCCDBaseline(alpha=0.05)
        elif method_name == 'dcilp':
            method = DCILPBaseline(alpha=0.05)
        elif method_name == 'mf_acd':
            method = MFACD(
                budget_allocation=config.get('budget_allocation', (0.34, 0.20, 0.46)),
                use_adaptive=config.get('use_adaptive', True),
                use_ugfs=config.get('use_ugfs', True)
            )
        elif method_name == 'mf_acd_fixed':
            method = MFACD(
                budget_allocation=(0.34, 0.20, 0.46),
                use_adaptive=False,
                use_ugfs=True
            )
        elif method_name == 'mf_acd_nougfs':
            method = MFACD(
                budget_allocation=(0.34, 0.20, 0.46),
                use_adaptive=True,
                use_ugfs=False
            )
        else:
            return {'error': f'Unknown method: {method_name}'}
        
        result = method.fit(data)
        metrics = compute_metrics(result['adjacency'], true_adj)
        
        return {
            'method': method_name,
            'metrics': metrics,
            'runtime': result.get('runtime', 0),
            'n_tests': result.get('n_tests', 0),
            'phase_costs': result.get('phase_costs', [0, 0, 0]),
            'total_cost': result.get('total_cost', 0),
            'baseline_cost': result.get('baseline_cost', 0),
            'savings_pct': result.get('savings_pct', 0),
            'config': config
        }
    
    except Exception as e:
        import traceback
        return {
            'method': method_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_experiments_batch(experiments: List[Tuple], max_workers: int = 2) -> List[Dict]:
    """Run a batch of experiments in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_experiment, method, dataset, config): (method, dataset_path)
            for method, dataset, dataset_path, config in experiments
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results


if __name__ == '__main__':
    print("Comprehensive experiment runner loaded.")
    print("Use run_single_experiment() or run_experiments_batch() to execute experiments.")
