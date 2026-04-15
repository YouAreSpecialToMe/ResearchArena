"""
Main experiment runner for CAD-DiT evaluation.
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_cad import (
    SimpleCADPipeline, DeepCachePipeline, DeltaDiTPipeline,
    GlobalExitPipeline, CADDiTPipeline
)
from metrics import InceptionScore, FIDScore


class ExperimentRunner:
    """Run experiments and collect results."""
    
    def __init__(
        self,
        output_dir: str,
        num_samples: int = 1000,
        num_seeds: int = 3,
        device: str = "cuda",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.num_seeds = num_seeds
        self.device = device
        
        # Metrics
        self.is_metric = InceptionScore(device=device)
        self.fid_metric = FIDScore(device=device)
        
        # ImageNet class indices (subset)
        np.random.seed(42)
        self.class_labels = torch.randint(0, 1000, (num_samples,))
    
    def run_experiment(
        self,
        method_name: str,
        method_class,
        method_kwargs: Dict[str, Any] = None,
        seeds: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            method_name: Name of the method
            method_class: Pipeline class
            method_kwargs: Additional kwargs for the method
            seeds: List of random seeds
        
        Returns:
            Results dictionary
        """
        if seeds is None:
            seeds = [42, 123, 2024][:self.num_seeds]
        
        if method_kwargs is None:
            method_kwargs = {}
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {method_name}")
        print(f"Seeds: {seeds}")
        print(f"Samples per seed: {self.num_samples}")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            # Initialize method
            method = method_class(
                device=self.device,
                num_inference_steps=50,
                guidance_scale=1.0,
                **method_kwargs
            )
            
            # Generate images in batches
            batch_size = 4
            num_batches = (self.num_samples + batch_size - 1) // batch_size
            
            all_images = []
            all_times = []
            all_flops_reductions = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.num_samples)
                actual_batch = end_idx - start_idx
                
                batch_labels = self.class_labels[start_idx:end_idx]
                
                result = method.generate(
                    class_labels=batch_labels,
                    num_images=actual_batch,
                    seed=seed + batch_idx,
                    return_stats=True,
                )
                
                all_images.append(result['images'])
                all_times.append(result['time_seconds'])
                if 'stats' in result:
                    all_flops_reductions.append(result['stats'].get('flops_reduction', 0.0))
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Progress: {end_idx}/{self.num_samples} images")
            
            # Concatenate images
            all_images = torch.cat(all_images, dim=0)
            
            # Compute IS for this seed
            print("Computing Inception Score...")
            is_mean, is_std = self.is_metric.compute(all_images)
            
            seed_result = {
                'seed': seed,
                'is_mean': float(is_mean),
                'is_std': float(is_std),
                'total_time': sum(all_times),
                'time_per_image': sum(all_times) / len(all_images),
                'num_images': len(all_images),
                'avg_flops_reduction': float(np.mean(all_flops_reductions)) if all_flops_reductions else 0.0,
            }
            
            print(f"Seed {seed} results:")
            print(f"  IS: {is_mean:.2f} ± {is_std:.2f}")
            print(f"  Time: {seed_result['total_time']:.1f}s ({seed_result['time_per_image']:.3f}s/img)")
            print(f"  FLOPs reduction: {seed_result['avg_flops_reduction']*100:.1f}%")
            
            all_results.append(seed_result)
        
        # Aggregate results across seeds
        aggregated = {
            'experiment': method_name,
            'seeds': seeds,
            'num_samples': self.num_samples,
            'metrics': {
                'is_mean': {
                    'mean': float(np.mean([r['is_mean'] for r in all_results])),
                    'std': float(np.std([r['is_mean'] for r in all_results])),
                },
                'is_std': {
                    'mean': float(np.mean([r['is_std'] for r in all_results])),
                    'std': float(np.std([r['is_std'] for r in all_results])),
                },
                'time_per_image': {
                    'mean': float(np.mean([r['time_per_image'] for r in all_results])),
                    'std': float(np.std([r['time_per_image'] for r in all_results])),
                },
                'flops_reduction_percent': {
                    'mean': float(np.mean([r['avg_flops_reduction'] for r in all_results]) * 100),
                    'std': float(np.std([r['avg_flops_reduction'] for r in all_results]) * 100),
                },
            },
            'per_seed_results': all_results,
            'config': {
                'num_samples': self.num_samples,
                'num_seeds': len(seeds),
                'method_kwargs': method_kwargs,
            },
        }
        
        # Save results
        results_file = self.output_dir / f"{method_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        print(f"\nAggregated results for {method_name}:")
        print(f"  IS: {aggregated['metrics']['is_mean']['mean']:.2f} ± {aggregated['metrics']['is_mean']['std']:.2f}")
        print(f"  Time: {aggregated['metrics']['time_per_image']['mean']:.3f} ± {aggregated['metrics']['time_per_image']['std']:.3f} s/img")
        print(f"  FLOPs reduction: {aggregated['metrics']['flops_reduction_percent']['mean']:.1f}%")
        
        return aggregated
    
    def compute_fid_between_methods(
        self,
        method1_name: str,
        method2_name: str,
        seed: int = 42,
        num_samples: int = 1000,
    ) -> float:
        """Compute FID between two methods."""
        print(f"\nComputing FID between {method1_name} and {method2_name}...")
        
        # This would require generating images from both methods
        # For now, return a placeholder
        return 0.0


def run_all_experiments(output_dir: str = "exp", num_samples: int = 1000, device: str = "cuda"):
    """Run all experiments from the plan."""
    
    runner = ExperimentRunner(
        output_dir=output_dir,
        num_samples=num_samples,
        num_seeds=3,
        device=device,
    )
    
    all_results = {}
    
    # 1. Full DiT baseline
    print("\n" + "="*60)
    print("EXPERIMENT 1: Full DiT Baseline")
    print("="*60)
    results_full = runner.run_experiment(
        method_name="dit_full",
        method_class=SimpleCADPipeline,
        seeds=[42, 123, 2024],
    )
    all_results['dit_full'] = results_full
    
    # 2. DeepCache baseline
    print("\n" + "="*60)
    print("EXPERIMENT 2: DeepCache Baseline")
    print("="*60)
    results_deepcache = runner.run_experiment(
        method_name="dit_deepcache",
        method_class=DeepCachePipeline,
        seeds=[42, 123, 2024],
    )
    all_results['dit_deepcache'] = results_deepcache
    
    # 3. Delta-DiT baseline
    print("\n" + "="*60)
    print("EXPERIMENT 3: Delta-DiT Baseline")
    print("="*60)
    results_delta = runner.run_experiment(
        method_name="dit_delta_dit",
        method_class=DeltaDiTPipeline,
        seeds=[42, 123, 2024],
    )
    all_results['dit_delta_dit'] = results_delta
    
    # 4. Global Exit baseline
    print("\n" + "="*60)
    print("EXPERIMENT 4: Global Early Exit Baseline")
    print("="*60)
    results_global = runner.run_experiment(
        method_name="dit_global_exit",
        method_class=GlobalExitPipeline,
        seeds=[42, 123, 2024],
    )
    all_results['dit_global_exit'] = results_global
    
    # 5. CAD-DiT main method
    print("\n" + "="*60)
    print("EXPERIMENT 5: CAD-DiT Main Method")
    print("="*60)
    results_cad = runner.run_experiment(
        method_name="dit_cad_dit",
        method_class=CADDiTPipeline,
        method_kwargs={'tau_base': 0.95, 'alpha': 0.3},
        seeds=[42, 123, 2024],
    )
    all_results['dit_cad_dit'] = results_cad
    
    # 6. Ablation: tau_base sweep (single seed for speed)
    print("\n" + "="*60)
    print("EXPERIMENT 6: Ablation - tau_base sweep")
    print("="*60)
    for tau in [0.90, 0.93, 0.95, 0.97]:
        results = runner.run_experiment(
            method_name=f"dit_ablation_tau_{tau}",
            method_class=CADDiTPipeline,
            method_kwargs={'tau_base': tau, 'alpha': 0.3},
            seeds=[42],  # Single seed for ablations
        )
        all_results[f'dit_ablation_tau_{tau}'] = results
    
    # 7. Ablation: alpha sweep (single seed)
    print("\n" + "="*60)
    print("EXPERIMENT 7: Ablation - alpha sweep")
    print("="*60)
    for alpha in [0.2, 0.3, 0.4]:
        results = runner.run_experiment(
            method_name=f"dit_ablation_alpha_{alpha}",
            method_class=CADDiTPipeline,
            method_kwargs={'tau_base': 0.95, 'alpha': alpha},
            seeds=[42],
        )
        all_results[f'dit_ablation_alpha_{alpha}'] = results
    
    # Save aggregated results
    output_path = Path(output_dir) / "all_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="exp")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_all_experiments(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )
