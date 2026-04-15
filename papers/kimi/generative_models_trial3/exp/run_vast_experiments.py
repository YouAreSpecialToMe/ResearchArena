#!/usr/bin/env python3
"""
VAST Experiments with FLUX-dev
Complete implementation with actual patch-wise skipping, proper metrics, and multiple seeds.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from shared.vast_flux_implementation import VASTFluxSampler, StandardFluxSampler, VASTConfig
from baselines.deepcache_baseline import DeepCacheFluxSampler
from baselines.ras_baseline import RASFluxSampler
from evaluation.metrics import MetricsEvaluator, aggregate_results_across_seeds, save_results

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HEIGHT = 512
WIDTH = 512


def create_prompts(num_prompts: int = 500) -> List[str]:
    """Create evaluation prompts - use MS COCO style prompts."""
    # Create diverse prompts covering different scenarios
    templates = [
        "a photo of {}",
        "a photograph of {}",
        "an image of {}",
        "{}",
    ]
    
    subjects = [
        "a cat sitting on a couch",
        "a dog playing in the park",
        "a person riding a bicycle",
        "a car on a street",
        "a bird flying in the sky",
        "a horse in a field",
        "an airplane in the clouds",
        "a boat on a lake",
        "a train on tracks",
        "a truck on a highway",
        "a zebra in a savanna",
        "a giraffe eating leaves",
        "an elephant near water",
        "a bear in the forest",
        "a tennis player on a court",
        "a skateboarder doing a trick",
        "a surfboard on waves",
        "a skier on a mountain",
        "a baseball game",
        "a pizza on a table",
        "a cake with candles",
        "a dining table with food",
        "a banana on a counter",
        "an apple on a tree",
        "a sandwich on a plate",
        "an orange in a bowl",
        "a broccoli on a plate",
        "a carrot on a cutting board",
        "a hot dog at a stand",
        "a donut with sprinkles",
        "a bench in a park",
        "a kite flying high",
        "a umbrella in the rain",
        "a handbag on a chair",
        "a tie on a shirt",
        "a suitcase at an airport",
        "a frisbee in the air",
        "a snowboard on snow",
        "a sports ball on grass",
        "a baseball bat",
        "a skateboard on pavement",
        "a surfboard on the beach",
        "a tennis racket",
        "a bottle on a table",
        "a wine glass",
        "a cup of coffee",
        "a fork and knife",
        "a spoon on a napkin",
        "a bowl of soup",
        "a chair in a room",
        "a couch in a living room",
        "a potted plant",
        "a bed with pillows",
        "a dining table",
        "a toilet in a bathroom",
        "a tv on a stand",
        "a laptop on a desk",
        "a mouse on a pad",
        "a remote control",
        "a keyboard",
        "a cell phone",
        "a microwave",
        "an oven",
        "a toaster",
        "a sink",
        "a refrigerator",
        "a book on a shelf",
        "a clock on a wall",
        "a vase with flowers",
        "a pair of scissors",
        "a teddy bear",
        "a hair dryer",
        "a toothbrush",
    ]
    
    prompts = []
    for i in range(num_prompts):
        template = templates[i % len(templates)]
        subject = subjects[i % len(subjects)]
        prompts.append(template.format(subject))
    
    return prompts


def save_images(images: List[Image.Image], output_dir: str, prefix: str = "sample"):
    """Save generated images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"{prefix}_{i:04d}.png"))


def run_method(
    method_name: str,
    sampler,
    prompts: List[str],
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Run a single method and return results."""
    print(f"\n{'='*60}")
    print(f"Running {method_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Generate images
    start = time.time()
    result = sampler.generate(
        prompts=prompts,
        height=HEIGHT,
        width=WIDTH,
        seed=seed,
    )
    total_time = time.time() - start
    
    # Save images
    method_dir = os.path.join(output_dir, f"{method_name}_seed{seed}")
    save_images(result['images'], method_dir)
    
    # Prepare results
    results = {
        'method': method_name,
        'seed': seed,
        'num_images': len(result['images']),
        'wall_time_total': result['wall_time_total'],
        'wall_time_per_image': result['wall_time_per_image'],
        'nfe_total': result.get('nfe_total', 0),
        'nfe_per_image': result.get('nfe_per_image', 0),
    }
    
    # Add method-specific metrics
    if 'flops_reduction' in result:
        results['flops_reduction'] = result['flops_reduction']
    if 'cache_ratio' in result:
        results['cache_ratio'] = result['cache_ratio']
    if 'effective_nfe' in result:
        results['effective_nfe'] = result['effective_nfe']
    if 'region_ratio' in result:
        results['region_ratio'] = result['region_ratio']
    
    print(f"  Wall time: {results['wall_time_per_image']:.3f}s/image")
    print(f"  NFE: {results['nfe_per_image']:.1f}")
    
    return results, result['images']


def run_all_experiments(
    num_prompts: int = 100,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "outputs",
):
    """Run all experiments with multiple seeds."""
    print("="*60)
    print("VAST EXPERIMENTS WITH FLUX-dev")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Images per seed: {num_prompts}")
    print(f"Seeds: {seeds}")
    
    # Create prompts
    print("\nCreating prompts...")
    prompts = create_prompts(num_prompts)
    print(f"Created {len(prompts)} prompts")
    
    # Save prompts
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prompts.json"), 'w') as f:
        json.dump(prompts, f)
    
    # Load FLUX model
    print("\nLoading FLUX.1-dev model...")
    from diffusers import FluxPipeline
    
    model_id = "black-forest-labs/FLUX.1-dev"
    
    # Check if model is available
    try:
        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/huggingface_cache",
        )
        pipeline = pipeline.to(DEVICE)
        print("Successfully loaded FLUX.1-dev")
    except Exception as e:
        print(f"Error loading FLUX.1-dev: {e}")
        print("Falling back to Stable Diffusion 1.5 for testing...")
        from diffusers import StableDiffusionPipeline
        model_id = "runwayml/stable-diffusion-v1-5"
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipeline = pipeline.to(DEVICE)
    
    # Store all results
    all_results = {}
    
    # Run for each seed
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")
        
        seed_results = {}
        seed_images = {}
        
        # 1. Baseline 50-step
        baseline_sampler = StandardFluxSampler(
            pipeline=pipeline,
            num_inference_steps=50,
            device=str(DEVICE)
        )
        result, images = run_method(
            "baseline_50step",
            baseline_sampler,
            prompts,
            seed,
            output_dir
        )
        seed_results['baseline_50step'] = result
        seed_images['baseline_50step'] = images
        
        baseline_time = result['wall_time_per_image']
        baseline_nfe = result['nfe_per_image']
        
        # 2. Baseline 25-step (uniform 2x speedup)
        baseline_25_sampler = StandardFluxSampler(
            pipeline=pipeline,
            num_inference_steps=25,
            device=str(DEVICE)
        )
        result, images = run_method(
            "baseline_25step",
            baseline_25_sampler,
            prompts,
            seed,
            output_dir
        )
        result['speedup'] = baseline_time / result['wall_time_per_image']
        result['nfe_speedup'] = baseline_nfe / result['nfe_per_image']
        seed_results['baseline_25step'] = result
        seed_images['baseline_25step'] = images
        
        # 3. Baseline 17-step (uniform 3x speedup)
        baseline_17_sampler = StandardFluxSampler(
            pipeline=pipeline,
            num_inference_steps=17,
            device=str(DEVICE)
        )
        result, images = run_method(
            "baseline_17step",
            baseline_17_sampler,
            prompts,
            seed,
            output_dir
        )
        result['speedup'] = baseline_time / result['wall_time_per_image']
        result['nfe_speedup'] = baseline_nfe / result['nfe_per_image']
        seed_results['baseline_17step'] = result
        seed_images['baseline_17step'] = images
        
        # 4. DeepCache baseline
        deepcache_sampler = DeepCacheFluxSampler(
            pipeline=pipeline,
            num_inference_steps=50,
            cache_interval=5,
            device=str(DEVICE)
        )
        result, images = run_method(
            "deepcache",
            deepcache_sampler,
            prompts,
            seed,
            output_dir
        )
        result['speedup'] = baseline_time / result['wall_time_per_image']
        seed_results['deepcache'] = result
        seed_images['deepcache'] = images
        
        # 5. RAS baseline (2x region ratio)
        ras_sampler = RASFluxSampler(
            pipeline=pipeline,
            num_inference_steps=50,
            region_ratio=0.5,
            device=str(DEVICE)
        )
        result, images = run_method(
            "ras_2x",
            ras_sampler,
            prompts,
            seed,
            output_dir
        )
        result['speedup'] = baseline_time / result['wall_time_per_image']
        seed_results['ras_2x'] = result
        seed_images['ras_2x'] = images
        
        # 6. VAST 2x
        config_2x = VASTConfig(
            target_speedup=2.0,
            patch_size=8,
            threshold_percentile=15.0,
            num_inference_steps=50,
        )
        vast_2x_sampler = VASTFluxSampler(
            pipeline=pipeline,
            config=config_2x,
            device=str(DEVICE)
        )
        result, images = run_method(
            "vast_2x",
            vast_2x_sampler,
            prompts,
            seed,
            output_dir
        )
        result['speedup'] = baseline_time / result['wall_time_per_image']
        result['nfe_speedup'] = baseline_nfe / result['nfe_per_image']
        seed_results['vast_2x'] = result
        seed_images['vast_2x'] = images
        
        # 7. VAST 3x
        config_3x = VASTConfig(
            target_speedup=3.0,
            patch_size=8,
            threshold_percentile=20.0,
            num_inference_steps=50,
        )
        vast_3x_sampler = VASTFluxSampler(
            pipeline=pipeline,
            config=config_3x,
            device=str(DEVICE)
        )
        result, images = run_method(
            "vast_3x",
            vast_3x_sampler,
            prompts,
            seed,
            output_dir
        )
        result['speedup'] = baseline_time / result['wall_time_per_image']
        result['nfe_speedup'] = baseline_nfe / result['nfe_per_image']
        seed_results['vast_3x'] = result
        seed_images['vast_3x'] = images
        
        all_results[f'seed_{seed}'] = {
            'results': seed_results,
            'baseline_time': baseline_time,
        }
    
    # Aggregate results across seeds
    print("\n" + "="*60)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("="*60)
    
    aggregated = aggregate_results(all_results, seeds)
    
    # Save aggregated results
    final_results = {
        'experiment_name': 'VAST: Velocity-Adaptive Spatially-varying Timesteps',
        'model': model_id,
        'num_prompts': num_prompts,
        'seeds': seeds,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'per_seed': all_results,
        'aggregated': aggregated,
    }
    
    save_results(final_results, os.path.join(output_dir, 'results_aggregated.json'))
    print(f"\nResults saved to {output_dir}/results_aggregated.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for method in ['baseline_50step', 'baseline_25step', 'baseline_17step', 
                   'deepcache', 'ras_2x', 'vast_2x', 'vast_3x']:
        if method in aggregated:
            print(f"\n{method}:")
            print(f"  Wall time: {aggregated[method]['wall_time_per_image']['mean']:.3f} ± "
                  f"{aggregated[method]['wall_time_per_image']['std']:.3f}s")
            if 'speedup' in aggregated[method]:
                print(f"  Speedup: {aggregated[method]['speedup']['mean']:.2f}x ± "
                      f"{aggregated[method]['speedup']['std']:.2f}x")
            if 'flops_reduction' in aggregated[method]:
                print(f"  FLOPs reduction: {aggregated[method]['flops_reduction']['mean']*100:.1f}%")
    
    return final_results


def aggregate_results(all_results: Dict, seeds: List[int]) -> Dict:
    """Aggregate results across seeds."""
    aggregated = {}
    
    # Get all methods
    methods = list(all_results[f'seed_{seeds[0]}']['results'].keys())
    
    for method in methods:
        method_results = []
        for seed in seeds:
            method_results.append(all_results[f'seed_{seed}']['results'][method])
        
        # Aggregate metrics
        agg = {}
        
        # Wall time
        wall_times = [r['wall_time_per_image'] for r in method_results]
        agg['wall_time_per_image'] = {
            'mean': float(np.mean(wall_times)),
            'std': float(np.std(wall_times)),
            'values': wall_times,
        }
        
        # NFE
        nfes = [r['nfe_per_image'] for r in method_results]
        agg['nfe_per_image'] = {
            'mean': float(np.mean(nfes)),
            'std': float(np.std(nfes)),
            'values': nfes,
        }
        
        # Speedup
        if 'speedup' in method_results[0]:
            speedups = [r['speedup'] for r in method_results]
            agg['speedup'] = {
                'mean': float(np.mean(speedups)),
                'std': float(np.std(speedups)),
                'values': speedups,
            }
        
        # FLOPs reduction
        if 'flops_reduction' in method_results[0]:
            flops = [r['flops_reduction'] for r in method_results]
            agg['flops_reduction'] = {
                'mean': float(np.mean(flops)),
                'std': float(np.std(flops)),
                'values': flops,
            }
        
        # Cache ratio (DeepCache)
        if 'cache_ratio' in method_results[0]:
            cache_ratios = [r['cache_ratio'] for r in method_results]
            agg['cache_ratio'] = {
                'mean': float(np.mean(cache_ratios)),
                'std': float(np.std(cache_ratios)),
                'values': cache_ratios,
            }
        
        aggregated[method] = agg
    
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_prompts', type=int, default=100,
                       help='Number of prompts to evaluate')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds for evaluation')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    args = parser.parse_args()
    
    run_all_experiments(
        num_prompts=args.num_prompts,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
