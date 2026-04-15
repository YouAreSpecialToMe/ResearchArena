"""
Quick inference latency benchmark for CASS-ViM and baselines.
Measures actual GPU/CPU inference time.
"""
import torch
import torch.nn as nn
import time
import json
import numpy as np
from src.minimal_models import MinimalCASSViM, MinimalVMamba, MinimalLocalMamba

def benchmark_model(model, input_size=(1, 3, 32, 32), n_warmup=20, n_runs=100, device='cuda'):
    """Benchmark inference latency."""
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'p99_ms': float(np.percentile(times, 99)),
        'n_runs': n_runs
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Use smaller architecture matching the trained models
    embed_dims = [32, 64, 128, 256]
    depths = [2, 2, 2, 2]
    
    # Batch sizes to test
    batch_sizes = [1, 8]
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")
        
        results[f'batch_{batch_size}'] = {}
        
        # CASS-ViM-4D
        print("\nBenchmarking CASS-ViM-4D...")
        try:
            model = MinimalCASSViM(num_classes=100, num_directions=4,
                                   embed_dims=embed_dims, depths=depths)
            results[f'batch_{batch_size}']['cassvim_4d'] = benchmark_model(
                model, input_size=(batch_size, 3, 32, 32), device=device, n_runs=100
            )
            r = results[f'batch_{batch_size}']['cassvim_4d']
            print(f"  Mean: {r['mean_ms']:.3f} ± {r['std_ms']:.3f} ms")
            print(f"  Throughput: {1000/r['mean_ms']*batch_size:.1f} images/sec")
        except Exception as e:
            print(f"  Error: {e}")
        
        # CASS-ViM-8D
        print("\nBenchmarking CASS-ViM-8D...")
        try:
            model = MinimalCASSViM(num_classes=100, num_directions=8,
                                   embed_dims=embed_dims, depths=depths)
            results[f'batch_{batch_size}']['cassvim_8d'] = benchmark_model(
                model, input_size=(batch_size, 3, 32, 32), device=device, n_runs=100
            )
            r = results[f'batch_{batch_size}']['cassvim_8d']
            print(f"  Mean: {r['mean_ms']:.3f} ± {r['std_ms']:.3f} ms")
            print(f"  Throughput: {1000/r['mean_ms']*batch_size:.1f} images/sec")
        except Exception as e:
            print(f"  Error: {e}")
        
        # VMamba
        print("\nBenchmarking VMamba...")
        try:
            model = MinimalVMamba(num_classes=100, embed_dims=embed_dims, depths=depths)
            results[f'batch_{batch_size}']['vmamba'] = benchmark_model(
                model, input_size=(batch_size, 3, 32, 32), device=device, n_runs=100
            )
            r = results[f'batch_{batch_size}']['vmamba']
            print(f"  Mean: {r['mean_ms']:.3f} ± {r['std_ms']:.3f} ms")
            print(f"  Throughput: {1000/r['mean_ms']*batch_size:.1f} images/sec")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save results
    with open('latency_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch':<8} {'Model':<15} {'Latency (ms)':<20} {'Throughput':<15}")
    print("-"*60)
    for bs in batch_sizes:
        key = f'batch_{bs}'
        if key in results:
            for model_name, r in results[key].items():
                print(f"{bs:<8} {model_name:<15} {r['mean_ms']:>6.3f}±{r['std_ms']:<10.3f} {1000/r['mean_ms']*bs:>8.1f} img/s")
    
    print("\nResults saved to latency_benchmark.json")

if __name__ == '__main__':
    main()
