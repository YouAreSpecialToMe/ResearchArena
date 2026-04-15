"""Run all model evaluations sequentially."""

import json
import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'exp', 'shared'))
from inference import run_inference

DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def load_dataset(seed_name):
    path = os.path.join(DATA_DIR, seed_name, 'flipbench.json')
    with open(path) as f:
        return json.load(f)


# Model configurations
MODELS = [
    {
        'name': 'microsoft/Phi-3.5-mini-instruct',
        'short': 'phi35',
        'max_tokens': 512,
        'seeds': ['seed_42'],
        'max_model_len': 4096,
    },
    {
        'name': 'meta-llama/Llama-3.1-8B-Instruct',
        'short': 'llama31_8b',
        'max_tokens': 512,
        'seeds': ['seed_42', 'seed_123', 'seed_456'],
        'max_model_len': 4096,
    },
    {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'short': 'qwen25_7b',
        'max_tokens': 512,
        'seeds': ['seed_42'],
        'max_model_len': 4096,
    },
    {
        'name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'short': 'deepseek_r1_7b',
        'max_tokens': 1024,
        'seeds': ['seed_42', 'seed_123', 'seed_456'],
        'max_model_len': 4096,
    },
    {
        'name': 'Qwen/Qwen2.5-32B-Instruct-AWQ',
        'short': 'qwen25_32b',
        'max_tokens': 512,
        'seeds': ['seed_42'],
        'quantization': 'awq',
        'max_model_len': 4096,
    },
]


def main():
    total_start = time.time()
    all_results = {}

    for model_cfg in MODELS:
        model_name = model_cfg['name']
        short_name = model_cfg['short']
        max_tokens = model_cfg['max_tokens']
        quantization = model_cfg.get('quantization')
        max_model_len = model_cfg.get('max_model_len', 4096)

        for seed_name in model_cfg['seeds']:
            # Check if already done
            parsed_path = os.path.join(RESULTS_DIR, 'parsed',
                                       f'{short_name}_{seed_name}.json')
            if os.path.exists(parsed_path):
                print(f"\nSkipping {short_name} on {seed_name} (already done)")
                with open(parsed_path) as f:
                    all_results[f'{short_name}_{seed_name}'] = json.load(f)
                continue

            dataset = load_dataset(seed_name)
            metrics = run_inference(
                model_name=model_name,
                dataset=dataset,
                output_dir=RESULTS_DIR,
                model_short_name=short_name,
                seed_name=seed_name,
                max_tokens=max_tokens,
                quantization=quantization,
                max_model_len=max_model_len,
            )
            all_results[f'{short_name}_{seed_name}'] = metrics

    total_time = time.time() - total_start
    print(f"\n\nTotal evaluation time: {total_time/60:.1f} minutes")

    # Save combined results
    outpath = os.path.join(RESULTS_DIR, 'aggregated', 'all_model_results.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {outpath}")


if __name__ == '__main__':
    main()
