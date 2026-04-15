"""Run CoT ablation: evaluate models with Chain-of-Thought prompting."""

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


COT_MODELS = [
    {
        'name': 'meta-llama/Llama-3.1-8B-Instruct',
        'short': 'llama31_8b',
        'max_tokens': 1024,
        'max_model_len': 4096,
    },
    {
        'name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'short': 'deepseek_r1_7b',
        'max_tokens': 1024,
        'max_model_len': 4096,
    },
]


def main():
    dataset = load_dataset('seed_42')
    total_start = time.time()
    cot_results = {}

    for model_cfg in COT_MODELS:
        short_name = model_cfg['short']
        parsed_path = os.path.join(RESULTS_DIR, 'parsed',
                                   f'{short_name}_cot_seed_42.json')
        if os.path.exists(parsed_path):
            print(f"\nSkipping {short_name} CoT (already done)")
            with open(parsed_path) as f:
                cot_results[short_name] = json.load(f)
            continue

        metrics = run_inference(
            model_name=model_cfg['name'],
            dataset=dataset,
            output_dir=RESULTS_DIR,
            model_short_name=short_name,
            seed_name='seed_42',
            use_cot=True,
            max_tokens=model_cfg['max_tokens'],
            max_model_len=model_cfg.get('max_model_len', 4096),
        )
        cot_results[short_name] = metrics

    total_time = time.time() - total_start
    print(f"\nCoT ablation total time: {total_time/60:.1f} minutes")

    # Save CoT ablation results
    outpath = os.path.join(RESULTS_DIR, 'aggregated', 'cot_ablation.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(cot_results, f, indent=2)
    print(f"CoT ablation results saved to {outpath}")


if __name__ == '__main__':
    main()
