"""Inference engine using vLLM for FlipBench evaluation."""

import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prompts import format_chat_messages
from parse_answers import parse_answer, check_answer
from metrics import compute_metrics


def run_inference(model_name, dataset, output_dir, model_short_name,
                  seed_name="seed_42", use_cot=False, max_tokens=512,
                  gpu_memory_utilization=0.85, quantization=None,
                  max_model_len=4096):
    """Run vLLM inference on FlipBench dataset.

    Args:
        model_name: HuggingFace model name
        dataset: list of FlipBench instances
        output_dir: base output directory
        model_short_name: short name for file naming
        seed_name: dataset seed name
        use_cot: whether to use CoT prompting
        max_tokens: max generation tokens
        gpu_memory_utilization: fraction of GPU memory to use
        quantization: quantization method (e.g., 'awq')
        max_model_len: maximum model context length
    """
    from vllm import LLM, SamplingParams

    suffix = "_cot" if use_cot else ""
    raw_path = os.path.join(output_dir, 'raw', f'{model_short_name}{suffix}_{seed_name}.jsonl')
    parsed_path = os.path.join(output_dir, 'parsed', f'{model_short_name}{suffix}_{seed_name}.json')

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(parsed_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running inference: {model_short_name}{suffix} on {seed_name}")
    print(f"Model: {model_name}")
    print(f"Dataset size: {len(dataset)} instances")
    print(f"Max tokens: {max_tokens}, CoT: {use_cot}")
    print(f"{'='*60}")

    # Load model
    t0 = time.time()
    llm_kwargs = dict(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    if quantization:
        llm_kwargs['quantization'] = quantization

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Prepare prompts
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=["\n\nQuestion:", "\n\nProblem:", "\n\n---"]
    )

    # Format as chat messages and apply template
    prompts = []
    for inst in dataset:
        messages = format_chat_messages(inst, use_cot=use_cot)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\n"
        prompts.append(prompt)

    # Run inference
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - t1
    print(f"Inference completed in {inference_time:.1f}s "
          f"({inference_time/len(dataset)*1000:.1f}ms per instance)")

    # Parse and evaluate
    results = []
    parse_failures = 0

    with open(raw_path, 'w') as raw_f:
        for inst, output in zip(dataset, outputs):
            raw_text = output.outputs[0].text
            parsed, success = parse_answer(raw_text, inst['domain'], inst['direction'])

            if not success:
                parse_failures += 1

            correct = check_answer(parsed, inst['answer'], inst['domain'], inst['direction'])

            result = {
                'id': inst['id'],
                'domain': inst['domain'],
                'difficulty': inst['difficulty'],
                'direction': inst['direction'],
                'gold_answer': inst['answer'],
                'parsed_answer': str(parsed) if parsed is not None else None,
                'correct': correct,
                'parse_success': success,
                'matched_pair_id': inst['matched_pair_id']
            }
            results.append(result)

            # Write raw output
            raw_entry = {**result, 'raw_output': raw_text}
            raw_f.write(json.dumps(raw_entry) + '\n')

    # Compute metrics
    metrics = compute_metrics(results, dataset)
    metrics['meta'] = {
        'model': model_name,
        'model_short_name': model_short_name,
        'seed': seed_name,
        'use_cot': use_cot,
        'max_tokens': max_tokens,
        'load_time_s': round(load_time, 1),
        'inference_time_s': round(inference_time, 1),
        'parse_failures': parse_failures,
        'total_instances': len(dataset)
    }

    with open(parsed_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n--- Results Summary ---")
    print(f"Parse failures: {parse_failures}/{len(dataset)}")
    print(f"Overall: FA={metrics['overall']['forward_accuracy']:.3f}, "
          f"BA={metrics['overall']['backward_accuracy']:.3f}, "
          f"DRG={metrics['overall']['drg']:.3f}")
    for domain in ['propositional_logic', 'arithmetic_reasoning',
                   'relational_reasoning', 'function_computation']:
        d = metrics[domain]
        print(f"  {domain:25s}: FA={d['forward_accuracy']:.3f} BA={d['backward_accuracy']:.3f} "
              f"DRG={d['drg']:.3f} CR={d['consistency_rate']:.3f}")

    # Cleanup GPU memory
    del llm
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return metrics
