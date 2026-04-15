"""Data loading utilities for LayerSelect experiments."""
import os
import json
import random
import torch
from typing import List, Dict, Optional
from datasets import load_dataset
from transformers import AutoTokenizer


def load_calibration_data(n_samples: int = 256, seq_len: int = 4096, 
                          tokenizer=None, cache_dir: str = "data/calibration") -> List[str]:
    """Load or create calibration data for profiling."""
    cache_file = os.path.join(cache_dir, f"calibration_{n_samples}_{seq_len}.jsonl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached calibration data from {cache_file}")
        with open(cache_file, 'r') as f:
            return [json.loads(line)["text"] for line in f]
    
    print(f"Generating calibration data: {n_samples} samples x {seq_len} tokens")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use C4 subset as calibration data
    try:
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    except:
        # Fallback to a simpler dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    samples = []
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    for i, example in enumerate(dataset):
        if len(samples) >= n_samples:
            break
        text = example.get("text", example.get("content", ""))
        if len(text) > 100:  # Only use substantial texts
            # Tokenize and check length
            tokens = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
            if tokens["input_ids"].shape[1] >= seq_len // 2:  # At least half the desired length
                samples.append(text[:4000])  # Truncate to reasonable length
    
    # Save calibration data
    with open(cache_file, 'w') as f:
        for text in samples[:n_samples]:
            f.write(json.dumps({"text": text}) + "\n")
    
    return samples[:n_samples]


def create_ruler_niah_single(seq_len: int = 4096, num_samples: int = 10) -> List[Dict]:
    """Create Needle-in-Haystack (single) tasks for RULER-style evaluation."""
    samples = []
    
    # Templates for needle insertion
    templates = [
        "The secret number is {needle}. Remember this number.",
        "Important: {needle}. Keep this in mind.",
        "Note: {needle}. Don't forget.",
    ]
    
    # Generate haystack text (repeated passages)
    haystack_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has revolutionized computer vision.",
        "Transformers have become the dominant architecture in NLP.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Large language models can generate human-like text.",
        "The development of AI has accelerated in recent years.",
        "Data quality is crucial for training effective models.",
        "Optimization algorithms help models learn from data.",
    ] * 100  # Repeat to ensure enough content
    
    for i in range(num_samples):
        needle = f"[{i*1000 + 1234}]"  # Unique needle for each sample
        template = templates[i % len(templates)]
        needle_text = template.format(needle=needle)
        
        # Create haystack by joining sentences
        haystack = " ".join(haystack_sentences[:seq_len//10])
        
        # Insert needle at random position
        insert_pos = random.randint(len(haystack)//4, 3*len(haystack)//4)
        text = haystack[:insert_pos] + " " + needle_text + " " + haystack[insert_pos:]
        
        # Truncate to exact sequence length
        text = text[:seq_len * 4]  # Approximate chars to tokens ratio
        
        samples.append({
            "text": text,
            "needle": needle,
            "question": "What is the secret number?",
            "answer": needle,
            "task": "niah_single"
        })
    
    return samples


def create_ruler_qa_task(seq_len: int = 4096, num_samples: int = 10) -> List[Dict]:
    """Create QA tasks for evaluation."""
    samples = []
    
    contexts = [
        {
            "text": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "qa_pairs": [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is Paris known for?", "answer": "Eiffel Tower"}
            ]
        },
        {
            "text": "Python was created by Guido van Rossum in 1991. It is a popular programming language.",
            "qa_pairs": [
                {"question": "Who created Python?", "answer": "Guido van Rossum"},
                {"question": "When was Python created?", "answer": "1991"}
            ]
        },
        {
            "text": "The Earth orbits around the Sun. One orbit takes approximately 365 days.",
            "qa_pairs": [
                {"question": "What does Earth orbit around?", "answer": "Sun"},
                {"question": "How long does one orbit take?", "answer": "365 days"}
            ]
        },
        {
            "text": "Shakespeare wrote Romeo and Juliet. It is a famous tragedy about two lovers.",
            "qa_pairs": [
                {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
                {"question": "What is Romeo and Juliet about?", "answer": "two lovers"}
            ]
        },
        {
            "text": "The Great Wall of China was built over many centuries. It is thousands of miles long.",
            "qa_pairs": [
                {"question": "How long is the Great Wall?", "answer": "thousands of miles"},
                {"question": "How long did it take to build the Great Wall?", "answer": "many centuries"}
            ]
        },
    ]
    
    # Repeat contexts and add filler text to reach desired sequence length
    filler = " This is additional context to increase sequence length. " * 1000
    
    for i in range(num_samples):
        context = contexts[i % len(contexts)]
        qa = context["qa_pairs"][i % len(context["qa_pairs"])]
        
        # Build long context with filler
        long_context = context["text"] + filler[:seq_len * 3]
        long_context = long_context[:seq_len * 3]
        
        samples.append({
            "text": long_context,
            "question": qa["question"],
            "answer": qa["answer"],
            "task": "qa"
        })
    
    return samples


def prepare_ruler_benchmark(seq_lengths: List[int] = [4096, 8192], 
                            output_dir: str = "data/ruler_processed") -> Dict[str, List[Dict]]:
    """Prepare RULER benchmark subsets."""
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark = {}
    for seq_len in seq_lengths:
        niah_samples = create_ruler_niah_single(seq_len, num_samples=10)
        qa_samples = create_ruler_qa_task(seq_len, num_samples=10)
        
        benchmark[f"niah_{seq_len}"] = niah_samples
        benchmark[f"qa_{seq_len}"] = qa_samples
        
        # Save to file
        with open(os.path.join(output_dir, f"niah_{seq_len}.json"), 'w') as f:
            json.dump(niah_samples, f, indent=2)
        with open(os.path.join(output_dir, f"qa_{seq_len}.json"), 'w') as f:
            json.dump(qa_samples, f, indent=2)
    
    return benchmark


def load_benchmark(benchmark_name: str, seq_len: int = 4096, 
                   data_dir: str = "data/ruler_processed") -> List[Dict]:
    """Load a specific benchmark."""
    if benchmark_name == "ruler":
        file_path = os.path.join(data_dir, f"niah_{seq_len}.json")
        if not os.path.exists(file_path):
            prepare_ruler_benchmark([seq_len], data_dir)
        with open(file_path, 'r') as f:
            return json.load(f)
    elif benchmark_name == "longbench":
        # Simplified longbench - just use QA tasks
        file_path = os.path.join(data_dir, f"qa_{seq_len}.json")
        if not os.path.exists(file_path):
            prepare_ruler_benchmark([seq_len], data_dir)
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
