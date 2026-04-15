"""Model loading and inference utilities for ESR experiments."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def load_model(model_name: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {param_count:.2f}B parameters")
    
    return model, tokenizer


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def compute_varentropy(logits: torch.Tensor) -> float:
    """Compute varentropy (variance of entropy) from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    
    # Varentropy = E[(log p + H)^2]
    varentropy = (probs * (log_probs + entropy) ** 2).sum().item()
    return varentropy


def compute_uncertainty_metrics(logits: torch.Tensor) -> Dict[str, float]:
    """Compute both entropy and varentropy."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    entropy = -(probs * log_probs).sum()
    varentropy = (probs * (log_probs + entropy) ** 2).sum()
    
    # Also compute max probability
    max_prob = probs.max().item()
    
    return {
        "entropy": entropy.item(),
        "varentropy": varentropy.item(),
        "max_prob": max_prob
    }


def classify_uncertainty_regime(entropy: float, varentropy: float, 
                                 tau_h: float, tau_v: float) -> str:
    """Classify uncertainty regime based on entropy and varentropy."""
    if entropy < tau_h:
        return "confident"
    elif varentropy > tau_v:
        return "fork"
    else:
        return "confused"


def generate_with_uncertainty_tracking(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    device: str = "cuda"
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate text while tracking uncertainty at each step."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generated_ids = inputs["input_ids"]
    uncertainty_trace = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids,
                output_attentions=False,
                return_dict=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Compute uncertainty metrics
            metrics = compute_uncertainty_metrics(next_token_logits[0])
            
            # Sample or greedy
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            metrics["token_id"] = next_token.item()
            metrics["token"] = tokenizer.decode(next_token.item())
            uncertainty_trace.append(metrics)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Remove the prompt from generated text
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    new_text = generated_text[len(prompt_text):]
    
    return new_text, uncertainty_trace


def generate_vanilla_cot(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    device: str = "cuda"
) -> Tuple[str, int]:
    """Standard chain-of-thought generation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    new_text = generated_text[len(prompt_text):]
    
    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    
    return new_text, num_tokens


def detect_step_boundary(text: str, tokenizer: Any) -> List[int]:
    """Detect reasoning step boundaries in text."""
    # Simple heuristic: newlines followed by numbers or bullet points
    lines = text.split('\n')
    boundaries = []
    pos = 0
    for i, line in enumerate(lines):
        if i > 0:
            # Check if line starts with step indicator
            stripped = line.strip()
            if (re.match(r'^(Step\s+\d+|\d+[.):\-]|\-|\*)', stripped, re.IGNORECASE) or
                any(word in stripped.lower() for word in ['first', 'second', 'third', 'next', 'then', 'finally'])):
                boundaries.append(pos)
        pos += len(line) + 1  # +1 for newline
    return boundaries


import re
