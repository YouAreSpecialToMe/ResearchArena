"""
Model loading and activation extraction utilities.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import numpy as np

def load_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    return model, tokenizer

def get_layer_activations(model, tokenizer, text: str, layer_indices: List[int], 
                          device: str = "cuda") -> Dict[int, torch.Tensor]:
    """Extract activations from specific layers."""
    activations = {}
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # Hook function to capture activations
    def hook_fn(layer_idx):
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Take last token's activation
            activations[layer_idx] = hidden_states[:, -1, :].detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        h = layer.register_forward_hook(hook_fn(layer_idx))
        hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return activations

def generate_with_steering(model, tokenizer, prompt: str, 
                           steering_vectors: Optional[Dict[int, torch.Tensor]] = None,
                           steering_weights: Optional[Dict[int, float]] = None,
                           max_new_tokens: int = 256, temperature: float = 0.0,
                           device: str = "cuda") -> str:
    """Generate text with optional activation steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    # Prepare hooks if steering vectors provided
    hooks = []
    if steering_vectors and steering_weights:
        def steering_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    # Apply steering
                    if layer_idx in steering_vectors and layer_idx in steering_weights:
                        vector = steering_vectors[layer_idx].to(hidden_states.device)
                        weight = steering_weights[layer_idx]
                        hidden_states = hidden_states + weight * vector.unsqueeze(0).unsqueeze(0)
                    return (hidden_states,) + output[1:]
                else:
                    if layer_idx in steering_vectors and layer_idx in steering_weights:
                        vector = steering_vectors[layer_idx].to(output.device)
                        weight = steering_weights[layer_idx]
                        output = output + weight * vector.unsqueeze(0).unsqueeze(0)
                    return output
            return hook
        
        for layer_idx in steering_vectors.keys():
            layer = model.model.layers[layer_idx]
            h = layer.register_forward_hook(steering_hook(layer_idx))
            hooks.append(h)
    
    # Generate
    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Decode only the generated part
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

def extract_contrast_vector(model, tokenizer, positive_texts: List[str], 
                            negative_texts: List[str], layer_idx: int,
                            device: str = "cuda") -> torch.Tensor:
    """Extract contrast vector from positive and negative examples."""
    positive_acts = []
    negative_acts = []
    
    # Extract activations for positive examples
    for text in positive_texts:
        acts = get_layer_activations(model, tokenizer, text, [layer_idx], device)
        if layer_idx in acts:
            positive_acts.append(acts[layer_idx])
    
    # Extract activations for negative examples
    for text in negative_texts:
        acts = get_layer_activations(model, tokenizer, text, [layer_idx], device)
        if layer_idx in acts:
            negative_acts.append(acts[layer_idx])
    
    if not positive_acts or not negative_acts:
        # Return zero vector if extraction failed
        dummy_input = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            dummy_output = model(**dummy_input, output_hidden_states=True)
            hidden_size = dummy_output.hidden_states[-1].shape[-1]
        return torch.zeros(hidden_size)
    
    # Compute mean difference
    positive_mean = torch.cat(positive_acts, dim=0).mean(dim=0)
    negative_mean = torch.cat(negative_acts, dim=0).mean(dim=0)
    contrast_vector = positive_mean - negative_mean
    
    # Normalize
    contrast_vector = F.normalize(contrast_vector, p=2, dim=0)
    
    return contrast_vector

def compute_perplexity(model, tokenizer, text: str, device: str = "cuda") -> float:
    """Compute perplexity of text under model."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

if __name__ == "__main__":
    print("Testing models module...")
    print("This module requires GPU and model download, skipping direct tests.")
