"""
Data loading utilities for SAE training and robustness evaluation.
Handles activation extraction and semantic perturbation generation.
"""
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import random


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_openwebtext_data(num_tokens=2_000_000, seq_len=512, seed=42):
    """Load and tokenize OpenWebText dataset."""
    set_seed(seed)
    
    # Calculate number of sequences needed
    num_sequences = num_tokens // seq_len
    
    # Load dataset
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token
    
    all_tokens = []
    buffer = ""
    
    for example in tqdm(dataset, desc="Loading OpenWebText", total=num_sequences):
        buffer += example['text'] + " "
        
        # Tokenize when buffer is large enough
        if len(buffer) > 10000:
            tokens = tokenizer(buffer, return_tensors="pt", truncation=True, 
                             max_length=seq_len, padding="max_length")
            all_tokens.append(tokens['input_ids'])
            buffer = ""
            
            if len(all_tokens) >= num_sequences:
                break
    
    # Stack all tokens
    all_tokens = torch.cat(all_tokens[:num_sequences], dim=0)
    
    # Split into train/val
    n_train = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]
    
    return train_tokens, val_tokens, tokenizer


def extract_activations(model_name, layer_idx, tokens, batch_size=8, device='cuda'):
    """Extract activations from a specific layer of the model."""
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        output_hidden_states=True
    )
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Extracting activations"):
            batch = tokens[i:i+batch_size].to(device)
            outputs = model(batch, output_hidden_states=True)
            
            # Get activations from specified layer
            # Shape: [batch, seq_len, hidden_dim]
            activations = outputs.hidden_states[layer_idx]
            all_activations.append(activations.cpu().float())
    
    # Concatenate all batches
    all_activations = torch.cat(all_activations, dim=0)
    
    # Reshape to [num_samples * seq_len, hidden_dim]
    batch_size_total, seq_len, hidden_dim = all_activations.shape
    all_activations = all_activations.reshape(-1, hidden_dim)
    
    return all_activations


def create_semantic_perturbations(tokens, tokenizer, perturbation_type='dropout', dropout_rate=0.3):
    """
    Create perturbed versions of input tokens for consistency training.
    
    Args:
        tokens: [batch, seq_len] token ids
        tokenizer: tokenizer object
        perturbation_type: 'dropout', 'shuffle', or ' synonym'
        dropout_rate: probability of dropping a token
    """
    perturbed = tokens.clone()
    
    if perturbation_type == 'dropout':
        # Random token dropout (like denoising)
        mask = torch.rand_like(perturbed.float()) > dropout_rate
        perturbed = perturbed * mask.long()
        # Replace dropped tokens with pad token
        perturbed[perturbed == 0] = tokenizer.pad_token_id
        
    elif perturbation_type == 'shuffle':
        # Shuffle tokens within each sequence slightly
        batch_size, seq_len = tokens.shape
        for i in range(batch_size):
            # Shuffle 10% of positions
            num_shuffle = max(1, seq_len // 10)
            shuffle_idx = torch.randperm(seq_len)[:num_shuffle]
            shuffled = perturbed[i, shuffle_idx][torch.randperm(num_shuffle)]
            perturbed[i, shuffle_idx] = shuffled
    
    return perturbed


def get_art_science_prompts():
    """Return art and science prompts for population-level attack evaluation."""
    art_prompts = [
        "The painting depicts a serene landscape with vibrant colors",
        "The sculpture captures the essence of human emotion",
        "Renaissance art revolutionized perspective and composition",
        "Abstract expressionism emphasizes spontaneous creation",
        "The museum houses masterpieces from various artistic movements",
        "Impressionist painters focused on capturing light and atmosphere",
        "The artist used bold brushstrokes to convey movement",
        "Baroque art is characterized by dramatic lighting and emotion",
        "The portrait reveals the subject's inner psychology",
        "Contemporary art challenges traditional aesthetic boundaries",
        "The fresco covers the entire ceiling of the chapel",
        "Cubism deconstructs objects into geometric forms",
        "The gallery opening attracted art enthusiasts and collectors",
        "Surrealism explores the unconscious mind through imagery",
        "The color palette evokes a sense of melancholy and nostalgia",
    ]
    
    science_prompts = [
        "The experiment demonstrated quantum entanglement principles",
        "Researchers discovered a new species in the Amazon rainforest",
        "The chemical reaction produced a stable crystalline compound",
        "Astrophysicists measured the redshift of distant galaxies",
        "The DNA sequencing revealed genetic mutations associated with disease",
        "Climate models predict significant temperature increases",
        "The particle accelerator produced evidence of the Higgs boson",
        "Neuroscientists mapped neural pathways in the visual cortex",
        "The mathematical proof resolves a long-standing conjecture",
        "Biologists studied cellular mechanisms of protein synthesis",
        "The telescope captured images of exoplanet atmospheres",
        "Chemists synthesized a new polymer with unique properties",
        "Physicists observed superconductivity at room temperature",
        "The epidemiological study tracked disease transmission patterns",
        "Geologists analyzed rock formations to understand plate tectonics",
    ]
    
    return art_prompts, science_prompts


class ActivationDataset(torch.utils.data.Dataset):
    """Dataset for pre-extracted activations."""
    
    def __init__(self, activations):
        self.activations = activations
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


class ConsistencyDataset(torch.utils.data.Dataset):
    """Dataset that provides original and perturbed activations."""
    
    def __init__(self, tokens, model, layer_idx, tokenizer, 
                 perturbation_type='dropout', dropout_rate=0.3, device='cuda'):
        self.tokens = tokens
        self.model = model
        self.layer_idx = layer_idx
        self.tokenizer = tokenizer
        self.perturbation_type = perturbation_type
        self.dropout_rate = dropout_rate
        self.device = device
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        # Get original tokens
        original_tokens = self.tokens[idx:idx+1].to(self.device)
        
        # Create perturbed version
        perturbed_tokens = create_semantic_perturbations(
            original_tokens, self.tokenizer, 
            self.perturbation_type, self.dropout_rate
        )
        
        # Extract activations
        with torch.no_grad():
            orig_out = self.model(original_tokens, output_hidden_states=True)
            pert_out = self.model(perturbed_tokens, output_hidden_states=True)
            
            orig_act = orig_out.hidden_states[self.layer_idx][0].mean(dim=0).cpu()
            pert_act = pert_out.hidden_states[self.layer_idx][0].mean(dim=0).cpu()
        
        return orig_act, pert_act
