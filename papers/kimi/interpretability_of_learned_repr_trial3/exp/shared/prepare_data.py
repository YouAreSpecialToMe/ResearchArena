"""
Data preparation script: Extract activations from Pythia-70M.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import os

# Set seeds
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "EleutherAI/pythia-70m-deduped"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
    output_hidden_states=True
)
model.eval()

# Get model dimensions
d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
print(f"Model has {n_layers} layers, d_model={d_model}")

# Target layer (middle)
layer_idx = n_layers // 2  # Layer 3 for 6-layer model
print(f"Extracting from layer {layer_idx}")

# Load dataset - use smaller subset for faster processing
print("Loading dataset...")
num_sequences = 2000  # Reduced for faster processing
seq_len = 128  # Shorter sequences

# Try to load OpenWebText, fall back to a simpler dataset
try:
    dataset = load_dataset("stas/openwebtext-10k", split="train", streaming=False)
except:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

train_tokens = []
val_tokens = []

print("Tokenizing data...")
buffer = ""
for i, example in enumerate(tqdm(dataset, total=num_sequences)):
    if i >= num_sequences:
        break
    
    text = example.get('text', example.get('content', str(example)))
    buffer += text + " "
    
    if len(buffer) > 1000:
        tokens = tokenizer(buffer[:1000], return_tensors="pt", 
                          truncation=True, max_length=seq_len)
        if tokens['input_ids'].shape[1] == seq_len:
            if len(train_tokens) < int(0.9 * num_sequences):
                train_tokens.append(tokens['input_ids'])
            else:
                val_tokens.append(tokens['input_ids'])
        buffer = ""

print(f"Collected {len(train_tokens)} train, {len(val_tokens)} val sequences")

# Handle case where val_tokens is empty
if len(val_tokens) == 0 and len(train_tokens) > 100:
    # Use last 10% of train as val
    split_idx = int(0.9 * len(train_tokens))
    val_tokens = train_tokens[split_idx:]
    train_tokens = train_tokens[:split_idx]
    print(f"Split train into {len(train_tokens)} train, {len(val_tokens)} val")

# Stack tokens
train_tokens = torch.cat(train_tokens, dim=0) if train_tokens else torch.zeros((1, seq_len), dtype=torch.long)
val_tokens = torch.cat(val_tokens, dim=0) if val_tokens else torch.zeros((1, seq_len), dtype=torch.long)

print(f"Train tokens: {train_tokens.shape}")
print(f"Val tokens: {val_tokens.shape}")

# Extract activations
def extract_activations(tokens, batch_size=4):
    all_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Extracting"):
            batch = tokens[i:i+batch_size].to(device)
            outputs = model(batch, output_hidden_states=True)
            
            # Get activations from target layer
            acts = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden]
            all_activations.append(acts.cpu().float())
    
    # Concatenate and reshape
    all_activations = torch.cat(all_activations, dim=0)
    batch_size_total, seq_len, hidden_dim = all_activations.shape
    all_activations = all_activations.reshape(-1, hidden_dim)
    
    return all_activations

print("Extracting training activations...")
train_activations = extract_activations(train_tokens)
print(f"Train activations: {train_activations.shape}")

print("Extracting validation activations...")
val_activations = extract_activations(val_tokens)
print(f"Val activations: {val_activations.shape}")

# Save activations
os.makedirs('data', exist_ok=True)
torch.save({
    'train': train_activations,
    'val': val_activations,
    'layer_idx': layer_idx,
    'd_model': d_model
}, 'data/activations_pythia70m_layer3.pt')

print("Saved activations to data/activations_pythia70m_layer3.pt")
print(f"Train shape: {train_activations.shape}")
print(f"Val shape: {val_activations.shape}")
print(f"Mean: {train_activations.mean():.4f}, Std: {train_activations.std():.4f}")
