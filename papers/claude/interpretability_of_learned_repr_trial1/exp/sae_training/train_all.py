"""Train TopK SAEs on GPT-2 Small, Pythia-160M, and Pythia-410M with multiple seeds."""
import sys, os, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from shared.sae import TopKSAE

DEVICE = "cuda"
DTYPE = torch.float32

# Configuration
MODELS = {
    "gpt2_small": {"name": "gpt2", "layer": 6, "d_model": 768},
    "pythia_160m": {"name": "EleutherAI/pythia-160m", "layer": 6, "d_model": 768},
    "pythia_410m": {"name": "EleutherAI/pythia-410m", "layer": 12, "d_model": 1024},
}
SEEDS = [42, 123, 456, 789, 1024]
N_FEATURES = 16384
K = 64
BATCH_SIZE = 4096  # tokens per batch
SEQ_LEN = 128
# Use 10M tokens per SAE (practical for time budget; still enough for convergence)
N_TOKENS_TRAIN = 10_000_000
LR = 3e-4
WARMUP_STEPS = 500
EVAL_TOKENS = 50000  # for activation collection

SAVE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_tokenized_data(tokenizer, n_tokens):
    """Load wikitext-103 and tokenize."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    all_ids = []
    total = 0
    for ex in ds:
        text = ex["text"].strip()
        if len(text) < 50:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]
        all_ids.append(ids)
        total += len(ids)
        if total >= n_tokens + 200000:  # extra for eval
            break

    all_ids = torch.cat(all_ids)
    return all_ids


def collect_activations_streaming(model, token_ids, layer_idx, batch_size=64, seq_len=SEQ_LEN):
    """Collect residual stream activations from a specific layer."""
    model.eval()
    all_acts = []
    n_seqs = len(token_ids) // seq_len
    token_ids = token_ids[:n_seqs * seq_len].reshape(n_seqs, seq_len)

    for i in tqdm(range(0, len(token_ids), batch_size), desc="Collecting activations"):
        batch = token_ids[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            acts = outputs.hidden_states[layer_idx + 1]
            all_acts.append(acts.reshape(-1, acts.shape[-1]).cpu().to(torch.float32))

    return torch.cat(all_acts, dim=0)


def train_sae(activations, d_model, seed, n_features=N_FEATURES, k=K):
    """Train a single TopK SAE on pre-collected activations."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    sae = TopKSAE(d_model, n_features, k).to(DEVICE).to(DTYPE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    n_tokens = len(activations)
    n_batches = n_tokens // BATCH_SIZE
    total_steps = n_batches

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.033, 0.5 * (1 + np.cos(np.pi * progress)))  # decay to lr/30

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    losses = []
    l0s = []

    perm = torch.randperm(n_tokens)
    for batch_i in tqdm(range(n_batches), desc=f"Seed {seed}", leave=False):
        idx = perm[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
        x = activations[idx].to(DEVICE)

        result = sae(x)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder()

        if batch_i % 200 == 0:
            losses.append(loss.item())
            with torch.no_grad():
                topk_vals, _ = sae.encode(x[:256])
                active = (topk_vals > 0).float().sum(dim=-1).mean().item()
                l0s.append(active)

    # Final eval
    sae.eval()
    with torch.no_grad():
        eval_x = activations[:5000].to(DEVICE)
        result = sae(eval_x)
        final_mse = result["recon_loss"].item()
        var_x = eval_x.var(dim=0).sum().item()
        var_residual = (eval_x - result["x_hat"]).var(dim=0).sum().item()
        explained_var = 1 - var_residual / var_x

        full_acts = sae.get_activations(eval_x)
        dead_frac = (full_acts.sum(dim=0) == 0).float().mean().item()

    metrics = {
        "final_mse": final_mse,
        "explained_variance": explained_var,
        "mean_l0": l0s[-1] if l0s else 0,
        "dead_feature_fraction": dead_frac,
        "n_training_tokens": n_tokens,
        "loss_curve": losses,
    }

    return sae, metrics


def collect_eval_activations(sae, model_acts, n_eval=EVAL_TOKENS):
    """Collect SAE feature activations on eval data."""
    sae.eval()
    acts_list = []
    for i in range(0, min(n_eval, len(model_acts)), 2048):
        batch = model_acts[i:i+2048].to(DEVICE)
        with torch.no_grad():
            z = sae.get_activations(batch)
            acts_list.append(z.cpu())
    return torch.cat(acts_list, dim=0)[:n_eval]


def main():
    all_metrics = {}
    total_start = time.time()

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_key} ({model_cfg['name']})")
        print(f"{'='*60}")

        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"]).to(DEVICE)
        model.eval()

        # Load and tokenize dataset
        print("Tokenizing dataset...")
        token_ids = get_tokenized_data(tokenizer, N_TOKENS_TRAIN)
        print(f"  Total tokens: {len(token_ids):,}")

        # Split into train and eval
        train_tokens = token_ids[:N_TOKENS_TRAIN]
        eval_tokens = token_ids[N_TOKENS_TRAIN:N_TOKENS_TRAIN + 200000]

        # Collect model activations
        print(f"Collecting activations from layer {model_cfg['layer']}...")
        t0 = time.time()
        train_acts = collect_activations_streaming(model, train_tokens, model_cfg["layer"])
        eval_acts_model = collect_activations_streaming(model, eval_tokens, model_cfg["layer"])
        print(f"  Train acts: {train_acts.shape}, Eval acts: {eval_acts_model.shape}")
        print(f"  Time: {time.time()-t0:.0f}s")

        # Free model memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Train SAEs with different seeds
        for seed in SEEDS:
            print(f"\n--- Training SAE: {model_key}, seed={seed} ---")
            t0 = time.time()
            sae, metrics = train_sae(train_acts, model_cfg["d_model"], seed)
            elapsed = time.time() - t0
            metrics["runtime_minutes"] = elapsed / 60

            print(f"  MSE: {metrics['final_mse']:.6f}, "
                  f"Explained Var: {metrics['explained_variance']:.4f}, "
                  f"L0: {metrics['mean_l0']:.1f}, "
                  f"Dead: {metrics['dead_feature_fraction']:.3f}, "
                  f"Time: {elapsed/60:.1f} min")

            # Save SAE
            save_path = os.path.join(SAVE_DIR, f"{model_key}_seed{seed}.pt")
            torch.save({
                "state_dict": sae.state_dict(),
                "config": {"d_model": model_cfg["d_model"], "n_features": N_FEATURES, "k": K},
                "metrics": {k: v for k, v in metrics.items() if k != "loss_curve"},
            }, save_path)

            # Collect and save SAE eval activations
            print("  Collecting eval activations...")
            sae_eval_acts = collect_eval_activations(sae, eval_acts_model)
            eval_path = os.path.join(SAVE_DIR, f"activations_{model_key}_seed{seed}.pt")
            torch.save(sae_eval_acts, eval_path)

            all_metrics[f"{model_key}_seed{seed}"] = {
                k: v for k, v in metrics.items() if k != "loss_curve"
            }

            del sae, sae_eval_acts
            gc.collect()
            torch.cuda.empty_cache()

        del train_acts, eval_acts_model
        gc.collect()
        torch.cuda.empty_cache()

    # Save all metrics
    with open(os.path.join(RESULTS_DIR, "sae_training_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    total_time = (time.time() - total_start) / 60
    print(f"\n\nAll SAEs trained in {total_time:.1f} minutes!")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
