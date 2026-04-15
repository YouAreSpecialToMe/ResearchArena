"""Train multiple TopK SAEs on Pythia-160M with different random seeds."""

import sys
import os
import json
import time
import uuid

# Monkey-patch wandb.util before sae_lens imports it
import types
import wandb
if not hasattr(wandb, 'util'):
    util_module = types.ModuleType('wandb.util')
    util_module.generate_id = lambda length=8: uuid.uuid4().hex[:length]
    wandb.util = util_module
    sys.modules['wandb.util'] = util_module

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from sae_lens.config import LoggingConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig


def train_single_sae(layer: int, seed: int, output_dir: str, dict_size: int = DICT_SIZE,
                      k: int = TOPK_K, n_tokens: int = N_TRAINING_TOKENS,
                      architecture: str = "topk", l1_coeff: float = 3e-3):
    """Train a single SAE and return training summary."""
    os.makedirs(output_dir, exist_ok=True)

    hook_point = f"blocks.{layer}.hook_resid_post"

    if architecture == "topk":
        sae_cfg = TopKTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=dict_size,
            k=k,
            device="cuda",
            dtype="float32",
        )
    else:
        sae_cfg = StandardTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=dict_size,
            l1_coefficient=l1_coeff,
            device="cuda",
            dtype="float32",
        )

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=MODEL_NAME,
        hook_name=hook_point,
        dataset_path=DATASET_PATH,
        streaming=True,
        context_size=CONTEXT_SIZE,
        is_dataset_tokenized=False,
        prepend_bos=True,
        training_tokens=n_tokens,
        train_batch_size_tokens=BATCH_SIZE,
        store_batch_size_prompts=32,
        n_batches_in_buffer=64,
        logger=LoggingConfig(log_to_wandb=False),
        device="cuda",
        seed=seed,
        dtype="float32",
        checkpoint_path=output_dir,
        lr=LR,
        lr_warm_up_steps=500,
        output_path=output_dir,
        save_final_checkpoint=True,
        n_checkpoints=0,
        verbose=False,
    )

    start_time = time.time()
    sae = SAETrainingRunner(cfg).run()
    elapsed = time.time() - start_time

    # Save decoder weights for efficient matching later
    W_dec = sae.W_dec.detach().cpu()
    torch.save(W_dec, os.path.join(output_dir, "W_dec.pt"))

    # Also save encoder weights and biases for full SAE reconstruction
    state = {
        "W_enc": sae.W_enc.detach().cpu(),
        "W_dec": W_dec,
        "b_enc": sae.b_enc.detach().cpu(),
        "b_dec": sae.b_dec.detach().cpu(),
    }
    torch.save(state, os.path.join(output_dir, "sae_weights.pt"))

    summary = {
        "layer": layer,
        "seed": seed,
        "dict_size": dict_size,
        "architecture": architecture,
        "training_time_minutes": elapsed / 60,
        "output_dir": output_dir,
    }

    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Layer {layer}, Seed {seed}: {elapsed/60:.1f} min")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--dict_size", type=int, default=DICT_SIZE)
    parser.add_argument("--architecture", type=str, default="topk")
    parser.add_argument("--n_tokens", type=int, default=N_TRAINING_TOKENS)
    parser.add_argument("--l1_coeff", type=float, default=3e-3)
    parser.add_argument("--output_base", type=str, default=None)
    args = parser.parse_args()

    if args.output_base is None:
        args.output_base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            args.architecture
        )

    all_summaries = []
    total_start = time.time()

    for layer in args.layers:
        for seed in args.seeds:
            output_dir = os.path.join(
                args.output_base,
                f"layer_{layer}",
                f"seed_{seed}"
            )

            if os.path.exists(os.path.join(output_dir, "W_dec.pt")):
                print(f"  Skipping layer {layer}, seed {seed} (already trained)")
                summary_path = os.path.join(output_dir, "training_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        all_summaries.append(json.load(f))
                continue

            summary = train_single_sae(
                layer=layer,
                seed=seed,
                output_dir=output_dir,
                dict_size=args.dict_size,
                architecture=args.architecture,
                n_tokens=args.n_tokens,
                l1_coeff=args.l1_coeff,
            )
            all_summaries.append(summary)
            torch.cuda.empty_cache()

    total_time = (time.time() - total_start) / 60
    print(f"\nTotal training time: {total_time:.1f} minutes")

    results_path = os.path.join(args.output_base, "training_summary.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "summaries": all_summaries,
            "total_time_minutes": total_time,
        }, f, indent=2)


if __name__ == "__main__":
    main()
