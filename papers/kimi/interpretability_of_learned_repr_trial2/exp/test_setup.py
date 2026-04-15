"""Test script to verify environment setup and model loading."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import torch
from exp.shared import get_model_and_sae, set_seed

def main():
    print("Testing environment setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    set_seed(42)
    
    print("\nLoading GPT-2 Small and SAE...")
    try:
        model, sae, cfg_dict, hook_point = get_model_and_sae(
            model_name="gpt2-small",
            layer=8,
            device="cuda"
        )
        print(f"Model loaded successfully!")
        print(f"Hook point: {hook_point}")
        print(f"SAE config: {cfg_dict}")
        print(f"SAE d_in: {sae.cfg.d_in}")
        print(f"SAE d_sae: {sae.cfg.d_sae}")
        
        # Test forward pass
        test_prompt = "Hello, world!"
        tokens = model.to_tokens(test_prompt)
        print(f"\nTest prompt: '{test_prompt}'")
        print(f"Tokens shape: {tokens.shape}")
        
        with torch.no_grad():
            logits = model(tokens)
            print(f"Logits shape: {logits.shape}")
            
            # Test SAE encoding
            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            acts = cache[hook_point]
            feature_acts = sae.encode(acts)
            print(f"Feature activations shape: {feature_acts.shape}")
            
            # Count active features
            active = (feature_acts > 0).sum().item()
            print(f"Active features: {active} / {sae.cfg.d_sae}")
        
        print("\n✓ Setup test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
