"""
Download pre-trained SAEs for Qwen2.5-7B-Instruct.
"""
import os
import json

def main():
    print("=" * 60)
    print("Setting up SAEs for Qwen2.5-7B")
    print("=" * 60)
    
    os.makedirs('models/sae_cache', exist_ok=True)
    
    # SAE configuration for Qwen2.5-7B
    # Note: sae_lens may not have Qwen2.5 SAEs available yet
    # We'll create a configuration file and attempt to load
    
    target_layers = [15, 16, 17, 18, 19, 20]
    
    sae_info = {
        'model': 'qwen2.5-7b',
        'layers': target_layers,
        'sae_ids': {},
        'status': {},
        'note': 'SAE baseline may use mock/proxy features if pretrained SAEs unavailable'
    }
    
    # Try to load SAEs from sae_lens
    try:
        from sae_lens import SAE
        
        for layer in target_layers:
            print(f"\nAttempting to load SAE for layer {layer}...")
            try:
                # Try to load SAE - Qwen2.5 SAEs might not be publicly available
                # Use generic loading approach
                sae = SAE.from_pretrained(
                    release='qwen2.5-7b-res-jb',
                    sae_id=f'blocks.{layer}.hook_resid_pre',
                    device='cpu'
                )
                sae_info['sae_ids'][layer] = f'blocks.{layer}.hook_resid_pre'
                sae_info['status'][layer] = 'loaded'
                print(f"  Successfully loaded SAE for layer {layer}")
            except Exception as e:
                sae_info['status'][layer] = f'not_available: {str(e)[:50]}'
                print(f"  SAE not available for layer {layer}")
    
    except ImportError:
        print("sae_lens not available, will use mock SAE features for baseline")
        for layer in target_layers:
            sae_info['status'][layer] = 'sae_lens_not_imported'
    
    # Save info
    with open('models/sae_cache/sae_info.json', 'w') as f:
        json.dump(sae_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SAE Setup Summary:")
    print(json.dumps(sae_info['status'], indent=2))
    print("=" * 60)
    
    # Check if we have any SAEs loaded
    loaded_count = sum(1 for v in sae_info['status'].values() if v == 'loaded')
    print(f"\nLoaded {loaded_count}/{len(target_layers)} SAEs")
    
    if loaded_count == 0:
        print("\nNote: No SAEs available. Will implement alternative baseline comparison.")

if __name__ == "__main__":
    main()
