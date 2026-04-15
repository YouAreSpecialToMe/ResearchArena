"""Prepare all datasets for experiments."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

from exp.shared.data_loader import prepare_all_datasets

def main():
    print("Preparing datasets...")
    data = prepare_all_datasets(cache_dir="data", max_samples=500)
    
    print(f"\nPrepared datasets:")
    for key in data:
        print(f"  - {key}: {len(data[key])} items")
    
    print("\n✓ Datasets prepared and saved to data/all_datasets.json")

if __name__ == "__main__":
    main()
