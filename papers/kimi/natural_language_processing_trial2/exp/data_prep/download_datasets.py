"""
Download and preprocess datasets for ACT-DRS experiments.
"""
import sys
sys.path.append('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01')

from exp.shared.data_loader import load_mgsm_data, load_msvamp_data, set_seed, FOCUS_LANGUAGES
import json
import os

def main():
    set_seed(42)
    
    print("=" * 60)
    print("Downloading and Preprocessing Datasets")
    print("=" * 60)
    
    # Create directories
    os.makedirs('data/mgsm', exist_ok=True)
    os.makedirs('data/msvamp', exist_ok=True)
    
    # Download MGSM
    print("\n1. Loading MGSM dataset...")
    mgsm_data = load_mgsm_data(split='test', languages=FOCUS_LANGUAGES)
    
    for lang in FOCUS_LANGUAGES:
        if lang in mgsm_data:
            output_path = f'data/mgsm/{lang}_test.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mgsm_data[lang], f, ensure_ascii=False, indent=2)
            print(f"  Saved {lang}: {len(mgsm_data[lang])} examples -> {output_path}")
    
    # Download MSVAMP
    print("\n2. Loading MSVAMP dataset...")
    msvamp_data = load_msvamp_data(split='test', languages=FOCUS_LANGUAGES)
    
    for lang in FOCUS_LANGUAGES:
        if lang in msvamp_data and len(msvamp_data[lang]) > 0:
            output_path = f'data/msvamp/{lang}_test.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(msvamp_data[lang], f, ensure_ascii=False, indent=2)
            print(f"  Saved {lang}: {len(msvamp_data[lang])} examples -> {output_path}")
        else:
            print(f"  {lang}: No data available")
    
    # Create validation split from MGSM training data
    print("\n3. Creating validation split...")
    try:
        from datasets import load_dataset
        for lang in FOCUS_LANGUAGES:
            try:
                train_data = load_dataset('juletxara/mgsm', f'mgsm_{lang}', split='train', trust_remote_code=True)
                # Take first 50 for validation
                val_items = []
                for i, item in enumerate(train_data):
                    if i >= 50:
                        break
                    val_items.append({
                        'question': item['question'],
                        'answer': str(item['answer_number']),
                        'lang': lang
                    })
                
                output_path = f'data/mgsm/{lang}_val.json'
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(val_items, f, ensure_ascii=False, indent=2)
                print(f"  {lang}: {len(val_items)} validation examples")
            except Exception as e:
                print(f"  {lang}: Error - {e}")
    except Exception as e:
        print(f"Error creating validation split: {e}")
    
    # Create statistics
    stats = {
        'mgsm': {},
        'msvamp': {},
        'languages': FOCUS_LANGUAGES,
        'num_languages': len(FOCUS_LANGUAGES)
    }
    
    for lang in FOCUS_LANGUAGES:
        stats['mgsm'][lang] = len(mgsm_data.get(lang, []))
        stats['msvamp'][lang] = len(msvamp_data.get(lang, []))
    
    with open('data/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data Statistics:")
    print(json.dumps(stats, indent=2))
    print("=" * 60)
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    main()
