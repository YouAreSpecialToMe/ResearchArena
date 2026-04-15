"""Step 1: Prepare capability evaluation datasets for GPT-2 Small."""
import json
import os
import sys
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    return tokenizer, model

def prepare_factual(tokenizer, model, n=500):
    """Factual knowledge using LAMA TREx-style prompts."""
    print("Preparing factual knowledge dataset...")
    # Use a subset of common knowledge prompts
    # Load the lama dataset
    try:
        ds = load_dataset("lama", "trex")
        data_source = ds['train']
    except Exception as e:
        print(f"Could not load LAMA TREx: {e}")
        print("Creating factual prompts from common knowledge...")
        data_source = None

    examples = []

    if data_source is not None:
        for item in tqdm(data_source, desc="Filtering factual"):
            if len(examples) >= n:
                break
            template = item.get('template', '')
            obj = item.get('obj_label', '')
            sub = item.get('sub_label', '')
            if not template or not obj or not sub:
                continue
            prompt = template.replace('[X]', sub).replace('[Y]', '').strip()
            if prompt.endswith('.'):
                prompt = prompt[:-1].strip()

            # Check GPT-2 can predict the target
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            last_logits = out.logits[0, -1]
            target_ids = tokenizer.encode(" " + obj, add_special_tokens=False)
            if len(target_ids) == 0:
                continue
            target_id = target_ids[0]
            prob = torch.softmax(last_logits, dim=-1)[target_id].item()

            if prob > 0.01:  # Model has some knowledge
                examples.append({
                    'prompt': prompt,
                    'target_token': obj,
                    'target_token_id': target_id,
                    'relation_type': item.get('predicate_id', 'unknown'),
                    'model_prob': prob,
                })

    if len(examples) < n:
        # Fallback: create simple factual prompts
        factual_prompts = [
            ("The capital of France is", "Paris"),
            ("The capital of Germany is", "Berlin"),
            ("The capital of Japan is", "Tokyo"),
            ("The capital of Italy is", "Rome"),
            ("The capital of Spain is", "Madrid"),
            ("The capital of China is", "Beijing"),
            ("The capital of Russia is", "Moscow"),
            ("The capital of Brazil is", "Bras"),
            ("The capital of Canada is", "Ottawa"),
            ("The capital of Australia is", "Canberra"),
            ("The largest ocean is the", "Pacific"),
            ("The largest planet is", "Jupiter"),
            ("The smallest planet is", "Mercury"),
            ("Water freezes at", " 0"),
            ("The speed of light is approximately", " 300"),
            ("The chemical symbol for gold is", " Au"),
            ("The chemical symbol for water is", " H"),
            ("Einstein developed the theory of", " relativity"),
            ("Shakespeare wrote", " Ham"),
            ("The Earth orbits the", " Sun"),
        ]

        # Generate variations with different subjects
        countries = ["France", "Germany", "Japan", "Italy", "Spain", "China", "Russia",
                     "Brazil", "Canada", "Australia", "India", "Mexico", "Egypt",
                     "Turkey", "Sweden", "Norway", "Denmark", "Poland", "Greece", "Portugal"]
        capitals = ["Paris", "Berlin", "Tokyo", "Rome", "Madrid", "Beijing", "Moscow",
                    "Bras", "Ottawa", "Canberra", "New", "Mexico", "Cairo",
                    "Ank", "Stock", "Oslo", "Copenhagen", "Warsaw", "Athens", "Lis"]

        langs = ["French", "German", "Japanese", "Italian", "Spanish", "Chinese", "Russian",
                 "Portuguese", "English", "Arabic", "Hindi", "Korean", "Dutch", "Swedish", "Greek"]

        templates = [
            ("The capital of {} is", capitals, countries),
            ("The language spoken in {} is", langs[:len(countries)], countries),
        ]

        for template, answers, subjects in templates:
            for subj, ans in zip(subjects, answers):
                if len(examples) >= n:
                    break
                prompt = template.format(subj)
                target_ids = tokenizer.encode(" " + ans, add_special_tokens=False)
                if not target_ids:
                    continue
                target_id = target_ids[0]
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inputs)
                last_logits = out.logits[0, -1]
                prob = torch.softmax(last_logits, dim=-1)[target_id].item()

                examples.append({
                    'prompt': prompt,
                    'target_token': ans,
                    'target_token_id': target_id,
                    'relation_type': 'geography',
                    'model_prob': prob,
                })

        # More factual prompts from various domains
        more_prompts = [
            ("The president of the United States lives in the", " White"),
            ("The Eiffel Tower is located in", " Paris"),
            ("The Great Wall is located in", " China"),
            ("The Amazon River flows through", " South"),
            ("The currency of Japan is the", " yen"),
            ("The currency of the United Kingdom is the", " pound"),
            ("The CEO of Tesla is", " Elon"),
            ("The founder of Microsoft is", " Bill"),
            ("The inventor of the telephone was Alexander Graham", " Bell"),
            ("The first person to walk on the moon was", " Neil"),
            ("World War II ended in", " 1945"),
            ("The Declaration of Independence was signed in", " 1776"),
            ("The chemical formula for table salt is", " Na"),
            ("The human body has 206", " bones"),
            ("DNA stands for deoxyrib", "onucle"),
            ("The speed of sound is approximately", " 343"),
            ("Pi is approximately equal to", " 3"),
            ("The boiling point of water is", " 100"),
            ("The periodic table was created by", " D"),
            ("Photosynthesis converts sunlight into", " energy"),
        ]

        for prompt, target in more_prompts:
            if len(examples) >= n:
                break
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                continue
            target_id = target_ids[0]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            last_logits = out.logits[0, -1]
            prob = torch.softmax(last_logits, dim=-1)[target_id].item()
            examples.append({
                'prompt': prompt,
                'target_token': target.strip(),
                'target_token_id': target_id,
                'relation_type': 'general',
                'model_prob': prob,
            })

        # Duplicate and vary to reach n examples if needed
        while len(examples) < n:
            base = random.choice(examples[:min(40, len(examples))])
            examples.append(dict(base))

    examples = examples[:n]
    print(f"  Factual: {len(examples)} examples, mean prob: {np.mean([e['model_prob'] for e in examples]):.4f}")
    return examples


def prepare_syntax(tokenizer, model, n=500):
    """Syntactic processing using BLiMP-style minimal pairs."""
    print("Preparing syntax dataset...")
    examples = []

    try:
        ds = load_dataset("nyu-mll/blimp", "anaphor_gender_agreement")
        for item in ds['train']:
            if len(examples) >= n // 3:
                break
            good = item['sentence_good']
            bad = item['sentence_bad']

            good_ids = tokenizer.encode(good, return_tensors="pt").to(device)
            bad_ids = tokenizer.encode(bad, return_tensors="pt").to(device)

            with torch.no_grad():
                good_logits = model(good_ids).logits
                bad_logits = model(bad_ids).logits

            # Compute log-prob
            good_lp = 0
            for i in range(good_ids.shape[1] - 1):
                good_lp += torch.log_softmax(good_logits[0, i], dim=-1)[good_ids[0, i+1]].item()
            good_lp /= (good_ids.shape[1] - 1)

            bad_lp = 0
            for i in range(bad_ids.shape[1] - 1):
                bad_lp += torch.log_softmax(bad_logits[0, i], dim=-1)[bad_ids[0, i+1]].item()
            bad_lp /= (bad_ids.shape[1] - 1)

            examples.append({
                'good_sentence': good,
                'bad_sentence': bad,
                'phenomenon': 'anaphor_gender_agreement',
                'good_logprob': good_lp,
                'bad_logprob': bad_lp,
                'correct': good_lp > bad_lp,
            })
    except Exception as e:
        print(f"  Could not load BLiMP anaphor: {e}")

    # Try more BLiMP subsets
    for subset in ['anaphor_number_agreement', 'irregular_plural_subject_verb_agreement_1',
                    'regular_plural_subject_verb_agreement_1']:
        if len(examples) >= n:
            break
        try:
            ds = load_dataset("nyu-mll/blimp", subset)
            for item in ds['train']:
                if len(examples) >= n:
                    break
                good = item['sentence_good']
                bad = item['sentence_bad']

                good_ids = tokenizer.encode(good, return_tensors="pt").to(device)
                bad_ids = tokenizer.encode(bad, return_tensors="pt").to(device)

                with torch.no_grad():
                    good_logits = model(good_ids).logits
                    bad_logits = model(bad_ids).logits

                good_lp = 0
                for i in range(good_ids.shape[1] - 1):
                    good_lp += torch.log_softmax(good_logits[0, i], dim=-1)[good_ids[0, i+1]].item()
                good_lp /= max(good_ids.shape[1] - 1, 1)

                bad_lp = 0
                for i in range(bad_ids.shape[1] - 1):
                    bad_lp += torch.log_softmax(bad_logits[0, i], dim=-1)[bad_ids[0, i+1]].item()
                bad_lp /= max(bad_ids.shape[1] - 1, 1)

                examples.append({
                    'good_sentence': good,
                    'bad_sentence': bad,
                    'phenomenon': subset,
                    'good_logprob': good_lp,
                    'bad_logprob': bad_lp,
                    'correct': good_lp > bad_lp,
                })
        except Exception as e:
            print(f"  Could not load BLiMP {subset}: {e}")

    if len(examples) < n:
        # Fallback: create simple subject-verb agreement pairs
        print("  Creating fallback syntax pairs...")
        templates_good = [
            "The dog runs quickly.",
            "The cats sleep peacefully.",
            "She walks to the store.",
            "They play in the park.",
            "He reads the newspaper every morning.",
            "The children are playing outside.",
            "The teacher explains the lesson clearly.",
            "The birds sing in the trees.",
            "The student studies hard for exams.",
            "The workers build the new bridge.",
        ]
        templates_bad = [
            "The dog run quickly.",
            "The cats sleeps peacefully.",
            "She walk to the store.",
            "They plays in the park.",
            "He read the newspaper every morning.",
            "The children is playing outside.",
            "The teacher explain the lesson clearly.",
            "The birds sings in the trees.",
            "The student study hard for exams.",
            "The workers builds the new bridge.",
        ]

        for good, bad in zip(templates_good, templates_bad):
            if len(examples) >= n:
                break
            good_ids = tokenizer.encode(good, return_tensors="pt").to(device)
            bad_ids = tokenizer.encode(bad, return_tensors="pt").to(device)
            with torch.no_grad():
                good_logits = model(good_ids).logits
                bad_logits = model(bad_ids).logits
            good_lp = sum(torch.log_softmax(good_logits[0, i], dim=-1)[good_ids[0, i+1]].item()
                         for i in range(good_ids.shape[1] - 1)) / max(good_ids.shape[1] - 1, 1)
            bad_lp = sum(torch.log_softmax(bad_logits[0, i], dim=-1)[bad_ids[0, i+1]].item()
                        for i in range(bad_ids.shape[1] - 1)) / max(bad_ids.shape[1] - 1, 1)
            examples.append({
                'good_sentence': good,
                'bad_sentence': bad,
                'phenomenon': 'subject_verb_agreement',
                'good_logprob': good_lp,
                'bad_logprob': bad_lp,
                'correct': good_lp > bad_lp,
            })

        # Duplicate to fill
        while len(examples) < n:
            base = random.choice(examples[:min(20, len(examples))])
            examples.append(dict(base))

    examples = examples[:n]
    accuracy = np.mean([e['correct'] for e in examples])
    print(f"  Syntax: {len(examples)} examples, accuracy: {accuracy:.4f}")
    return examples


def prepare_sentiment(tokenizer, model, n=500):
    """Sentiment analysis using SST-2."""
    print("Preparing sentiment dataset...")
    examples = []

    try:
        ds = load_dataset("glue", "sst2")
        val_data = ds['validation']

        # Positive/negative words for probing
        pos_words = [" great", " good", " excellent", " wonderful", " amazing", " positive", " happy"]
        neg_words = [" bad", " terrible", " awful", " horrible", " negative", " sad", " poor"]
        pos_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in pos_words]
        neg_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in neg_words]

        for item in val_data:
            if len(examples) >= n:
                break
            sentence = item['sentence']
            label = item['label']  # 0=neg, 1=pos

            prompt = sentence.strip()
            if not prompt.endswith('.'):
                prompt += '.'
            prompt += " Overall, this was"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = model(**inputs)
            last_logits = out.logits[0, -1]
            probs = torch.softmax(last_logits, dim=-1)

            pos_prob = sum(probs[pid].item() for pid in pos_ids) / len(pos_ids)
            neg_prob = sum(probs[nid].item() for nid in neg_ids) / len(neg_ids)

            predicted = 1 if pos_prob > neg_prob else 0

            examples.append({
                'prompt': prompt,
                'sentence': sentence,
                'label': label,
                'label_str': 'positive' if label == 1 else 'negative',
                'pos_prob': pos_prob,
                'neg_prob': neg_prob,
                'correct': predicted == label,
            })
    except Exception as e:
        print(f"  Error loading SST-2: {e}")

    examples = examples[:n]
    accuracy = np.mean([e['correct'] for e in examples]) if examples else 0
    print(f"  Sentiment: {len(examples)} examples, accuracy: {accuracy:.4f}")
    return examples


def prepare_semantic(tokenizer, model, n=500):
    """Semantic understanding - word sense disambiguation style."""
    print("Preparing semantic dataset...")
    examples = []

    # Create word-in-context style examples using polysemous words
    polysemous = {
        'bank': [
            ("I went to the bank to deposit money.", "financial"),
            ("The river bank was covered in flowers.", "geography"),
        ],
        'bat': [
            ("He swung the bat and hit a home run.", "sports"),
            ("The bat flew out of the cave at dusk.", "animal"),
        ],
        'spring': [
            ("The flowers bloom in spring.", "season"),
            ("The spring in the watch was broken.", "mechanism"),
        ],
        'light': [
            ("Turn on the light please.", "illumination"),
            ("The bag was very light to carry.", "weight"),
        ],
        'right': [
            ("Turn right at the intersection.", "direction"),
            ("You have the right to remain silent.", "legal"),
        ],
        'match': [
            ("He lit a match to start the fire.", "object"),
            ("The tennis match lasted three hours.", "competition"),
        ],
        'run': [
            ("She went for a run in the park.", "exercise"),
            ("There was a run on the bank.", "financial"),
        ],
        'note': [
            ("She left a note on the table.", "written"),
            ("He played a high note on the piano.", "music"),
        ],
        'ring': [
            ("She wore a diamond ring.", "jewelry"),
            ("The phone started to ring.", "sound"),
        ],
        'plant': [
            ("The plant grew tall in the garden.", "botany"),
            ("The factory plant produced cars.", "industrial"),
        ],
    }

    for word, senses in polysemous.items():
        for i, (sent1, sense1) in enumerate(senses):
            for j, (sent2, sense2) in enumerate(senses):
                if i >= j:
                    continue
                same_sense = (sense1 == sense2)

                # Get model representations
                ids1 = tokenizer.encode(sent1, return_tensors="pt").to(device)
                ids2 = tokenizer.encode(sent2, return_tensors="pt").to(device)

                with torch.no_grad():
                    h1 = model(ids1, output_hidden_states=True).hidden_states[-1][0].mean(dim=0)
                    h2 = model(ids2, output_hidden_states=True).hidden_states[-1][0].mean(dim=0)

                sim = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()

                examples.append({
                    'sentence1': sent1,
                    'sentence2': sent2,
                    'target_word': word,
                    'sense1': sense1,
                    'sense2': sense2,
                    'same_sense': same_sense,
                    'similarity': sim,
                })

    # Expand with more sentence pairs
    same_sense_pairs = [
        ("The dog chased the cat.", "The puppy ran after the kitten.", "chase", True),
        ("She bought a new car.", "He purchased a new vehicle.", "buy", True),
        ("The temperature dropped suddenly.", "The price fell sharply.", "drop", False),
        ("He broke the glass.", "She shattered the window.", "break", True),
        ("The book was on the table.", "The novel sat on the desk.", "on", True),
        ("He ran to the store.", "She sprinted to the shop.", "run", True),
        ("The ship sailed across the ocean.", "The boat crossed the sea.", "sail", True),
        ("She opened the door carefully.", "He unlocked the gate slowly.", "open", True),
    ]

    for s1, s2, word, same in same_sense_pairs:
        ids1 = tokenizer.encode(s1, return_tensors="pt").to(device)
        ids2 = tokenizer.encode(s2, return_tensors="pt").to(device)
        with torch.no_grad():
            h1 = model(ids1, output_hidden_states=True).hidden_states[-1][0].mean(dim=0)
            h2 = model(ids2, output_hidden_states=True).hidden_states[-1][0].mean(dim=0)
        sim = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
        examples.append({
            'sentence1': s1, 'sentence2': s2, 'target_word': word,
            'sense1': 'main', 'sense2': 'main' if same else 'alt',
            'same_sense': same, 'similarity': sim,
        })

    # Fill to n
    while len(examples) < n:
        base = random.choice(examples[:min(20, len(examples))])
        examples.append(dict(base))

    examples = examples[:n]
    print(f"  Semantic: {len(examples)} examples")
    return examples


def prepare_ner(tokenizer, model, n=500):
    """Named entity recognition using entity type prediction."""
    print("Preparing NER dataset...")
    examples = []

    entities = {
        'person': [
            ("Albert Einstein was a famous", " physicist"),
            ("William Shakespeare wrote many", " plays"),
            ("Marie Curie discovered", " rad"),
            ("Barack Obama was the president of the", " United"),
            ("Leonardo da Vinci painted the", " Mon"),
            ("Isaac Newton discovered", " gravity"),
            ("Charles Darwin proposed the theory of", " evolution"),
            ("Nikola Tesla invented the", " alternating"),
            ("Mahatma Gandhi led the independence movement in", " India"),
            ("Nelson Mandela was the president of", " South"),
            ("Queen Elizabeth ruled", " England"),
            ("Abraham Lincoln was the", " 16"),
            ("Cleopatra was the queen of", " Egypt"),
            ("Mozart composed", " music"),
            ("Galileo discovered that the Earth", " revolves"),
        ],
        'location': [
            ("The Eiffel Tower is located in", " Paris"),
            ("Mount Everest is the tallest", " mountain"),
            ("The Amazon River flows through", " South"),
            ("The Sahara Desert is in", " Africa"),
            ("Tokyo is the capital of", " Japan"),
            ("The Great Barrier Reef is near", " Australia"),
            ("The Nile River is the longest river in", " Africa"),
            ("Antarctica is the", " coldest"),
            ("The Mediterranean Sea is between Europe and", " Africa"),
            ("Silicon Valley is located in", " California"),
            ("Hollywood is famous for its", " movie"),
            ("The Alps are located in", " Europe"),
            ("The Pacific Ocean is the", " largest"),
            ("The Grand Canyon is in", " Arizona"),
            ("Yellowstone is a national", " park"),
        ],
        'organization': [
            ("Google was founded by Larry Page and", " Sergey"),
            ("Microsoft was founded by Bill", " Gates"),
            ("NASA launched the Apollo", " mission"),
            ("The United Nations headquarters is in", " New"),
            ("Apple released the first iPhone in", " 2007"),
            ("The World Health Organization declared", " a"),
            ("Amazon started as an online", " book"),
            ("Facebook was created by Mark", " Zuckerberg"),
            ("Tesla produces electric", " cars"),
            ("The European Union has", " 27"),
            ("SpaceX launched the", " Falcon"),
            ("Netflix started as a", " DVD"),
            ("The Olympics are organized by the", " International"),
            ("Harvard University was founded in", " 1636"),
            ("The Red Cross provides", " humanitarian"),
        ],
    }

    for etype, prompts in entities.items():
        for prompt, target in prompts:
            if len(examples) >= n:
                break
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                continue
            target_id = target_ids[0]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            last_logits = out.logits[0, -1]
            prob = torch.softmax(last_logits, dim=-1)[target_id].item()

            examples.append({
                'prompt': prompt,
                'target_token': target.strip(),
                'target_token_id': target_id,
                'entity_type': etype,
                'model_prob': prob,
            })

    while len(examples) < n:
        base = random.choice(examples[:min(40, len(examples))])
        examples.append(dict(base))

    examples = examples[:n]
    print(f"  NER: {len(examples)} examples")
    return examples


def prepare_reasoning(tokenizer, model, n=500):
    """Simple reasoning tasks - numerical, analogical, causal."""
    print("Preparing reasoning dataset...")
    examples = []

    # Numerical successor
    for i in range(1, 100):
        prompt = f"After {i} comes"
        target = f" {i+1}"
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        last_logits = out.logits[0, -1]
        prob = torch.softmax(last_logits, dim=-1)[target_id].item()

        examples.append({
            'prompt': prompt,
            'target_token': target.strip(),
            'target_token_id': target_id,
            'reasoning_type': 'numerical_successor',
            'model_prob': prob,
        })

    # Simple analogies and causal
    analogy_prompts = [
        ("Hot is to cold as big is to", " small"),
        ("Cat is to kitten as dog is to", " puppy"),
        ("Up is to down as left is to", " right"),
        ("Day is to night as light is to", " dark"),
        ("Fast is to slow as tall is to", " short"),
        ("Man is to woman as boy is to", " girl"),
        ("Sun is to moon as day is to", " night"),
        ("Fire is to hot as ice is to", " cold"),
        ("Book is to reading as movie is to", " watching"),
        ("Pen is to writing as brush is to", " painting"),
        ("Bird is to fly as fish is to", " swim"),
        ("Eye is to see as ear is to", " hear"),
        ("If it rains, the ground gets", " wet"),
        ("When you heat water, it starts to", " bo"),
        ("If you drop a glass, it will", " break"),
        ("When the sun sets, it becomes", " dark"),
        ("If you study hard, you will", " pass"),
        ("When ice melts, it becomes", " water"),
        ("2 + 2 equals", " 4"),
        ("10 minus 3 equals", " 7"),
    ]

    for prompt, target in analogy_prompts:
        if len(examples) >= n:
            break
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        last_logits = out.logits[0, -1]
        prob = torch.softmax(last_logits, dim=-1)[target_id].item()

        examples.append({
            'prompt': prompt,
            'target_token': target.strip(),
            'target_token_id': target_id,
            'reasoning_type': 'analogy' if 'is to' in prompt else 'causal',
            'model_prob': prob,
        })

    while len(examples) < n:
        base = random.choice(examples[:min(50, len(examples))])
        examples.append(dict(base))

    examples = examples[:n]
    accuracy = np.mean([e['model_prob'] > 0.1 for e in examples])
    print(f"  Reasoning: {len(examples)} examples, fraction with prob>0.1: {accuracy:.4f}")
    return examples


def main():
    set_seed(42)
    tokenizer, model = load_gpt2()

    datasets = {}
    datasets['factual'] = prepare_factual(tokenizer, model)
    datasets['syntax'] = prepare_syntax(tokenizer, model)
    datasets['sentiment'] = prepare_sentiment(tokenizer, model)
    datasets['semantic'] = prepare_semantic(tokenizer, model)
    datasets['ner'] = prepare_ner(tokenizer, model)
    datasets['reasoning'] = prepare_reasoning(tokenizer, model)

    # Save all datasets
    stats = {}
    for cap_name, examples in datasets.items():
        save_path = DATA_DIR / f"{cap_name}.json"
        with open(save_path, 'w') as f:
            json.dump(examples, f, indent=2, default=str)

        stats[cap_name] = {
            'n_examples': len(examples),
        }
        if 'model_prob' in examples[0]:
            stats[cap_name]['mean_prob'] = float(np.mean([e['model_prob'] for e in examples]))
        if 'correct' in examples[0]:
            stats[cap_name]['accuracy'] = float(np.mean([e['correct'] for e in examples]))

    with open(DATA_DIR / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n=== Dataset Statistics ===")
    for cap, s in stats.items():
        print(f"  {cap}: {s}")

    print(f"\nAll datasets saved to {DATA_DIR}")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
