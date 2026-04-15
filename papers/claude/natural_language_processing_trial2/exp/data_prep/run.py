"""Step 1: Download and preprocess evaluation datasets."""
import os
import sys
import json
import random
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, save_json, load_json, DATA_DIR, BASE_DIR

set_seed(42)


def prepare_truthfulqa():
    """Download and subset TruthfulQA."""
    from datasets import load_dataset
    print("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    indices = random.sample(range(len(ds)), min(200, len(ds)))
    data = []
    for i in indices:
        item = ds[i]
        data.append({
            "id": f"tqa_{i}",
            "source": "truthfulqa",
            "prompt": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "category": item.get("category", "general"),
        })
    save_json(data, os.path.join(DATA_DIR, "truthfulqa_subset.json"))
    print(f"  TruthfulQA: {len(data)} questions saved")
    return data


def prepare_factscore():
    """Create FActScore-style biography prompts from a curated entity list."""
    print("Preparing FActScore biography entities...")
    # Use a curated list of entities with varying popularity
    # Mix of well-known and less common figures to test hallucination rates
    entities_popular = [
        "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare",
        "Nikola Tesla", "Cleopatra", "Isaac Newton", "Charles Darwin",
        "Mahatma Gandhi", "Nelson Mandela", "Martin Luther King Jr.", "Abraham Lincoln",
        "Napoleon Bonaparte", "Queen Elizabeth II", "Winston Churchill",
        "Alexander the Great", "Julius Caesar", "Aristotle", "Galileo Galilei",
        "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "Pablo Picasso",
        "Frida Kahlo", "Ada Lovelace", "Alan Turing",
    ]
    entities_medium = [
        "Rosalind Franklin", "Hedy Lamarr", "Emmy Noether", "Lise Meitner",
        "Rachel Carson", "Grace Hopper", "Katherine Johnson", "Dorothy Hodgkin",
        "Barbara McClintock", "Chien-Shiung Wu", "Mary Anning", "Hypatia",
        "Srinivasa Ramanujan", "Niels Bohr", "Max Planck", "Erwin Schrodinger",
        "Werner Heisenberg", "Paul Dirac", "Richard Feynman", "Enrico Fermi",
        "John von Neumann", "Claude Shannon", "Norbert Wiener", "Andrey Kolmogorov",
        "Henri Poincare",
    ]
    entities_rare = [
        "Maryam Mirzakhani", "Terence Tao", "Grigori Perelman", "Karen Uhlenbeck",
        "Ingrid Daubechies", "Cathleen Synge Morawetz", "Emmy Noether",
        "Sofia Kovalevskaya", "Shakuntala Devi", "Mary Cartwright",
        "Olga Ladyzhenskaya", "Julia Robinson", "Karen Keskulla Uhlenbeck",
        "Sun-Yung Alice Chang", "Fan Chung", "Nalini Joshi",
        "Dusa McDuff", "Lai-Sang Young", "Amalie Emmy Noether",
        "Ruth Lyttle Satter", "Vera Rubin", "Jocelyn Bell Burnell",
        "Cecilia Payne-Gaposchkin", "Annie Jump Cannon", "Williamina Fleming",
        "Henrietta Swan Leavitt", "Maria Mitchell", "Caroline Herschel",
        "Mary Somerville", "Florence Nightingale", "Nettie Stevens",
        "Alice Ball", "Chien-Shiung Wu", "Maria Goeppert Mayer",
        "Dorothy Crowfoot Hodgkin", "Gertrude Elion", "Tu Youyou",
        "Jennifer Doudna", "Emmanuelle Charpentier", "Donna Strickland",
        "Andrea Ghez", "Katalin Kariko", "Frances Arnold",
        "Elizabeth Blackburn", "Carol Greider", "May-Britt Moser",
        "Christiane Nusslein-Volhard", "Linda Buck", "Françoise Barré-Sinoussi",
        "Ada Yonath", "Youyou Tu",
    ]
    # Deduplicate
    all_entities = list(dict.fromkeys(entities_popular + entities_medium + entities_rare))
    random.shuffle(all_entities)
    selected = all_entities[:100]

    data = []
    for i, entity in enumerate(selected):
        pop = "popular" if entity in entities_popular else ("medium" if entity in entities_medium else "rare")
        data.append({
            "id": f"fs_{i}",
            "source": "factscore",
            "prompt": f"Tell me a bio of {entity}",
            "entity": entity,
            "popularity": pop,
            "category": "biography",
        })
    save_json(data, os.path.join(DATA_DIR, "factscore_subset.json"))
    print(f"  FActScore: {len(data)} entities saved")
    return data


def prepare_longfact():
    """Create LongFact-style prompts covering diverse topics."""
    print("Preparing LongFact prompts...")
    prompts = [
        # Science
        "Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
        "Describe the structure and function of DNA, including its role in protein synthesis.",
        "What are black holes? Explain their formation, properties, and how they are detected.",
        "Explain the theory of plate tectonics, including evidence supporting it.",
        "Describe the life cycle of stars, from nebula to their final stages.",
        # History
        "Describe the causes and major events of World War I.",
        "Explain the history and significance of the Silk Road.",
        "What were the main causes and consequences of the French Revolution?",
        "Describe the history of the Roman Empire from its founding to its fall.",
        "Explain the significance of the Industrial Revolution and its impact on society.",
        # Geography
        "Describe the geography and ecosystems of the Amazon Rainforest.",
        "Explain the formation and features of the Grand Canyon.",
        "Describe the major ocean currents and their effects on global climate.",
        "What are the main geographical features of Antarctica?",
        "Explain the geography and geological features of the Himalayas.",
        # Technology
        "Explain how the internet works, from basic protocols to data transmission.",
        "Describe the history and evolution of artificial intelligence.",
        "How do nuclear power plants generate electricity? Explain the process in detail.",
        "Explain the development and impact of the printing press.",
        "Describe how GPS technology works, including the satellite system.",
        # Culture
        "Describe the history and cultural significance of the Olympic Games.",
        "Explain the origins and evolution of jazz music.",
        "What is the history of the English language and how has it evolved?",
        "Describe the major schools of philosophy in ancient Greece.",
        "Explain the history and significance of the Renaissance period.",
        # Medicine
        "Describe the human immune system and how vaccines work.",
        "Explain the discovery and impact of antibiotics on modern medicine.",
        "What is CRISPR and how does it work for gene editing?",
        "Describe the history and development of organ transplantation.",
        "Explain how the human brain processes and stores memories.",
        # Mathematics
        "Explain the significance of prime numbers in mathematics and cryptography.",
        "Describe the development of calculus and its key contributors.",
        "What is the Riemann Hypothesis and why is it important?",
        "Explain the concept of infinity in mathematics and its different types.",
        "Describe the history and applications of probability theory.",
        # Economics/Politics
        "Explain the causes and effects of the 2008 global financial crisis.",
        "Describe the history and structure of the United Nations.",
        "What is the history of democracy and how has it evolved?",
        "Explain the economic principles behind supply and demand.",
        "Describe the history and impact of globalization on world economies.",
        # Environment
        "Explain the science behind climate change and its major effects.",
        "Describe the causes and consequences of deforestation worldwide.",
        "What are renewable energy sources and how do they work?",
        "Explain the water cycle and its importance to Earth's ecosystems.",
        "Describe the major extinction events in Earth's history.",
        # Space
        "Explain the history of space exploration, from early rockets to modern missions.",
        "Describe the solar system, including all planets and their characteristics.",
        "What is dark matter and dark energy? Explain current scientific understanding.",
        "Describe the Big Bang theory and evidence supporting it.",
        "Explain how telescopes work and the major astronomical discoveries they enabled.",
    ]
    data = []
    topics = ["science", "history", "geography", "technology", "culture",
              "medicine", "mathematics", "economics", "environment", "space"]
    for i, prompt in enumerate(prompts):
        data.append({
            "id": f"lf_{i}",
            "source": "longfact",
            "prompt": prompt,
            "category": topics[i // 5],
        })
    save_json(data, os.path.join(DATA_DIR, "longfact_subset.json"))
    print(f"  LongFact: {len(data)} prompts saved")
    return data


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    tqa = prepare_truthfulqa()
    fs = prepare_factscore()
    lf = prepare_longfact()

    stats = {
        "truthfulqa": {"n_prompts": len(tqa), "type": "QA"},
        "factscore": {"n_prompts": len(fs), "type": "biography"},
        "longfact": {"n_prompts": len(lf), "type": "long-form"},
    }
    save_json(stats, os.path.join(DATA_DIR, "dataset_stats.json"))
    print("Dataset preparation complete.")
