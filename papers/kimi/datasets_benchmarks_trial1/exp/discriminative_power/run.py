"""
Discriminative Power Experiment (RQ2)
Evaluate models across difficulty levels to demonstrate >30% accuracy gap.
"""

import sys
sys.path.insert(0, '../shared')

import json
import re
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from utils import set_seed, save_json, calculate_accuracy, compute_statistics
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import cairosvg
import io


class VLMEvaluator:
    """Evaluates Vision-Language Models on CompViz tasks."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the VLM model."""
        print(f"Loading {self.model_name}...")
        
        if "qwen" in self.model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        elif "internvl" in self.model_name.lower():
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print(f"Model loaded on {self.device}")
    
    def svg_to_pil(self, svg_string: str) -> Image.Image:
        """Convert SVG to PIL Image."""
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB')
    
    def predict(self, image: Image.Image, question: str, temperature: float = 0.0) -> str:
        """Generate prediction for an image-question pair."""
        prompt = f"Answer the following question with a single word (yes/no) or number. Question: {question}"
        
        # Prepare inputs
        if "qwen" in self.model_name.lower():
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self.device)
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                num_return_sequences=1
            )
        
        # Decode
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract answer
        answer = self.extract_answer(generated_text)
        return answer
    
    def extract_answer(self, text: str) -> str:
        """Extract yes/no/number from model output."""
        text_lower = text.lower().strip()
        
        # Direct match
        if text_lower in ["yes", "no", "true", "false"]:
            return text_lower.replace("true", "yes").replace("false", "no")
        
        # Check first word
        first_word = text_lower.split()[0] if text_lower else ""
        if first_word in ["yes", "no"]:
            return first_word
        
        # Extract number
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[0]
        
        # Check for answer pattern
        if "answer:" in text_lower:
            after = text_lower.split("answer:")[1].strip()
            if after.startswith("yes"):
                return "yes"
            elif after.startswith("no"):
                return "no"
        
        return "unknown"


def evaluate_dataset(
    model: VLMEvaluator,
    dataset_path: str,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate model on a dataset."""
    set_seed(seed)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    instances = dataset.get("instances", [])
    if max_samples:
        instances = instances[:max_samples]
    
    predictions = []
    ground_truth = []
    
    print(f"Evaluating {len(instances)} samples from {dataset_path}...")
    
    for i, inst in enumerate(instances):
        # Convert SVG to image
        from scene_generator import SceneGenerator, scene_to_dict
        scene_dict = inst["scene"]
        
        # Recreate SVG from scene dict
        scene_gen = SceneGenerator(inst.get("seed", 42))
        shapes = []
        for s in scene_dict["shapes"]:
            from scene_generator import Shape
            shape = Shape(
                id=s["id"],
                shape_type=s["shape_type"],
                color=s["color"],
                size=s["size"],
                x=s["x"],
                y=s["y"],
                rotation=s.get("rotation", 0),
                texture=s.get("texture", "solid"),
                opacity=s.get("opacity", 1.0)
            )
            shapes.append(shape)
        
        from scene_generator import Relation
        relations = [Relation(r["from"], r["to"], r["type"]) for r in scene_dict["relations"]]
        
        # Create SVG
        svg_string = scene_gen._render_svg(shapes, (400, 400))
        image = model.svg_to_pil(svg_string)
        
        # Predict
        question = inst["query"]["text"]
        pred = model.predict(image, question)
        
        predictions.append(pred)
        ground_truth.append(inst["answer"].lower())
        
        if (i + 1) % 50 == 0:
            acc = calculate_accuracy(predictions, ground_truth)
            print(f"  Processed {i+1}/{len(instances)} - Accuracy: {acc:.3f}")
    
    accuracy = calculate_accuracy(predictions, ground_truth)
    
    return {
        "accuracy": accuracy,
        "num_samples": len(instances),
        "predictions": predictions,
        "ground_truth": ground_truth
    }


def run_discriminative_experiment(
    model_names: List[str],
    data_dir: str = "../../data/scenes",
    output_dir: str = "../../results",
    max_samples_per_level: int = 100
):
    """Run discriminative power experiment across difficulty levels."""
    
    print("=" * 60)
    print("DISCRIMINATIVE POWER EXPERIMENT (RQ2)")
    print("=" * 60)
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        
        try:
            model = VLMEvaluator(model_name)
            
            model_results = {
                "model": model_name,
                "levels": {}
            }
            
            for level in range(1, 5):
                dataset_path = f"{data_dir}/level{level}_existential_n500_s{100+level-1}.json"
                
                if not Path(dataset_path).exists():
                    print(f"  Dataset not found: {dataset_path}")
                    continue
                
                result = evaluate_dataset(model, dataset_path, max_samples=max_samples_per_level)
                model_results["levels"][f"level_{level}"] = result
                print(f"  Level {level}: {result['accuracy']:.3f}")
            
            # Calculate gap
            level_accs = [model_results["levels"][f"level_{l}"]["accuracy"] for l in range(1, 5)]
            if len(level_accs) >= 2:
                gap = level_accs[0] - level_accs[-1]
                model_results["accuracy_gap_l1_l4"] = gap
                model_results["hypothesis_met"] = gap > 0.30
                print(f"  L1-L4 Gap: {gap:.3f} - {'PASS' if gap > 0.30 else 'FAIL'}")
            
            all_results[model_name] = model_results
            
            # Clean up GPU memory
            del model.model
            del model.processor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save results
    save_json(all_results, f"{output_dir}/discriminative_power.json")
    
    # Create visualization
    create_accuracy_plot(all_results, f"{output_dir}/../figures/accuracy_by_difficulty.png")
    
    return all_results


def create_accuracy_plot(results: Dict, output_path: str):
    """Create accuracy by difficulty plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    levels = [1, 2, 3, 4]
    
    for model_name, model_results in results.items():
        if "error" in model_results:
            continue
        
        accuracies = []
        for level in levels:
            level_key = f"level_{level}"
            if level_key in model_results.get("levels", {}):
                accuracies.append(model_results["levels"][level_key]["accuracy"])
            else:
                accuracies.append(0)
        
        ax.plot(levels, accuracies, marker='o', label=model_name.split('/')[-1])
    
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Across Difficulty Levels')
    ax.set_xticks(levels)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Accuracy plot saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', default=["Qwen/Qwen2-VL-7B-Instruct"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="../../results")
    args = parser.parse_args()
    
    run_discriminative_experiment(args.models, max_samples_per_level=args.max_samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
