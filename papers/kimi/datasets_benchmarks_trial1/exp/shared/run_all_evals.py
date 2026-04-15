"""
Run comprehensive VLM evaluations on all datasets.
"""

import sys
sys.path.insert(0, '.')

import json
import torch
import re
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import io
import cairosvg

from scene_generator import SceneGenerator, Shape, Relation
from utils import set_seed, calculate_accuracy


class QwenEvaluator:
    """Qwen2-VL evaluator with correct answer extraction."""
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        print(f"Loading {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded!")
    
    def scene_to_image(self, scene_dict):
        scene_gen = SceneGenerator(42)
        shapes = []
        for s in scene_dict["shapes"]:
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
        
        relations = [Relation(r["from"], r["to"], r["type"]) for r in scene_dict["relations"]]
        svg_string = scene_gen._render_svg(shapes, (400, 400))
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB')
    
    def extract_answer(self, text):
        """Extract yes/no from model output."""
        if "assistant" in text.lower():
            parts = text.split("assistant")
            text = parts[-1].strip()
        
        text = text.lower().strip()
        
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        
        if "yes" in text.split():
            return "yes"
        if "no" in text.split():
            return "no"
        
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[0]
        
        return "unknown"
    
    def predict(self, image, question):
        prompt = f"Answer with just 'yes' or 'no': {question}"
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
        
        generated = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return self.extract_answer(generated)
    
    def evaluate_dataset(self, dataset_path: str, max_samples: int = None) -> Dict:
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        instances = dataset["instances"]
        if max_samples:
            instances = instances[:max_samples]
        
        predictions = []
        ground_truth = []
        
        print(f"Evaluating {len(instances)} samples from {Path(dataset_path).name}...")
        
        for i, inst in enumerate(instances):
            image = self.scene_to_image(inst["scene"])
            pred = self.predict(image, inst["query"]["text"])
            gt = inst["answer"].lower()
            
            predictions.append(pred)
            ground_truth.append(gt)
            
            if (i + 1) % 20 == 0:
                acc = calculate_accuracy(predictions, ground_truth)
                print(f"  {i+1}/{len(instances)} - Accuracy: {acc:.3f}")
        
        accuracy = calculate_accuracy(predictions, ground_truth)
        print(f"  Final accuracy: {accuracy:.3f}")
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "num_samples": len(instances)
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data/scenes")
    parser.add_argument("--output", default="../../results/vlm_evaluation.json")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    
    evaluator = QwenEvaluator()
    
    data_dir = Path(args.data_dir)
    datasets = sorted(data_dir.glob("*.json"))
    
    results = {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "datasets": {}
    }
    
    for dataset_file in datasets:
        print(f"\n{'='*60}")
        try:
            result = evaluator.evaluate_dataset(str(dataset_file), max_samples=args.max_samples)
            results["datasets"][dataset_file.stem] = result
        except Exception as e:
            print(f"Error: {e}")
            results["datasets"][dataset_file.stem] = {"error": str(e)}
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nSummary:")
    for name, res in sorted(results["datasets"].items()):
        if "accuracy" in res:
            print(f"  {name}: {res['accuracy']:.3f}")


if __name__ == "__main__":
    main()
