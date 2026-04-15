"""
Shared VLM Evaluation Utilities
"""

import sys
sys.path.insert(0, '.')

import json
import re
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

from scene_generator import SceneGenerator, Shape, Relation
from utils import set_seed, calculate_accuracy


class VLMEvaluator:
    """Unified VLM evaluator for CompViz experiments."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load model and processor."""
        print(f"Loading {self.model_name}...")
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            if "qwen" in self.model_name.lower():
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
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
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def scene_to_image(self, scene_dict: Dict) -> Image.Image:
        """Convert scene dict to PIL Image."""
        import cairosvg
        
        # Recreate SVG from scene dict
        scene_gen = SceneGenerator(42)  # Seed doesn't matter for rendering
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
        
        # Convert to PNG
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB')
    
    def predict(self, image: Image.Image, question: str, temperature: float = 0.0) -> str:
        """Generate prediction."""
        prompt = f"Answer with a single word (yes/no) or number: {question}"
        
        try:
            # Prepare inputs based on model type
            if "qwen" in self.model_name.lower():
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]}
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
                )
            
            # Decode
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return self.extract_answer(generated_text)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error"
    
    def extract_answer(self, text: str) -> str:
        """Extract yes/no/number from text."""
        text_lower = text.lower().strip()
        
        # Direct match
        if text_lower in ["yes", "no", "true", "false"]:
            return text_lower.replace("true", "yes").replace("false", "no")
        
        # First word
        words = text_lower.split()
        if words and words[0] in ["yes", "no"]:
            return words[0]
        
        # Extract number
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[0]
        
        return "unknown"
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        max_samples: Optional[int] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        set_seed(seed)
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        instances = dataset.get("instances", [])
        if max_samples:
            instances = instances[:max_samples]
        
        predictions = []
        ground_truth = []
        
        print(f"Evaluating {len(instances)} samples...")
        
        for i, inst in enumerate(instances):
            image = self.scene_to_image(inst["scene"])
            pred = self.predict(image, inst["query"]["text"])
            
            predictions.append(pred)
            ground_truth.append(inst["answer"].lower())
            
            if (i + 1) % 20 == 0:
                acc = calculate_accuracy(predictions, ground_truth)
                print(f"  {i+1}/{len(instances)} - Accuracy: {acc:.3f}")
        
        accuracy = calculate_accuracy(predictions, ground_truth)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "num_samples": len(instances)
        }
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()


def evaluate_all_datasets(
    model_name: str,
    data_dir: str,
    output_path: str,
    max_samples: int = 100
):
    """Evaluate model on all datasets and save results."""
    
    evaluator = VLMEvaluator(model_name)
    
    # Find all dataset files
    data_path = Path(data_dir)
    datasets = list(data_path.glob("*.json"))
    
    all_results = {
        "model": model_name,
        "datasets": {}
    }
    
    for dataset_file in sorted(datasets):
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_file.name}")
        print(f"{'='*60}")
        
        try:
            result = evaluator.evaluate_dataset(str(dataset_file), max_samples=max_samples)
            all_results["datasets"][dataset_file.stem] = result
            print(f"Final accuracy: {result['accuracy']:.3f}")
        except Exception as e:
            print(f"Error: {e}")
            all_results["datasets"][dataset_file.stem] = {"error": str(e)}
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    evaluator.cleanup()
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    
    evaluate_all_datasets(args.model, args.data_dir, args.output, args.max_samples)
