"""
Debug batch evaluation.
"""

import sys
sys.path.insert(0, '.')

import json
import torch
import re
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import io
import cairosvg

from scene_generator import SceneGenerator, Shape, Relation

# Load model
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("Model loaded!")

# Load dataset
with open("../../data/scenes/level1_existential_n500_s100.json") as f:
    dataset = json.load(f)

def scene_to_image(scene_dict):
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

def extract_answer(text):
    """Extract yes/no from model output."""
    # Get the last part after 'assistant'
    if "assistant" in text.lower():
        parts = text.split("assistant")
        text = parts[-1].strip()
    
    text = text.lower().strip()
    
    # Look for yes/no
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    
    return "unknown"

# Test on 20 samples
instances = dataset["instances"][:20]
correct = 0

print("\nTesting on 20 samples:")
print("=" * 60)

for i, inst in enumerate(instances):
    image = scene_to_image(inst["scene"])
    question = inst["query"]["text"]
    gt = inst["answer"].lower()
    
    prompt = f"Answer with just 'yes' or 'no': {question}"
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    pred = extract_answer(generated)
    
    match = pred == gt
    if match:
        correct += 1
    
    print(f"{i+1:2d}. Q: {question[:40]:40s} | Pred: {pred:6s} | GT: {gt:3s} | {'✓' if match else '✗'}")

print("=" * 60)
print(f"Accuracy: {correct}/{len(instances)} = {correct/len(instances):.2%}")
