"""
Debug model output format.
"""

import sys
sys.path.insert(0, '.')

import json
import torch
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

# Load a sample dataset
with open("../../data/scenes/level1_existential_n500_s100.json") as f:
    dataset = json.load(f)

inst = dataset["instances"][0]

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

image = scene_to_image(inst["scene"])
question = inst["query"]["text"]
gt = inst["answer"]

print(f"\nQuestion: {question}")
print(f"Ground truth: {gt}")
print(f"\nScene has {len(inst['scene']['shapes'])} shapes:")
for s in inst['scene']['shapes'][:5]:
    print(f"  - {s['color']} {s['shape_type']}")

# Create prompt
prompt = f"Look at the image and answer with just 'yes' or 'no'. {question}"

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nPrompt template:\n{text[:200]}...")

inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

# Print full output
print(f"\nFull output tokens: {outputs[0]}")

# Decode different parts
full_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\nFull decoded (with special tokens):\n{full_text}")

generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"\nGenerated text (skip_special_tokens=True):\n{generated_text}")

# Try to find where the assistant response starts
if "assistant" in generated_text.lower():
    parts = generated_text.split("assistant")
    if len(parts) > 1:
        print(f"\nAfter 'assistant':\n{parts[1].strip()}")
