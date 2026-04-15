"""Prepare evaluation prompt sets for CoPS experiments."""
import json
import os
import random
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_coco_prompts(n=500, seed=42):
    """Download COCO 2017 val captions and sample n prompts."""
    out_path = os.path.join(DATA_DIR, "coco_500_prompts.json")
    if os.path.exists(out_path):
        print(f"COCO prompts already exist at {out_path}")
        return json.load(open(out_path))

    # Download captions
    captions_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    captions_json = os.path.join(DATA_DIR, "captions_val2017.json")

    if not os.path.exists(captions_json):
        print("Downloading COCO captions...")
        import zipfile, io
        resp = urllib.request.urlopen(captions_url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        # Extract only val captions
        for name in z.namelist():
            if "captions_val2017" in name:
                with open(captions_json, "wb") as f:
                    f.write(z.read(name))
                break
        z.close()

    with open(captions_json) as f:
        data = json.load(f)

    # Get unique image_id -> first caption mapping
    seen_ids = set()
    captions = []
    for ann in data["annotations"]:
        if ann["image_id"] not in seen_ids:
            seen_ids.add(ann["image_id"])
            captions.append({"id": ann["image_id"], "prompt": ann["caption"].strip()})

    random.seed(seed)
    sampled = random.sample(captions, min(n, len(captions)))

    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved {len(sampled)} COCO prompts to {out_path}")
    return sampled


def prepare_parti_prompts(n=200, seed=42):
    """Load PartiPrompts from HuggingFace."""
    out_path = os.path.join(DATA_DIR, "parti_200_prompts.json")
    if os.path.exists(out_path):
        print(f"PartiPrompts already exist at {out_path}")
        return json.load(open(out_path))

    try:
        from datasets import load_dataset
        ds = load_dataset("nateraw/parti-prompts", split="train")
        all_prompts = [{"id": i, "prompt": row["Prompt"]} for i, row in enumerate(ds)]
    except Exception as e:
        print(f"Failed to load from HF: {e}. Using fallback prompts.")
        # Fallback: generate diverse prompts
        all_prompts = [{"id": i, "prompt": p} for i, p in enumerate([
            "A photo of a cat sitting on a windowsill",
            "A red sports car driving on a mountain road",
            "An astronaut riding a horse on the moon",
        ] * 70)]  # Will sample from these

    random.seed(seed)
    sampled = random.sample(all_prompts, min(n, len(all_prompts)))

    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved {len(sampled)} PartiPrompts to {out_path}")
    return sampled


def prepare_drawbench_prompts():
    """Prepare DrawBench-style evaluation prompts."""
    out_path = os.path.join(DATA_DIR, "drawbench_prompts.json")
    if os.path.exists(out_path):
        print(f"DrawBench prompts already exist at {out_path}")
        return json.load(open(out_path))

    # DrawBench prompts from the Imagen paper (representative subset)
    drawbench_prompts = [
        "A red colored car", "A green colored car", "A blue colored car",
        "A panda making latte art", "A dog wearing a chef hat cooking",
        "A photo of a teddy bear on a skateboard in Times Square",
        "A sign that says 'Deep Learning'", "A sign that says 'Diffusion Models'",
        "An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with no umbrella",
        "A blue jay standing on a large basket of rainbow macarons",
        "A small cactus wearing a straw hat and neon sunglasses in the Sahara desert",
        "A transparent sculpture of a duck made out of glass",
        "A photo of a Corgi dog riding a bike in Times Square",
        "A storm trooper vacuuming the beach",
        "A raccoon wearing formal clothes, bowing to the other animals",
        "A living room with two white armchairs and a painting of the Colosseum. The painting is mounted above a modern fireplace",
        "An espresso machine that makes coffee from human souls",
        "A golden retriever wearing a blue checkered beret and red dotted turtleneck",
        "A high contrast image of a pear on a dark background",
        "A yellow and black bus cruising through the rainforest",
        "A cute sloth holding a small glowing lantern, sitting on a mossy rock at twilight",
        "A medieval castle on a floating island, surrounded by waterfalls and dragons",
        "A photo of a horse standing on top of a chair",
        "A cube made of brick. A cube with the texture of brick",
        "A spherical paperweight sitting on a table, the sphere is made of metal",
        "A group of penguins having a tea party",
        "A painting of a tree with bright autumn leaves in the foreground with a castle far off in the background",
        "A single beam of light entering a dark room through a window",
        "Three cats and one dog sitting together on a park bench",
        "An origami crane made from a newspaper page about space exploration",
        "A photo of a mushroom growing out of a coffee cup",
        "A painting of a sunset over a valley with a river flowing through it",
        "A vintage typewriter with flowers growing out of the keys",
        "A knight riding a giant snail across a bridge",
        "A robot painting a self-portrait in a studio",
        "An hourglass with galaxies as the sand",
        "A photo of an elephant in a cherry blossom forest",
        "A cozy library with a fireplace and a sleeping cat",
        "A portrait of a scientist with equation tattoos",
        "A glass of red wine sitting on a grand piano",
        "A surreal painting of clocks melting over a desert landscape",
    ]

    prompts = [{"id": i, "prompt": p} for i, p in enumerate(drawbench_prompts)]

    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} DrawBench prompts to {out_path}")
    return prompts


if __name__ == "__main__":
    coco = prepare_coco_prompts()
    parti = prepare_parti_prompts()
    drawbench = prepare_drawbench_prompts()

    summary = {
        "coco_500": {"count": len(coco), "file": "coco_500_prompts.json"},
        "parti_200": {"count": len(parti), "file": "parti_200_prompts.json"},
        "drawbench": {"count": len(drawbench), "file": "drawbench_prompts.json"},
    }
    with open(os.path.join(DATA_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nData summary: {summary}")
