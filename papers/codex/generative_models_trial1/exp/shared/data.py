from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from exp.shared.parser import PromptParse, Slot, build_parse
from exp.shared.utils import ensure_dir, write_json, write_jsonl


def _load_geneval(repo_root: Path) -> list[dict]:
    path = repo_root / "external" / "geneval" / "prompts" / "evaluation_metadata.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            record = json.loads(line)
            tag = record.get("tag", "")
            if tag not in {"counting", "colors", "position"}:
                continue
            parse = _build_geneval_parse(record)
            if not parse.supported:
                continue
            rows.append(
                {
                    "dataset": "geneval",
                    "source_category": tag,
                    "prompt_id": f"geneval_{idx:04d}",
                    "prompt": record["prompt"],
                    "metadata": record,
                    "parse": parse,
                }
            )
    return rows


def _build_geneval_parse(record: dict) -> PromptParse:
    groups = []
    count_atoms = []
    attribute_atoms = []
    relation_atoms = []
    slots = []
    for idx, item in enumerate(record["include"]):
        group_id = f"g{idx}"
        attrs = [item["color"]] if "color" in item else []
        noun = item["class"]
        count = int(item.get("count", 1))
        groups.append({"group_id": group_id, "noun": noun, "count": count, "attrs": attrs})
        count_atoms.append({"group_id": group_id, "noun": noun, "count": count})
        for attr in attrs:
            attribute_atoms.append({"group_id": group_id, "noun": noun, "attribute": attr})
        for demand_idx in range(count):
            slots.append(Slot(slot_id=f"{group_id}#{demand_idx}", noun=noun, attrs=attrs, group_id=group_id, demand_index=demand_idx))
        if "position" in item:
            rel_phrase, ref_idx = item["position"]
            rel_map = {"left of": "left_of", "right of": "right_of", "above": "above", "below": "below"}
            relation_atoms.append({"lhs_group_id": group_id, "rhs_group_id": f"g{ref_idx}", "relation": rel_map[rel_phrase]})
    return PromptParse(
        prompt=record["prompt"],
        object_groups=groups,
        count_atoms=count_atoms,
        attribute_atoms=attribute_atoms,
        relation_atoms=relation_atoms,
        slots=slots,
        supported=True,
        source_parser="geneval_metadata",
    )


def _load_t2i(repo_root: Path) -> list[dict]:
    base = repo_root / "external" / "T2I-CompBench" / "examples" / "dataset"
    configs = [
        ("color.txt", "attribute_binding"),
        ("spatial.txt", "relation"),
        ("numeracy.txt", "count"),
    ]
    rows = []
    for filename, category in configs:
        with (base / filename).open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                prompt = line.strip()
                if not prompt:
                    continue
                parse = build_parse(prompt, source_hint="t2icompbench")
                if not parse.supported:
                    continue
                rows.append(
                    {
                        "dataset": "t2icompbench",
                        "source_category": category,
                        "prompt_id": f"t2icompbench_{filename.replace('.txt', '')}_{idx:04d}",
                        "prompt": prompt,
                        "metadata": {"source_file": filename},
                        "parse": parse,
                    }
                )
    return rows


def _sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[:n]


def _transfer_prompts() -> list[str]:
    return [
        "a red apple and a green bowl",
        "two blue cups",
        "a cat on the left of a chair",
        "a yellow bus and a brown dog",
        "three red books",
        "a boat on the right of a lighthouse",
        "a green bottle and a pink cake",
        "two horses and a brown bench",
        "a sheep on the top of a car",
        "four oranges",
        "a red chair and a blue vase",
        "a dog on the left of a bicycle",
        "three blue bowls",
        "a green train and a red bench",
        "a phone on the right of a person",
        "two apples and a bowl",
        "a yellow clock and a blue car",
        "a bird on the top of a lamp",
        "four cups",
        "a brown horse and a blue suitcase",
        "a backpack on the left of a bicycle",
        "three dogs",
        "a red boat and a green elephant",
        "a rabbit on the right of a cake",
        "two bananas",
        "a cow on the left of a train",
        "a blue bowl and a red cup",
        "three cameras",
        "a balloon on the top of a giraffe",
        "a green bench and a yellow cat",
    ]


def build_splits(repo_root: Path, seed: int = 17) -> dict:
    geneval = _load_geneval(repo_root)
    t2i = _load_t2i(repo_root)

    dev = _sample([r for r in geneval if r["source_category"] in {"counting", "colors", "position"}], 20, seed)
    dev += _sample([r for r in t2i if r["source_category"] in {"attribute_binding", "relation"}], 20, seed + 1)

    main = _sample([r for r in geneval if r["source_category"] in {"counting", "colors", "position"}], 120, seed + 2)
    main += _sample([r for r in t2i if r["source_category"] in {"attribute_binding", "relation"}], 80, seed + 3)

    candidate_budget = _sample(main, 40, seed + 4)

    transfer = []
    for idx, prompt in enumerate(_transfer_prompts()):
        transfer.append(
            {
                "dataset": "transfer",
                "source_category": "transfer",
                "prompt_id": f"transfer_{idx:04d}",
                "prompt": prompt,
                "metadata": {},
                "parse": build_parse(prompt, source_hint="transfer"),
            }
        )

    return {"dev": dev, "test": main, "candidate_budget": candidate_budget, "transfer": transfer}


def write_splits(repo_root: Path) -> None:
    splits = build_splits(repo_root)
    data_dir = ensure_dir(repo_root / "data" / "splits")
    serializable = {}
    for name, rows in splits.items():
        serializable[name] = rows
        write_jsonl(data_dir / f"{name}.jsonl", rows)
    write_json(data_dir / "manifest.json", {k: len(v) for k, v in serializable.items()})
    summary_rows = ["split,dataset,source_category,num_rows"]
    for split_name, rows in serializable.items():
        counts = Counter((row["dataset"], row["source_category"]) for row in rows)
        for (dataset, source_category), count in sorted(counts.items()):
            summary_rows.append(f"{split_name},{dataset},{source_category},{count}")
    (data_dir / "summary.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
