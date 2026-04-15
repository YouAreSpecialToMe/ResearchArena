from __future__ import annotations

import re
from dataclasses import dataclass, field


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}

COLORS = {
    "red",
    "green",
    "blue",
    "brown",
    "yellow",
    "gold",
    "pink",
    "black",
    "white",
    "orange",
    "purple",
    "gray",
}

REL_MAP = {
    "left of": "left_of",
    "right of": "right_of",
    "top of": "above",
    "bottom of": "below",
    "above": "above",
    "below": "below",
}


@dataclass
class Slot:
    slot_id: str
    noun: str
    attrs: list[str] = field(default_factory=list)
    group_id: str = ""
    demand_index: int = 0


@dataclass
class PromptParse:
    prompt: str
    object_groups: list[dict]
    count_atoms: list[dict]
    attribute_atoms: list[dict]
    relation_atoms: list[dict]
    slots: list[Slot]
    supported: bool = True
    source_parser: str = "heuristic"


def normalize_noun(noun: str, singularize: bool = True) -> str:
    noun = noun.strip().lower()
    noun = re.sub(r"^(a|an|the)\s+", "", noun)
    noun = noun.replace("television", "tv")
    noun = noun.replace("computer ", "computer ")
    noun = noun.replace("tv remote", "tv remote")
    if not singularize:
        return noun
    if noun.endswith("ies"):
        return noun[:-3] + "y"
    if noun.endswith("ves"):
        return noun[:-3] + "f"
    if noun.endswith(("ss", "us", "is")):
        return noun
    if noun.endswith("s"):
        return noun[:-1]
    return noun


def parse_np(text: str) -> tuple[list[str], str, int]:
    text = text.strip().lower()
    count = 1
    words = [w for w in text.split() if w]
    if words and words[0] in NUMBER_WORDS:
        count = NUMBER_WORDS[words[0]]
        words = words[1:]
    words = [w for w in words if w not in {"a", "an", "the", "photo", "of"}]
    attrs = [w for w in words[:-1] if w in COLORS]
    noun = normalize_noun(" ".join(words[len(attrs) :]), singularize=count != 1)
    return attrs, noun, count


def build_parse(prompt: str, source_hint: str = "") -> PromptParse:
    text = prompt.strip().lower()
    text = re.sub(r"^a photo of\s+", "", text)
    relation_atoms = []
    groups = []
    count_atoms = []
    attribute_atoms = []
    slots: list[Slot] = []

    relation_match = None
    for rel_phrase, rel_name in REL_MAP.items():
        pattern = rf"(.+?)\s+on the {re.escape(rel_phrase)}\s+(.+)"
        relation_match = re.fullmatch(pattern, text)
        if relation_match:
            left, right = relation_match.group(1), relation_match.group(2)
            left_attrs, left_noun, left_count = parse_np(left)
            right_attrs, right_noun, right_count = parse_np(right)
            groups = [
                {"group_id": "g0", "noun": left_noun, "count": left_count, "attrs": left_attrs},
                {"group_id": "g1", "noun": right_noun, "count": right_count, "attrs": right_attrs},
            ]
            for g in groups:
                count_atoms.append({"group_id": g["group_id"], "noun": g["noun"], "count": g["count"]})
                for attr in g["attrs"]:
                    attribute_atoms.append({"group_id": g["group_id"], "noun": g["noun"], "attribute": attr})
                for i in range(g["count"]):
                    slots.append(
                        Slot(
                            slot_id=f"{g['group_id']}#{i}",
                            noun=g["noun"],
                            attrs=g["attrs"],
                            group_id=g["group_id"],
                            demand_index=i,
                        )
                    )
            relation_atoms.append({"lhs_group_id": "g0", "rhs_group_id": "g1", "relation": rel_name})
            return PromptParse(prompt=prompt, object_groups=groups, count_atoms=count_atoms, attribute_atoms=attribute_atoms, relation_atoms=relation_atoms, slots=slots, source_parser="relation")

    parts = [p.strip() for p in text.split(" and ")]
    if not parts:
        return PromptParse(prompt=prompt, object_groups=[], count_atoms=[], attribute_atoms=[], relation_atoms=[], slots=[], supported=False)

    for idx, part in enumerate(parts):
        attrs, noun, count = parse_np(part)
        if not noun:
            return PromptParse(prompt=prompt, object_groups=[], count_atoms=[], attribute_atoms=[], relation_atoms=[], slots=[], supported=False)
        group_id = f"g{idx}"
        groups.append({"group_id": group_id, "noun": noun, "count": count, "attrs": attrs})
        count_atoms.append({"group_id": group_id, "noun": noun, "count": count})
        for attr in attrs:
            attribute_atoms.append({"group_id": group_id, "noun": noun, "attribute": attr})
        for i in range(count):
            slots.append(Slot(slot_id=f"{group_id}#{i}", noun=noun, attrs=attrs, group_id=group_id, demand_index=i))

    return PromptParse(
        prompt=prompt,
        object_groups=groups,
        count_atoms=count_atoms,
        attribute_atoms=attribute_atoms,
        relation_atoms=relation_atoms,
        slots=slots,
        source_parser=source_hint or "coordination",
    )
