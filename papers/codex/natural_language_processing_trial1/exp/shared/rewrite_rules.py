import re


VARIETY_RULES = {
    "Australian English": [
        (r"\bfriend\b", "mate"),
        (r"\bvery\b", "right"),
        (r"\bmaybe\b", "maybe"),
    ],
    "Irish English": [
        (r"\bdo not\b", "don't"),
        (r"\bis not\b", "isn't"),
        (r"\bmust\b", "has to"),
    ],
    "Scottish English": [
        (r"\bsmall\b", "wee"),
        (r"\bknow\b", "ken"),
        (r"\bchild\b", "bairn"),
    ],
    "Appalachian English": [
        (r"\bwas not\b", "warn't"),
        (r"\bgoing to\b", "fixin to"),
        (r"\bmight have\b", "mighta"),
    ],
    "East Anglian English": [
        (r"\bnothing\b", "nowt"),
        (r"\banything\b", "owt"),
        (r"\bgoing\b", "goin"),
    ],
    "Ozark English": [
        (r"\bcarrying\b", "toting"),
        (r"\bvery\b", "mighty"),
        (r"\bchildren\b", "young'uns"),
    ],
    "Newfoundland English": [
        (r"\bfriends\b", "b'ys"),
        (r"\byes\b", "yes b'y"),
        (r"\bvery\b", "right"),
    ],
    "Welsh English": [
        (r"\bsmall\b", "little"),
        (r"\blook\b", "have a look"),
        (r"\bjust\b", "just now"),
    ],
    "Southwest England English": [
        (r"\bare not\b", "bain't"),
        (r"\bgoing to\b", "goin to"),
        (r"\bfriend\b", "mucker"),
    ],
    "New Zealand English": [
        (r"\bvery\b", "heaps"),
        (r"\bfriend\b", "mate"),
        (r"\bexcellent\b", "choice"),
    ],
}

PARAPHRASE_RULES = [
    (r"\bTherefore\b", "So"),
    (r"\bHowever\b", "Still"),
    (r"\bWhich of the following\b", "Which option"),
    (r"\bAccording to the passage\b", "Based on the passage"),
    (r"\bWhat can be inferred\b", "What follows"),
]

STANDARDIZE_RULES = [
    ("mate", "friend"),
    ("wee", "small"),
    ("ken", "know"),
    ("warn't", "was not"),
    ("fixin to", "going to"),
    ("nowt", "nothing"),
    ("owt", "anything"),
    ("young'uns", "children"),
    ("b'ys", "friends"),
]


def apply_rules(text: str, rules):
    out = text
    for pattern, repl in rules:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def variety_prefix(variety: str) -> str:
    short = {
        "Australian English": "In Aussie-style wording, ",
        "Irish English": "In Irish-style wording, ",
        "Scottish English": "In Scots-influenced wording, ",
        "Appalachian English": "In Appalachian-style wording, ",
        "East Anglian English": "In East Anglian-style wording, ",
        "Ozark English": "In Ozark-style wording, ",
        "Newfoundland English": "In Newfoundland-style wording, ",
        "Welsh English": "In Welsh-style wording, ",
        "Southwest England English": "In Southwest England-style wording, ",
        "New Zealand English": "In New Zealand-style wording, ",
    }
    return short.get(variety, "")

