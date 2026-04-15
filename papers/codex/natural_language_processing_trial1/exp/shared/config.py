from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_AUDIT_DIR = ARTIFACTS_DIR / "data_audit"
SUMMARY_DIR = ARTIFACTS_DIR / "summary"
FIGURES_DIR = ROOT / "figures"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 768
TRAIN_SEEDS = [13, 21, 42]
SEEN_VARIETIES = [
    "Australian English",
    "Irish English",
    "Scottish English",
    "Appalachian English",
    "East Anglian English",
    "Ozark English",
]
HELD_OUT_VARIETIES = ["Newfoundland English", "Welsh English"]
RESERVE_VARIETIES = ["Southwest England English", "New Zealand English"]
ALL_VARIETIES = SEEN_VARIETIES + HELD_OUT_VARIETIES + RESERVE_VARIETIES

SYSTEM_DISPLAY = {
    "base": "Base",
    "sae_only_sft": "SAE-only SFT",
    "dialect_augmentation": "Dialect augmentation",
    "paraphrase_pair_control": "Paraphrase-pair control",
    "rewrite_then_answer": "Rewrite-then-answer",
    "svpt": "SVPT",
    "single_view_svpt": "single-view SVPT",
}

SYSTEMS_MAIN = [
    "base",
    "sae_only_sft",
    "dialect_augmentation",
    "paraphrase_pair_control",
    "rewrite_then_answer",
    "svpt",
]

TRAINABLE_SYSTEMS = [
    "sae_only_sft",
    "dialect_augmentation",
    "paraphrase_pair_control",
    "svpt",
]

LABEL_TEMPLATES = {
    "mcq": ["A", "B", "C", "D", "E"],
    "binary_logic": ["necessarily true", "necessarily false", "neither"],
}

PROMPT_HEADER = (
    "Answer the reasoning question. Reply with exactly one valid answer label "
    "inside <answer></answer>."
)

