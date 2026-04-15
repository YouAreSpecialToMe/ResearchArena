from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.run_llm_condition import run_condition


if __name__ == "__main__":
    run_condition("thread_only")
