from __future__ import annotations

import json
import traceback
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _read_code(path: Path) -> str:
    return path.read_text()


def execute_item(item_dir: Path, code_override: str | None = None) -> dict:
    item_dir = item_dir.resolve()
    item_id = item_dir.name
    namespace: dict[str, object] = {}
    setup_code = _read_code(item_dir / "setup_code.py")
    answer_code = code_override if code_override is not None else _read_code(item_dir / "answer_code.py")
    check_code = _read_code(item_dir / "check_code.py")
    old_cwd = Path.cwd()
    try:
        sink = io.StringIO()
        os.chdir(item_dir)
        with redirect_stdout(sink), redirect_stderr(sink):
            exec(setup_code, namespace, namespace)
            exec(answer_code, namespace, namespace)
            exec(check_code, namespace, namespace)
        result = {
            "item_id": item_id,
            "passed": bool(namespace["passed"]),
            "exception_type": None,
            "result": namespace["result"],
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "item_id": item_id,
            "passed": False,
            "exception_type": type(exc).__name__,
            "result": {"traceback": traceback.format_exc(limit=5)},
        }
    finally:
        os.chdir(old_cwd)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("item_dir")
    parser.add_argument("--code-file", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code = Path(args.code_file).read_text() if args.code_file else None
    output = execute_item(Path(args.item_dir), code)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
