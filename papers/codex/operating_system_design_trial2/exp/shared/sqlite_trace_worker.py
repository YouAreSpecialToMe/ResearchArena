from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from exp.shared.common import set_reproducible
from exp.shared.workloads import sqlite_query_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tenant", type=int, required=True)
    parser.add_argument("--target-refs", type=int, required=True)
    parser.add_argument("--out-queries", required=True)
    args = parser.parse_args()

    rng = set_reproducible(args.seed * 17 + args.tenant)
    templates = sqlite_query_bank()
    con = sqlite3.connect(f"file:{args.db}?immutable=1", uri=True)
    cur = con.cursor()
    queries = []
    shared_customers = [1 + ((args.seed * 101 + i * 389) % 80_000) for i in range(512)]
    shared_orders = [1 + ((args.seed * 313 + i * 977) % 260_000) for i in range(512)]
    if args.family == "SQLiteTraceMix-2T":
        template_sequences = {
            0: [0, 2, 5, 3],
            1: [2, 5, 4, 1],
        }
    else:
        template_sequences = {
            0: [0, 2, 5, 3, 1],
            1: [2, 5, 4, 1, 3],
            2: [5, 4, 3, 0, 1],
        }
    template_ids = template_sequences[args.tenant]
    for step, template_id in enumerate(template_ids):
        template = templates[template_id]
        customer_id = shared_customers[(step * 13 + args.tenant * 7 + rng.randrange(64)) % len(shared_customers)]
        order_id = shared_orders[(step * 17 + args.tenant * 11 + rng.randrange(64)) % len(shared_orders)]
        query = template.format(
            order_id=order_id,
            customer_id=customer_id,
        )
        queries.append(query)
        cur.execute(query)
        cur.fetchall()
    Path(args.out_queries).write_text(json.dumps({"queries": queries}, indent=2))


if __name__ == "__main__":
    main()
