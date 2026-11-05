"""Run csrc/wspsolve over instances × rules in a process pool (paired, same batch).

usage: uv run python run_c.py OUT.jsonl TIME_LIMIT WORKERS RULE[,RULE...] INSTANCE...
RULE = builtin name (relpscost, fullstrong, ...) or a net file nets/x.txt
"""

from __future__ import annotations

import json
import subprocess
import sys
from multiprocessing.pool import ThreadPool

BIN = "csrc/wspsolve"


def solve(job):
    path, rule, tlimit = job
    cmd = [BIN, path, "net", str(tlimit), rule] if rule.endswith(".txt") else [BIN, path, rule, str(tlimit)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if out.returncode or not lines:
        return {"instance": path, "rule": rule, "status": "error", "nodes": 0, "time": 0.0,
                "gap": 0.0, "obj": None, "err": out.stderr[-300:]}
    return json.loads(lines[-1])


if __name__ == "__main__":
    out, tlimit, workers, rules = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    # interleave rules per instance so every rule sees the same machine load
    jobs = [(p, r, tlimit) for p in sys.argv[5:] for r in rules.split(",")]
    with ThreadPool(workers) as pool, open(out, "a") as f:
        for res in pool.imap_unordered(solve, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['instance'].split('/')[-1]:16} {res['rule']:16} {res['status']:9} "
                  f"nodes={res['nodes']:>7} t={res['time']:7.1f}s", flush=True)
