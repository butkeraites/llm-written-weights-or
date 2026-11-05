"""Step 0: solve instances with SCIP's built-in branching rules and record tree size / time.

usage: uv run python run_builtin.py OUT.jsonl TIME_LIMIT WORKERS RULE[,RULE...] INSTANCE.lp...
Rules: relpscost (SCIP default), fullstrong (gold standard), pscost, mostinf, random.
Settings follow Gasse et al. (2019): no restarts, cuts only at the root.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

from pyscipopt import Model


def solve(job):
    path, rule, tlimit = job
    m = Model()
    m.hideOutput()
    try:
        m.readProblem(path)
    except OSError:
        return {"instance": path, "rule": "?", "status": "readerror", "nodes": 0, "time": 0.0, "gap": 0.0, "obj": None}
    if not os.environ.get("SCIP_DEFAULTS"):  # Gasse et al. settings unless asked for SCIP defaults
        m.setIntParam("presolving/maxrestarts", 0)
        m.setIntParam("separating/maxrounds", 0)
    m.setRealParam("limits/time", tlimit)
    m.setIntParam("randomization/randomseedshift", 0)
    m.setIntParam(f"branching/{rule}/priority", 9_999_999)
    t = time.perf_counter()
    m.optimize()
    return {
        "instance": path,
        "rule": rule,
        "status": m.getStatus(),
        "nodes": m.getNNodes(),
        "time": time.perf_counter() - t,
        "gap": m.getGap(),
        "obj": m.getObjVal() if m.getNSols() else None,
    }


if __name__ == "__main__":
    out, tlimit, workers, rules = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    jobs = [(p, r, tlimit) for p in sys.argv[5:] for r in rules.split(",")]
    with Pool(workers) as pool, open(out, "a") as f:
        for res in pool.imap_unordered(solve, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['instance'].split('/')[-1]:16} {res['rule']:10} {res['status']:9} "
                  f"nodes={res['nodes']:>7} t={res['time']:7.1f}s gap={res['gap']:.4f}", flush=True)
