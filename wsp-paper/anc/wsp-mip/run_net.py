"""Solve instances with a net branching rule (same settings as run_builtin.py).

usage: uv run python run_net.py OUT.jsonl TIME_LIMIT WORKERS NET.json[,NET2.json] [--diag K:MAX] INSTANCE.lp...
--diag K:MAX  run strong branching at every K-th branching call (up to MAX samples) and log
              the net's rank vs the gold standard (slows the run; use on practice instances).
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from pyscipopt import Model

from branching import Net, NetBranching


def solve(job):
    path, netfile, tlimit, diag = job
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
    rule = NetBranching(m, Net(netfile), *diag)
    m.includeBranchrule(rule, "wsp", "weight-space programmed", priority=9_999_999,
                        maxdepth=-1, maxbounddist=1.0)
    t = time.perf_counter()
    m.optimize()
    res = {
        "instance": path,
        "rule": "net:" + Path(netfile).stem,
        "status": m.getStatus(),
        "nodes": m.getNNodes(),
        "time": time.perf_counter() - t,
        "gap": m.getGap(),
        "obj": m.getObjVal() if m.getNSols() else None,
    }
    if diag[0]:
        res["diag"] = rule.diag
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    diag = (0, 0)
    if "--diag" in args:
        i = args.index("--diag")
        k, mx = args[i + 1].split(":")
        diag = (int(k), int(mx))
        del args[i : i + 2]
    out, tlimit, workers, nets = args[0], float(args[1]), int(args[2]), args[3]
    jobs = [(p, n, tlimit, diag) for p in args[4:] for n in nets.split(",")]
    with Pool(workers) as pool, open(out, "a") as f:
        for res in pool.imap_unordered(solve, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            extra = ""
            if res.get("diag"):
                d = res["diag"]
                top1 = sum(s["rank"] == 0 for s in d) / len(d)
                ratio = sum(s["ratio"] for s in d) / len(d)
                extra = f" sb-top1={top1:.0%} sb-ratio={ratio:.2f} (n={len(d)})"
            print(f"{res['instance'].split('/')[-1]:16} {res['rule']:14} {res['status']:9} "
                  f"nodes={res['nodes']:>7} t={res['time']:7.1f}s{extra}", flush=True)
