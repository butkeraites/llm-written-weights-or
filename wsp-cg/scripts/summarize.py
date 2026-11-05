"""Summarize CG runs (jsonl) per instance and strategy.

usage: python3 scripts/summarize.py results/file.jsonl [more.jsonl ...] [--limit SECS] [--only a,b,c]
Timeouts count as the time limit (PAR1). Prints a per-instance table and aggregates.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def main() -> None:
    args = sys.argv[1:]
    limit = 120.0
    only: set[str] | None = None
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--limit":
            limit = float(args[i + 1])
            i += 2
        elif args[i] == "--only":
            only = set(args[i + 1].split(","))
            i += 2
        else:
            files.append(args[i])
            i += 1
    rows = [json.loads(line) for f in files for line in open(f) if line.strip()]
    by = defaultdict(dict)
    strategies: list[str] = []
    for r in rows:
        name = r["instance"]
        if only and name not in only:
            continue
        by[name][r["strategy"]] = r
        if r["strategy"] not in strategies:
            strategies.append(r["strategy"])
    names = sorted(by, key=lambda s: int(s.split("-n")[1]))

    def t(r):
        return r["t_total"] if r["status"] == "optimal" else limit

    hdr = f"{'instance':<11}" + "".join(f"{s:>10}" for s in strategies) + "   lb/bks"
    print(hdr)
    for n in names:
        cells = []
        lb = None
        for s in strategies:
            r = by[n].get(s)
            if r is None:
                cells.append(f"{'-':>10}")
                continue
            mark = "" if r["status"] == "optimal" else "*"
            cells.append(f"{t(r):>9.1f}{mark or ' '}")
            if r["status"] == "optimal":
                lb = r["lp_bound"] / r["bks"] if r.get("bks") else None
        print(f"{n:<11}" + "".join(cells) + (f"   {lb:.3f}" if lb else "   -"))
    print("\n* = timeout (counted as limit)")
    print(f"{'strategy':<10}{'solved':>8}{'geo-mean t':>12}{'sum t':>10}{'exact calls':>13}")
    for s in strategies:
        rs = [by[n][s] for n in names if s in by[n]]
        solved = sum(r["status"] == "optimal" for r in rs)
        gm = math.exp(sum(math.log(max(t(r), 0.01)) for r in rs) / max(len(rs), 1))
        st = sum(t(r) for r in rs)
        ex = sum(r["exact_calls"] for r in rs)
        print(f"{s:<10}{solved:>5}/{len(rs):<3}{gm:>11.2f}s{st:>9.0f}s{ex:>13}")


if __name__ == "__main__":
    main()
