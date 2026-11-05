"""Feature profile of arcs the reduced graph dropped but exact pricing needed.

This is the "forward-pass feedback" the weight-writer reads before editing a net.
usage: python3 scripts/misses.py results/file.jsonl [--strategy rc8] [--only a,b]
"""

from __future__ import annotations

import json
import sys

FEATURES = [
    "rc_sym", "dist", "pi_i", "pi_j", "rank_out", "rank_in",
    "tw_slack", "min_wait", "demand_j", "tw_width_j",
]


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main() -> None:
    args = sys.argv[1:]
    strat = None
    only = None
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--strategy":
            strat = args[i + 1]
            i += 2
        elif args[i] == "--only":
            only = set(args[i + 1].split(","))
            i += 2
        else:
            files.append(args[i])
            i += 1
    misses = []
    per_inst = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if strat and r["strategy"] != strat:
                continue
            if only and r["instance"] not in only:
                continue
            per_inst.append((r["instance"], r["strategy"], r["miss_count"], r["exact_calls"]))
            misses.extend(m["features"] for m in r["misses"])
    print(f"{'instance':<11}{'strategy':<8}{'misses':>8}{'exact':>7}")
    for row in per_inst:
        print(f"{row[0]:<11}{row[1]:<8}{row[2]:>8}{row[3]:>7}")
    print(f"\n{len(misses)} sampled missed arcs; feature percentiles (p10 / p50 / p90):")
    for k, name in enumerate(FEATURES):
        v = [m[k] for m in misses]
        print(f"  {name:<11} {pct(v, .1):>8.3f} {pct(v, .5):>8.3f} {pct(v, .9):>8.3f}")


if __name__ == "__main__":
    main()
