"""Read strong-branching diagnostics (the forward-pass feedback).

For every sampled node we have each candidate's features and its strong-branching score.
Reports, per class: the logged rule's top-1 agreement and mean score ratio, and — for each single
feature (and its negation) — how often its argmax would have hit the SB-best candidate and the
mean SB-score ratio it would have achieved. Read to guide hand-written weights; never fitted.

usage: uv run python diag.py results/diag_*.jsonl
"""

from __future__ import annotations

import collections
import json
import sys

import numpy as np

from branching import FEATURES


def main() -> None:
    samples = collections.defaultdict(list)
    rule_stats = collections.defaultdict(list)
    for f in sys.argv[1:]:
        for line in open(f):
            r = json.loads(line)
            cls = r["instance"].split("/")[-2]
            for s in r.get("diag", []):
                samples[cls].append(s)
                rule_stats[(cls, r["rule"])].append((s["rank"] == 0, s["ratio"]))
    for (cls, rule), v in sorted(rule_stats.items()):
        print(f"{cls:9} {rule:12} top1 {np.mean([a for a, _ in v]):.0%}  ratio {np.mean([b for _, b in v]):.2f}  (n={len(v)})")
    for cls, ss in sorted(samples.items()):
        print(f"\n== {cls}: {len(ss)} nodes, median candidates {int(np.median([s['ncands'] for s in ss]))}")
        print(f"   {'feature':8} {'argmax top1':>12} {'ratio':>6} | {'argmin top1':>12} {'ratio':>6}")
        rand = np.mean([np.mean(np.array(s["sb"]) / max(max(s["sb"]), 1e-12)) for s in ss])
        for k, name in enumerate(FEATURES):
            res = []
            for sign in (1, -1):
                hits, ratios = [], []
                for s in ss:
                    x = np.array(s["features"])[:, k] * sign
                    sb = np.array(s["sb"])
                    pick = int(np.argmax(x))
                    hits.append(sb[pick] >= sb.max() - 1e-12)
                    ratios.append(sb[pick] / sb.max() if sb.max() > 0 else 1.0)
                res.append((np.mean(hits), np.mean(ratios)))
            print(f"   {name:8} {res[0][0]:>12.0%} {res[0][1]:>6.2f} | {res[1][0]:>12.0%} {res[1][1]:>6.2f}")
        print(f"   random pick: mean ratio {rand:.2f}")


if __name__ == "__main__":
    main()
