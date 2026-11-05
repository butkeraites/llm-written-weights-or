"""Per-class summary: solved, shifted geometric mean of nodes and time (shift 10 nodes / 1 s).
usage: uv run python summarize.py results/*.jsonl [--limit 300]"""
import json, math, sys, collections
args = sys.argv[1:]; limit = 300.0
if "--limit" in args:
    i = args.index("--limit"); limit = float(args[i+1]); del args[i:i+2]
R = collections.defaultdict(list)
for f in args:
    for l in open(f):
        r = json.loads(l); cls = r["instance"].split("/")[-2]
        R[(cls, r["rule"])].append(r)
def sgm(v, s): return math.exp(sum(math.log(x + s) for x in v) / len(v)) - s
for cls in sorted({c for c, _ in R}):
    print(f"== {cls}")
    for (c, rule), rs in sorted(R.items(), key=lambda kv: kv[0][1]):
        if c != cls: continue
        solved = sum(r["status"] == "optimal" for r in rs)
        t = [r["time"] if r["status"] == "optimal" else limit for r in rs]
        nodes = [r["nodes"] for r in rs]
        print(f"  {rule:16} solved {solved:2}/{len(rs):<2} nodes {sgm(nodes,10):9.1f}  time {sgm(t,1):7.1f}s")
