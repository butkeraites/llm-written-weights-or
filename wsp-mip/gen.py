"""Instance generators for the four classic learning-to-branch benchmarks (Gasse et al., 2019),
sizes calibrated so SCIP 10 default needs a real tree: set cover 1000x1000, combinatorial auction
100 items / 1500 bids, capacitated facility location 70x70, independent set 750 nodes (BA, affinity 4).

usage: uv run python gen.py OUTDIR N_PER_CLASS SEED
Writes OUTDIR/<class>/<class>_<k>.lp
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
from pyscipopt import Model, quicksum


def set_cover(rng: np.random.Generator, nrows=1000, ncols=1000, density=0.05, max_coef=100) -> Model:
    nnz = int(nrows * ncols * density)
    cols = np.concatenate([np.arange(2 * ncols) % ncols, rng.integers(0, ncols, nnz - 2 * ncols)])
    rows = np.concatenate([rng.permutation(nrows), rng.integers(0, nrows, nnz - nrows)])
    pairs = set(zip(rows.tolist(), cols.tolist()))
    cost = rng.integers(1, max_coef + 1, ncols)
    m = Model("setcover")
    x = [m.addVar(f"x{j}", vtype="B", obj=float(cost[j])) for j in range(ncols)]
    by_row: dict[int, list[int]] = {}
    for r, c in pairs:
        by_row.setdefault(r, []).append(c)
    for r in range(nrows):
        m.addCons(quicksum(x[c] for c in by_row[r]) >= 1, f"r{r}")
    m.setMinimize()
    return m


def comb_auction(rng: np.random.Generator, n_items=100, n_bids=1500, cont=0.8, spread=0.05) -> Model:
    # Simplified "arbitrary relationships" generator (Leyton-Brown et al., 2000): items have
    # private values and pairwise compatibilities; bundles grow by compatibility; price is
    # superadditive in the bundle value plus noise.
    values = rng.uniform(1, 100, n_items)
    compat = rng.uniform(0, 1, (n_items, n_items))
    compat = (compat + compat.T) / 2
    np.fill_diagonal(compat, 0)
    m = Model("cauction")
    bids = []
    while len(bids) < n_bids:
        first = int(rng.integers(n_items))
        bundle = [first]
        while rng.uniform() < cont and len(bundle) < n_items:
            w = compat[bundle].sum(axis=0)
            w[bundle] = 0
            nxt = int(rng.choice(n_items, p=w / w.sum()))
            bundle.append(nxt)
        price = values[bundle].sum() * (1 + 0.2 * len(bundle) ** 0.2) + rng.uniform(0, 10)
        if spread > 0:
            price = values[bundle].sum() * rng.uniform(1 - spread, 1 + spread)
        bids.append((sorted(set(bundle)), price))
    x = [m.addVar(f"b{k}", vtype="B", obj=float(p)) for k, (_, p) in enumerate(bids)]
    for i in range(n_items):
        holders = [k for k, (b, _) in enumerate(bids) if i in b]
        if holders:
            m.addCons(quicksum(x[k] for k in holders) <= 1, f"i{i}")
    m.setMaximize()
    return m


def facility_location(rng: np.random.Generator, n_cust=70, n_fac=70, ratio=5) -> Model:
    # Cornuejols et al. (1991), as in Gasse et al.
    cx, cy, fx, fy = (rng.uniform(size=n) for n in (n_cust, n_cust, n_fac, n_fac))
    demand = rng.integers(5, 36, n_cust)
    cap = rng.integers(10, 161, n_fac)
    fixed = (rng.integers(100, 111, n_fac) * np.sqrt(cap) + rng.integers(0, 91, n_fac)).astype(int)
    cap = (cap * ratio * demand.sum() / cap.sum()).astype(int)
    trans = np.sqrt((cx[:, None] - fx[None, :]) ** 2 + (cy[:, None] - fy[None, :]) ** 2)
    trans = trans * 10 * demand[:, None]
    m = Model("cfl")
    y = [m.addVar(f"y{j}", vtype="B", obj=float(fixed[j])) for j in range(n_fac)]
    x = [[m.addVar(f"x{i}_{j}", lb=0, ub=1, obj=float(trans[i, j])) for j in range(n_fac)]
         for i in range(n_cust)]
    for i in range(n_cust):
        m.addCons(quicksum(x[i]) == 1, f"d{i}")
    for j in range(n_fac):
        m.addCons(quicksum(demand[i] * x[i][j] for i in range(n_cust)) <= cap[j] * y[j], f"c{j}")
    m.addCons(quicksum(cap[j] * y[j] for j in range(n_fac)) >= int(demand.sum()), "tot")
    for i in range(n_cust):
        for j in range(n_fac):
            m.addCons(x[i][j] <= y[j], f"t{i}_{j}")
    m.setMinimize()
    return m


def indep_set(rng: np.random.Generator, n=750, affinity=4) -> Model:
    g = nx.barabasi_albert_graph(n, affinity, seed=int(rng.integers(2**31)))
    m = Model("indset")
    x = [m.addVar(f"v{v}", vtype="B", obj=1.0) for v in range(n)]
    # greedy clique partition of the edges (as in Gasse et al.)
    remaining = {v: set(g.neighbors(v)) for v in g.nodes}
    k = 0
    while any(remaining.values()):
        v = max(remaining, key=lambda u: len(remaining[u]))
        clique = [v]
        for u in sorted(remaining[v], key=lambda u: -len(remaining[u])):
            if all(u in remaining[w] for w in clique):
                clique.append(u)
        for a in clique:
            for b in clique:
                if a != b:
                    remaining[a].discard(b)
        m.addCons(quicksum(x[u] for u in clique) <= 1, f"k{k}")
        k += 1
    m.setMaximize()
    return m


GENERATORS = {
    "setcover": set_cover,
    "cauction": comb_auction,
    "cfl": facility_location,
    "indset": indep_set,
}

MEDIUM = {
    "setcover": (set_cover, {"nrows": 1250}),
    "indset": (indep_set, {"n": 1000}),
}

if __name__ == "__main__":
    out, n, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    medium = len(sys.argv) > 4 and sys.argv[4] == "medium"
    items = MEDIUM.items() if medium else ((k, (g, {})) for k, g in GENERATORS.items())
    for name, (gen, kw) in items:
        (out / name).mkdir(parents=True, exist_ok=True)
        for k in range(n):
            rng = np.random.default_rng([seed, k, sum(map(ord, name))])
            m = gen(rng, **kw)
            m.writeProblem(str(out / name / f"{name}_{k}.lp"), verbose=False)
        print(name, "done")
