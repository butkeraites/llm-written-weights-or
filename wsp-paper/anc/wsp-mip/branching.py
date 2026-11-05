"""Weight-space-programmed branching for SCIP.

A tiny MLP (weights in JSON, written by an LLM — no data, no training) scores each fractional
candidate from cheap features; SCIP branches on the argmax. Branch-and-bound stays exact whatever
the weights are: a bad net only grows the tree.

Diagnostic mode: at sampled nodes, also run full strong branching on every candidate (the gold
standard) and log where the net's choice ranks. That log is the "forward-pass feedback" the
weight writer reads; it is never fitted.
"""

from __future__ import annotations

import json

import numpy as np
from pyscipopt import SCIP_RESULT, Branchrule

FEATURES = [
    "frac",   # 2*min(f, 1-f): 1 = exactly half
    "ps",     # pseudocost score / max over candidates (SCIP's running estimate of bound gain)
    "obj",    # |c_j| / max |c| over candidates
    "deg",    # nonzeros of column j in the LP / max over candidates
    "tight",  # share of j's rows that are tight at the LP optimum
    "dualm",  # sum_i |y_i * a_ij| / max over candidates (how much LP dual pressure j carries)
    "val",    # LP value of x_j (asymmetric: 0.9 vs 0.1)
    # --- structural features for set-partitioning-like rows (added for crew scheduling) ---
    "fconf",  # LP mass of the OTHER fractional candidates sharing a row with j (each counted
              # once) / max: what the up-branch x_j=1 forces out
    "alt",    # 1/(m-1), m = fewest fractional candidates in any of j's rows: a row with only one
              # alternative makes the down-branch x_j=0 force that alternative
    "rowf",   # number of j's rows that are fractional (touched by >1 fractional cand) / max
    "eff",    # cost per row, best over candidates / this candidate's (1 = most efficient duty)
]


class Net:
    def __init__(self, path: str):
        spec = json.load(open(path))
        self.note = spec.get("note", "")
        # optional policy knob: strong-branch only the net's top-k candidates down to maxdepth
        self.lookahead = spec.get("lookahead")
        self.layers = [
            (np.array(layer["w"], dtype=float), np.array(layer["b"], dtype=float), layer["act"])
            for layer in spec["layers"]
        ]
        self.inputs = self.layers[0][0].shape[1]
        assert self.inputs <= len(FEATURES)
        width = self.inputs
        for w, b, _ in self.layers:
            assert w.shape == (len(b), width), f"layer shape {w.shape} vs in={width}"
            width = len(b)
        assert width == 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x[:, : self.inputs]
        for w, b, act in self.layers:
            h = h @ w.T + b
            if act == "relu":
                h = np.maximum(h, 0)
            elif act == "tanh":
                h = np.tanh(h)
        return h[:, 0]


class NetBranching(Branchrule):
    def __init__(self, model, net: Net, diag_every: int = 0, diag_max: int = 0):
        self.model = model
        self.net = net
        self.diag_every = diag_every
        self.diag_max = diag_max
        self.diag: list[dict] = []
        self.calls = 0
        self._cache_nrows = -1
        self._colrows: dict[int, list[tuple[int, float]]] = {}
        self._rows = []

    def _structure(self):
        rows = self.model.getLPRowsData()
        if len(rows) == self._cache_nrows:
            return
        self._rows = rows
        self._cache_nrows = len(rows)
        colrows: dict[int, list[tuple[int, float]]] = {}
        for r, row in enumerate(rows):
            for col, val in zip(row.getCols(), row.getVals()):
                colrows.setdefault(col.getLPPos(), []).append((r, val))
        self._colrows = colrows

    def features(self, cands, sols):
        m = self.model
        self._structure()
        rows = self._rows
        y = np.array([row.getDualsol() for row in rows])
        tight = np.zeros(len(rows), dtype=bool)
        for r, row in enumerate(rows):
            act = m.getRowLPActivity(row)
            tight[r] = (abs(act - row.getRhs()) < 1e-6) or (abs(act - row.getLhs()) < 1e-6)
        n = len(cands)
        x = np.zeros((n, len(FEATURES)))
        for k, (var, v) in enumerate(zip(cands, sols)):
            f = v - np.floor(v)
            entries = self._colrows.get(var.getCol().getLPPos(), [])
            idx = np.fromiter((r for r, _ in entries), dtype=int, count=len(entries))
            vals = np.fromiter((a for _, a in entries), dtype=float, count=len(entries))
            x[k, 0] = 2 * min(f, 1 - f)
            x[k, 1] = m.getVarPseudocostScore(var, v)
            x[k, 2] = abs(var.getObj())
            x[k, 3] = len(entries)
            x[k, 4] = tight[idx].mean() if len(entries) else 0.0
            x[k, 5] = np.abs(y[idx] * vals).sum() if len(entries) else 0.0
            x[k, 6] = v
        # structural features over the fractional candidates
        cand_rows = [
            [r for r, _ in self._colrows.get(var.getCol().getLPPos(), [])] for var in cands
        ]
        row_cands: dict[int, list[int]] = {}
        for k, rs in enumerate(cand_rows):
            for r in rs:
                row_cands.setdefault(r, []).append(k)
        for k, rs in enumerate(cand_rows):
            others = {o for r in rs for o in row_cands[r] if o != k}
            x[k, 7] = sum(sols[o] for o in others)
            m_ = min((len(row_cands[r]) for r in rs), default=1)
            x[k, 8] = 1.0 / (m_ - 1) if m_ > 1 else 0.0
            x[k, 9] = sum(len(row_cands[r]) > 1 for r in rs)
            x[k, 10] = abs(cands[k].getObj()) / max(len(rs), 1)
        e = x[:, 10]
        x[:, 10] = np.where(e > 0, e.min() / np.maximum(e, 1e-12), 1.0) if (e > 0).any() else 1.0
        for c in (1, 2, 3, 5, 7, 9):
            mx = x[:, c].max()
            if mx > 0:
                x[:, c] /= mx
        return x

    def _strong_scores(self, cands, itlim: int = 10**6):
        m = self.model
        lpobj = m.getLPObjVal()
        m.startStrongbranch()
        out = []
        for var in cands:
            down, up, dv, uv, dinf, uinf, *_ = m.getVarStrongbranch(var, itlim)
            dg = 1e9 if dinf else max(down - lpobj, 0.0)
            ug = 1e9 if uinf else max(up - lpobj, 0.0)
            out.append(max(dg, 1e-6) * max(ug, 1e-6))
        m.endStrongbranch()
        return np.array(out)

    def branchexeclp(self, allowaddcons):
        m = self.model
        cands, sols, _fracs, _n, nprio, *_ = m.getLPBranchCands()
        cands, sols = cands[:nprio], sols[:nprio]
        x = self.features(cands, sols)
        scores = self.net(x)
        best = int(np.argmax(scores))
        la = self.net.lookahead
        if la and len(cands) > 1 and m.getCurrentNode().getDepth() <= la.get("maxdepth", 10**9):
            top = np.argsort(-scores)[: la["k"]]
            sb = self._strong_scores([cands[i] for i in top], la.get("itlim", 10**6))
            best = int(top[int(np.argmax(sb))])
        self.calls += 1
        if (
            self.diag_every
            and len(self.diag) < self.diag_max
            and (self.calls - 1) % self.diag_every == 0
            and len(cands) > 1
        ):
            sb = self._strong_scores(cands)
            order = np.argsort(-sb)
            rank = int(np.where(order == best)[0][0])
            self.diag.append({
                "depth": m.getCurrentNode().getDepth(),
                "ncands": len(cands),
                "rank": rank,
                "ratio": float(sb[best] / sb.max()) if sb.max() > 0 else 1.0,
                "features": x.tolist(),
                "sb": sb.tolist(),
                "chosen": best,
            })
        m.branchVar(cands[best])
        return {"result": SCIP_RESULT.BRANCHED}
