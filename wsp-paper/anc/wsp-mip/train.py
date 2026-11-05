"""Trained baseline: same 7 features, same JSON format, weights fitted by imitation of strong
branching (listwise softmax cross-entropy over each node's candidates; target = SB-best).
This isolates "LLM-written weights" vs "weights learned from data".

usage: uv run python train.py results/sb_train.jsonl OUT.json [--classes setcover,indset]
       [--hidden 16] [--epochs 300] [--seed 0]
"""

from __future__ import annotations

import json
import sys

import numpy as np


def load(path: str, classes: set[str]):
    groups = []
    for line in open(path):
        r = json.loads(line)
        cls = r["instance"].split("/")[-2]
        if cls not in classes:
            continue
        for s in r.get("diag", []):
            x = np.array(s["features"], dtype=float)
            sb = np.array(s["sb"], dtype=float)
            if len(sb) < 2:
                continue
            t = (sb >= sb.max() * (1 - 1e-9)).astype(float)
            groups.append((r["instance"], x, t / t.sum(), sb))
    return groups


def forward(params, x):
    w1, b1, w2, b2 = params
    z = x @ w1.T + b1
    h = np.maximum(z, 0)
    return (h @ w2.T + b2)[:, 0], (z, h)


def loss_grad(params, groups):
    w1, b1, w2, b2 = params
    g = [np.zeros_like(p) for p in params]
    total = 0.0
    for _, x, t, _ in groups:
        s, (z, h) = forward(params, x)
        s = s - s.max()
        p = np.exp(s) / np.exp(s).sum()
        total += -(t * np.log(p + 1e-12)).sum()
        ds = (p - t)[:, None]  # d loss / d s
        g[2] += ds.T @ h
        g[3] += ds.sum(0)
        dh = ds @ w2 * (z > 0)
        g[0] += dh.T @ x
        g[1] += dh.sum(0)
    n = len(groups)
    return total / n, [gi / n for gi in g]


def evaluate(params, groups):
    top1, ratio = [], []
    for _, x, _, sb in groups:
        k = int(np.argmax(forward(params, x)[0]))
        top1.append(sb[k] >= sb.max() * (1 - 1e-9))
        ratio.append(sb[k] / sb.max() if sb.max() > 0 else 1.0)
    return float(np.mean(top1)), float(np.mean(ratio))


def main() -> None:
    args = sys.argv[1:]
    opts = {"--classes": "setcover,indset", "--hidden": "16", "--epochs": "300", "--seed": "0"}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i : i + 2]
    data, out = args
    groups = load(data, set(opts["--classes"].split(",")))
    rng = np.random.default_rng(int(opts["--seed"]))
    insts = sorted({g[0] for g in groups})
    rng.shuffle(insts)
    val_insts = set(insts[: max(1, len(insts) // 5)])
    train = [g for g in groups if g[0] not in val_insts]
    val = [g for g in groups if g[0] in val_insts]
    hdim = int(opts["--hidden"])
    params = [rng.normal(0, 0.5, (hdim, 7)), np.zeros(hdim), rng.normal(0, 0.5, (1, hdim)), np.zeros(1)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    best = (-1.0, None)
    lr, b1, b2 = 1e-2, 0.9, 0.999
    for ep in range(1, int(opts["--epochs"]) + 1):
        loss, grads = loss_grad(params, train)
        for i, gr in enumerate(grads):
            m[i] = b1 * m[i] + (1 - b1) * gr
            v[i] = b2 * v[i] + (1 - b2) * gr * gr
            params[i] -= lr * (m[i] / (1 - b1**ep)) / (np.sqrt(v[i] / (1 - b2**ep)) + 1e-8)
        if ep % 25 == 0 or ep == 1:
            vt, vr = evaluate(params, val)
            if vr > best[0]:
                best = (vr, [p.copy() for p in params])
            print(f"ep {ep:4} loss {loss:.4f} val top1 {vt:.0%} ratio {vr:.3f}", flush=True)
    params = best[1]
    tt, tr = evaluate(params, train)
    vt, vr = evaluate(params, val)
    print(f"best: train top1 {tt:.0%} ratio {tr:.3f} | val top1 {vt:.0%} ratio {vr:.3f}"
          f" | {len(train)} train / {len(val)} val nodes")
    w1, bb1, w2, bb2 = params
    json.dump({
        "note": f"TRAINED baseline (imitation of strong branching), classes={opts['--classes']}, "
                f"hidden={hdim}, {len(train)} train nodes; val ratio {vr:.3f}",
        "layers": [
            {"act": "relu", "w": w1.tolist(), "b": bb1.tolist()},
            {"act": "linear", "w": w2.tolist(), "b": bb2.tolist()},
        ],
    }, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
