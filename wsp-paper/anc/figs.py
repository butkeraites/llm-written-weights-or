"""Build every data figure of the paper from the raw result files (JSONL) of the two POC repos.

Each figure is computed from logged runs only; nothing is typed in by hand except the strong-
branching agreement ratios, which are recomputed here from the logged diagnostics.
Also writes figs/summary.json, the machine-readable numbers quoted in the text.

usage: uv run --with matplotlib --with numpy python scripts/figs.py
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
# works from the monorepo (wsp-paper/scripts/), side-by-side checkouts, or the arXiv anc/ folder
ROOT = next(b for b in (HERE, HERE.parent, HERE.parent.parent) if (b / "wsp-mip").exists())
MIP = ROOT / "wsp-mip"
CG = ROOT / "wsp-cg"
OUT = HERE.parent / "figs"
OUT.mkdir(exist_ok=True)
SUMMARY: dict = {}

# colour follows the entity, in every figure (validated palette, light mode)
C = {
    "default": "#7a7974",   # SCIP default (relpscost) — neutral baseline
    "wsp": "#2a78d6",       # LLM-written nets
    "trained": "#eb6834",   # gradient-trained nets / plain pseudocost (other cheap rules)
    "strong": "#1baf7a",    # full strong branching (gold standard)
    "look": "#6250d6",      # net-guided lookahead
    "pscost": "#eda100",
}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7.5, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load(path):
    import gzip
    path = Path(path)
    if not path.exists() and path.with_suffix(".jsonl.gz").exists():
        return [json.loads(line) for line in gzip.open(path.with_suffix(".jsonl.gz"), "rt") if line.strip()]
    return [json.loads(line) for line in open(path) if line.strip()]


def sgm(xs, shift):
    xs = list(xs)
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift


def by_inst(rows, key=lambda r: r["rule"]):
    d = collections.defaultdict(dict)
    for r in rows:
        d[r["instance"]][key(r)] = r
    return d


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 1. headroom (step 0)
def fig_headroom():
    rows = load(MIP / "results/step0.jsonl")
    classes = ["setcover", "indset", "cfl", "cauction"]
    names = {"setcover": "Set\ncover", "indset": "Indep.\nset", "cfl": "Facility\nlocation",
             "cauction": "Comb.\nauction"}
    rules = ["relpscost", "fullstrong", "pscost", "mostinf"]
    lab = {"relpscost": "SCIP default (relpscost)", "fullstrong": "Full strong branching",
           "pscost": "Pseudocost", "mostinf": "Most infeasible"}
    col = {"relpscost": C["default"], "fullstrong": C["strong"], "pscost": C["pscost"],
           "mostinf": C["trained"]}
    tab = {}
    for cl in classes:
        for ru in rules:
            rs = [r for r in rows if r["instance"].split("/")[-2] == cl and r["rule"] == ru]
            tab[(cl, ru)] = (sgm([r["nodes"] for r in rs], 10), sgm([r["time"] for r in rs], 1),
                             sum(r["status"] == "optimal" for r in rs), len(rs))
    SUMMARY["headroom"] = {f"{c}/{r}": {"nodes": round(v[0], 1), "time": round(v[1], 2),
                                        "solved": v[2], "n": v[3]} for (c, r), v in tab.items()}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    w = 0.2
    x = np.arange(len(classes))
    for k, ru in enumerate(rules):
        for ax, idx in zip(axes, (0, 1)):
            vals = [tab[(cl, ru)][idx] for cl in classes]
            ax.bar(x + (k - 1.5) * w, vals, w - 0.03, color=col[ru], label=lab[ru])
    for ax, t in zip(axes, ("Nodes (shifted geo. mean)", "Time, s (shifted geo. mean)")):
        ax.set_yscale("log")
        ax.set_xticks(x, [names[c] for c in classes])
        ax.set_title(t, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    save(fig, "headroom")


# ---------------------------------------------------------------- 2. synthetic paired: v1 vs default
def fig_synthetic_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    out = {}
    for ax, cl, title in zip(axes, ("indset", "setcover"), ("Independent set", "Set cover")):
        for f, mk, setname in (("gen_test", "o", "test"), ("gen_medium", "^", "medium")):
            d = by_inst([r for r in load(MIP / f"results/{f}.jsonl")
                         if r["instance"].split("/")[-2] == cl])
            pts = [(v["relpscost"]["time"], v["net:v1"]["time"]) for v in d.values()
                   if "relpscost" in v and "net:v1" in v]
            a = np.array(pts)
            ax.scatter(a[:, 0], a[:, 1], s=22, marker=mk, color=C["wsp"], edgecolor="white",
                       linewidth=0.6, label=f"{setname} set (n={len(a)})", zorder=3)
            out[f"{cl}/{setname}"] = {
                "time_default": round(sgm(a[:, 0], 1), 2), "time_v1": round(sgm(a[:, 1], 1), 2),
                "v1_faster": int((a[:, 1] < a[:, 0]).sum()), "n": len(a)}
        lo, hi = 1, 400
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        ax.text(hi * 0.6, hi * 0.75, "equal time", color=INK2, fontsize=7, ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("SCIP default time (s)"); ax.set_ylabel("LLM-written v1 time (s)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
    SUMMARY["synthetic_v1_vs_default"] = out
    save(fig, "synthetic_scatter")


# ---------------------------------------------------------------- 3. SB agreement (imitation quality)
def sb_ratio_of(net_json, samples):
    import sys
    sys.path.insert(0, str(MIP))
    from branching import Net  # noqa: E402
    net = Net(str(net_json))
    r = []
    for s in samples:
        x = np.array(s["features"]); sb = np.array(s["sb"])
        k = int(np.argmax(net(x)))
        r.append(sb[k] / sb.max() if sb.max() > 0 else 1.0)
    return float(np.mean(r))


def fig_agreement():
    samples = collections.defaultdict(list)
    for r in load(MIP / "results/sb_train.jsonl"):
        samples[r["instance"].split("/")[-2]].extend(r.get("diag", []))
    nets = [("ps", "Pseudocost only", C["pscost"]), ("v0", "LLM v0", C["wsp"]),
            ("v1", "LLM v1", C["wsp"]), ("trained_joint", "Trained (joint)", C["trained"])]
    per = {"setcover": "trained_setcover", "indset": "trained_indset"}
    tab = {}
    for cl in ("setcover", "indset"):
        for n, _, _ in nets:
            tab[(cl, n)] = sb_ratio_of(MIP / f"nets/{n}.json", samples[cl])
        tab[(cl, "per")] = sb_ratio_of(MIP / f"nets/{per[cl]}.json", samples[cl])
        rnd = np.mean([np.mean(np.array(s["sb"]) / max(max(s["sb"]), 1e-12)) for s in samples[cl]])
        tab[(cl, "random")] = float(rnd)
    SUMMARY["sb_agreement_train"] = {f"{c}/{n}": round(v, 3) for (c, n), v in tab.items()}
    SUMMARY["sb_train_nodes"] = {c: len(v) for c, v in samples.items()}
    order = [("random", "Random pick", "#c3c2b7")] + nets + [("per", "Trained (per class)",
                                                               C["trained"])]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    x = np.arange(2)
    w = 0.13
    for k, (n, lab, col) in enumerate(order):
        vals = [tab[(cl, n)] for cl in ("setcover", "indset")]
        hatch = "////" if n in ("v0", "per") else None
        bars = ax.bar(x + (k - 2.5) * w, vals, w - 0.02, color=col, hatch=hatch,
                      edgecolor="white", linewidth=0, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                    fontsize=6.5, color=INK2)
    ax.set_xticks(x, ["Set cover", "Independent set"])
    ax.set_ylabel("Mean SB-score ratio of pick")
    ax.set_ylim(0, 0.6)
    ax.legend(ncol=3, loc="upper left", fontsize=7)
    save(fig, "agreement")


# ---------------------------------------------------------------- 4. hand-written vs trained
def fig_hand_vs_trained():
    rules = [("relpscost", "SCIP default", C["default"], None),
             ("net:v1", "LLM-written v1", C["wsp"], None),
             ("net:trained_joint", "Trained (joint)", C["trained"], None),
             ("per", "Trained (per class)", C["trained"], "////")]
    cells = []
    tab = {}
    for f, setname in (("vs_trained_test", "test"), ("vs_trained_medium", "medium")):
        rows = load(MIP / f"results/{f}.jsonl")
        for cl in ("setcover", "indset"):
            rs = [r for r in rows if r["instance"].split("/")[-2] == cl]
            for ru, *_ in rules:
                name = f"net:trained_{cl}" if ru == "per" else ru
                sel = [r for r in rs if r["rule"] == name]
                tab[(setname, cl, ru)] = (sgm([r["nodes"] for r in sel], 10),
                                          sgm([r["time"] for r in sel], 1), len(sel))
            cells.append((setname, cl))
    SUMMARY["hand_vs_trained"] = {f"{s}/{c}/{r}": {"nodes": round(v[0]), "time": round(v[1], 2),
                                                   "n": v[2]} for (s, c, r), v in tab.items()}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(len(cells))
    w = 0.2
    for k, (ru, lab, col, hatch) in enumerate(rules):
        nodes = [tab[(s, c, ru)][0] for s, c in cells]
        rel = [tab[(s, c, ru)][1] / tab[(s, c, "relpscost")][1] for s, c in cells]
        axes[0].bar(x + (k - 1.5) * w, nodes, w - 0.03, color=col, hatch=hatch,
                    edgecolor="white", linewidth=0, label=lab)
        axes[1].bar(x + (k - 1.5) * w, rel, w - 0.03, color=col, hatch=hatch,
                    edgecolor="white", linewidth=0)
    names = [f"{c.replace('setcover', 'SC').replace('indset', 'IS')}\n{s}" for s, c in cells]
    for ax in axes:
        ax.set_xticks(x, names)
    axes[0].set_title("Nodes (shifted geo. mean)", loc="left")
    axes[1].set_title("Time relative to SCIP default (same batch)", loc="left")
    axes[1].axhline(1, color=INK2, lw=0.8, ls="--")
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
    save(fig, "hand_vs_trained")


# ---------------------------------------------------------------- 5. MIPLIB
def fig_miplib():
    rows = load(MIP / "results/miplib.jsonl") + load(MIP / "results/miplib2.jsonl")
    d = by_inst([r for r in rows if r["rule"] in ("relpscost", "net:v1")])
    items = []
    for inst, v in d.items():
        if "relpscost" not in v or "net:v1" not in v:
            continue
        a, b = v["relpscost"], v["net:v1"]
        if max(a["time"], b["time"]) < 10:
            continue
        if a["status"] != "optimal" and b["status"] != "optimal":
            continue  # both hit the limit: time ratio is 1 by construction (gaps in the table)
        name = Path(inst).name.replace(".mps.gz", "")
        items.append((name, b["time"] / a["time"], a["status"], b["status"], a, b))
    items.sort(key=lambda t: t[1])
    both = [t for t in items]
    solved_def = sum(t[4]["status"] == "optimal" for t in d.values() if False)  # placeholder
    allp = [(v["relpscost"], v["net:v1"]) for v in d.values() if "relpscost" in v and "net:v1" in v]
    bs = [(a, b) for a, b in allp if a["status"] == b["status"] == "optimal" and max(a["nodes"], b["nodes"]) > 1]
    SUMMARY["miplib"] = {
        "instances": len(allp),
        "solved_default": sum(a["status"] == "optimal" for a, _ in allp),
        "solved_v1": sum(b["status"] == "optimal" for _, b in allp),
        "time_default": round(sgm([a["time"] for a, _ in allp], 1), 1),
        "time_v1": round(sgm([b["time"] for _, b in allp], 1), 1),
        "nodes_both_solved_default": round(sgm([a["nodes"] for a, _ in bs], 10)),
        "nodes_both_solved_v1": round(sgm([b["nodes"] for _, b in bs], 10)),
        "both_solved_nontrivial": len(bs),
    }
    fig, ax = plt.subplots(figsize=(6.6, 0.2 * len(items) + 0.6))
    y = np.arange(len(items))
    cols = [C["wsp"] if r < 1 else C["default"] for _, r, *_ in items]
    ax.barh(y, [math.log2(r) for _, r, *_ in items], color=cols, height=0.7)
    ax.set_yticks(y, [f"{n}{'*' if s1 != 'optimal' or s2 != 'optimal' else ''}"
                      for n, _, s1, s2, *_ in items], fontsize=7)
    ticks = [-2, -1, 0, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks, [f"{2.0 ** t:g}×" for t in ticks])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Time of LLM-written v1 relative to SCIP default (log scale; left = v1 faster)")
    ax.text(-2.1, len(items) - 1, "v1 faster", color=C["wsp"], fontsize=7.5, va="center")
    ax.text(4.9, 0, "default faster", color=INK2, fontsize=7.5, va="center", ha="right")
    ax.grid(axis="y", visible=False)
    save(fig, "miplib")


# ---------------------------------------------------------------- 6. MARTA: feature diagnostic
def fig_marta_features():
    samples = [s for r in load(MIP / "results/marta_diag2.jsonl") for s in r.get("diag", [])]
    names = ["frac", "ps", "obj", "deg", "tight", "dualm", "val", "fconf", "alt", "rowf", "eff"]
    res = []
    for k, n in enumerate(names):
        rr = []
        for s in samples:
            x = np.array(s["features"])[:, k]; sb = np.array(s["sb"])
            p = int(np.argmax(x)); rr.append(sb[p] / sb.max() if sb.max() > 0 else 1)
        res.append(float(np.mean(rr)))
    rnd = float(np.mean([np.mean(np.array(s["sb"]) / max(max(s["sb"]), 1e-12)) for s in samples]))
    prun = float(np.mean([max(s["sb"]) >= 1e8 for s in samples]))
    SUMMARY["marta_feature_argmax_ratio"] = dict(zip(names, [round(v, 3) for v in res]))
    SUMMARY["marta_random_ratio"] = round(rnd, 3)
    SUMMARY["marta_nodes_with_prunable_child"] = round(prun, 3)
    SUMMARY["marta_diag_nodes"] = len(samples)
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    x = np.arange(len(names))
    cols = [C["wsp"] if k >= 7 else "#9fc3ef" for k in range(len(names))]
    ax.bar(x, res, 0.7, color=cols)
    ax.axhline(rnd, color=INK2, ls="--", lw=0.8, label=f"random pick ({rnd:.2f})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.set_xticks(x, names)
    ax.set_ylabel("SB-score ratio of argmax")
    ax.set_ylim(0, 0.4)
    ax.text(8, 0.36, "set-partitioning features (added)", color=C["wsp"], ha="center", fontsize=7)
    ax.text(3, 0.36, "generic features", color=INK2, ha="center", fontsize=7)
    save(fig, "marta_features")


# ---------------------------------------------------------------- 7. MARTA practice frontier
def fig_marta_practice():
    rows = load(MIP / "results/c_prac.jsonl")
    lab = {"c:relpscost": ("SCIP default", C["default"], "s"),
           "c:m3": ("net only (m3)", C["wsp"], "o"),
           "c:c_la3": ("net + SB top-3", C["look"], "^"),
           "c:c_la3i100": ("net + SB top-3, 100 it.", C["look"], "D"),
           "c:c_la8i100": ("net + SB top-8, 100 it.", C["look"], "v"),
           "c:c_la5d10": ("net + SB top-5, depth≤10", C["look"], "P")}
    pts = {}
    for ru in lab:
        rs = [r for r in rows if r["rule"] == ru]
        pts[ru] = (sgm([r["nodes"] for r in rs], 10), sgm([r["time"] for r in rs], 1))
    SUMMARY["marta_practice_c"] = {k: {"nodes": round(v[0]), "time": round(v[1], 1)}
                                   for k, v in pts.items()}
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for ru, (lb, col, mk) in lab.items():
        n, t = pts[ru]
        ax.scatter(n, t, s=46, color=col, marker=mk, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(lb, (n, t), textcoords="offset points", xytext=(6, 3), fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Nodes (shifted geo. mean, log)")
    ax.set_ylabel("Time, s (shifted geo. mean)")
    ax.set_xlim(150, 2500)
    ax.set_xticks([200, 300, 500, 1000, 2000], ["200", "300", "500", "1000", "2000"])
    ax.minorticks_off()
    save(fig, "marta_practice")


# ---------------------------------------------------------------- 8. MARTA test (decisive)
def fig_marta_test():
    ctrl = by_inst(load(MIP / "results/c_ctrl.jsonl"))
    look = by_inst(load(MIP / "results/c_test.jsonl"))
    series = [("c:pscost", ctrl, "SCIP pseudocost (no SB)", C["pscost"], "-"),
              ("c:v1", ctrl, "LLM v1 (generic)", C["wsp"], "--"),
              ("c:vm_fast", ctrl, "LLM vm_fast (MARTA)", C["wsp"], "-"),
              ("c:vm_la", look, "LLM vm_la (net + SB top-3)", C["look"], "-")]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    summ = {}
    for ru, d, lb, col, ls in series:
        ratios = sorted(v[ru]["time"] / v["c:relpscost"]["time"] for v in d.values()
                        if ru in v and "c:relpscost" in v)
        y = np.arange(1, len(ratios) + 1) / len(ratios)
        ax.step(np.log2(ratios), y, where="post", color=col, ls=ls, lw=1.8, label=lb)
        summ[ru] = {"median_time_ratio": round(float(np.median(ratios)), 3),
                    "faster_share": round(float(np.mean(np.array(ratios) < 1)), 3)}
    for name, d in (("ctrl", ctrl), ("lookahead_batch", look)):
        rules = sorted({ru for v in d.values() for ru in v})
        summ[name] = {ru: {"solved": sum(v[ru]["status"] == "optimal" for v in d.values()),
                           "time": round(sgm([v[ru]["time"] for v in d.values()], 1), 1),
                           "nodes": round(sgm([v[ru]["nodes"] for v in d.values()], 10))}
                      for ru in rules}
    SUMMARY["marta_test_c"] = summ
    ax.axvline(0, color=INK, lw=0.8)
    ticks = [-3, -2, -1, 0, 1, 2]
    ax.set_xticks(ticks, [f"{2.0 ** t:g}×" for t in ticks])
    ax.set_xlabel("Time relative to SCIP default on the same garage (log scale)")
    ax.set_ylabel("Share of 32 test garages")
    ax.legend(loc="lower right")
    save(fig, "marta_test")


# ---------------------------------------------------------------- 9. column generation
def fig_cg():
    rows = load(CG / "results/sweep1.jsonl")
    d = by_inst(rows, key=lambda r: r["strategy"])
    strat = ["full", "rc4", "rc8", "rc16", "net8"]
    lab = {"full": "no reduction", "rc4": "rc top-4", "rc8": "rc top-8", "rc16": "rc top-16",
           "net8": "LLM net top-8"}
    parts = [("t_exact", "exact pricing", C["default"]), ("t_heur", "heuristic pricing", C["wsp"]),
             ("t_lp", "master LP", C["trained"]), ("t_reduce", "graph reduction", C["strong"])]
    tot = {s: {p: sum(v[s][p] for v in d.values() if s in v) for p, *_ in parts} for s in strat}
    SUMMARY["cg_time_breakdown_s"] = {s: {p: round(x, 1) for p, x in v.items()} for s, v in tot.items()}
    SUMMARY["cg_instances"] = len(d)
    rr = [d[i]["net8"]["t_total"] / d[i]["rc8"]["t_total"] for i in d if "net8" in d[i] and "rc8" in d[i]]
    SUMMARY["cg_net8_vs_rc8_time_geomean"] = round(float(np.exp(np.mean(np.log(rr)))), 3)
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    y = np.arange(len(strat))
    left = np.zeros(len(strat))
    for p, lb, col in parts:
        vals = np.array([tot[s][p] for s in strat])
        ax.barh(y, vals, left=left, color=col, height=0.6, label=lb, edgecolor="white",
                linewidth=1)
        left += vals
    ax.set_yticks(y, [lab[s] for s in strat])
    ax.set_xlabel(f"Total time over {len(d)} XTW instances (s)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.28))
    ax.grid(axis="y", visible=False)
    save(fig, "cg_breakdown")


# ---------------------------------------------------------------- 10. MARTA garage map
def fig_marta_map():
    import csv
    import pickle
    sys_path = MIP
    trips = pickle.load(open(sys_path / "data/gtfs_marta/trips.pkl", "rb"))
    head = open(MIP / "data/marta_practice/marta_s5_06.lp").readline()
    routes = eval(head.split("routes ")[1].split(" trips")[0])  # noqa: S307 — our own header
    g = MIP / "data/gtfs_marta"
    rid = {r["route_short_name"]: r["route_id"] for r in csv.DictReader(open(g / "routes.txt"))}
    shapes = collections.defaultdict(set)
    for r in csv.DictReader(open(g / "trips.txt")):
        if r["service_id"] == "5":
            shapes[r["route_id"]].add(r["shape_id"])
    want = {s: k for k in routes for s in shapes[rid[k]]}
    pts = collections.defaultdict(list)
    for r in csv.DictReader(open(g / "shapes.txt")):
        if r["shape_id"] in want:
            pts[r["shape_id"]].append((int(r["shape_pt_sequence"]), float(r["shape_pt_lon"]),
                                       float(r["shape_pt_lat"])))
    allbus = collections.defaultdict(list)
    sample = list({s for v in shapes.values() for s in list(v)[:1]})
    keep = set(sample)
    for r in csv.DictReader(open(g / "shapes.txt")):
        if r["shape_id"] in keep and r["shape_id"] not in want:
            allbus[r["shape_id"]].append((int(r["shape_pt_sequence"]), float(r["shape_pt_lon"]),
                                          float(r["shape_pt_lat"])))
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for v in allbus.values():
        v.sort(); a = np.array(v)
        ax.plot(a[:, 1], a[:, 2], color="#d9d8d3", lw=0.5)
    cols = [C["wsp"], C["trained"], C["strong"]]
    for k, rt in enumerate(sorted(routes)):
        for s, r in want.items():
            if r != rt:
                continue
            v = sorted(pts[s]); a = np.array(v)
            ax.plot(a[:, 1], a[:, 2], color=cols[k], lw=1.4)
        ax.plot([], [], color=cols[k], lw=1.6, label=f"route {rt}")
    tr = [t for t in trips if t["service"] == "5" and t["route"] in routes]
    ends = np.array([t["to"] for t in tr] + [t["from"] for t in tr])
    ax.scatter(ends[:, 1], ends[:, 0], s=10, color=INK, zorder=4, label="terminals (relief)")
    ax.set_aspect(1 / math.cos(math.radians(33.75)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper left", fontsize=6.5, bbox_to_anchor=(0.0, 1.0))
    SUMMARY["marta_map_garage"] = {"routes": sorted(routes), "trips": len(tr)}
    save(fig, "marta_map")


if __name__ == "__main__":
    for f in (fig_headroom, fig_synthetic_scatter, fig_agreement, fig_hand_vs_trained, fig_miplib,
              fig_marta_features, fig_marta_practice, fig_marta_test, fig_cg, fig_marta_map):
        f()
        print("ok", f.__name__, flush=True)
    json.dump(SUMMARY, open(OUT / "summary.json", "w"), indent=1)


# ---------------------------------------------------------------- paired significance tests
def tests():
    from scipy.stats import wilcoxon
    out = {}

    def paired(rows, a, b, name, pred=lambda inst: True):
        d = by_inst(rows)
        pairs = [(v[a]["time"], v[b]["time"]) for i, v in d.items() if a in v and b in v and pred(i)]
        x = np.log([p[0] for p in pairs]); y = np.log([p[1] for p in pairs])
        diff = y - x
        nz = diff[np.abs(diff) > 1e-9]
        p = float(wilcoxon(nz).pvalue) if len(nz) > 5 else None
        out[name] = {"n": len(pairs), "geomean_time_ratio": round(float(np.exp(diff.mean())), 3),
                     "wilcoxon_p": None if p is None else float(f"{p:.3g}")}

    syn = load(MIP / "results/gen_test.jsonl") + load(MIP / "results/gen_medium.jsonl")
    for cl in ("indset", "setcover"):
        rows = [r for r in syn if r["instance"].split("/")[-2] == cl]
        d = collections.defaultdict(list)
        paired(rows, "relpscost", "net:v1", f"synthetic_{cl}_v1_vs_default")
    mip = load(MIP / "results/miplib.jsonl") + load(MIP / "results/miplib2.jsonl")
    paired(mip, "relpscost", "net:v1", "miplib_v1_vs_default")
    ctrl = load(MIP / "results/c_ctrl.jsonl")
    paired(ctrl, "c:relpscost", "c:vm_fast", "marta_vmfast_vs_default")
    paired(ctrl, "c:relpscost", "c:pscost", "marta_pscost_vs_default")
    paired(ctrl, "c:pscost", "c:vm_fast", "marta_vmfast_vs_pscost")
    paired(ctrl, "c:v1", "c:vm_fast", "marta_vmfast_vs_v1")
    paired(load(MIP / "results/c_test.jsonl"), "c:relpscost", "c:vm_la", "marta_vmla_vs_default")
    SUMMARY["paired_tests_time"] = out


tests()
json.dump(SUMMARY, open(OUT / "summary.json", "w"), indent=1)
