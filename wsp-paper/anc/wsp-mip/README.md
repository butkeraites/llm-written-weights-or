# wsp-mip — weight-space programming for MILP branching (POC)

Question: can an LLM *write the weights* of a cheap branching-score network (no data, no
training) that approaches strong branching's tree sizes at pseudocost-like cost per node, and so
beats SCIP's default (reliability pseudocost branching)?

Follows the lesson of `~/dev/wsp-cg`: measure the headroom first.

## Setup

- SCIP 10 via PySCIPOpt 6.3 (`uv sync`). Settings as Gasse et al. (2019): no restarts, cuts only
  at the root, fixed random seed.
- `gen.py` — the four classic learning-to-branch classes, sized so SCIP default needs a tree:
  set cover 1000×1000 (d=0.05), combinatorial auction (simplified arbitrary-relationships,
  100 items / 1500 bids), capacitated facility location 70×70, independent set (BA, 750 nodes).
  `data/practice` (4/class, seed 1) for forward passes; `data/test` (10/class, seed 2) held out.
- `run_builtin.py` — SCIP's own rules (relpscost default, fullstrong gold standard, pscost, mostinf).
- `branching.py` / `run_net.py` — the WSP rule: 7 cheap features per candidate
  (`frac, ps, obj, deg, tight, dualm, val`) → MLP from `nets/*.json` → branch on argmax.
  `--diag K:MAX` runs strong branching at sampled nodes and logs where the net's pick ranks.
- `diag.py` — reads that log (per-feature agreement with strong branching). Read, never fitted.
- `summarize.py` — per-class solved / shifted geometric means of nodes and time.

Python overhead: the net rule computes features in Python (~ms per node), so time comparisons
against SCIP's C rules are pessimistic for the net; nodes are the clean metric.

## Step 0 — headroom (test set, 300 s)

| class | fullstrong / relpscost nodes | time |
|---|---|---|
| setcover | 0.12× | 5.1× |
| indset | 0.09× | 4.3× |
| cfl | 0.74× | 1.1× |
| cauction | 0.53× | 1.3× (trees of ~7 nodes) |

Gold standard trees are ~10× smaller on setcover/indset → real headroom. Focus there.

## Net v0 (nets/v0.json) — written by hand, one forward-pass read of SB agreement on practice

Diagnostics on practice (mean SB-score ratio of the chosen variable vs the best): pseudocost
rule 0.24 (indset) / 0.26 (setcover) → v0 0.31 / 0.41.

Held-out test set, **paired run** (SCIP default and v0 in the same batch, 4 workers each,
300 s; shifted geometric means; optima identical on every instance solved by both):

| class | nodes v0/default | time v0/default | v0 faster on |
|---|---|---|---|
| indset | 0.85× | **0.43×** | 10/10 |
| setcover | 1.46× | 0.96× | 2/10 (and 9/10 solved vs 10/10) |
| cfl | 3.39× | 0.78× | 9/10 |
| cauction | 19.3× | 0.67× | 10/10 (trees of ~7 nodes: default pays root strong branching) |

A first unpaired test run (v0 vs step-0 numbers) looked better on setcover (0.76×); the paired
run shows that was load noise. Indset is the robust win: same-or-smaller trees than the default
without any strong branching, ~2.3× faster even with Python feature extraction.

## v1 + generalization (paired, held-out)

v1 = v0 edited after reading where v0 disagreed with strong branching (less pseudocost, no obj,
+LP value). Test + medium (setcover 1250 rows, indset 1000 nodes, new seed), vs SCIP default:
indset time 0.37× on both (10/10, 8/8), nodes 0.87× / **0.56×**; setcover time ~0.9×, nodes 1.3–1.4×.

## Hand-written vs trained (same 7 features, same MLP format)

Trained = imitation of strong branching, listwise softmax CE, 2,166 labelled nodes from a separate
training set (seed 4), joint and per-class. Paired batch, nodes = shifted geo-mean:

| set / class | default | v1 (hand) | trained joint | trained per-class |
|---|---|---|---|---|
| test setcover | 3337 | 4444 | 6136 | 6261 |
| test indset | 339 | **276** | 285 | 370 |
| medium setcover | 5798 | 7087 | 9683 | 9504 |
| medium indset | 2786 | 1426 | 1700 | **1170** |

Time vs default in that batch: v1 0.55× / 0.25× / 0.65× / 0.31×; trained joint 0.60× / 0.25× /
0.72× / 0.33×; per-class 1.00× / 0.45× / 0.98× / 0.44×. Optima identical everywhere.
Hand-written weights match or beat the trained ones in 3 of 4 cells (all but medium-indset nodes)
despite the trained nets imitating strong branching better on their own data (0.44–0.46 vs 0.39).
Caveat: small trained baseline (2k nodes, same features) — not a full Gasse-style GNN.
Absolute time ratios vary between batches (load); compare within a batch, and nodes across.

## MIPLIB 2017 (40-instance subset, paired, 300 s, Gasse settings)

v1 (written for setcover/indset) applied unchanged to heterogeneous real instances, 39 readable:
solved 19 vs default 24; nodes on the 17 both-solved non-trivial instances 5080 vs 1989
(shifted geo-mean); time 131 s vs 111 s; faster on 9, slower on 12. Wins: nw04 (107→40 s), qap10
(95→35 s), mcsched, eil33-2, air05, istanbul, eilA101-2 gap 0.67→0.26, neos-662469 gap 0.49→0.001.
Losses: neos-911970 (6 s→timeout), h80x6320d, rail507, mik-250, binkar10_1, n5-3.
Optima identical where both solved. Reading: a hand-written rule is family-specific — it is not a
general-purpose replacement for relpscost. The pitch is per-family specialization.

## Real application: MARTA bus driver scheduling (`marta_gen.py`, `csrc/wspsolve.c`)

Instances from MARTA's real GTFS (blocks, trips, terminals; labour rules synthetic, 3-route
"garages"): set partitioning of trips into 1–2-piece duties. Practice 8 garages, test 32 garages
(new seed, weekday/Saturday/Sunday). Headroom: strong branching trees 10–30× smaller than default.
Added 4 set-partitioning features (fconf, alt, rowf, eff) and a net-guided lookahead (strong-branch
only the net's top-k, with SB domain reductions). Ported to a native SCIP plugin (C, SCIP 10.1) so
time is fair; SCIP factory settings; paired, 300 s.

Test (32 garages), shifted geo-mean time / solved:
relpscost 88.9 s, 24 | **pscost 57.2 s, 24** | v1 56.9 s, 25 | vm_fast (MARTA net) 56.6 s, 24 |
vm_la (net + top-3 lookahead) 93.8 s, 22. Optima identical.

Verdict: the ~35% speed-up over SCIP's default is entirely "skip strong branching on this family" —
SCIP's own plain pseudocost rule gets the same. The WSP nets (generic or MARTA-specialised) tie with
it; the net-guided lookahead halved trees on practice (272 vs 487 nodes) but overfit and lost on
test. No WSP-specific gain on a real application. Practical takeaway for the client problem: set
`branching/pscost/priority` high (one line), not a learned rule.
