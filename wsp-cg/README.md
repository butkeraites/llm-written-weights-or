# wsp-cg — weight-space programming for column-generation pricing (POC)

Question: can an LLM *write the weights* of a small arc-scoring network (no training, no data)
that makes VRPTW column generation faster than the standard reduced-cost heuristic?
Inspired by "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, 2026) and evaluated
on the new XTW benchmark (Uchoa et al., 2026; CVRPLib).

## What is here

- `src/instance.rs` — XTW parser (EUC_2D rounded integer distances, depot = node 1).
- `src/master.rs` — restricted master LP (set covering over ng-routes) kept alive in HiGHS,
  warm-started dual simplex; column pool purged/rebuilt past `--pool × n` columns.
- `src/pricing.rs` — ng-route labeling (ng = 8), dominance on cost/time/load/ng-memory.
  Same code runs heuristic pricing (reduced graph, `--cap` labels per node) and exact pricing
  (full graph, no cap).
- `src/scorer.rs` — strategies that turn duals into a reduced graph:
  `full` (no reduction), `rc:K` (K best in/out arcs per customer by symmetric reduced cost —
  the classic expert rule), `net:FILE:K` (same selection, ranked by an MLP whose weights live in
  `FILE`, written by an LLM).
- `src/cg.rs` — the loop. Heuristic pricing first; exact pricing only when the heuristic finds
  nothing. **The final bound is exact for every strategy** — a bad network costs time, never
  correctness. Records the "misses": arcs a reduced graph dropped that exact pricing then used.
- `nets/` — hand-written weight files (the WSP artifacts).
- `scripts/summarize.py`, `scripts/misses.py` — tables and miss-feature profiles.

## Run

```bash
cargo build --release
./target/release/wsp-cg info data/xtw/*.vrp
./target/release/wsp-cg run --strategy rc:8 --strategy net:nets/v0.json:8 --cap 8 \
  --time-limit 120 --threads 6 --out results/x.jsonl data/xtw/XTW-n101.vrp
python3 scripts/summarize.py results/x.jsonl
```

Objective: distance only (the exact-method convention); `--fixed-cost F` adds a per-route cost
to approximate XTW's hierarchical objective. LP bounds are therefore ≤ BKS distance.

## Data

`data/xtw/` and `data/bks/` from CVRPLib (XTW set, released 2026-11-02).

## Findings so far (2026-11-04)

1. Arc scoring (`net:nets/v0.json:8`) ties the rc top-K rule on 34 XTW instances (n ≤ 256).
   57–82% of the time is exact pricing; heuristic failures come from the label cap, not arc choice.
2. Label eviction nets (`--label-net nets/label_v*.json`): v0 (potential bonus) made the
   heuristic fail early (bonus dominated cost when duals were large); v1/v2 (bonus as tie-break)
   approach but do not beat "keep cheapest" (exact calls 78/72 vs 71 on the 6 practice instances).
3. Headroom check: raising the cap from 8 to 64 labels per node cuts exact time only ~13% and
   multiplies heuristic time ~10×. Eviction policy cannot buy much here.
4. ng-set selection (`--ng-net nets/ng_v0.json`): drop neighbours that can never form a cycle
   (time window, reachability, capacity — ~5–93%, median ~18%, of distance-nearest ng-8 sets)
   and refill with the next feasible ones. Bound: +0.15% (ng4), +0.05% (ng8), +0.03% (ng12)
   on average. Time: first sweep suggested −17%/−21% at ng8/ng12, but a best-of-3 replicate at
   ng8 gives 0.93× time and 0.99× exact labels (deterministic) → mostly noise.

Verdict (2026-11-04): on XTW n ≤ 256 none of the three levers (arc scoring, label eviction,
ng selection) yields a material speedup; exact pricing dominates and the expert rules are near
their ceiling. Gains found are small and also reachable by a 3-line rule (no net needed).
5. Large instances (10 largest XTW, 819–1001 customers), heuristic-only CG (`--no-exact`),
   300 s + 120 s price-and-branch IP. At equal iteration the arc net reaches a lower master
   objective than rc8 on 7/10 (≥2% on 5/10, up to −8%), and both beat `full` (graph reduction
   matters at scale). But the net's unoptimized scoring eats 40–197 s, so at equal time it loses,
   and every IP is 1.5–20× the BKS: heuristic CG in 300 s is far from competitive at this scale.
   Pre-agreed criterion (≥2% on 7/10 and visible in IP) not met.
