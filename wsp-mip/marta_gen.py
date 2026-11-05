"""Bus driver scheduling (crew scheduling) instances from MARTA's real GTFS feed.

Real data: trips, vehicle blocks (block_id), times and terminals come from MARTA's public GTFS.
Synthetic part: the labour rules below (typical North-American transit union rules, simplified)
and the choice of which routes form a "garage" (GTFS does not publish garage assignment).

Model (set partitioning, the classic run-cutting MILP):
  rows    = every trip of the garage's blocks, covered exactly once
  columns = candidate duties; a duty is 1 or 2 "pieces", a piece = consecutive trips of one block
            (relief = driver change at a trip boundary, at the terminal stop)
  cost    = FIXED per driver + paid minutes (+ split / tripper penalties)

Duty rules:
  piece      >= 1 trip, <= MAX_PIECE minutes, sign-on 10 min / sign-off 5 min
  straight   single piece, work in [MIN_STRAIGHT, MAX_WORK]
  2-piece    second piece starts at a terminal within RELIEF_M metres of where the first ended;
             break in [30, 90] min = meal break duty, break in (90, 300] = split duty (penalty);
             total work in [MIN_STRAIGHT, MAX_WORK], spread <= MAX_SPREAD
  tripper    any single piece shorter than MIN_STRAIGHT (part-timer), heavy penalty
             (also guarantees feasibility)
  paid       max(work, GUARANTEE)

usage: uv run python marta_gen.py OUTDIR N SEED SERVICE [ROUTES_PER_GARAGE]
  SERVICE: 5 = weekday, 3 = saturday, 4 = sunday, 28 = alternate weekday
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GTFS = Path(__file__).parent / "data" / "gtfs_marta"
CACHE = GTFS / "trips.pkl"

SIGN_ON, SIGN_OFF = 10, 5
MAX_PIECE = 330
MIN_STRAIGHT, MAX_WORK, MAX_SPREAD = 300, 540, 720
GUARANTEE = 480
FIXED = 240
SPLIT_PEN, TRIPPER_PEN = 60, 300
RELIEF_M = 400
BREAK_MEAL, BREAK_SPLIT = (30, 90), (91, 300)


def hms(s: str) -> int:
    h, m, _ = s.strip().split(":")
    return int(h) * 60 + int(m)


def load_trips():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    bus = {r["route_id"] for r in csv.DictReader(open(GTFS / "routes.txt")) if r["route_type"] == "3"}
    short = {r["route_id"]: r["route_short_name"] for r in csv.DictReader(open(GTFS / "routes.txt"))}
    stops = {r["stop_id"]: (float(r["stop_lat"]), float(r["stop_lon"]))
             for r in csv.DictReader(open(GTFS / "stops.txt"))}
    meta = {r["trip_id"]: r for r in csv.DictReader(open(GTFS / "trips.txt"))
            if r["route_id"] in bus and r["block_id"]}
    first, last = {}, {}
    for r in csv.DictReader(open(GTFS / "stop_times.txt")):
        t = r["trip_id"]
        if t not in meta:
            continue
        seq = int(r["stop_sequence"])
        if t not in first or seq < first[t][0]:
            first[t] = (seq, hms(r["departure_time"]), r["stop_id"])
        if t not in last or seq > last[t][0]:
            last[t] = (seq, hms(r["arrival_time"]), r["stop_id"])
    trips = []
    for t, r in meta.items():
        if t not in first:
            continue
        trips.append({
            "id": t, "service": r["service_id"], "route": short[r["route_id"]],
            "block": r["block_id"], "start": first[t][1], "end": last[t][1],
            "from": stops[first[t][2]], "to": stops[last[t][2]],
        })
    pickle.dump(trips, open(CACHE, "wb"))
    return trips


def dist_m(a, b) -> float:
    dy = (a[0] - b[0]) * 111_000
    dx = (a[1] - b[1]) * 111_000 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def build(trips, routes: set[str], service: str):
    blocks = defaultdict(list)
    for t in trips:
        if t["service"] == service and t["route"] in routes:
            blocks[t["block"]].append(t)
    # keep a block only if all its trips are in the garage's routes (no partial blocks)
    allb = defaultdict(set)
    for t in trips:
        if t["service"] == service:
            allb[t["block"]].add(t["route"])
    rows, pieces = [], []
    for b, ts in blocks.items():
        if not allb[b] <= routes:
            continue
        ts.sort(key=lambda t: t["start"])
        base = len(rows)
        rows.extend(ts)
        n = len(ts)
        for i in range(n):
            for j in range(i, n):
                work = ts[j]["end"] - ts[i]["start"] + SIGN_ON + SIGN_OFF
                if work > MAX_PIECE:
                    break
                pieces.append({
                    "rows": list(range(base + i, base + j + 1)),
                    "start": ts[i]["start"] - SIGN_ON, "end": ts[j]["end"] + SIGN_OFF,
                    "work": work, "from": ts[i]["from"], "to": ts[j]["to"],
                })
    return rows, pieces


def duties(pieces, rng, max_pairs_per_piece: int):
    cols = []
    for p in pieces:
        if p["work"] >= MIN_STRAIGHT:
            cols.append((p["rows"], FIXED + max(p["work"], GUARANTEE)))
        else:
            cols.append((p["rows"], FIXED + TRIPPER_PEN + max(p["work"], GUARANTEE // 2)))
    by_start = sorted(range(len(pieces)), key=lambda k: pieces[k]["start"])
    starts = np.array([pieces[k]["start"] for k in by_start])
    for p in pieces:
        lo = np.searchsorted(starts, p["end"] + BREAK_MEAL[0])
        hi = np.searchsorted(starts, p["end"] + BREAK_SPLIT[1], side="right")
        cand = []
        for idx in range(lo, hi):
            q = pieces[by_start[idx]]
            work = p["work"] + q["work"]
            if work < MIN_STRAIGHT or work > MAX_WORK or q["end"] - p["start"] > MAX_SPREAD:
                continue
            if dist_m(p["to"], q["from"]) > RELIEF_M:
                continue
            brk = q["start"] - p["end"]
            pen = SPLIT_PEN if brk > BREAK_MEAL[1] else 0
            cand.append((p["rows"] + q["rows"], FIXED + pen + max(work, GUARANTEE)))
        if len(cand) > max_pairs_per_piece:
            keep = rng.choice(len(cand), max_pairs_per_piece, replace=False)
            cand = [cand[k] for k in keep]
        cols.extend(cand)
    # dedupe identical row sets (keep the cheapest)
    best = {}
    for r, c in cols:
        key = tuple(r)
        if key not in best or c < best[key]:
            best[key] = c
    return list(best.items())


def write_lp(path: Path, nrows: int, cols, name: str) -> None:
    rows = defaultdict(list)
    for j, (r, _) in enumerate(cols):
        for i in r:
            rows[i].append(j)
    with open(path, "w") as f:
        f.write(f"\\ {name}\nminimize\nobj: ")
        f.write(" + ".join(f"{c} x{j}" for j, (_, c) in enumerate(cols)))
        f.write("\nsubject to\n")
        for i in range(nrows):
            f.write(f"t{i}: " + " + ".join(f"x{j}" for j in rows[i]) + " = 1\n")
        f.write("binary\n" + " ".join(f"x{j}" for j in range(len(cols))) + "\nend\n")


def main() -> None:
    out, n, seed, service = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    out.mkdir(parents=True, exist_ok=True)
    trips = load_trips()
    # routes whose blocks never interline with other routes would be trivial garages; sample
    # routes among those active in this service
    active = sorted({t["route"] for t in trips if t["service"] == service})
    rng = np.random.default_rng([seed, int(service)])
    made = 0
    while made < n:
        routes = set(rng.choice(active, k, replace=False).tolist())
        rows, pieces = build(trips, routes, service)
        if not 100 <= len(rows) <= 200:
            continue
        cols = duties(pieces, rng, max_pairs_per_piece=15)
        if len(cols) > 12_000:
            continue
        name = f"marta_s{service}_{made:02d}"
        write_lp(out / f"{name}.lp", len(rows), cols, f"routes {sorted(routes)} trips {len(rows)}")
        print(f"{name}: routes {sorted(routes)} trips {len(rows)} pieces {len(pieces)} duties {len(cols)}",
              flush=True)
        made += 1


if __name__ == "__main__":
    main()
