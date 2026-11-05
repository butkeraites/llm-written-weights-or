//! ESPPRC pricing relaxed to ng-routes: mono-directional labeling with dominance on
//! (reduced cost, time, load, ng-memory). Runs on any arc subset, so the same code does
//! heuristic pricing (reduced graph) and exact pricing (full feasible graph).

use crate::instance::Instance;
use crate::labels::LabelScorer;
use crate::master::Column;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

pub struct Graph {
    /// succ[i] = customers reachable from i (depot = 0 included as a source).
    pub succ: Vec<Vec<u32>>,
}

impl Graph {
    pub fn num_arcs(&self) -> usize {
        self.succ.iter().map(|s| s.len()).sum()
    }
}

/// All individually feasible arcs.
pub fn full_graph(inst: &Instance) -> Graph {
    let n = inst.n;
    let mut succ = vec![Vec::new(); n];
    for i in 0..n {
        for j in 1..n {
            if inst.arc_feasible(i, j) {
                succ[i].push(j as u32);
            }
        }
    }
    Graph { succ }
}

/// ng-neighbourhoods: each customer plus its (ng-1) nearest customers.
pub fn ng_sets(inst: &Instance, size: usize) -> Vec<Vec<u32>> {
    let n = inst.n;
    let mut out = vec![Vec::new(); n];
    for i in 1..n {
        let mut others: Vec<u32> = (1..n as u32).filter(|&j| j as usize != i).collect();
        others.sort_by_key(|&j| inst.d(i, j as usize));
        let mut set = vec![i as u32];
        set.extend(others.into_iter().take(size.saturating_sub(1)));
        out[i] = set;
    }
    out
}

#[derive(Clone, Copy)]
struct Label {
    node: u32,
    parent: u32,
    time: i32,
    load: i32,
    mem: u32,
    alive: bool,
    len: u16,
    cost: f64,
    /// Eviction score in heuristic mode (higher = keep).
    score: f64,
}

pub struct PricingOut {
    pub columns: Vec<Column>,
    pub labels: usize,
    pub complete: bool,
    pub min_rc: f64,
}

pub struct PricingParams {
    pub max_cols: usize,
    pub max_labels: usize,
    pub deadline: Instant,
    /// Heuristic mode: stop as soon as `max_cols` negative routes were completed.
    pub early_stop: bool,
    /// Heuristic mode: keep at most this many (cheapest) labels per node.
    pub bucket_cap: usize,
}

const EPS: f64 = 1e-7;

pub fn price(
    inst: &Instance,
    g: &Graph,
    ng: &[Vec<u32>],
    duals: &[f64],
    fixed_cost: f64,
    prm: &PricingParams,
    ls: Option<&LabelScorer>,
) -> PricingOut {
    let n = inst.n;
    let horizon = inst.horizon();
    let mut arena: Vec<Label> = Vec::with_capacity(1 << 16);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
    arena.push(Label {
        node: 0,
        parent: u32::MAX,
        time: inst.tw_e[0],
        load: 0,
        mem: 0,
        alive: true,
        len: 0,
        cost: fixed_cost,
        score: 0.0,
    });
    heap.push(Reverse((0, 0)));
    let mut finished: Vec<(f64, u32)> = Vec::new();
    let mut complete = true;
    let mut min_rc = f64::INFINITY;
    let mut popped = 0usize;

    while let Some(Reverse((_, idx))) = heap.pop() {
        let l = arena[idx as usize];
        if !l.alive {
            continue;
        }
        popped += 1;
        if popped % 4096 == 0 && Instant::now() > prm.deadline {
            complete = false;
            break;
        }
        if arena.len() > prm.max_labels {
            complete = false;
            break;
        }
        let i = l.node as usize;
        // close the route
        if i != 0 {
            let rc = l.cost + inst.d(i, 0) as f64;
            if rc < min_rc {
                min_rc = rc;
            }
            if rc < -EPS {
                finished.push((rc, idx));
                if prm.early_stop && finished.len() >= prm.max_cols {
                    complete = false;
                    break;
                }
            }
        }
        let depart = l.time + inst.service[i];
        for &j32 in &g.succ[i] {
            let j = j32 as usize;
            // ng check: j forbidden if it is in i's memory
            if i != 0 {
                if let Some(p) = ng[i].iter().position(|&v| v == j32) {
                    if l.mem & (1 << p) != 0 {
                        continue;
                    }
                }
            }
            let load = l.load + inst.demand[j];
            if load > inst.capacity {
                continue;
            }
            let t = (depart + inst.d(i, j)).max(inst.tw_e[j]);
            if t > inst.tw_l[j] || t + inst.service[j] + inst.d(j, 0) > horizon {
                continue;
            }
            let cost = l.cost + inst.d(i, j) as f64 - duals[j];
            // new memory over ng[j]
            let mut mem = 1u32; // position 0 is j itself
            for (p, &k) in ng[j].iter().enumerate().skip(1) {
                if i != 0 {
                    if let Some(q) = ng[i].iter().position(|&v| v == k) {
                        if l.mem & (1 << q) != 0 {
                            mem |= 1 << p;
                        }
                    }
                }
            }
            // dominance
            let bucket = &mut buckets[j];
            let mut dominated = false;
            let mut w = 0;
            for r in 0..bucket.len() {
                let mi = bucket[r] as usize;
                let m = arena[mi];
                if !dominated
                    && m.cost <= cost + EPS
                    && m.time <= t
                    && m.load <= load
                    && (m.mem & !mem) == 0
                {
                    dominated = true;
                }
                if !dominated
                    && cost <= m.cost + EPS
                    && t <= m.time
                    && load <= m.load
                    && (mem & !m.mem) == 0
                {
                    arena[mi].alive = false;
                    continue;
                }
                bucket[w] = bucket[r];
                w += 1;
            }
            bucket.truncate(w);
            if dominated {
                continue;
            }
            let len = l.len + 1;
            let score = if bucket.len() + 1 >= prm.bucket_cap {
                match ls {
                    Some(sc) => sc.score(inst, j, cost, t, load, len),
                    None => -cost,
                }
            } else {
                f64::NAN // lazily computed only when the bucket gets full
            };
            if bucket.len() >= prm.bucket_cap {
                // make sure every label in the bucket has a score
                for &b in bucket.iter() {
                    let lb = &mut arena[b as usize];
                    if lb.score.is_nan() {
                        lb.score = match ls {
                            Some(sc) => sc.score(inst, j, lb.cost, lb.time, lb.load, lb.len),
                            None => -lb.cost,
                        };
                    }
                }
                let (wpos, wscore) = bucket
                    .iter()
                    .enumerate()
                    .map(|(p, &b)| (p, arena[b as usize].score))
                    .fold((usize::MAX, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
                if score <= wscore {
                    continue;
                }
                arena[bucket[wpos] as usize].alive = false;
                bucket.swap_remove(wpos);
            }
            let nidx = arena.len() as u32;
            arena.push(Label { node: j32, parent: idx, time: t, load, mem, alive: true, len, cost, score });
            bucket.push(nidx);
            heap.push(Reverse((t, nidx)));
        }
    }

    finished.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut columns = Vec::new();
    for (_, idx) in finished {
        let mut route = Vec::new();
        let mut cur = idx;
        while cur != u32::MAX && arena[cur as usize].node != 0 {
            route.push(arena[cur as usize].node);
            cur = arena[cur as usize].parent;
        }
        route.reverse();
        if seen.insert(route.clone()) {
            let cost = route_cost(inst, &route) + fixed_cost;
            columns.push(Column { cost, route });
            if columns.len() >= prm.max_cols {
                break;
            }
        }
    }
    PricingOut { columns, labels: arena.len(), complete, min_rc }
}

pub fn route_cost(inst: &Instance, route: &[u32]) -> f64 {
    let mut prev = 0usize;
    let mut c = 0i64;
    for &v in route {
        c += inst.d(prev, v as usize) as i64;
        prev = v as usize;
    }
    c += inst.d(prev, 0) as i64;
    c as f64
}
