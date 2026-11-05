//! Label scoring for heuristic pricing. When a node's bucket is full, the label with the lowest
//! score is evicted. The default rule (no net) is score = -reduced cost, i.e. keep the cheapest.
//! A label net (weights in JSON, written by an LLM) can value a label's *future*, not just its
//! cost so far.

use crate::instance::Instance;
use crate::pricing::Graph;
use crate::scorer::{Net, Scales};
use std::cell::RefCell;

pub const LABEL_FEATURES: [&str; 8] = [
    "cost",      // reduced cost so far / local_scale
    "time",      // service start at current node / H
    "rem_time",  // (H - time - s_j - d_j0) / H : slack left after returning to depot
    "load",      // load / Q
    "rem_cap",   // (Q - load) / Q
    "len",       // customers visited / 10
    "pot_time",  // rem_time_abs * rate_t(j) / local_scale  (dual gain still collectable, by time)
    "pot_cap",   // rem_cap_abs * rate_q(j) / local_scale   (dual gain still collectable, by capacity)
];

pub struct LabelScorer<'a> {
    net: &'a Net,
    rate_t: Vec<f64>,
    rate_q: Vec<f64>,
    local: f64,
    h: f64,
    q: f64,
    bufs: RefCell<(Vec<f64>, Vec<f64>)>,
}

const TOP: usize = 5;

impl<'a> LabelScorer<'a> {
    /// Precomputes, per node j, the dual gain per unit of time and per unit of capacity offered
    /// by its best few profitable successors under the current duals.
    pub fn new(inst: &Instance, full: &Graph, duals: &[f64], sc: &Scales, net: &'a Net) -> Self {
        let n = inst.n;
        let mut rate_t = vec![0.0; n];
        let mut rate_q = vec![0.0; n];
        let mut rt: Vec<f64> = Vec::new();
        let mut rq: Vec<f64> = Vec::new();
        for j in 1..n {
            rt.clear();
            rq.clear();
            for &k in &full.succ[j] {
                let k = k as usize;
                let p = duals[k] - inst.d(j, k) as f64;
                if p > 0.0 {
                    rt.push(p / ((inst.d(j, k) + inst.service[k]).max(1) as f64));
                    rq.push(p / (inst.demand[k].max(1) as f64));
                }
            }
            rt.sort_by(|a, b| b.partial_cmp(a).unwrap());
            rq.sort_by(|a, b| b.partial_cmp(a).unwrap());
            rate_t[j] = rt.iter().take(TOP).sum::<f64>() / TOP as f64;
            rate_q[j] = rq.iter().take(TOP).sum::<f64>() / TOP as f64;
        }
        LabelScorer {
            net,
            rate_t,
            rate_q,
            local: sc.local,
            h: sc.horizon,
            q: inst.capacity as f64,
            bufs: RefCell::new((Vec::new(), Vec::new())),
        }
    }

    pub fn features(
        &self,
        inst: &Instance,
        j: usize,
        cost: f64,
        time: i32,
        load: i32,
        len: u16,
    ) -> [f64; 8] {
        let rem_t = (inst.horizon() - time - inst.service[j] - inst.d(j, 0)).max(0) as f64;
        let rem_q = (inst.capacity - load).max(0) as f64;
        [
            cost / self.local,
            time as f64 / self.h,
            rem_t / self.h,
            load as f64 / self.q,
            rem_q / self.q,
            len as f64 / 10.0,
            rem_t * self.rate_t[j] / self.local,
            rem_q * self.rate_q[j] / self.local,
        ]
    }

    pub fn score(&self, inst: &Instance, j: usize, cost: f64, time: i32, load: i32, len: u16) -> f64 {
        let x = self.features(inst, j, cost, time, load, len);
        let mut b = self.bufs.borrow_mut();
        let (a, c) = &mut *b;
        self.net.forward(&x, a, c)
    }
}
