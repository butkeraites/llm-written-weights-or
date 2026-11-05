//! Arc scoring for heuristic pricing. A strategy turns the current duals into a reduced graph.
//!
//! `Net` is the weight-space-programming hook: a tiny MLP whose weights live in a JSON file
//! that an LLM writes/edits directly (no gradient training). Features are per arc (i,j).

use crate::instance::Instance;
use crate::pricing::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const FEATURES: [&str; 10] = [
    "rc_sym",     // (d_ij - (pi_i + pi_j)/2) / local_scale
    "dist",       // d_ij / local_scale
    "pi_i",       // pi_i / (2 * dmax)
    "pi_j",       // pi_j / (2 * dmax)
    "rank_out",   // rank of j among i's successors by rc_sym, / k
    "rank_in",    // rank of i among j's predecessors by rc_sym, / k
    "tw_slack",   // (l_j - (e_i + s_i + d_ij)) / H   (>= 0 on feasible arcs)
    "min_wait",   // max(0, e_j - (l_i + s_i + d_ij)) / H  (wait even leaving i as late as possible)
    "demand_j",   // q_j / Q
    "tw_width_j", // (l_j - e_j) / H
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub w: Vec<Vec<f64>>, // [out][in]
    pub b: Vec<f64>,
    pub act: String, // "tanh" | "relu" | "linear"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Net {
    #[serde(default)]
    pub note: String,
    /// ng selection only: candidates scoring <= threshold are excluded.
    #[serde(default)]
    pub threshold: Option<f64>,
    pub layers: Vec<Layer>,
}

impl Net {
    pub fn forward(&self, x: &[f64], buf_a: &mut Vec<f64>, buf_b: &mut Vec<f64>) -> f64 {
        buf_a.clear();
        buf_a.extend_from_slice(x);
        for layer in &self.layers {
            buf_b.clear();
            for (row, bias) in layer.w.iter().zip(&layer.b) {
                let mut s = *bias;
                for (w, v) in row.iter().zip(buf_a.iter()) {
                    s += w * v;
                }
                buf_b.push(match layer.act.as_str() {
                    "tanh" => s.tanh(),
                    "relu" => s.max(0.0),
                    _ => s,
                });
            }
            std::mem::swap(buf_a, buf_b);
        }
        buf_a[0]
    }

    pub fn validate(&self, n_in: usize) -> Result<(), String> {
        let mut width = n_in;
        for (li, l) in self.layers.iter().enumerate() {
            if l.w.len() != l.b.len() {
                return Err(format!("layer {li}: w rows != b len"));
            }
            for r in &l.w {
                if r.len() != width {
                    return Err(format!("layer {li}: row width {} != {width}", r.len()));
                }
            }
            width = l.w.len();
        }
        if width != 1 {
            return Err("last layer must have 1 output".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Strategy {
    Full,
    /// Keep, per customer, the k best out-arcs and k best in-arcs by symmetric reduced cost.
    RcTopK(usize),
    /// Same selection rule, but ranking by the network's score.
    Net(Box<Net>, usize),
}

impl Strategy {
    pub fn label(&self) -> String {
        match self {
            Strategy::Full => "full".into(),
            Strategy::RcTopK(k) => format!("rc{k}"),
            Strategy::Net(_, k) => format!("net{k}"),
        }
    }
}

pub struct Scales {
    pub local: f64,
    pub dmax: f64,
    pub horizon: f64,
}

pub fn scales(inst: &Instance) -> Scales {
    let n = inst.n;
    let mut acc: f64 = 0.0;
    let mut cnt: f64 = 0.0;
    for i in 1..n {
        let mut d: Vec<i32> = (1..n).filter(|&j| j != i).map(|j| inst.d(i, j)).collect();
        d.sort_unstable();
        if let Some(v) = d.get(4.min(d.len().saturating_sub(1))) {
            acc += *v as f64;
            cnt += 1.0;
        }
    }
    let dmax = (1..n).map(|i| inst.d(0, i)).max().unwrap_or(1) as f64;
    Scales { local: (acc / cnt.max(1.0)).max(1.0), dmax: dmax.max(1.0), horizon: inst.horizon() as f64 }
}

pub struct Reduced {
    pub graph: Graph,
    pub rank_out: Vec<HashMap<u32, f64>>,
    pub rank_in: Vec<HashMap<u32, f64>>,
    pub k: usize,
}

/// The 10 features of customer arc (i,j) under the current duals (order = FEATURES).
pub fn arc_features(
    inst: &Instance,
    duals: &[f64],
    sc: &Scales,
    red: &Reduced,
    i: usize,
    j: usize,
) -> [f64; 10] {
    let h = sc.horizon;
    let kf = red.k as f64;
    let dij = inst.d(i, j) as f64;
    let ri = red.rank_out[i].get(&(j as u32)).copied().unwrap_or(1e3);
    let rj = red.rank_in[j].get(&(i as u32)).copied().unwrap_or(1e3);
    [
        (dij - 0.5 * (duals[i] + duals[j])) / sc.local,
        dij / sc.local,
        duals[i] / (2.0 * sc.dmax),
        duals[j] / (2.0 * sc.dmax),
        ri / kf,
        rj / kf,
        (inst.tw_l[j] as f64 - (inst.tw_e[i] + inst.service[i]) as f64 - dij) / h,
        ((inst.tw_e[j] as f64) - (inst.tw_l[i] + inst.service[i]) as f64 - dij).max(0.0) / h,
        inst.demand[j] as f64 / inst.capacity as f64,
        (inst.tw_l[j] - inst.tw_e[j]) as f64 / h,
    ]
}

/// Builds the reduced graph. Depot arcs are always kept; only customer->customer arcs compete.
/// Per customer, the k best out-arcs and the k best in-arcs (by score) survive.
pub fn reduce(
    inst: &Instance,
    full: &Graph,
    duals: &[f64],
    sc: &Scales,
    strat: &Strategy,
) -> Reduced {
    let n = inst.n;
    let k = match strat {
        Strategy::Full => {
            let empty = vec![HashMap::new(); n];
            return Reduced { graph: Graph { succ: full.succ.clone() }, rank_out: empty.clone(), rank_in: empty, k: 0 };
        }
        Strategy::RcTopK(k) | Strategy::Net(_, k) => *k,
    };
    let rc = |i: usize, j: usize| (inst.d(i, j) as f64 - 0.5 * (duals[i] + duals[j])) / sc.local;

    let mut rank_out: Vec<HashMap<u32, f64>> = vec![HashMap::new(); n];
    let mut preds: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for i in 1..n {
        let mut v: Vec<(u32, f64)> =
            full.succ[i].iter().map(|&j| (j, rc(i, j as usize))).collect();
        for &(j, r) in &v {
            preds[j as usize].push((i as u32, r));
        }
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rank_out[i] = v.iter().enumerate().map(|(r, &(j, _))| (j, r as f64)).collect();
    }
    let mut rank_in: Vec<HashMap<u32, f64>> = vec![HashMap::new(); n];
    for j in 1..n {
        preds[j].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rank_in[j] = preds[j].iter().enumerate().map(|(r, &(i, _))| (i, r as f64)).collect();
    }
    let mut red = Reduced { graph: Graph { succ: Vec::new() }, rank_out, rank_in, k };

    let mut scored_out: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut scored_in: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    for i in 1..n {
        for &j32 in &full.succ[i] {
            let j = j32 as usize;
            let s = match strat {
                Strategy::RcTopK(_) => -rc(i, j),
                Strategy::Net(net, _) => {
                    let x = arc_features(inst, duals, sc, &red, i, j);
                    net.forward(&x, &mut ba, &mut bb)
                }
                Strategy::Full => unreachable!(),
            };
            scored_out[i].push((j32, s));
            scored_in[j].push((i as u32, s));
        }
    }
    let mut keep = vec![Vec::<u32>::new(); n];
    keep[0] = full.succ[0].clone();
    for i in 1..n {
        let v = &mut scored_out[i];
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        keep[i].extend(v.iter().take(k).map(|&(j, _)| j));
    }
    for j in 1..n {
        let v = &mut scored_in[j];
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(i, _) in v.iter().take(k) {
            keep[i as usize].push(j as u32);
        }
    }
    for s in keep.iter_mut() {
        s.sort_unstable();
        s.dedup();
    }
    red.graph = Graph { succ: keep };
    red
}
