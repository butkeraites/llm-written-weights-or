//! ng-set selection. Any ng-sets give a valid lower bound, so this is a safe place for a net:
//! the choice trades bound strength against label explosion in *exact* pricing.
//!
//! For customer j, each candidate k is scored by a net over pair features; the `size` best with
//! score above the net's threshold join N_j (j itself is always first).

use crate::instance::Instance;
use crate::scorer::{Net, Scales};

pub const NG_FEATURES: [&str; 7] = [
    "dist",        // d(j,k) / local_scale
    "dist_rank",   // rank of k among j's neighbours by distance / 8
    "cycle_slack", // (l_k - e_k - (s_k + d(k,j) + s_j + d(j,k))) / H  : <0 => no cycle k..j..k fits
    "reach_j",     // (l_j - (e_k + s_k + d(k,j))) / H  : <0 => j unreachable after k
    "cap_slack",   // (Q - 2 q_k - q_j) / Q             : <0 => k twice + j exceeds capacity
    "tw_width_k",  // (l_k - e_k) / H
    "tw_width_j",  // (l_j - e_j) / H
];

pub fn pair_features(inst: &Instance, sc: &Scales, j: usize, k: usize, rank: usize) -> [f64; 7] {
    let h = sc.horizon;
    let q = inst.capacity as f64;
    let (dkj, djk) = (inst.d(k, j) as f64, inst.d(j, k) as f64);
    let (sk, sj) = (inst.service[k] as f64, inst.service[j] as f64);
    let (ek, lk) = (inst.tw_e[k] as f64, inst.tw_l[k] as f64);
    let (ej, lj) = (inst.tw_e[j] as f64, inst.tw_l[j] as f64);
    [
        djk / sc.local,
        rank as f64 / 8.0,
        (lk - ek - (sk + dkj + sj + djk)) / h,
        (lj - (ek + sk + dkj)) / h,
        (q - 2.0 * inst.demand[k] as f64 - inst.demand[j] as f64) / q,
        (lk - ek) / h,
        (lj - ej) / h,
    ]
}

pub fn ng_sets_net(inst: &Instance, sc: &Scales, size: usize, net: &Net) -> Vec<Vec<u32>> {
    let n = inst.n;
    let threshold = net.threshold.unwrap_or(f64::NEG_INFINITY);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut out = vec![Vec::new(); n];
    for j in 1..n {
        let mut others: Vec<usize> = (1..n).filter(|&k| k != j).collect();
        others.sort_by_key(|&k| inst.d(j, k));
        // only the 3*size nearest are candidates (keeps it O(n * size))
        let mut scored: Vec<(f64, u32)> = others
            .iter()
            .take(3 * size.max(1))
            .enumerate()
            .map(|(r, &k)| {
                let x = pair_features(inst, sc, j, k, r);
                (net.forward(&x, &mut a, &mut b), k as u32)
            })
            .filter(|&(s, _)| s > threshold)
            .collect();
        scored.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        let mut set = vec![j as u32];
        set.extend(scored.into_iter().take(size.saturating_sub(1)).map(|(_, k)| k));
        out[j] = set;
    }
    out
}

/// Fraction of distance-based ng neighbours for which no 2-cycle k..j..k fits k's window.
pub fn wasted_fraction(inst: &Instance, sc: &Scales, ng: &[Vec<u32>]) -> f64 {
    let (mut wasted, mut total) = (0usize, 0usize);
    for j in 1..inst.n {
        for &k in ng[j].iter().skip(1) {
            let x = pair_features(inst, sc, j, k as usize, 0);
            total += 1;
            if x[2] < 0.0 || x[3] < 0.0 || x[4] < 0.0 {
                wasted += 1;
            }
        }
    }
    wasted as f64 / total.max(1) as f64
}
