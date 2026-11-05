//! Column generation loop: heuristic pricing on a reduced graph first, exact pricing on the full
//! graph only when the heuristic finds nothing. The bound is therefore exact for any strategy —
//! a bad scorer costs time, never correctness.

use crate::instance::Instance;
use crate::labels::LabelScorer;
use crate::master::{Column, Master};
use crate::pricing::{self, PricingParams};
use crate::scorer::{self, Strategy};
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Debug, Serialize)]
pub struct CgStats {
    pub instance: String,
    pub n: usize,
    pub strategy: String,
    pub status: String, // "optimal" | "timeout" | "error: ..."
    pub lp_bound: f64,
    pub iterations: usize,
    pub exact_calls: usize,
    pub heur_calls: usize,
    pub columns: usize,
    pub t_total: f64,
    pub t_lp: f64,
    pub t_reduce: f64,
    pub t_heur: f64,
    pub t_exact: f64,
    pub labels_heur: usize,
    pub labels_exact: usize,
    pub full_arcs: usize,
    pub avg_reduced_arcs: f64,
    /// LP bound trajectory: (seconds, objective) after each master solve.
    pub trace: Vec<(f64, f64)>,
    /// Arcs the reduced graph dropped but exact pricing then used (capped sample).
    pub misses: Vec<Miss>,
    pub miss_count: usize,
    pub purges: usize,
    pub misprices: usize,
    /// Each exact call: (iteration, min reduced cost found, columns found, seconds).
    pub exact_log: Vec<(usize, f64, usize, f64)>,
    pub ng_avg_size: f64,
    pub ng_wasted: f64,
    pub ip_cost: Option<f64>,
    pub ip_routes: Option<usize>,
    pub ip_status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Miss {
    pub iter: usize,
    pub i: u32,
    pub j: u32,
    pub features: [f64; 10],
}

pub struct CgParams {
    pub time_limit: Duration,
    pub ng_size: usize,
    pub cols_per_iter: usize,
    pub max_labels: usize,
    pub fixed_cost: f64,
    /// Rebuild the master when the pool exceeds pool_factor * customers columns.
    pub pool_factor: usize,
    /// Wentges smoothing factor (0 = off).
    pub alpha: f64,
    /// Heuristic pricing: max labels per node (usize::MAX = use early stop instead).
    pub bucket_cap: usize,
    /// Optional label-eviction net for heuristic pricing.
    pub label_net: Option<(String, crate::scorer::Net)>,
    /// Optional ng-set selection net (default: nearest by distance).
    pub ng_net: Option<(String, crate::scorer::Net)>,
    /// Heuristic mode: never call exact pricing; stop when the heuristic finds nothing.
    pub no_exact: bool,
    /// Seconds for the final price-and-branch IP (0 = skip).
    pub ip_time: f64,
}

pub fn run(inst: &Instance, strat: &Strategy, prm: &CgParams) -> CgStats {
    let t0 = Instant::now();
    let deadline = t0 + prm.time_limit;
    let full = pricing::full_graph(inst);
    let sc = scorer::scales(inst);
    let ng = match &prm.ng_net {
        Some((_, net)) => crate::ngsel::ng_sets_net(inst, &sc, prm.ng_size, net),
        None => pricing::ng_sets(inst, prm.ng_size),
    };
    let mut cols: Vec<Column> = inst
        .customers()
        .map(|i| {
            let route = vec![i as u32];
            Column { cost: pricing::route_cost(inst, &route) + prm.fixed_cost, route }
        })
        .collect();
    let mut st = CgStats {
        instance: inst.name.clone(),
        n: inst.n - 1,
        strategy: format!(
            "{}{}{}",
            strat.label(),
            prm.label_net.as_ref().map(|(t, _)| format!("+{t}")).unwrap_or_default(),
            match &prm.ng_net {
                Some((t, _)) => format!("+{t}{}", prm.ng_size),
                None => format!("+ng{}", prm.ng_size),
            }
        ),
        status: "timeout".into(),
        lp_bound: f64::NAN,
        iterations: 0,
        exact_calls: 0,
        heur_calls: 0,
        columns: 0,
        t_total: 0.0,
        t_lp: 0.0,
        t_reduce: 0.0,
        t_heur: 0.0,
        t_exact: 0.0,
        labels_heur: 0,
        labels_exact: 0,
        full_arcs: full.num_arcs(),
        avg_reduced_arcs: 0.0,
        trace: Vec::new(),
        misses: Vec::new(),
        miss_count: 0,
        purges: 0,
        misprices: 0,
        exact_log: Vec::new(),
        ng_avg_size: ng.iter().skip(1).map(|v| v.len() as f64).sum::<f64>() / (inst.n - 1) as f64,
        ng_wasted: crate::ngsel::wasted_fraction(inst, &sc, &ng),
        ip_cost: None,
        ip_routes: None,
        ip_status: None,
    };
    let mut reduced_arcs_acc = 0.0;
    let mut master = Master::new(inst.n, &cols);
    let mut prev_sep: Option<Vec<f64>> = None;
    let mut last_duals = vec![0.0; inst.n];
    loop {
        if Instant::now() > deadline {
            break;
        }
        st.iterations += 1;
        let t = Instant::now();
        let lp = match master.solve() {
            Ok(lp) => lp,
            Err(e) => {
                st.status = format!("error: {e}");
                break;
            }
        };
        st.t_lp += t.elapsed().as_secs_f64();
        st.lp_bound = lp.obj;
        last_duals = lp.duals.clone();
        st.trace.push((t0.elapsed().as_secs_f64(), lp.obj));

        let pp = PricingParams {
            max_cols: prm.cols_per_iter,
            max_labels: prm.max_labels,
            deadline,
            early_stop: false,
            bucket_cap: usize::MAX,
        };
        let hp = PricingParams { bucket_cap: prm.bucket_cap, early_stop: prm.bucket_cap == usize::MAX, ..pp };
        let mut new_cols: Vec<Column> = Vec::new();
        // Wentges smoothing: price first at a point between the previous separation duals and
        // the LP duals; on a mis-price fall back to the LP duals.
        let sep: Option<Vec<f64>> = match &prev_sep {
            Some(p) if prm.alpha > 0.0 => Some(
                p.iter().zip(&lp.duals).map(|(a, b)| prm.alpha * a + (1.0 - prm.alpha) * b).collect(),
            ),
            _ => None,
        };
        let mut red = None;
        for duals in [sep.as_ref(), Some(&lp.duals)].into_iter().flatten() {
            let t = Instant::now();
            let r = scorer::reduce(inst, &full, duals, &sc, strat);
            st.t_reduce += t.elapsed().as_secs_f64();
            reduced_arcs_acc += r.graph.num_arcs() as f64;
            let t = Instant::now();
            let ls = prm.label_net.as_ref().map(|(_, net)| LabelScorer::new(inst, &full, duals, &sc, net));
            let out = pricing::price(inst, &r.graph, &ng, duals, prm.fixed_cost, &hp, ls.as_ref());
            st.t_heur += t.elapsed().as_secs_f64();
            st.heur_calls += 1;
            st.labels_heur += out.labels;
            red = Some(r);
            new_cols = out
                .columns
                .into_iter()
                .filter(|c| {
                    c.cost - c.route.iter().map(|&v| lp.duals[v as usize]).sum::<f64>() < -1e-6
                })
                .collect();
            if !new_cols.is_empty() {
                prev_sep = Some(duals.clone());
                break;
            }
            st.misprices += 1;
        }
        let red = red.unwrap();
        let g = &red.graph;
        if new_cols.is_empty() && prm.no_exact {
            st.status = "heur-converged".into();
            break;
        }
        if new_cols.is_empty() {
            let t = Instant::now();
            let out = pricing::price(inst, &full, &ng, &lp.duals, prm.fixed_cost, &pp, None);
            st.t_exact += t.elapsed().as_secs_f64();
            st.exact_calls += 1;
            st.labels_exact += out.labels;
            st.exact_log.push((st.iterations, out.min_rc, out.columns.len(), t.elapsed().as_secs_f64()));
            if out.columns.is_empty() {
                if out.complete {
                    st.status = "optimal".into();
                }
                break;
            }
            new_cols = out.columns;
            prev_sep = None;
            if !matches!(strat, Strategy::Full) {
                for c in &new_cols {
                    let mut prev = 0usize;
                    for &v in &c.route {
                        let v = v as usize;
                        if prev != 0 && g.succ[prev].binary_search(&(v as u32)).is_err() {
                            st.miss_count += 1;
                            if st.misses.len() < 400 {
                                st.misses.push(Miss {
                                    iter: st.iterations,
                                    i: prev as u32,
                                    j: v as u32,
                                    features: scorer::arc_features(inst, &lp.duals, &sc, &red, prev, v),
                                });
                            }
                        }
                        prev = v;
                    }
                }
            }
        }
        cols.extend(new_cols.iter().cloned());
        let limit = prm.pool_factor * (inst.n - 1);
        if cols.len() > limit {
            // purge: keep the `limit/2` columns with lowest reduced cost under current duals
            let mut scored: Vec<(f64, usize)> = cols
                .iter()
                .enumerate()
                .map(|(r, c)| {
                    let rc = c.cost - c.route.iter().map(|&v| lp.duals[v as usize]).sum::<f64>();
                    (rc, r)
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let keep: Vec<Column> =
                scored.iter().take(limit / 2).map(|&(_, r)| cols[r].clone()).collect();
            // always keep single-customer routes so the master stays feasible
            let mut kept = keep;
            for i in inst.customers() {
                let route = vec![i as u32];
                kept.push(Column { cost: pricing::route_cost(inst, &route) + prm.fixed_cost, route });
            }
            cols = kept;
            master = Master::new(inst.n, &cols);
            st.purges += 1;
        } else {
            master.add(&new_cols);
        }
    }
    st.columns = cols.len();
    st.t_total = t0.elapsed().as_secs_f64();
    if prm.ip_time > 0.0 {
        if let Some(ip) = crate::master::solve_ip(inst.n, &cols, &last_duals, 3 * inst.n, prm.ip_time) {
            st.ip_cost = Some(ip.cost);
            st.ip_routes = Some(ip.routes);
            st.ip_status = Some(ip.status);
        }
    }
    if st.heur_calls > 0 {
        st.avg_reduced_arcs = reduced_arcs_acc / st.heur_calls as f64;
    }
    st
}
