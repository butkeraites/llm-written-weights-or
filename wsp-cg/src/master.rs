//! Restricted master LP (set covering over ng-routes), kept alive in HiGHS between iterations so
//! each re-solve warm-starts from the previous basis (dual simplex after adding columns).

use highs::{ColProblem, HighsModelStatus, Model, Row, Sense, SolvedModel};

#[derive(Debug, Clone)]
pub struct Column {
    pub cost: f64,
    /// Customer sequence (node indices, depot excluded).
    pub route: Vec<u32>,
}

pub struct LpResult {
    pub obj: f64,
    /// Dual of each node's covering row; index 0 (depot) is 0.
    pub duals: Vec<f64>,
}

pub struct Master {
    n_nodes: usize,
    rows: Vec<Row>,
    model: Option<Model>,
    pub num_cols: usize,
}

fn coefficients(rows: &[Row], route: &[u32]) -> Vec<(Row, f64)> {
    let mut sorted: Vec<usize> = route.iter().map(|&v| v as usize - 1).collect();
    sorted.sort_unstable();
    let mut out: Vec<(usize, f64)> = Vec::new();
    for r in sorted {
        match out.last_mut() {
            Some((last, c)) if *last == r => *c += 1.0,
            _ => out.push((r, 1.0)),
        }
    }
    out.into_iter().map(|(r, c)| (rows[r], c)).collect()
}

impl Master {
    /// min sum c_r x_r  s.t.  sum_r a_ir x_r >= 1 (i customers), x >= 0.
    pub fn new(n_nodes: usize, initial: &[Column]) -> Self {
        let mut pb = ColProblem::default();
        let rows: Vec<Row> = (1..n_nodes).map(|_| pb.add_row(1.0..)).collect();
        for c in initial {
            pb.add_column(c.cost, 0.0.., coefficients(&rows, &c.route));
        }
        let mut model = pb.optimise(Sense::Minimise);
        model.make_quiet();
        model.set_option("threads", 1);
        model.set_option("presolve", "off");
        model.set_option("solver", "simplex");
        model.set_option("simplex_strategy", std::env::var("WSP_SIMPLEX").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(4));
        Master { n_nodes, rows, model: Some(model), num_cols: initial.len() }
    }

    pub fn add(&mut self, cols: &[Column]) {
        let model = self.model.as_mut().unwrap();
        for c in cols {
            model.add_col(c.cost, 0.0.., coefficients(&self.rows, &c.route));
        }
        self.num_cols += cols.len();
    }

    pub fn solve(&mut self) -> Result<LpResult, String> {
        let model = self.model.take().unwrap();
        let solved: SolvedModel = model.try_solve().map_err(|e| format!("highs: {e:?}"))?;
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            return Err(format!("LP status {status:?}"));
        }
        let obj = solved.objective_value();
        let sol = solved.get_solution();
        let mut duals = vec![0.0; self.n_nodes];
        for (i, d) in sol.dual_rows().iter().enumerate() {
            duals[i + 1] = d.max(0.0);
        }
        self.model = Some(solved.into());
        Ok(LpResult { obj, duals })
    }
}

pub struct IpResult {
    pub cost: f64,
    pub routes: usize,
    pub status: String,
}

/// Price-and-branch: the restricted master as a set-covering IP over elementary columns.
pub fn solve_ip(
    n_nodes: usize,
    cols: &[Column],
    duals: &[f64],
    max_cols: usize,
    time_limit: f64,
) -> Option<IpResult> {
    let mut order: Vec<(f64, usize)> = cols
        .iter()
        .enumerate()
        .map(|(r, c)| (c.cost - c.route.iter().map(|&v| duals[v as usize]).sum::<f64>(), r))
        .collect();
    order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut pick: Vec<&Column> = order.iter().take(max_cols).map(|&(_, r)| &cols[r]).collect();
    pick.extend(cols.iter().filter(|c| c.route.len() == 1)); // keep it feasible
    let cols = pick;
    let mut pb = ColProblem::default();
    let rows: Vec<Row> = (1..n_nodes).map(|_| pb.add_row(1.0..)).collect();
    let mut used = Vec::new();
    for c in cols {
        let mut seen = c.route.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != c.route.len() {
            continue; // ng-route with a cycle: not a real route
        }
        pb.add_integer_column(c.cost, 0.0..=1.0, coefficients(&rows, &c.route));
        used.push(c);
    }
    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1);
    model.set_option("time_limit", time_limit);
    model.set_option("mip_rel_gap", 1e-4);
    let solved = model.try_solve().ok()?;
    let status = format!("{:?}", solved.status());
    let sol = solved.get_solution();
    let x = sol.columns();
    if x.is_empty() {
        return None;
    }
    let mut cost = 0.0;
    let mut routes = 0;
    for (v, c) in x.iter().zip(&used) {
        if *v > 0.5 {
            cost += c.cost;
            routes += 1;
        }
    }
    Some(IpResult { cost, routes, status })
}
