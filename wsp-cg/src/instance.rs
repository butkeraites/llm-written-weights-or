//! XTW instance parsing (TSPLIB-style VRPTW, EUC_2D rounded distances, depot = node 1).

use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Instance {
    pub name: String,
    /// Number of nodes including the depot (index 0).
    pub n: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub demand: Vec<i32>,
    pub service: Vec<i32>,
    pub tw_e: Vec<i32>,
    pub tw_l: Vec<i32>,
    pub capacity: i32,
    /// Row-major n*n integer travel distances/times.
    pub dist: Vec<i32>,
}

impl Instance {
    #[inline]
    pub fn d(&self, i: usize, j: usize) -> i32 {
        self.dist[i * self.n + j]
    }

    pub fn horizon(&self) -> i32 {
        self.tw_l[0]
    }

    pub fn customers(&self) -> std::ops::Range<usize> {
        1..self.n
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut name = String::new();
        let mut n = 0usize;
        let mut capacity = 0i32;
        let mut section = "";
        let (mut x, mut y, mut demand, mut service, mut tw_e, mut tw_l) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                let k = k.trim();
                let v = v.trim();
                match k {
                    "NAME" => name = v.to_string(),
                    "DIMENSION" => {
                        n = v.parse().map_err(|_| "bad DIMENSION")?;
                        x = vec![0.0; n];
                        y = vec![0.0; n];
                        demand = vec![0; n];
                        service = vec![0; n];
                        tw_e = vec![0; n];
                        tw_l = vec![0; n];
                    }
                    "CAPACITY" => capacity = v.parse().map_err(|_| "bad CAPACITY")?,
                    _ => {}
                }
                continue;
            }
            if line.ends_with("SECTION") {
                section = match line {
                    "NODE_COORD_SECTION" => "coord",
                    "DEMAND_SECTION" => "demand",
                    "SERVICE_TIME_SECTION" => "service",
                    "TIME_WINDOW_SECTION" => "tw",
                    "DEPOT_SECTION" => "depot",
                    _ => "",
                };
                continue;
            }
            if line == "EOF" {
                break;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            let num = |s: &str| -> Result<i64, String> {
                s.parse::<i64>().map_err(|_| format!("bad number '{s}' in {}", path.display()))
            };
            match section {
                "coord" => {
                    let i = num(f[0])? as usize - 1;
                    x[i] = num(f[1])? as f64;
                    y[i] = num(f[2])? as f64;
                }
                "demand" => demand[num(f[0])? as usize - 1] = num(f[1])? as i32,
                "service" => service[num(f[0])? as usize - 1] = num(f[1])? as i32,
                "tw" => {
                    let i = num(f[0])? as usize - 1;
                    tw_e[i] = num(f[1])? as i32;
                    tw_l[i] = num(f[2])? as i32;
                }
                "depot" => {
                    let v = num(f[0])?;
                    if v != -1 && v != 1 {
                        return Err(format!("{}: depot must be node 1", path.display()));
                    }
                }
                _ => {}
            }
        }
        if n == 0 {
            return Err(format!("{}: no DIMENSION", path.display()));
        }
        let mut dist = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = x[i] - x[j];
                let dy = y[i] - y[j];
                dist[i * n + j] = ((dx * dx + dy * dy).sqrt() + 0.5).floor() as i32;
            }
        }
        Ok(Instance { name, n, x, y, demand, service, tw_e, tw_l, capacity, dist })
    }

    /// Arc (i,j) between customers is time-feasible on its own.
    pub fn arc_feasible(&self, i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }
        if i != 0 && j != 0 && self.demand[i] + self.demand[j] > self.capacity {
            return false;
        }
        let arrive = self.tw_e[i] + self.service[i] + self.d(i, j);
        if arrive > self.tw_l[j] {
            return false;
        }
        if j != 0 {
            let start = arrive.max(self.tw_e[j]);
            if start + self.service[j] + self.d(j, 0) > self.horizon() {
                return false;
            }
        }
        true
    }
}

/// Reads the `Cost N` line of a CVRPLib .sol file.
pub fn load_bks_cost(path: &Path) -> Option<i64> {
    let text = fs::read_to_string(path).ok()?;
    text.lines()
        .find_map(|l| l.trim().strip_prefix("Cost").map(|v| v.trim().parse::<i64>().ok()))
        .flatten()
}

/// Number of routes and average customers per route of a .sol file.
pub fn load_bks_routes(path: &Path) -> Option<(usize, f64)> {
    let text = fs::read_to_string(path).ok()?;
    let lens: Vec<usize> = text
        .lines()
        .filter(|l| l.starts_with("Route"))
        .map(|l| l.split_once(':').map(|(_, r)| r.split_whitespace().count()).unwrap_or(0))
        .collect();
    if lens.is_empty() {
        return None;
    }
    let total: usize = lens.iter().sum();
    Some((lens.len(), total as f64 / lens.len() as f64))
}
