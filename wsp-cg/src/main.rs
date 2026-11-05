mod cg;
mod instance;
mod labels;
mod master;
mod ngsel;
mod pricing;
mod scorer;

use rayon::prelude::*;
use scorer::{Net, Strategy};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn usage() -> ! {
    eprintln!(
        "usage:
  wsp-cg run [--strategy full|rc:K|net:FILE:K]... [--time-limit SECS] [--threads N]
             [--ng N] [--cols N] [--fixed-cost F] [--out FILE.jsonl] INSTANCES...
  wsp-cg info INSTANCES...      (size, horizon, BKS routes / customers per route)"
    );
    std::process::exit(2)
}

fn parse_strategy(s: &str) -> Strategy {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.as_slice() {
        ["full"] => Strategy::Full,
        ["rc", k] => Strategy::RcTopK(k.parse().unwrap_or_else(|_| usage())),
        ["net", file, k] => {
            let text = std::fs::read_to_string(file).expect("read net file");
            let net: Net = serde_json::from_str(&text).expect("parse net json");
            net.validate(scorer::FEATURES.len()).expect("net shape");
            Strategy::Net(Box::new(net), k.parse().unwrap_or_else(|_| usage()))
        }
        _ => usage(),
    }
}

fn bks_path(inst_path: &Path) -> PathBuf {
    let stem = inst_path.file_stem().unwrap().to_string_lossy().to_string();
    inst_path.parent().unwrap().parent().unwrap().join("bks").join(format!("{stem}.sol"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let cmd = args[0].as_str();
    let mut strategies = Vec::new();
    let mut time_limit = 60.0;
    let mut threads = 8;
    let mut ng = 8;
    let mut cols = 200;
    let mut fixed_cost = 0.0;
    let mut pool = 30usize;
    let mut alpha = 0.0f64;
    let mut cap = 32usize;
    let mut label_net: Option<(String, Net)> = None;
    let mut ng_net: Option<(String, Net)> = None;
    let mut no_exact = false;
    let mut ip_time = 0.0f64;
    let mut out: Option<String> = None;
    let mut files = Vec::new();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| usage());
        match a.as_str() {
            "--strategy" => strategies.push(parse_strategy(&val())),
            "--time-limit" => time_limit = val().parse().unwrap(),
            "--threads" => threads = val().parse().unwrap(),
            "--ng" => ng = val().parse().unwrap(),
            "--cols" => cols = val().parse().unwrap(),
            "--fixed-cost" => fixed_cost = val().parse().unwrap(),
            "--pool" => pool = val().parse().unwrap(),
            "--alpha" => alpha = val().parse().unwrap(),
            "--cap" => cap = val().parse().unwrap(),
            "--no-exact" => no_exact = true,
            "--ip-time" => ip_time = val().parse().unwrap(),
            "--ng-net" => {
                let f = val();
                let net: Net = serde_json::from_str(&std::fs::read_to_string(&f).expect("read ng net"))
                    .expect("parse ng net");
                net.validate(ngsel::NG_FEATURES.len()).expect("ng net shape");
                let tag = Path::new(&f).file_stem().unwrap().to_string_lossy().to_string();
                ng_net = Some((tag, net));
            }
            "--label-net" => {
                let f = val();
                let net: Net = serde_json::from_str(&std::fs::read_to_string(&f).expect("read label net"))
                    .expect("parse label net");
                net.validate(labels::LABEL_FEATURES.len()).expect("label net shape");
                let tag = Path::new(&f).file_stem().unwrap().to_string_lossy().to_string();
                label_net = Some((tag, net));
            }
            "--out" => out = Some(val()),
            f => files.push(PathBuf::from(f)),
        }
    }
    let insts: Vec<(PathBuf, instance::Instance)> = files
        .iter()
        .map(|p| (p.clone(), instance::Instance::load(p).unwrap_or_else(|e| panic!("{e}"))))
        .collect();

    match cmd {
        "info" => {
            println!("{:<12} {:>5} {:>7} {:>6} {:>6} {:>9} {:>8} {:>8}", "instance", "n", "H", "Q", "routes", "cust/rt", "bks", "ngwaste");
            for (p, inst) in &insts {
                let bp = bks_path(p);
                let (r, avg) = instance::load_bks_routes(&bp).unwrap_or((0, f64::NAN));
                let c = instance::load_bks_cost(&bp).unwrap_or(-1);
                let sc = scorer::scales(inst);
                let w = ngsel::wasted_fraction(inst, &sc, &pricing::ng_sets(inst, ng));
                println!(
                    "{:<12} {:>5} {:>7} {:>6} {:>6} {:>9.1} {:>8} {:>7.0}%",
                    inst.name, inst.n - 1, inst.horizon(), inst.capacity, r, avg, c, 100.0 * w
                );
            }
        }
        "run" => {
            if strategies.is_empty() {
                strategies.push(Strategy::Full);
            }
            let prm = cg::CgParams {
                time_limit: Duration::from_secs_f64(time_limit),
                ng_size: ng,
                cols_per_iter: cols,
                max_labels: 20_000_000,
                fixed_cost,
                pool_factor: pool,
                alpha,
                bucket_cap: if cap == 0 { usize::MAX } else { cap },
                label_net: label_net.clone(),
                ng_net: ng_net.clone(),
                no_exact,
                ip_time,
            };
            let jobs: Vec<(usize, usize)> = (0..insts.len())
                .flat_map(|i| (0..strategies.len()).map(move |s| (i, s)))
                .collect();
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let writer = std::sync::Mutex::new(out.as_ref().map(|f| {
                std::fs::OpenOptions::new().create(true).append(true).open(f).expect("open out")
            }));
            pool.install(|| {
                jobs.par_iter().for_each(|&(ii, si)| {
                    let (p, inst) = &insts[ii];
                    let st = cg::run(inst, &strategies[si], &prm);
                    let bks = instance::load_bks_cost(&bks_path(p));
                    eprintln!(
                        "{:<10} {:<6} {:<8} lb={:>10.1} ip={:>9.0}/{:<3} bks={:>7} it={:>4} exact={:>3} t={:>7.1}s (lp {:.1} red {:.1} heur {:.1} exact {:.1})",
                        st.instance, st.strategy, st.status, st.lp_bound,
                        st.ip_cost.unwrap_or(f64::NAN), st.ip_routes.unwrap_or(0),
                        bks.map(|b| b.to_string()).unwrap_or("-".into()),
                        st.iterations, st.exact_calls, st.t_total, st.t_lp, st.t_reduce, st.t_heur, st.t_exact
                    );
                    if let Some(f) = writer.lock().unwrap().as_mut() {
                        let mut v = serde_json::to_value(&st).unwrap();
                        v["bks"] = serde_json::json!(bks);
                        writeln!(f, "{}", v).unwrap();
                    }
                });
            });
        }
        _ => usage(),
    }
}
