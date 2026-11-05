#!/bin/zsh
# Paired generalization run: SCIP default vs v0 vs v1 on test (300 s) and medium (600 s).
cd "$(dirname "$0")"
T="$(ls data/test/setcover/*.lp data/test/indset/*.lp)"
M="$(ls data/medium/*/*.lp)"
uv run python run_builtin.py results/gen_test.jsonl 300 3 relpscost ${=T} > /dev/null 2>&1 &
uv run python run_net.py results/gen_test.jsonl 300 5 nets/v0.json,nets/v1.json ${=T} > /dev/null 2>&1
wait
uv run python run_builtin.py results/gen_medium.jsonl 600 3 relpscost ${=M} > /dev/null 2>&1 &
uv run python run_net.py results/gen_medium.jsonl 600 5 nets/v0.json,nets/v1.json ${=M} > /dev/null 2>&1
wait
