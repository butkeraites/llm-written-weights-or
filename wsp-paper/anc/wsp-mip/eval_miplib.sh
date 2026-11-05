#!/bin/zsh
# Paired MIPLIB run: SCIP default (relpscost) and the hand-written v1 net, same batch.
cd ${0:A:h}
F=(data/miplib/*.mps.gz)
uv run python run_builtin.py results/miplib.jsonl 300 4 relpscost $F > results/miplib_builtin.log 2>&1 &
uv run python run_net.py results/miplib.jsonl 300 4 nets/v1.json $F > results/miplib_net.log 2>&1 &
wait
