#!/bin/zsh
# Paired: SCIP default vs hand-written v1 vs trained (joint, per-class) on test (300 s) and medium (600 s).
cd "$(dirname "$0")"
for set in test medium; do
  L=300; [ $set = medium ] && L=600
  OUT=results/vs_trained_$set.jsonl
  uv run python run_builtin.py $OUT $L 2 relpscost data/$set/setcover/*.lp data/$set/indset/*.lp > /dev/null 2>&1 &
  uv run python run_net.py $OUT $L 2 nets/v1.json,nets/trained_joint.json data/$set/setcover/*.lp data/$set/indset/*.lp > /dev/null 2>&1 &
  uv run python run_net.py $OUT $L 2 nets/trained_setcover.json data/$set/setcover/*.lp > /dev/null 2>&1 &
  uv run python run_net.py $OUT $L 2 nets/trained_indset.json data/$set/indset/*.lp > /dev/null 2>&1 &
  wait
done
