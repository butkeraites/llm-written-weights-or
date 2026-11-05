#!/bin/zsh
# ng-set sweep: distance vs ng_v0 net, sizes 4/8/12, on the small XTW set.
cd "$(dirname "$0")/.."
OUT=results/sweep_ng.jsonl
rm -f $OUT
for size in 4 8 12; do
  ./target/release/wsp-cg run --strategy full --cap 8 --ng $size --time-limit 120 --threads 6 --out $OUT $(cat results/small.txt)
  ./target/release/wsp-cg run --strategy full --cap 8 --ng $size --ng-net nets/ng_v0.json --time-limit 120 --threads 6 --out $OUT $(cat results/small.txt)
done
