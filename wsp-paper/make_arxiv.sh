#!/bin/zsh
# Build the arXiv upload: source + .bbl + figures + ancillary files (anc/).
set -e
cd ${0:A:h}
latexmk -pdf -interaction=nonstopmode -quiet main.tex
rm -rf build && mkdir -p build/figs
cp main.tex main.bbl build/
cp figs/*.pdf build/figs/
cp -R anc build/anc
(cd build && tar czf ../arxiv-upload.tar.gz .)
grep -n "TO-CONFIRM" main.tex && echo "WARNING: placeholders remain — fill them before uploading" || true
ls -la arxiv-upload.tar.gz
