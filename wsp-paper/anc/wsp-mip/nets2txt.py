"""Convert nets/*.json to the plain-text format read by csrc/wspsolve.

usage: uv run python nets2txt.py nets/a.json [nets/b.json ...]   (writes nets/a.txt ...)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for p in sys.argv[1:]:
    spec = json.load(open(p))
    layers = spec["layers"]
    la = spec.get("lookahead") or {}
    out = [f"{len(layers[0]['w'][0])} {len(layers)}",
           f"{la.get('k', 0)} {la.get('maxdepth', 10**9)} {la.get('itlim', 2**31 - 1)}"]
    act = {"linear": 0, "relu": 1, "tanh": 2}
    for layer in layers:
        w = layer["w"]
        out.append(f"{len(w)} {len(w[0])} {act[layer['act']]}")
        out.append(" ".join(repr(float(v)) for row in w for v in row))
        out.append(" ".join(repr(float(v)) for v in layer["b"]))
    Path(p).with_suffix(".txt").write_text("\n".join(out) + "\n")
