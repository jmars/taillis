#!/usr/bin/env python3
"""Which exact cell fires first in the rule's ARM C iteration order (prefill,
96 convs, K_ALL=(8,16,32))?  Iteration order: for k in K_ALL: for x sorted:
for y sorted: cells l in layer_sets[(x,y)] in ascending l; sides (full, held)."""
import os, sys
import numpy as np
os.environ["T22B_TRACE_DIR"] = "/var/lib/taillis/trace-a++/conversations"
os.environ["T22B_TRACES"] = open("/tmp/q1/conv_names.txt").read().strip()
import importlib.util
spec = importlib.util.spec_from_file_location("rule", os.path.expanduser(
    "~/futaie/taillis/tools/t22b_cross_request.py"))
rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
rule.K_ALL = (8, 16, 32)
rng = np.random.default_rng(rule.SEED)
masks = rule.draw_masks(rng)
parsed = {n: rule.parse_rows(n) for n in rule.TRACES}
dup = rule.build_dup_index(parsed)
raw = {n: rule.assemble(n, parsed[n], dup) for n in parsed}
prepped = {n: rule.prep_trace(raw[n]) for n in parsed}
r = "prefill"
names = [n for n in parsed if any(prepped[n][(r, l)]["tok_total"] >= 100
         for l in range(36) if (r, l) in prepped[n])]
names.sort()
pairs = [(x, y) for x in names for y in names if x != y]
per_k = {k: [0, None] for k in rule.K_ALL}   # [count>4, first cell in order]
for k in rule.K_ALL:
    for (x, y) in pairs:
        L = [l for l in range(36) if (r, l) in prepped[x] and (r, l) in prepped[y]
             and prepped[x][(r, l)]["tok_total"] >= 100
             and prepped[y][(r, l)]["n_fit"] >= 100 and prepped[y][(r, l)]["n_held"] >= 100]
        for l in L:
            b = prepped[y][(r, l)]
            for side in ("full", "held"):
                v = b["c"] if side == "full" else b["ch"]
                n = 4 * (b["tok_total"] if side == "full" else b["n_held"])
                rc = v[masks[l][k]].sum(axis=1) / n
                if rc.std() == 0: continue
                z = abs(rc.mean() - k / 128) / (rc.std() / np.sqrt(200))
                if z > 4.0:
                    per_k[k][0] += 1
                    if per_k[k][1] is None:
                        per_k[k][1] = (x, y, l, side, round(float(z), 3))
for k in rule.K_ALL:
    print(f"k={k}: cells>4se = {per_k[k][0]}, first in eval order = {per_k[k][1]}")
