#!/usr/bin/env python3
"""Q1 preflight: compute EVERY fail() abort condition of t22b_cross_request.py
for a given trace set at K_ALL=(8,16,32), using the rule's own code paths and
the exact masks the run will draw (same SEED, same K_ALL -> same masks).
Abort conditions are deterministic functions of (masks, count vectors)."""
import os, sys
import numpy as np

os.environ["T22B_TRACE_DIR"] = sys.argv[1]
os.environ["T22B_TRACES"] = sys.argv[2]
label = sys.argv[3]

import importlib.util
spec = importlib.util.spec_from_file_location("rule", os.path.expanduser(
    "~/futaie/taillis/tools/t22b_cross_request.py"))
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)
rule.K_ALL = (8, 16, 32)          # same restriction as the census runs -> same masks

rng = np.random.default_rng(rule.SEED)
masks = rule.draw_masks(rng)

parsed = {}
for n in rule.TRACES:
    p = rule.parse_rows(n)
    assert not p["corrupt"], n
    parsed[n] = p
used = tuple(parsed)
dup = rule.build_dup_index(parsed)
raw = {n: rule.assemble(n, parsed[n], dup) for n in used}
prepped = {n: rule.prep_trace(raw[n]) for n in used}
N_PERM, NE, MIN = rule.N_PERM, rule.N_EXPERTS, rule.MIN_LAYER_TOKENS

worst = {"capture": (0.0, None), "ceiling": (0.0, None), "overlap": (0.0, None)}
zero_std = []
for r in rule.REGIMES:
    names = [n for n in used if any(prepped[n][(r, l)]["tok_total"] >= MIN
             for l in range(rule.N_LAYERS) if (r, l) in prepped[n])]
    print(f"{label} regime {r}: eligible={len(names)}/{len(used)}")
    if len(names) < 2:
        continue
    pairs = [(x, y) for x in names for y in names if x != y]
    layer_sets = {}
    for (x, y) in pairs:
        layer_sets[(x, y)] = [l for l in range(rule.N_LAYERS)
                              if (r, l) in prepped[x] and (r, l) in prepped[y]
                              and prepped[x][(r, l)]["tok_total"] >= MIN
                              and prepped[y][(r, l)]["n_fit"] >= MIN
                              and prepped[y][(r, l)]["n_held"] >= MIN]
    # ARM C capture controls + ceiling: z depends on eval trace y only
    for y in names:
        for l in range(rule.N_LAYERS):
            if (r, l) not in prepped[y]:
                continue
            b = prepped[y][(r, l)]
            nsy, nhy = 4 * b["tok_total"], 4 * b["n_held"]
            for k in rule.K_ALL:
                M = masks[l][k]
                rc = b["c"][M].sum(axis=1) / nsy
                if rc.std() == 0: zero_std.append((r, y, l, k, "full"))
                else:
                    z = abs(rc.mean() - k / NE) / (rc.std() / np.sqrt(N_PERM))
                    if z > worst["capture"][0]: worst["capture"] = (z, (r, y, l, k, "full"))
                rw = b["ch"][M].sum(axis=1) / nhy
                if rw.std() == 0: zero_std.append((r, y, l, k, "held"))
                else:
                    z = abs(rw.mean() - k / NE) / (rw.std() / np.sqrt(N_PERM))
                    if z > worst["capture"][0]: worst["capture"] = (z, (r, y, l, k, "held"))
                    if k in (8, 16) and z > worst["ceiling"][0]:
                        worst["ceiling"] = (z, (r, y, l, k))
    # ARM B pooled overlap null per unordered pair
    for k in rule.K_ALL:
        for (x, y) in pairs:
            if x > y: continue
            L = layer_sets[(x, y)]
            if not L: continue
            null_ov = np.empty((len(L), N_PERM))
            for i, l in enumerate(L):
                B = prepped[y][(r, l)]["order"][:k]
                Bm = np.zeros(NE, dtype=bool); Bm[B] = True
                null_ov[i] = Bm[masks[l][k]].sum(axis=1) / k
            nmean = float(null_ov.mean(axis=0).mean())
            nse = float(null_ov.mean(axis=0).std()) / np.sqrt(N_PERM)
            if nse > 0:
                z = abs(nmean - k / NE) / nse
                if z > worst["overlap"][0]: worst["overlap"] = (z, (r, k, x, y, len(L)))
print(f"{label} zero-std control cells: {len(zero_std)}")
for key, (z, where) in worst.items():
    print(f"{label} max |z| {key}: {z:.3f} at {where}")
print(f"{label} ABORT-FREE at k<=32: {all(w[0] < 4.0 for w in worst.values()) and not zero_std}")
