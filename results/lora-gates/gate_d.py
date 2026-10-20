#!/usr/bin/env python3
"""GATE (d) — ALPHA ABSENT MEANS SCALE 1.0, PROVEN BY A CASE, not by reading the code.

llama.cpp's scale is `alpha ? adapter_scale*alpha/rank : adapter_scale`
(llama-graph.cpp:1625), so an adapter whose config carries NO lora_alpha merges at 1.0 —
NOT at 1/rank, and not at some other default.  This is what PEFT writes when it defaults
alpha to r, and it is the case where a tool that assumed "alpha/rank with alpha
defaulting to 1" would be wrong by exactly r.

One expert of one tensor, so the case is cheap; the oracle is the same independent numpy
reading as gate (b).  THE GATE CAN FAIL: the 1/rank arm must NOT match.
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/lora-gate")
import common as C

C.OUTDIR.mkdir(parents=True, exist_ok=True)
C.WORK.mkdir(parents=True, exist_ok=True)
LAYER, IDS, RANK = 20, [0], 8

sw = C.make_adapter("noalpha", "--layer", str(LAYER), "--experts", "0", "--rank",
                    str(RANK), "--alpha", "none")
print(f"adapter     : {sw}")
print(f"  config    : {(sw.parent / 'adapter_config.json').read_text().strip()}")

r, out = C.build(C.OUTDIR / "gate-d-noalpha.gguf", adapter=sw,
                 overrides=(f"{LAYER}:ffn_down_exps:0",))
print(f"\nbuild exit  : {r.returncode}")
if r.returncode != 0:
    print(r.stdout[-4000:] + r.stderr[-4000:])
    sys.exit(1)
C.check("alpha=absent (scale 1.0)" in r.stdout, "the tool DECLARES alpha absent, scale 1.0",
        [l.strip() for l in r.stdout.splitlines() if l.startswith("adapter")])
C.check("alpha=absent,rank=8,scale=1," in r.stdout,
        "merge_delta stamps alpha=absent,rank=8,scale=1",
        [l.strip() for l in r.stdout.splitlines() if "merge_delta" in l])

alpha, rank, scale = C.adapter_scale(sw)
C.check(alpha is None and rank == RANK and scale == 1.0,
        "the config really has no lora_alpha (read back from the file)", 
        f"alpha={alpha} rank={rank} scale={scale}")

slabs = C.Slabs(C.F16)
W = slabs.slab(f"blk.{LAYER}.ffn_down_exps.weight", IDS[0])
A, B = C.adapter_pairs(sw, LAYER, IDS[0])[C.HF["w2"]]
got = C.read_slice_f16(C.WORK / out.stem /
                       f"blk.{LAYER}.ffn_down_exps.ov0.weight.f16.gguf")
C.check(got.shape == (1, 4096, 2048), "merged slice shape", f"{got.shape}")

want_1 = (W + np.float32(1.0) * (B @ A)).astype(np.float16)
want_r = (W + np.float32(1.0 / rank) * (B @ A)).astype(np.float16)
n1, m1 = C.ulp_stats(got[0], want_1)
nr, mr = C.ulp_stats(got[0], want_r)
C.check(n1 == 0, "EXACT match to W + 1.0*B@A (the alpha-absent branch)",
        f"{n1}/{want_1.size} element(s) differ, max {m1} ulp")
C.check(nr > 0, "ASYMMETRY: W + (1/rank)*B@A MUST mismatch (the scale is read, not assumed)",
        f"{nr}/{want_r.size} element(s) differ, max {mr} ulp")
d_rms = float(np.sqrt(np.mean((want_1.astype(np.float32) - W) ** 2)))
C.check(d_rms > 0, "the delta is not a no-op (a zero delta would make both arms match)",
        f"delta rms={d_rms:.6g}")
C.finish("gate-d")
