#!/usr/bin/env python3
"""Hand-build a PEFT-format safetensors ADAPTER with KNOWN factors, for the LORA gates.

No trainer exists in this project, so a gate adapter is synthetic BY CONSTRUCTION and its
factors are written with a fixed seed — the oracle recomputes B@A from these same bytes,
independently of the tool.

Expert-case module paths use the checkpoint's own spelling (layout.md §6, `w1/w2/w3`);
the dense case uses the HF spelling under BOTH PEFT's `base_model.model.` prefix and the
checkpoint's `language_model.model.` prefix, so the prefix branches are exercised too.

  make_adapter.py --out DIR --layer 20 --experts 0,1 --rank 4 --alpha 2.0
                  [--zero-b] [--proj w1,w3,w2] [--dense] [--seed 7]
                  [--broken shared|b-out|rank|dup|incomplete|garbage|rslora|dora|alpha-pattern]
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file

PREFIX = "base_model.model.language_model.model."
HF = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
NE0 = {"gate_up": 4096, "down": 2048}
HALF = 2048          # w1 and w3 each own 2048 rows of the fused 4096-row out axis
OUT = {"w1": HALF, "w3": HALF, "w2": 4096}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--experts", required=True)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", default="2.0", help="float, or 'none' to omit lora_alpha")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--zero-b", action="store_true",
                   help="lora_B is ALL ZEROS (gate a: the merge must be a no-op)")
    p.add_argument("--proj", default="w1,w3,w2")
    p.add_argument("--dense", action="store_true",
                   help="also carry dense-module pairs the overlay path cannot express")
    p.add_argument("--broken", default=None,
                   choices=("shared", "b-out", "rank", "rank0", "ndim", "dup",
                            "incomplete", "garbage", "rslora", "dora", "alpha-pattern"),
                   help="emit a DELIBERATELY refused adapter, one failure class each")
    a = p.parse_args()

    experts = [int(x) for x in a.experts.split(",")]
    projs = [x for x in a.proj.split(",") if x]
    rng = np.random.default_rng(a.seed)
    T = {}

    def pair(name, nin, nout, rank):
        # PEFT: lora_A.weight is (r, in), lora_B.weight is (out, r).
        A = (rng.uniform(-1, 1, size=(rank, nin)) * 0.05).astype(np.float32)
        B = (rng.uniform(-1, 1, size=(nout, rank)) * 0.05).astype(np.float32)
        if a.zero_b:
            B = np.zeros_like(B)
        T[name + ".lora_A.weight"] = A
        T[name + ".lora_B.weight"] = B

    for e in experts:
        for pr in projs:
            nout = OUT[pr]
            if a.broken == "b-out" and pr == "w1" and e == experts[0]:
                nout = HALF + 32              # wrong out axis -> lora:shape-mismatch
            rank = a.rank + (1 if (a.broken == "rank" and e == experts[-1]) else 0)
            pair(f"{PREFIX}layers.{a.layer}.mlp.experts.{e}.{HF[pr]}",
                 NE0["down" if pr == "w2" else "gate_up"], nout, rank)
            if a.broken == "dup" and e == experts[0] and pr == "w1":
                # the SAME (expert, projection) under the checkpoint's own spelling: BOTH
                # factors, so whichever order they are read in, one is a duplicate
                base = f"{PREFIX}layers.{a.layer}.mlp.experts.{e}.{HF[pr]}"
                for f in (".lora_A.weight", ".lora_B.weight"):
                    T[f"layers.{a.layer}.experts.{e}.w1{f}"] = T[base + f]
    if a.broken == "ndim":
        # a STACKED 3-D per-expert pair -- the GGUF-expressible shape a converter-minded
        # user would hand over, and not a weight update this path can place
        k = f"{PREFIX}layers.{a.layer}.mlp.experts.{experts[0]}.{HF[projs[0]]}.lora_A.weight"
        T[k] = np.zeros((1, a.rank, T[k].shape[1]), dtype=np.float32)
    if a.broken == "rank0":
        # a (0, in) lora_A is constructible; rank 0 makes alpha/rank undefined
        for pr in projs:
            k = f"{PREFIX}layers.{a.layer}.mlp.experts.{experts[0]}.{HF[pr]}"
            T[k + ".lora_A.weight"] = np.zeros((0, T[k + ".lora_A.weight"].shape[1]),
                                               dtype=np.float32)
            T[k + ".lora_B.weight"] = np.zeros((T[k + ".lora_B.weight"].shape[0], 0),
                                               dtype=np.float32)
    if a.broken == "incomplete":
        del T[f"{PREFIX}layers.{a.layer}.mlp.experts.{experts[0]}.{HF[projs[0]]}.lora_B.weight"]
    if a.dense:
        for mod, nin, nout in (("self_attn.q_proj", 4096, 1024),
                               ("mlp.shared_experts.down_proj", 2048, 4096)):
            pair(f"{PREFIX}layers.{a.layer}.{mod}", nin, nout, a.rank)
    if a.broken == "shared":
        # NO expert index: ONE 2-D pair against a 3-D expert tensor -- the shape
        # llama-export-lora aborts on.  Named as the HF fused module it would target.
        pair(f"{PREFIX}layers.{a.layer}.mlp.experts.gate_up_proj", 4096, 4096, a.rank)
    if a.broken == "garbage":
        pair(f"{PREFIX}layers.{a.layer}.wat.w9", 4096, 4096, a.rank)

    os.makedirs(a.out, exist_ok=True)
    save_file(T, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": a.rank,
           "target_modules": ["gate_proj", "up_proj", "down_proj"],
           "base_model_name_or_path": "Mistral-Small-4-119B-2603"}
    if a.alpha != "none":
        cfg["lora_alpha"] = float(a.alpha)
    if a.broken == "rslora":
        cfg["use_rslora"] = True
    if a.broken == "dora":
        cfg["use_dora"] = True
    if a.broken == "alpha-pattern":
        cfg["alpha_pattern"] = {"layers.20.mlp.experts": 4.0}
    if a.broken == "rank":
        cfg["r"] = a.rank          # disagrees with the tensors' rank -> lora:rank-mismatch
    with open(os.path.join(a.out, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {a.out}: {len(T)} tensors, rank={a.rank}, alpha={a.alpha}, "
          f"experts={experts}, projs={projs}, zero_b={a.zero_b}, broken={a.broken}")


main()
