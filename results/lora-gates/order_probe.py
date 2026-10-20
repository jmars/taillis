#!/usr/bin/env python3
"""MEASURE the fused gate_up row order in the F16 GGUF against the SOURCE's per-expert
w1 (gate) / w3 (up), dequantised from FP8 (E4M3) with the source's own qscale_weight.

Question: in blk.<l>.ffn_gate_up_exps.weight (ne=(in,out,n_exp)), expert e's slab is
numpy (out, in) = (4096, 4096).  Which 2048-row block is w1 and which is w3?  This is
the load-bearing measurement for the adapter mapping: a wrong half is invisible once
merged, and no check in the format can see it.

RE-WRITTEN after a machine reboot wiped /tmp mid-unit; the numbers are re-measured here.
"""
import json
import os
import struct

import numpy as np
import torch
from gguf import GGUFReader

SRC = os.environ.get("SRC", "/home/jaye/futaie/taillis/models/Mistral-Small-4-119B-2603")
GGUFDIR = os.environ.get("GGUFDIR", "/home/jaye/futaie/taillis/models")
LAYER = int(os.environ.get("LAYER", "0"))
EXPERT = int(os.environ.get("EXPERT", "0"))


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def read_tensor(shard, off_base, hdr, name):
    info = hdr[name]
    b, e = info["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(off_base + b)
        raw = f.read(e - b)
    shape, dt = tuple(info["shape"]), info["dtype"]
    if dt == "F8_E4M3":
        return torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).reshape(shape) \
                    .float().numpy()
    if dt == "BF16":
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape) \
                    .float().numpy()
    if dt == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    raise SystemExit(f"unhandled dtype {dt} for {name}")


def source_dequant(name):
    idx = json.load(open(os.path.join(SRC, "consolidated.safetensors.index.json")))
    shard = os.path.join(SRC, idx["weight_map"][name])
    hdr, base = st_header(shard)
    w = read_tensor(shard, base, hdr, name)
    s = read_tensor(shard, base, hdr, name.replace(".weight", ".qscale_weight"))
    return w * float(s.reshape(()))


def corr(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def slab(path, r, name, expert):
    t = {x.name: x for x in r.tensors}[name]
    ne = tuple(int(x) for x in t.shape)
    stride = t.n_bytes // ne[2]
    with open(path, "rb") as f:
        f.seek(t.data_offset + expert * stride)
        buf = f.read(stride)
    return ne, np.frombuffer(buf, dtype=np.float16).reshape(ne[1], ne[0]).astype(np.float32)


def main():
    path = os.path.join(GGUFDIR, "mistral-small-4-2L-f16.gguf")
    r = GGUFReader(path, "r")
    ne, s = slab(path, r, f"blk.{LAYER}.ffn_gate_up_exps.weight", EXPERT)
    print(f"GGUF tensor : blk.{LAYER}.ffn_gate_up_exps.weight ne={ne} type=F16")
    print(f"expert {EXPERT} slab: numpy {s.shape} (out, in), rms="
          f"{np.sqrt(np.mean(s ** 2)):.6f}")

    w1 = source_dequant(f"layers.{LAYER}.experts.{EXPERT}.w1.weight")   # (2048, 4096)
    w3 = source_dequant(f"layers.{LAYER}.experts.{EXPERT}.w3.weight")
    print(f"source w1 {w1.shape} rms={np.sqrt(np.mean(w1 ** 2)):.6f}; "
          f"w3 {w3.shape} rms={np.sqrt(np.mean(w3 ** 2)):.6f}")

    h = ne[1] // 2
    top, bot = s[:h], s[h:]
    print()
    print(f"GGUF rows [0:{h})    ~ w1 corr={corr(top, w1):.8f}   ~ w3 corr={corr(top, w3):.8f}")
    print(f"GGUF rows [{h}:{ne[1]}) ~ w1 corr={corr(bot, w1):.8f}   ~ w3 corr={corr(bot, w3):.8f}")
    print(f"max|top-w1| = {np.max(np.abs(top - w1)):.6g}   max|bot-w3| = "
          f"{np.max(np.abs(bot - w3)):.6g}")
    print(f"max|top-w3| = {np.max(np.abs(top - w3)):.6g}   max|bot-w1| = "
          f"{np.max(np.abs(bot - w1)):.6g}")
    w1n = source_dequant(f"layers.{LAYER}.experts.{EXPERT + 1}.w1.weight")
    print(f"CONTROL top ~ w1(expert {EXPERT + 1}) corr={corr(top, w1n):.8f}")

    ned, sd = slab(path, r, f"blk.{LAYER}.ffn_down_exps.weight", EXPERT)
    print(f"\nGGUF blk.{LAYER}.ffn_down_exps.weight ne={ned}")
    w2 = source_dequant(f"layers.{LAYER}.experts.{EXPERT}.w2.weight")   # (4096, 2048)
    print(f"source w2 {w2.shape} (out, in); GGUF slab {sd.shape} (ne1, ne0)")
    print(f"corr(slab, w2) = {corr(sd, w2):.8f}   corr(slab, w2.T) = {corr(sd, w2.T):.8f}")
    print(f"max|slab-w2| = {np.max(np.abs(sd - w2)):.6g}")


main()
