#!/usr/bin/env python3
"""GATE (b) — KNOWN ARITHMETIC, against an INDEPENDENT numpy oracle.

An adapter with known A, B, alpha and rank targets experts 0,1 of LAYER 20.  The merged
F16 must equal W + (alpha/rank)*B@A EXACTLY, per expert and per fused block, must be
UNCHANGED on every other expert and every other tensor, and the oracle here reads the
adapter's safetensors and the F16 base's slabs ITSELF — it never imports the tool.

The oracle also runs in FLOAT64 and reports its distance from the tool's FLOAT32
arithmetic in F16 ULPs: the two can differ only at an F16 rounding tie, and that number
is what makes the FLOAT32 arm's exactness interpretable rather than assumed.

THE GATE CAN FAIL — three wrong conventions run through the same comparator and MUST
mismatch: the wrong scale (1/rank, i.e. alpha dropped), the fused halves SWAPPED (w1 into
w3's rows), and one expert's delta placed in another expert's slab.
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/lora-gate")
import common as C

C.OUTDIR.mkdir(parents=True, exist_ok=True)
C.WORK.mkdir(parents=True, exist_ok=True)
LAYER = 20
IDS = [0, 1]
UNDECLARED = [2, 3]          # pairs exist in the adapter, NO override declares them
ALPHA, RANK = "2.0", 4

sw = C.make_adapter("known", "--layer", str(LAYER), "--experts", "0,1,2,3",
                    "--rank", str(RANK), "--alpha", ALPHA)
print(f"adapter     : {sw}  experts 0,1,2,3  rank {RANK}  alpha {ALPHA}\n")

base_sha_before = C.sha256(C.BASE)
print(f"base sha256 : {base_sha_before}")

r, out = C.build(C.OUTDIR / "gate-b-known.gguf", adapter=sw)
print(f"build exit  : {r.returncode}")
if r.returncode != 0:
    print(r.stdout[-4000:] + r.stderr[-4000:])
    sys.exit(1)

C.check(C.sha256(C.BASE) == base_sha_before, "base file UNCHANGED by the build", "")

warng = [l for l in r.stdout.splitlines() if l.startswith("WARNING")]
C.check(any("6 adapter pair(s) are NOT applied" in l for l in warng),
        "the 6 undeclared expert-pair sets are DECLARED as not applied", f"{warng}")
C.check(any("dense_skipped=0" in l for l in r.stdout.splitlines()),
        "merge_delta declares dense_skipped=0 (this adapter carries no dense module)",
        [l for l in r.stdout.splitlines() if "merge_delta" in l])

alpha, rank, scale = C.adapter_scale(sw)
print(f"\noracle      : alpha={alpha} rank={rank} scale={scale} (alpha/rank; read from the "
      f"config + the tensors, NEVER from the tool)\n")

slabs = C.Slabs(C.F16)
wd = C.WORK / out.stem

for logical, k in (("ffn_gate_up_exps", 0), ("ffn_down_exps", 1)):
    slab_name = f"blk.{LAYER}.{logical}.weight"
    ne0 = int(slabs.by_name[slab_name].shape[0])
    ne1 = int(slabs.by_name[slab_name].shape[1])
    half = ne1 // 2
    got = C.read_slice_f16(wd / f"blk.{LAYER}.{logical}.ov{k}.weight.f16.gguf")
    C.check(got.shape == (len(IDS), ne1, ne0), f"{logical}: merged slice shape", f"{got.shape}")
    for j, e in enumerate(IDS):
        W = slabs.slab(slab_name, e)                     # (ne1, ne0), values exact
        blocks = {"w1": (0, half), "w3": (half, ne1)} if logical == "ffn_gate_up_exps" \
            else {"w2": (0, ne1)}

        def delta_of(expert, sc, swap=False):
            D = np.zeros(W.shape, dtype=np.float32)
            pr = C.adapter_pairs(sw, LAYER, expert)
            order = (("w3", "w1") if swap else ("w1", "w3")) if len(blocks) == 2 else ("w2",)
            for proj, (lo, hi) in zip(order, blocks.values()):
                A, B = pr[C.HF[proj]]
                D[lo:hi] = np.float32(sc) * (B @ A)
            return D

        D32 = delta_of(e, scale)
        pr = C.adapter_pairs(sw, LAYER, e)
        D64 = np.zeros(W.shape, dtype=np.float64)
        for proj, (lo, hi) in blocks.items():
            A, B = pr[C.HF[proj]]
            D64[lo:hi] = float(scale) * (B.astype(np.float64) @ A.astype(np.float64))
        want32 = (W + D32).astype(np.float16)
        want64 = (W.astype(np.float64) + D64).astype(np.float16)
        n32, m32 = C.ulp_stats(got[j], want32)
        n64, m64 = C.ulp_stats(got[j], want64)
        C.check(n32 == 0, f"{logical} e={e}: EXACT vs the f32 oracle (bit-identical F16)",
                f"{n32} differing element(s), max {m32} ulp of {want32.size}")
        print(f"     f64 oracle arm: {n64}/{want64.size} element(s) differ, max {m64} ulp "
              f"(F16 rounding ties: alpha/rank*B@A is not exactly representable on the "
              f"boundary)")

        arms = [("scale=1/rank (alpha dropped)", delta_of(e, 1.0 / rank)),
                (f"this slab holding expert {IDS[1 - j]}'s delta",
                 delta_of(IDS[1 - j], scale))]
        if len(blocks) == 2:
            arms.append(("fused halves SWAPPED (w1 into w3's rows)",
                         delta_of(e, scale, swap=True)))
        for label, Dw in arms:
            n, m = C.ulp_stats(got[j], (W + Dw).astype(np.float16))
            C.check(n > 0, f"  ASYMMETRY {logical} e={e}: {label} MUST mismatch",
                    f"{n} element(s), max {m} ulp")

# ---- the declared ids are the ONLY experts in the file, and nothing leaked ----------
tensors, _ = C.gguf_tensors(out)
fields = C.gguf_fields(out)
declared = {name: fields.get(f"taillis.overlay_experts.ov{name.split('.ov')[1].split('.')[0]}")
            for name in tensors if name.startswith("blk.")}
C.check(all(v == "0,1" for v in declared.values()) and len(declared) == 2,
        "declared ids are 0,1 in exactly two override tensors", f"{declared}")
for name, (arr, shape) in tensors.items():
    C.check(shape[2] == len(IDS),
            f"{name}: exactly {len(IDS)} expert slab(s) — no leak of undeclared {UNDECLARED}",
            f"ne={shape}")
C.check("taillis.provenance.merge_delta" in fields and
        all(f"taillis.overlay_digest.ov{i}" in fields for i in (0, 1)),
        "payload digests stamped for both tensors", "")
C.finish("gate-b")
