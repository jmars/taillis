#!/usr/bin/env python3
"""GATE (c) — INDEPENDENT EXECUTABLE CROSS-CHECK, DENSE CASE.

The same (A, B, alpha, r) is merged twice on the same F16 base tensor:

  * by llama.cpp's own `llama-export-lora`, from a HAND-BUILT GGUF LoRA
    (`general.type=adapter`, `adapter.type=lora`, `adapter.lora.alpha`, tensors
    `<base>.lora_a` ne=(in,r) / `.lora_b` ne=(r,out)) — the vendored converter tree cannot
    run, so it is written with gguf-py, exactly as the xcheck unit did;
  * by OUR functions from tools/repack_overlay.py: `lora_scale()` for the scale and
    `merged_f16()` for the merge, applied to `scale * B @ A`.

The comparison is BIT-EXACT on the F16 tensor, so there is no tolerance to hide behind.

THE SCOPE THIS COVERS, stated so it is not over-read: DENSE only, because a GGUF LoRA
CANNOT express a per-expert delta at all (llama-export-lora ABORTS on a shared 2-D pair
against a 3-D expert tensor — `GGML_ASSERT(ggml_can_repeat)`, exit 134, MEASURED by the
xcheck unit).  The EXPERT case is gate (b), an independent numpy oracle rather than an
executable reference.  What (c) pins is the arithmetic the two paths share: the scale,
the product order, the A orientation and the single F16 store.

THE GATE CAN FAIL: with the competing scale alpha/sqrt(rank) substituted, the same
comparison must mismatch.
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/tmp/lora-gate")
import common as C
sys.path.insert(0, str(C.TREE / "tools"))
import repack_overlay as R          # OUR side: the scale and the merge, not the loader

G = C.WORK / "dense"
G.mkdir(parents=True, exist_ok=True)
BASE = G / "tiny-base.gguf"
T = "blk.0.attn_q.weight"
N_EMBD = 64

if not BASE.exists():
    subprocess.run([str(C.PY), str(C.GATEDIR / "make_tiny_base.py"), str(BASE)],
                   check=True, capture_output=True)
    print(f"base        : {BASE} REBUILT (the pre-reboot tiny base was in tmpfs)")
else:
    print(f"base        : {BASE}")
print(f"             {T} ne=(64,64) F16; hand-built with gguf-py, seed 1234")

rng = np.random.default_rng(4242)
for case, (rank, alpha, expect) in {
        "alpha=2 r=8": (8, 2.0, 2.0 / 8),
        "alpha absent": (8, None, 1.0),
        "alpha=8 r=2": (2, 8.0, 8.0 / 2),
}.items():
    import gguf
    tag = case.replace("=", "").replace(" ", "-")
    A = (rng.uniform(-1, 1, size=(rank, N_EMBD)) * 0.1).astype(np.float32)
    B = (rng.uniform(-1, 1, size=(N_EMBD, rank)) * 0.1).astype(np.float32)

    lora = G / f"lora-{tag}.gguf"
    w = gguf.GGUFWriter(str(lora), arch="llama", use_temp_file=False)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_name("lora-gate-c")
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    if alpha is not None:
        w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
    w.add_tensor(T + ".lora_a", A, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.add_tensor(T + ".lora_b", B, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    merged = G / f"merged-{tag}.gguf"
    if merged.exists():
        merged.unlink()
    p = subprocess.run([str(C.EXPORT_LORA), "-m", str(BASE), "--lora", str(lora),
                        "-o", str(merged)], env=C.LLAMA_ENV, capture_output=True, text=True)
    print(f"\ncase        : {case} (r={rank}, alpha={alpha})")
    print(f"  export-lora exit {p.returncode}; "
          f"{[l.strip() for l in p.stdout.splitlines() if 'calculated_scale' in l]}")
    C.check(p.returncode == 0, f"{case}: llama-export-lora exited 0", f"exit {p.returncode}")
    if p.returncode != 0:
        continue

    W = C.raw_tensor(BASE, T)                       # F16 values as float32
    M = C.raw_tensor(merged, T)                     # export-lora's merged F16
    scale = R.lora_scale(alpha, rank)               # OUR scale (alpha/rank, else 1.0)
    C.check(abs(scale - expect) < 1e-12, f"{case}: our lora_scale() == {expect}", f"{scale}")

    ours, _ = R.merged_f16(W.astype(np.float16), scale * (B @ A))   # OUR merge
    n, m = C.ulp_stats(ours, M)
    C.check(n == 0, f"{case}: our merged F16 is BIT-IDENTICAL to llama-export-lora's",
            f"{n}/{M.size} element(s) differ, max {m} ulp")
    print(f"     |M-W| rms = {np.sqrt(np.mean((M - W) ** 2)):.8g}; scale measured back = "
          f"{float(np.sum((M - W) * (B @ A)) / np.sum((B @ A) ** 2)):.9f}")

    wrong = R.lora_scale((alpha if alpha is not None else 1.0) * np.sqrt(rank), rank)
    ours_w, _ = R.merged_f16(W.astype(np.float16), wrong * (B @ A))
    nw, mw = C.ulp_stats(ours_w, M)
    C.check(nw > 0, f"{case}: ASYMMETRY — alpha/sqrt(rank)={wrong:.6g} MUST mismatch",
            f"{nw}/{M.size} element(s) differ, max {mw} ulp")

C.finish("gate-c")
