#!/usr/bin/env python3
"""GATE (f) — REFUSALS, each NAMED, each BEFORE anything is written.

A tool that refuses wrongly is as dangerous as one that merges wrongly, so each case
asserts THREE things: a nonzero exit, the named failure code in the output, and that NO
output file was created.  The shared-pair case is the one this unit exists for: it is the
shape llama-export-lora ABORTS on (exit 134) and the shape that would apply one delta to
128 experts.
"""
import shutil
import sys

sys.path.insert(0, "/tmp/lora-gate")
import common as C

C.OUTDIR.mkdir(parents=True, exist_ok=True)
LAYER = 20
BOTH = (f"{LAYER}:ffn_gate_up_exps:0,1", f"{LAYER}:ffn_down_exps:0,1")


def A(name, *args):
    return C.make_adapter(name, "--layer", str(LAYER), *args, "--rank", "4", "--alpha", "2.0")


CASES = [
    ("unmappable module name", A("f-garbage", "--experts", "0", "--broken", "garbage"), [],
     "lora:unmappable-module", BOTH),
    ("SHARED 2-D pair on a 3-D expert tensor",
     A("f-shared", "--experts", "0", "--broken", "shared"), [],
     "lora:shared-pair-on-expert-tensor", BOTH),
    ("B's out axis wrong for the fused block",
     A("f-bout", "--experts", "0", "--broken", "b-out"), [], "lora:shape-mismatch",
     (f"{LAYER}:ffn_gate_up_exps:0", f"{LAYER}:ffn_down_exps:0")),
    ("rank disagrees with the config",
     A("f-rank", "--experts", "0", "--broken", "rank"), [], "lora:rank-mismatch", BOTH),
    ("an override expert the adapter does not cover",
     A("f-uncovered", "--experts", "0"), [], "lora:uncovered-expert", BOTH),
    ("lora_A without lora_B",
     A("f-incomplete", "--experts", "0", "--broken", "incomplete"), [],
     "lora:incomplete-pair", BOTH),
    ("two names for one (expert, projection)",
     A("f-dup", "--experts", "0", "--broken", "dup"), [], "lora:duplicate-pair", BOTH),
    ("rsLoRA (alpha/sqrt(rank) training)",
     A("f-rslora", "--experts", "0", "--broken", "rslora"), [],
     "lora:unsupported-config", BOTH),
    ("DoRA", A("f-dora", "--experts", "0", "--broken", "dora"), [],
     "lora:unsupported-config", BOTH),
    ("per-module alpha_pattern",
     A("f-apat", "--experts", "0", "--broken", "alpha-pattern"), [],
     "lora:unsupported-config", BOTH),
    ("--lora with --identity", A("f-ident", "--experts", "0"), ["--identity"],
     "a delta is merged into a QUANTISED payload", BOTH),
    ("--lora together with --delta-synth", A("f-two", "--experts", "0"),
     ["--delta-synth", "1:0.02"], "lora:delta", BOTH),
    ("--lora together with --delta", A("f-two-d", "--experts", "0"),
     ["--delta", "/nonexistent/delta.gguf"], "lora:delta", BOTH),
    ("a STACKED 3-D per-expert pair", A("f-ndim", "--experts", "0", "--broken", "ndim"),
     [], "lora:unsupported-tensor", BOTH),
    ("rank 0 (alpha/rank undefined)", A("f-rank0", "--experts", "0", "--broken", "rank0"),
     [], "lora:rank-mismatch", BOTH),
]

n_ok = 0
for label, sw, extra, code, ovr in CASES:
    out = C.OUTDIR / f"gate-f-{label.split()[0].strip('(),').lower()}.gguf"
    if out.exists():
        out.unlink()
    ident = "--identity" in extra
    r, out = C.build(out, adapter=sw, overrides=ovr, extra=extra,
                     requantise=not ident, f16=not ident, imatrix=not ident,
                     types=not ident, quantize=not ident)
    blob = r.stdout + r.stderr
    ok = r.returncode != 0 and code in blob and not out.exists()
    print(f"{'ok  ' if ok else 'FAIL'} {label:46s} exit={r.returncode} "
          f"out_written={out.exists()}  wants '{code}'")
    if ok:
        n_ok += 1
        line = [l.strip() for l in blob.splitlines() if code in l]
        print(f"     {line[0][:170] if line else ''}")
    else:
        C.FAILS.append(label)
        print("   ---- output ----")
        print("   " + "\n   ".join((blob.strip().splitlines() or ["<none>"])[-6:]))

# ---- missing config: the alpha cannot be guessed -----------------------------------
noconfig = C.ADAPTERS / "f-noconfig.d"
noconfig.mkdir(parents=True, exist_ok=True)
shutil.copyfile(A("f-noconfig-src", "--experts", "0"),
                noconfig / "adapter_model.safetensors")
out = C.OUTDIR / "gate-f-noconfig.gguf"
if out.exists():
    out.unlink()
r, out = C.build(out, adapter=noconfig / "adapter_model.safetensors", overrides=BOTH)
blob = r.stdout + r.stderr
ok = r.returncode != 0 and "lora:config-missing" in blob and not out.exists()
print(f"{'ok  ' if ok else 'FAIL'} {'no adapter_config.json beside the adapter':46s} "
      f"exit={r.returncode}  wants 'lora:config-missing'")
if ok:
    n_ok += 1
    print("     " + [l.strip() for l in blob.splitlines() if "lora:config-missing" in l][0][:170])
else:
    C.FAILS.append("config-missing")
    print("   " + "\n   ".join((blob.strip().splitlines() or ["<none>"])[-6:]))

print(f"\n{n_ok}/{len(CASES) + 1} refusals named, all cheap (no build started), "
      f"nothing written")
C.finish("gate-f")
