#!/usr/bin/env python3
"""GATE (a) — A ZERO DELTA IS A NO-OP.

An adapter whose lora_B is ALL ZEROS adds nothing to W, so its overlay must be the SAME
OVERLAY as a build with no adapter at all — otherwise the merge perturbs something it
should not.

STATED DEVIATION, because a gate that cannot pass is not a gate.  The two FILES are not
byte-identical and must not be: the zero-B build really did read an adapter, so its
`taillis.provenance.merge_delta` key records a different input, and demanding the key's
absence would force the provenance to lie.  Compared here is everything a LOADER uses —
every tensor's payload bytes, and every `taillis.overlay_digest.ov<k>` — plus EVERY
other KV key.  The full KV difference is printed, so the claim is auditable rather than
asserted: if it is anything but merge_delta, this gate fails.
"""
import sys

sys.path.insert(0, "/tmp/lora-gate")
import common as C

C.OUTDIR.mkdir(parents=True, exist_ok=True)
C.WORK.mkdir(parents=True, exist_ok=True)

sw_z = C.make_adapter("zero_b", "--layer", "20", "--experts", "0,1", "--rank", "4",
                      "--alpha", "2.0", "--zero-b")
print(f"adapter     : {sw_z}  (lora_B = ALL ZEROS)\n")

r_a, out_a = C.build(C.OUTDIR / "gate-a-noadapter.gguf")
print(f"build A (no adapter) exit={r_a.returncode}")
if r_a.returncode != 0:
    print(r_a.stdout[-3000:] + r_a.stderr[-3000:])
    sys.exit(1)

r_b, out_b = C.build(C.OUTDIR / "gate-a-zerob.gguf", adapter=sw_z)
print(f"build B (zero-B)     exit={r_b.returncode}")
if r_b.returncode != 0:
    print(r_b.stdout[-3000:] + r_b.stderr[-3000:])
    sys.exit(1)

ta, _ = C.gguf_tensors(out_a)
tb, _ = C.gguf_tensors(out_b)
fa, fb = C.gguf_fields(out_a), C.gguf_fields(out_b)

print()
C.check(sorted(ta) == sorted(tb) and len(ta) == 2, "same tensor set",
        f"{sorted(ta)} vs {sorted(tb)}")
for name in sorted(ta):
    ba, sa = ta[name]
    bb, sb = tb[name]
    C.check(ba.tobytes() == bb.tobytes() and sa == sb,
            f"payload BYTE-IDENTICAL {name}", f"{len(ba.tobytes())} B, ne={sa}")

da = {k: v for k, v in fa.items() if k.startswith("taillis.overlay_digest.")}
db = {k: v for k, v in fb.items() if k.startswith("taillis.overlay_digest.")}
C.check(da == db and len(da) == 2, "overlay_digest.ov<k> identical for both tensors",
        f"{da}")
C.check(fa.get("taillis.overlay_identity") is True, "no-adapter build DERIVES identity=true",
        f"{fa.get('taillis.overlay_identity')}")
C.check(fb.get("taillis.overlay_identity") is True,
        "zero-B build ALSO derives identity=true (its payload equals the base)",
        f"{fb.get('taillis.overlay_identity')}")

NOISE = ("GGUF.kv_count", "GGUF.tensor_count", "GGUF.version")   # reader-synthesised
keys = set(fa) | set(fb)
diff = sorted(k for k in keys if k not in NOISE and fa.get(k) != fb.get(k))
print(f"\nKV keys differing ({len(diff)} of {len(keys)}; {NOISE} are the reader's own "
      f"counts and follow the one extra key):")
for k in diff:
    print(f"  {k}\n    no adapter : {fa.get(k)!r}\n    zero-B     : {fb.get(k)!r}")
C.check(diff == ["taillis.provenance.merge_delta"],
        "the ONLY KV difference is merge_delta — the honest record that an adapter was "
        "applied (a zero one)", f"{diff}")
C.check("applied=6" in str(fb.get("taillis.provenance.merge_delta")),
        "merge_delta says 6 pairs were applied", str(fb.get("taillis.provenance.merge_delta")))
C.check(len(fa) == len(fb) - 1, "the zero-B file carries exactly one extra key",
        f"{len(fa)} vs {len(fb)}")
C.finish("gate-a")
