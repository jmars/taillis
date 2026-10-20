#!/usr/bin/env python3
"""GATE (e) — PROVENANCE, and the verification path that reads it.

The emitted overlay must carry the MANDATORY quantisation provenance (imatrix, type
file, quantiser, flags), a per-tensor `taillis.overlay_digest.ov<k>`, and the adapter it
was merged from in `taillis.provenance.merge_delta`.  `--verify` must ACCEPT it, must
REJECT a single flipped payload byte, and must REPRODUCE the payload byte-for-byte from
the declared spec INCLUDING the adapter — the only check that can see a corrupted byte in
an override whose bytes are supposed to differ from its base.  A --lora that is not the
adapter named in the file must be refused rather than used.
"""
import shutil
import subprocess
import sys

sys.path.insert(0, "/tmp/lora-gate")
import common as C

C.OUTDIR.mkdir(parents=True, exist_ok=True)
C.WORK.mkdir(parents=True, exist_ok=True)
LAYER = 20

sw = C.make_adapter("known", "--layer", str(LAYER), "--experts", "0,1", "--rank", "4",
                    "--alpha", "2.0")
other = C.make_adapter("known_other", "--layer", str(LAYER), "--experts", "0,1",
                       "--rank", "4", "--alpha", "2.0", "--seed", "11")
out = C.OUTDIR / "gate-e-lora.gguf"
r, out = C.build(out, adapter=sw, overrides=(f"{LAYER}:ffn_down_exps:0,1",))
print(f"build exit  : {r.returncode}")
if r.returncode != 0:
    print(r.stdout[-4000:] + r.stderr[-4000:])
    sys.exit(1)

f = C.gguf_fields(out)
C.check("taillis.provenance.imatrix_sha256" in f, "imatrix_sha256 present",
        f"{f.get('taillis.provenance.imatrix_sha256')}")
C.check("taillis.provenance.type_file_sha256" in f, "type_file_sha256 present",
        f"{f.get('taillis.provenance.type_file_sha256')}")
C.check("taillis.provenance.quantize_flags" in f and "taillis.provenance.quantize_tool" in f,
        "quantize_flags + quantize_tool present",
        f"{f.get('taillis.provenance.quantize_tool')}")
C.check("taillis.overlay_digest.ov0" in f, "per-tensor payload digest present",
        f"{f.get('taillis.overlay_digest.ov0')}")
C.check("taillis.overlay_identity" not in f,
        "overlay_identity ABSENT (derived: the payload really differs from the base)", "")
md = str(f.get("taillis.provenance.merge_delta"))
C.check(md.startswith("lora:sha256=") and C.sha256(sw) in md,
        "merge_delta names the adapter by sha256 (with config, alpha, rank, scale, map)",
        md)

v = subprocess.run([str(C.PY), str(C.TOOL), "--verify", "--base", str(C.BASE),
                    "--overlay", str(out)], capture_output=True, text=True)
print(f"\nverify (no --f16) exit {v.returncode}")
C.check(v.returncode == 0, "--verify accepts the lora-built overlay from its base alone",
        [l for l in v.stdout.splitlines() if l.startswith("payload")][:2])
C.check("GATES OK" in v.stdout, "--verify printed GATES OK", "")

# ---- one flipped payload byte must be REJECTED -----------------------------------
bad = C.WORK / "gate-e-corrupt.gguf"
bad.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(out, bad)
import gguf
rd = gguf.GGUFReader(str(out), "r")
t = [x for x in rd.tensors if x.name.endswith(".ov0.weight")][0]
with open(bad, "r+b") as fh:
    fh.seek(t.data_offset + 1000)
    b = fh.read(1)
    fh.seek(t.data_offset + 1000)
    fh.write(bytes([b[0] ^ 0x01]))
print(f"corrupted   : a copy of the overlay, byte {t.data_offset + 1000} flipped in {t.name}")
v2 = subprocess.run([str(C.PY), str(C.TOOL), "--verify", "--base", str(C.BASE),
                     "--overlay", str(bad)], capture_output=True, text=True)
print(f"verify (corrupt) exit {v2.returncode}")
C.check(v2.returncode != 0 and "PAYLOAD DIGEST MISMATCH" in v2.stdout,
        "--verify REJECTS one flipped payload byte, naming the digest",
        [l[:120] for l in v2.stdout.splitlines() if l.startswith("FAIL")][:1])
bad.unlink()

# ---- full reproduction from the declared spec, INCLUDING the adapter --------------
v3 = subprocess.run([str(C.PY), str(C.TOOL), "--verify", "--base", str(C.BASE),
                     "--overlay", str(out), "--f16", str(C.F16),
                     "--imatrix", str(C.IMATRIX), "--type-file", str(C.TYPES),
                     "--llama-quantize", str(C.QUANTIZE),
                     "--workdir", str(C.WORK / "gate-e-repro"), "--lora", str(sw)],
                    capture_output=True, text=True)
print(f"\nverify (reproduce from the adapter) exit {v3.returncode}")
C.check(v3.returncode == 0, "--verify --f16 --lora REPRODUCES the payload", "")
C.check("BYTE-IDENTICAL" in v3.stdout, "the reproduction is byte-identical",
        [l.strip() for l in v3.stdout.splitlines() if "reproduce" in l.lower()][-2:])
C.check("sha256 matches the declared merge_delta" in v3.stdout,
        "the adapter's sha256 was matched against the file's declaration", "")

# ---- a DIFFERENT adapter must be refused, not used --------------------------------
v4 = subprocess.run([str(C.PY), str(C.TOOL), "--verify", "--base", str(C.BASE),
                     "--overlay", str(out), "--f16", str(C.F16),
                     "--imatrix", str(C.IMATRIX), "--type-file", str(C.TYPES),
                     "--llama-quantize", str(C.QUANTIZE),
                     "--workdir", str(C.WORK / "gate-e-repro2"), "--lora", str(other)],
                    capture_output=True, text=True)
print(f"\nverify (WRONG adapter) exit {v4.returncode}")
C.check(v4.returncode != 0 and "not the adapter that built this overlay" in v4.stdout,
        "a --lora that is not the declared one is REFUSED",
        [l[:140] for l in v4.stdout.splitlines() if l.startswith("FAIL")][:1])
C.finish("gate-e")
