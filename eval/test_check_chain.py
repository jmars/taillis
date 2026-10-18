#!/usr/bin/env python3
"""Self-test for tools/check_chain.py (taillis/layout.md §9) — a test that CAN fail.

Builds SYNTHETIC gguf-py fixtures in a /tmp tempdir — a flat Q4_K parent (two
expert logicals, ne[2]=128, block-aligned raw bytes + two non-expert F32
tensors), a v1 MIXED artifact written directly (4 overridden members at
q8_0/q6_K with fake bytes, 8 sibling members carrying the PARENT SLAB bytes so
the positive path is honest), the uniform shadow built by the REAL
tools/group_experts.py, and a family of control/poison variants — then drives
tools/check_chain.py as a subprocess and asserts:

  POSITIVE  the full chain exits 0: G1 v1-derived grouping, G2 key required and
            present, G3 comparator match, G4 the SAME canonical sha from BOTH
            directions (synthesized-from-parent == shadow file) with
            12/12 shadow slabs == parent slabs, G5 8/8 siblings HARD;
  POSITIVE  --control with a byte-conserving equal-size permutation exits 0 on
            G6 (multiset conserved, |delta| <= 384 B, assignment differs);
  GOLDEN    the canonical-stream digest of the fixed synthetic shadow equals a
            constant pinned below — any future drift in the §9.2 encoding
            (name length field, ggml type, n_dims, ne[], payload length,
            payload bytes, table order, KV exclusion) breaks this loudly;
  NEGATIVES, each exit 1 NAMING its mechanism (a test that cannot fail is not
            a test — M6/N8):
    (a) mixed artifact WITHOUT the shadow key            -> "R3.2";
    (b) shadow key with one hex char flipped             -> "shadow identity mismatch";
    (c) --parent pointing at a different flat file       -> G3 names BOTH sha values;
    (d) UNIFORM artifact carrying a shadow key           -> "FORBIDDEN";
    (e) control moving the big type onto a different-SIZE member -> the ±384 B
        whole-file gate names the byte delta;
    (f) control identical to the allocated artifact      -> "IDENTICAL";
    (g) artifact with a non-uniform partition across logicals -> "non-uniform partition";
    (h) --shadow file with one flipped payload byte      -> "shadow identity mismatch"
        AND "shadow slab mismatch" (the writer's gate 1 re-run has teeth).

The negative variants are built by the SAME builder as the positive, changing
ONLY the mechanism under test — so a failure names the mechanism, it does not
fail incidentally.

Needs gguf-py (venvs/convert/bin/python3); re-execs itself under that venv
when gguf is not importable, exactly like eval/test_group_experts_v1.py.
No GPU, no model file, runs in a temp dir under /tmp.

Gate: this process's exit code. 0 = pass, nonzero = fail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "check_chain.py"
GROUPER = REPO / "tools" / "group_experts.py"

SHADOW_KEY = "taillis.uniform_shadow_sha256"
PARENT_KEY = "taillis.parent_sha256"
FORMAT_KEY = "taillis.format_version"
GROUP_KEY = "taillis.expert_groups"
ARM_KEY = "taillis.provenance.arm"
SEED_KEY = "taillis.provenance.shuffle_seed"
TYPE_FILE_KEY = "taillis.provenance.type_file_sha256"

SPEC = [43, 43, 42]
FAKE_TYPE_FILE_SHA = "e" * 64  # value never checked by the checker, presence is

# GOLDEN canonical-stream digest (layout.md §9.2) of the synthetic shadow built
# by tools/group_experts.py --src flat.gguf --spec 43,43,42 --format v0 from the
# deterministic fixture below.  Pins the byte-exact encoding: u32le(len(name))
# || name || u32le(ggml_type) || u32le(n_dims) || u32le(ne[]) ||
# u64le(payload_nbytes) || payload, over the shadow's table order, KV excluded.
# If this constant ever mismatches, the §9.2 definition drifted — re-derive the
# contract, do not re-pin the constant to whatever the code now computes.
GOLDEN = "a234b153740c567d26a9000b8dddce29b4fdd51602955207f03f9f5d5d8e0df6"


COUNT = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global COUNT
    COUNT += 1
    mark = "ok  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        sys.exit(f"FATAL: {name} failed {detail}")


def ensure_venv() -> None:
    """numpy and gguf-py both live in venvs/convert; re-exec under its python."""
    try:
        import numpy  # noqa: F401
        import gguf  # noqa: F401
    except ImportError:
        venv_py = REPO / "venvs" / "convert" / "bin" / "python3"
        if not venv_py.exists():
            sys.exit("FATAL: numpy/gguf not importable and venvs/convert missing")
        os.execv(str(venv_py), [str(venv_py), __file__, *sys.argv[1:]])


ensure_venv()

import numpy as np  # noqa: E402
sys.path.insert(0, str(REPO / "tools"))
import check_chain as cc  # noqa: E402


def run_tool(*argv: str, expect: int | None = 0) -> subprocess.CompletedProcess:
    cp = subprocess.run([sys.executable, str(TOOL), *argv],
                        capture_output=True, text=True)
    if expect is not None and cp.returncode != expect:
        sys.exit(f"FATAL: check_chain {' '.join(argv[:6])}... exited {cp.returncode}, "
                 f"expected {expect}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    return cp


# ---- deterministic raw-byte fixtures --------------------------------------

def q4_k_bytes(ne1: int, ne2: int, salt: int = 0) -> np.ndarray:
    """ne=(2048, ne1, ne2) elements -> (ne2, ne1, 1152) raw bytes (8 rows x 144 B)."""
    return ((np.arange(ne2 * ne1 * 1152, dtype=np.uint64) * 31 + salt) % 251).astype(np.uint8) \
        .reshape(ne2, ne1, 1152)


def down_q4_k_bytes(ne1: int, ne2: int, salt: int = 0) -> np.ndarray:
    """ne=(1024, ne1, ne2) elements -> (ne2, ne1, 576) raw bytes (4 rows x 144 B)."""
    return ((np.arange(ne2 * ne1 * 576, dtype=np.uint64) * 7 + 3 + salt) % 251).astype(np.uint8) \
        .reshape(ne2, ne1, 576)


def fake_q_bytes(ne1: int, ne2: int, row_bytes: int, salt: int) -> np.ndarray:
    """Overridden-member bytes: the right LENGTH for the type, wrong content
    (the checker does not verify overridden members against the parent — that
    is the whole point of the chain; the bytes only need to differ)."""
    n = ne2 * ne1 * row_bytes
    return ((np.arange(n, dtype=np.uint64) * 13 + salt) % 251).astype(np.uint8) \
        .reshape(ne2, ne1, row_bytes)


GU_ROW = {"q4_K": 1152, "q8_0": 2176, "q6_K": 1680}   # ne[0]=2048 elements per row
DN_ROW = {"q4_K": 576, "q6_K": 840, "q8_0": 1088}     # ne[0]=1024 elements per row


def build_flat(path: Path, salt: int = 0) -> None:
    """Synthetic FLAT parent: 2 expert logicals (ne[2]=128, Q4_K) + 2 F32."""
    from gguf import GGUFWriter, GGUFValueType
    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    w.add_key_value("general.architecture", "r3test", GGUFValueType.STRING)
    w.add_key_value("general.name", "check_chain self-test fixture", GGUFValueType.STRING)
    w.add_key_value("r3test.expert_count", np.uint32(128), GGUFValueType.UINT32)
    w.add_tensor("tok_emb.weight", np.arange(64, dtype=np.float32).reshape(8, 8) / 7.0)
    w.add_tensor("out_norm.weight", np.linspace(0.5, 2.0, 16, dtype=np.float32) + salt)
    w.add_tensor("blk.0.ffn_gate_up_exps.weight", q4_k_bytes(8, 128),
                 raw_dtype=__import__("gguf").GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.0.ffn_down_exps.weight", down_q4_k_bytes(4, 128),
                 raw_dtype=__import__("gguf").GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.1.ffn_gate_up_exps.weight", q4_k_bytes(8, 128, salt=11),
                 raw_dtype=__import__("gguf").GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.1.ffn_down_exps.weight", down_q4_k_bytes(4, 128, salt=17),
                 raw_dtype=__import__("gguf").GGMLQuantizationType.Q4_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_grouped(path: Path, flat: Path, parent_sha: str, shadow_key: str | None,
                  assignment: dict[str, dict[int, str]], control: bool,
                  spec: dict[str, list[int]], imatrix: bool = False) -> None:
    """Write a v1 grouped artifact DIRECTLY (the region path's output shape).

    assignment: logical -> {k: qtype}; members NOT named there carry q4_K and
    the PARENT SLAB bytes (honest siblings); named ones carry fake bytes of the
    right length (the requantised bytes, unverifiable against the parent).
    """
    from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType
    QT = {"q4_K": GGMLQuantizationType.Q4_K,
          "q8_0": GGMLQuantizationType.Q8_0,
          "q6_K": GGMLQuantizationType.Q6_K}
    gu_raw = q4_k_bytes(8, 128)      # (128, 8, 1152), expert axis first
    dn_raw = down_q4_k_bytes(4, 128)
    gu2_raw = q4_k_bytes(8, 128, salt=11)
    dn2_raw = down_q4_k_bytes(4, 128, salt=17)
    lo = {}
    ranges: dict[str, list[tuple[int, int]]] = {}
    for logical in ("blk.0.ffn_gate_up_exps", "blk.0.ffn_down_exps",
                    "blk.1.ffn_gate_up_exps", "blk.1.ffn_down_exps"):
        x = 0
        rs = []
        for s in spec[logical]:
            rs.append((x, x + s))
            x += s
        ranges[logical] = rs

    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    w.add_key_value("general.architecture", "r3test", GGUFValueType.STRING)
    w.add_key_value("r3test.expert_count", np.uint32(128), GGUFValueType.UINT32)
    w.add_key_value(FORMAT_KEY, np.uint32(1), GGUFValueType.UINT32)
    w.add_key_value(PARENT_KEY, parent_sha, GGUFValueType.STRING)
    if shadow_key is not None:
        w.add_key_value(SHADOW_KEY, shadow_key, GGUFValueType.STRING)
    w.add_key_value(ARM_KEY, "shuffled-control" if control else "allocated",
                    GGUFValueType.STRING)
    if control:
        w.add_key_value(SEED_KEY, "test-seed-1", GGUFValueType.STRING)
    w.add_key_value(TYPE_FILE_KEY, FAKE_TYPE_FILE_SHA, GGUFValueType.STRING)
    if imatrix:
        # flips G5 into its WARN-only mode (layout.md §9.4: outside the E2 proof,
        # the warning names both possibilities, it never silently passes)
        w.add_key_value("taillis.provenance.imatrix_sha256", "f" * 64,
                        GGUFValueType.STRING)
    w.add_tensor("tok_emb.weight", np.arange(64, dtype=np.float32).reshape(8, 8) / 7.0)
    w.add_tensor("out_norm.weight", np.linspace(0.5, 2.0, 16, dtype=np.float32))
    for logical, raw, rows in (("blk.0.ffn_gate_up_exps", gu_raw, GU_ROW),
                               ("blk.0.ffn_down_exps", dn_raw, DN_ROW),
                               ("blk.1.ffn_gate_up_exps", gu2_raw, GU_ROW),
                               ("blk.1.ffn_down_exps", dn2_raw, DN_ROW)):
        for k, (a, b) in enumerate(ranges[logical]):
            qtype = assignment.get(logical, {}).get(k, "q4_K")
            if qtype == "q4_K":
                bytes_arr = raw[a:b]                      # the parent slab, verbatim
            elif logical.endswith("gate_up_exps"):
                bytes_arr = fake_q_bytes(8, b - a, rows[qtype], salt=41 + k)
            else:
                bytes_arr = fake_q_bytes(4, b - a, rows[qtype], salt=97 + k)
            w.add_tensor(f"{logical}.g{k}.weight", bytes_arr, raw_dtype=QT[qtype])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def flip_member_byte(src: Path, dst: Path, member_name: str | None = None) -> str:
    """Copy src to dst with ONE byte flipped inside a member's payload; returns
    the member's name (the corruption is real, not asserted).  member_name
    selects the member; default = the first member in table order."""
    from gguf import GGUFReader
    r = GGUFReader(src)
    member = (next(t for t in r.tensors if t.name == member_name) if member_name
              else next(t for t in r.tensors if ".g" in t.name))
    raw = bytearray(src.read_bytes())
    raw[member.data_offset + 17] ^= 0xFF
    dst.write_bytes(bytes(raw))
    assert raw != bytearray(src.read_bytes())
    return member.name


def add_stray_member(src: Path, dst: Path, name: str) -> None:
    """Copy src (all KV + tensors verbatim) adding ONE member-shaped tensor the
    declaration does not claim.  Used for the v0 unclaimed-member negative."""
    from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
    r = GGUFReader(src)
    w = GGUFWriter(str(dst), arch="", use_temp_file=False)
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        main = f.types[0]
        if main == GGUFValueType.ARRAY:
            w.add_key_value(k, f.contents(), GGUFValueType.ARRAY, sub_type=f.types[-1])
        else:
            w.add_key_value(k, f.contents(), main)
    for t in r.tensors:
        w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)
    w.add_tensor(name, q4_k_bytes(8, 43), raw_dtype=GGMLQuantizationType.Q4_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    print(f"tool under test: {TOOL}")
    from gguf import GGUFReader

    with tempfile.TemporaryDirectory(prefix="taillis-r3-test-") as td:
        tmp = Path(td)
        flat = tmp / "flat.gguf"
        build_flat(flat)
        parent_sha = sha256_file(flat)

        # the artifact's shadow identity, computed with the module's OWN synth
        # direction (the golden below guards this very computation)
        spec_map = {"blk.0.ffn_gate_up_exps": SPEC, "blk.0.ffn_down_exps": SPEC,
                    "blk.1.ffn_gate_up_exps": SPEC, "blk.1.ffn_down_exps": SPEC}
        fd = os.open(flat, os.O_RDONLY)
        synth_sha, _ = cc.canonical_stream_sha(cc.shadow_entries(GGUFReader(flat), fd, spec_map))
        os.close(fd)

        mixed = tmp / "mixed.gguf"
        # the e2-mixed allocation shape: 4 overridden members, 8 q4_K siblings
        alloc = {"blk.0.ffn_gate_up_exps": {0: "q8_0"},
                 "blk.0.ffn_down_exps": {0: "q6_K"},
                 "blk.1.ffn_gate_up_exps": {0: "q6_K"},
                 "blk.1.ffn_down_exps": {2: "q8_0"}}
        build_grouped(mixed, flat, parent_sha, synth_sha, alloc, control=False, spec=spec_map)

        print("part 1: fixtures are what they claim")
        r = GGUFReader(mixed)
        names = [t.name for t in r.tensors]
        check("mixed artifact has 12 members in v1 shape",
              len(names) == 14 and sum(1 for n in names if ".g" in n) == 12, str(names))
        check("mixed types read back non-uniform",
              len({int(t.tensor_type) for t in r.tensors if ".g0.weight" in t.name}) == 3,
              str({int(t.tensor_type) for t in r.tensors if ".g" in t.name}))

        print("part 2: the uniform shadow, built by the REAL group_experts.py")
        shadow = tmp / "shadow.gguf"
        # --format v0 is EXPLICIT, not the default: part 8 below asserts the v0
        # declaration path, and the writer's default is v1 (layout.md §8.4). A case
        # that leaned on the default changed meaning silently when the default moved.
        cp = subprocess.run([sys.executable, str(GROUPER), "--src", str(flat),
                             "--out", str(shadow), "--spec", "43,43,42",
                             "--format", "v0"],
                            capture_output=True, text=True)
        check("group_experts build exits 0", cp.returncode == 0, cp.stdout + cp.stderr)
        fd2 = os.open(shadow, os.O_RDONLY)
        file_sha, _ = cc.canonical_stream_sha(cc.file_entries(fd2, GGUFReader(shadow)))
        os.close(fd2)
        check("BOTH canonical-stream directions agree on the synthetic fixture",
              file_sha == synth_sha, f"{file_sha} vs {synth_sha}")
        global GOLDEN
        if GOLDEN == "PENDING":
            print(f"  [pin ] GOLDEN canonical-stream digest = {synth_sha} "
                  "(pin this constant in the test file and re-run)")
            sys.exit("FATAL: GOLDEN constant not pinned yet — pin "
                     f"{synth_sha} and re-run")
        check("GOLDEN canonical-stream digest matches the pinned §9.2 constant",
              synth_sha == GOLDEN, f"got {synth_sha}, want {GOLDEN}")

        print("part 3: POSITIVE full chain")
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat),
                      "--shadow", str(shadow))
        check("positive chain exits 0", cp.returncode == 0, cp.stdout + cp.stderr)
        check("G1 reports 4 logicals / 12 members, v1-derived",
              "4 logical(s), 12 member(s), declaration v1-derived" in cp.stdout, cp.stdout)
        check("G4 synth digest printed and equals the pinned golden",
              f"G4: synth  {GOLDEN}  ==" in cp.stdout, cp.stdout)
        check("G4 file digest printed and EQUALS the synth digest",
              f"G4: file   {GOLDEN}" in cp.stdout, cp.stdout)
        check("G4 shadow slabs 12/12", "shadow slabs 12/12" in cp.stdout, cp.stdout)
        check("G5 siblings 8/8 HARD", "siblings 8/8" in cp.stdout and "HARD" in cp.stdout,
              cp.stdout)

        print("part 4: POSITIVE control (byte-conserving equal-size permutation)")
        ctrl_ok = tmp / "control_ok.gguf"
        # swap the q8_0 from g0 (43) to g1 (43) — equal ne[2], same logical
        # width: total bytes exactly conserved
        build_grouped(ctrl_ok, flat, parent_sha, synth_sha,
                      {**alloc, "blk.0.ffn_gate_up_exps": {0: "q4_K", 1: "q8_0"}},
                      control=True, spec=spec_map)
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat),
                      "--control", str(ctrl_ok))
        check("G6 control gates OK", "G6: control gates OK" in cp.stdout, cp.stdout)
        check("G6 multiset conserved", "type multiset conserved" in cp.stdout, cp.stdout)
        check("G6 assignment differs", "assignment differs" in cp.stdout, cp.stdout)

        print("part 5: the EIGHT negatives — each exit 1 NAMING its mechanism")
        # (a) mixed artifact without the shadow key
        nokey = tmp / "mixed_nokey.gguf"
        build_grouped(nokey, flat, parent_sha, None, alloc, control=False, spec=spec_map)
        cp = run_tool("--artifact", str(nokey), "--parent", str(flat), expect=1)
        check("(a) missing key FAILS naming R3.2", "R3.2" in cp.stdout and "REQUIRED" in cp.stdout,
              cp.stdout[-400:])

        # (b) shadow key with one hex char flipped
        badkey = tmp / "mixed_badkey.gguf"
        flipped = ("0" if synth_sha[0] != "0" else "1") + synth_sha[1:]
        build_grouped(badkey, flat, parent_sha, flipped, alloc, control=False, spec=spec_map)
        cp = run_tool("--artifact", str(badkey), "--parent", str(flat), expect=1)
        check("(b) flipped key FAILS naming 'shadow identity mismatch'",
              "shadow identity mismatch" in cp.stdout, cp.stdout[-400:])

        # (c) --parent pointing at a DIFFERENT flat file
        flat2 = tmp / "flat2.gguf"
        build_flat(flat2, salt=1.0)
        check("(c) the wrong parent really differs by bytes",
              sha256_file(flat2) != parent_sha)
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat2), expect=1)
        check("(c) wrong parent FAILS G3 naming BOTH sha values",
              f"{PARENT_KEY} {parent_sha} != sha256({flat2}) {sha256_file(flat2)}" in cp.stdout,
              cp.stdout[-500:])
        check("(c) wrong parent FAILS G4 naming 'shadow identity mismatch'",
              "shadow identity mismatch" in cp.stdout, cp.stdout[-500:])

        # (d) UNIFORM artifact carrying a shadow key (FORBIDDEN)
        uniform_key = tmp / "uniform_key.gguf"
        build_grouped(uniform_key, flat, parent_sha, synth_sha, {}, control=False,
                      spec=spec_map)
        cp = run_tool("--artifact", str(uniform_key), "--parent", str(flat), expect=1)
        check("(d) uniform artifact with the key FAILS naming FORBIDDEN",
              "FORBIDDEN" in cp.stdout and "R3.2" in cp.stdout, cp.stdout[-400:])

        # (e) control moving the big type onto a DIFFERENT-SIZE member:
        #     q8_0 2048-elem row = 2176 B vs q4_K 1152 B; 43 -> 42 experts
        #     changes total bytes by 1024 B > the 384 B bound
        ctrl_size = tmp / "control_size.gguf"
        build_grouped(ctrl_size, flat, parent_sha, synth_sha,
                      {**alloc, "blk.0.ffn_gate_up_exps": {0: "q4_K", 2: "q8_0"}},
                      control=True, spec=spec_map)
        check("(e) the size permutation really changes payload bytes",
              abs(ctrl_size.stat().st_size - ctrl_ok.stat().st_size) > 384)
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat),
                      "--control", str(ctrl_size), expect=1)
        check("(e) different-SIZE member FAILS the ±384 B gate naming the delta",
              "differs by" in cp.stdout and "384" in cp.stdout, cp.stdout[-500:])

        # (f) control identical to the allocated artifact
        ctrl_ident = tmp / "control_ident.gguf"
        build_grouped(ctrl_ident, flat, parent_sha, synth_sha, alloc, control=True,
                      spec=spec_map)
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat),
                      "--control", str(ctrl_ident), expect=1)
        check("(f) identical control FAILS naming IDENTICAL",
              "IDENTICAL" in cp.stdout, cp.stdout[-400:])

        # (g) non-uniform partition across logicals
        nonuni = tmp / "nonuniform.gguf"
        build_grouped(nonuni, flat, parent_sha, synth_sha, alloc, control=False,
                      spec={"blk.0.ffn_gate_up_exps": SPEC,
                            "blk.0.ffn_down_exps": [50, 50, 28],
                            "blk.1.ffn_gate_up_exps": SPEC,
                            "blk.1.ffn_down_exps": SPEC})
        cp = run_tool("--artifact", str(nonuni), "--parent", str(flat), expect=1)
        check("(g) non-uniform partition FAILS naming the mechanism",
              "non-uniform partition" in cp.stdout, cp.stdout[-400:])

        # (h) --shadow file with one flipped payload byte
        badshadow = tmp / "badshadow.gguf"
        h_member = flip_member_byte(shadow, badshadow)
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat),
                      "--shadow", str(badshadow), expect=1)
        check("(h) corrupted shadow FAILS naming 'shadow identity mismatch'",
              "shadow identity mismatch" in cp.stdout, cp.stdout[-500:])
        check("(h) corrupted shadow FAILS the slab re-run naming the member",
              "shadow slab mismatch" in cp.stdout and h_member in cp.stdout,
              cp.stdout[-500:])

        print("part 6: --rebuild-shadow against the REAL tool")
        cp = run_tool("--artifact", str(mixed), "--parent", str(flat),
                      "--rebuild-shadow", str(tmp / "rebuilt"))
        check("rebuild-shadow exits 0 (real group_experts.py + writer --verify)",
              cp.returncode == 0, cp.stdout + cp.stderr)
        check("rebuild stream equals the key",
              f"rebuild-shadow: == {SHADOW_KEY}" in cp.stdout, cp.stdout[-400:])
        rebuilt = tmp / "rebuilt" / "mixed.shadow-rebuilt.gguf"
        check("the rebuilt shadow file exists", rebuilt.exists(), str(rebuilt))

        print("part 7: --shadow WITHOUT --parent (layout.md §9.2's OR: the file "
              "direction alone verifies the key)")
        cp = run_tool("--artifact", str(mixed), "--shadow", str(shadow))
        check("--shadow without --parent exits 0", cp.returncode == 0,
              cp.stdout + cp.stderr)
        check("the file direction verified the key",
              f"G4: file   {GOLDEN}" in cp.stdout, cp.stdout)
        check("the skipped synth direction is VISIBLE, not a failure",
              "synth direction skipped" in cp.stdout and "cannot verify" not in cp.stdout,
              cp.stdout)

        print("part 8: the v0 artifact path (group-key declaration, layout.md §2/§8.1)")
        cp = run_tool("--artifact", str(shadow), "--parent", str(flat))
        check("a v0-declared artifact (the writer's own shadow) exits 0",
              cp.returncode == 0, cp.stdout + cp.stderr)
        check("G1 reports the v0 key declaration",
              "declaration v0 key" in cp.stdout, cp.stdout)
        check("G5 covers all 12 sibling members of the uniform shadow",
              "siblings 12/12" in cp.stdout, cp.stdout)
        v0stray = tmp / "shadow-stray.gguf"
        add_stray_member(shadow, v0stray, "blk.9.ffn_gate_up_exps.g7.weight")
        cp = run_tool("--artifact", str(v0stray), "--parent", str(flat), expect=1)
        check("v0 UNCLAIMED member-shaped tensor FAILS naming the mechanism",
              f"not named in {GROUP_KEY}" in cp.stdout and
              "blk.9.ffn_gate_up_exps.g7.weight" in cp.stdout, cp.stdout[-500:])

        print("part 9: G5 WARN mode (imatrix recorded — layout.md §9.4: it names both "
              "possibilities, it never silently passes)")
        mixed_im = tmp / "mixed_im.gguf"
        build_grouped(mixed_im, flat, parent_sha, synth_sha, alloc, control=False,
                      spec=spec_map, imatrix=True)
        im_bad = tmp / "mixed_im_bad.gguf"
        # corrupt a SIBLING member (q4_K, verified by G5) — corrupting an
        # override member (q8_0/q6_K) would be invisible to G5 by design: the
        # checker never compares overridden members against the parent (their
        # bytes are requantised; drift there is the token-line detectors' job,
        # layout.md §9.3.4)
        im_member = flip_member_byte(mixed_im, im_bad,
                                     "blk.0.ffn_gate_up_exps.g1.weight")
        cp = run_tool("--artifact", str(im_bad), "--parent", str(flat))
        check("G5 WARN run exits 0 (warn-only, never a silent pass)",
              cp.returncode == 0, cp.stdout + cp.stderr)
        check("the WARN line prints, naming the member and both possibilities",
              "WARN" in cp.stdout and im_member in cp.stdout
              and "wrong parent or wrong weights" in cp.stdout, cp.stdout)
        check("G5 counts 7/8 siblings under imatrix (the corrupt one named)",
              "siblings 7/8" in cp.stdout and "WARN-only" in cp.stdout, cp.stdout)
        # asymmetry: the SAME corruption WITHOUT the imatrix key is a HARD failure
        mixed_plain = tmp / "mixed_plain.gguf"
        build_grouped(mixed_plain, flat, parent_sha, synth_sha, alloc, control=False,
                      spec=spec_map, imatrix=False)
        plain_bad = tmp / "mixed_plain_bad.gguf"
        flip_member_byte(mixed_plain, plain_bad, "blk.0.ffn_gate_up_exps.g1.weight")
        cp = run_tool("--artifact", str(plain_bad), "--parent", str(flat), expect=1)
        check("asymmetry: the same corruption without imatrix_sha256 FAILS HARD",
              cp.returncode == 1 and "HARD" in cp.stdout
              and "wrong parent or wrong weights" in cp.stdout, cp.stdout[-400:])

    print(f"all {COUNT} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
