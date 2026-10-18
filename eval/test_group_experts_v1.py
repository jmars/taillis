#!/usr/bin/env python3
"""Self-test for tools/group_experts.py --format v1 (taillis/layout.md §8) — a test that CAN fail.

Builds a small SYNTHETIC FLAT GGUF fixture with gguf-py (two expert tensors with
ne[2] == expert_count == 128, two non-expert tensors; ordinary string/u32 KV),
then drives tools/group_experts.py as a subprocess and asserts:

  1. v1 build (--format v1 --groups 3): the output has format_version == 1
     (u32), has NO taillis.expert_groups, both logicals carry g0/g1/g2 members
     with ne[2] = 43+43+42 = 128 and no flat expert tensor survives, and
     --verify --format v1 passes (byte-conservation gates 1/5 against the flat
     fixture, R1 parent_sha256 == sha(src));
  1b. the DEFAULT format (no --format at all) is v1 — layout.md §8.3/§8.4: the
     writer emits format_version == 1 with no group key, and --verify with no
     --format accepts it. The one case that must NOT pass an explicit flag,
     because the default IS the thing under test;
  2. v0 build on the SAME fixture (--format v0, EXPLICIT — see below): the key is
     present exactly as today (6 entries), format_version == 0 (the pre-R2
     writer stamped 1; reclaimed, layout.md §8.1), and --verify --format v0
     passes;
  3. every malformed fixture from tools/make_malformed_grouped.py (the two v0
     and the four v1 ones) is REFUSED as a group_experts SOURCE — member-shaped
     tensor names mean "not a flat parent", which holds for v0 sources (key +
     names) and v1 sources (names only) alike; nothing is written to --out.
     The derivation-level defects the v1 fixtures carry are fixtures for
     AUBIER'S R2 READER (docs/R2-reader-request.md) — no such reader exists on
     this side, and the output says so rather than pretending to test one;
  4. --verify dispatch NEGATIVES: a v1 output with the group key re-added must
     exit 1 naming the key; a v0 output verified as v1 must exit 1; a v1
     output verified as v0 (--format v0, EXPLICIT) must exit 1; and two members'
     PAYLOAD SLABS exchanged (equal-size g0/g1, layout untouched) must fail
     gate 1 naming the payload sha mismatch — the derived ascending-k order is
     exactly the seam §8 exists to close, so the byte comparison along it must
     have teeth.

Every case except 1b passes --format EXPLICITLY, so no assertion changes meaning
when a default moves (2026-10-18: the default did move, v0 -> v1, layout.md §8.4;
this test was written against explicit flags and only case 2/4's absent-flag
shortcuts had to be made explicit).

The v1 key-re-add rewrite copies every KV and tensor payload verbatim (byte
gates still pass), so the ONLY failing gate is the v1 key-present one — the
negative names its mechanism, it does not fail incidentally.

Needs gguf-py (venvs/convert/bin/python3); re-execs itself under that venv
python when gguf is not importable here, exactly like eval/test_stamp_gguf.py.
No GPU, no model file, runs in a temp dir under /tmp.

Gate: this process's exit code. 0 = pass, nonzero = fail.

Failing-before / passing-after asymmetries:
  * for the v0 stamp history: the pre-R2 tool had no --format flag at all
    (`--format` → argparse error, exit 2), and its v0 output stamped
    format_version = 1, which THIS test asserts is 0 — the pre-R2 revision fails
    in parts 1/2/4;
  * for the DEFAULT flip (part 1b): the tool as of R2 (default v0) builds
    format_version == 0 WITH the group key when no --format is given, so part 1b
    fails on it — the default is asserted, not assumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "group_experts.py"
MALFORMED = REPO / "tools" / "make_malformed_grouped.py"
GROUP_KEY = "taillis.expert_groups"
FORMAT_KEY = "taillis.format_version"
PARENT_KEY = "taillis.parent_sha256"

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


def run_tool(*argv: str, expect: int | None = 0) -> subprocess.CompletedProcess:
    cp = subprocess.run([sys.executable, str(TOOL), *argv],
                        capture_output=True, text=True)
    if expect is not None and cp.returncode != expect:
        sys.exit(f"FATAL: group_experts {' '.join(argv[:8])}... exited {cp.returncode}, "
                 f"expected {expect}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    return cp


def build_fixture(path: Path) -> None:
    """A synthetic FLAT file: 2 expert tensors (ne[2]=128) + 2 non-expert ones."""
    from gguf import GGUFWriter, GGUFValueType

    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    w.add_key_value("general.architecture", "v1test", GGUFValueType.STRING)
    w.add_key_value("general.name", "group_experts v1 self-test fixture", GGUFValueType.STRING)
    w.add_key_value("v1test.expert_count", np.uint32(128), GGUFValueType.UINT32)
    # numpy shape (ne2, ne1, row) -> ggml ne = (row, ne1, ne2): ne[2] = 128 experts
    w.add_tensor("tok_emb.weight", np.arange(64, dtype=np.float16).reshape(8, 8) / 7.0)
    w.add_tensor("blk.0.attn_q_a.weight", np.linspace(0.5, 2.0, 32, dtype=np.float32).reshape(4, 8))
    w.add_tensor("blk.0.ffn_gate_up_exps.weight",
                 (np.arange(128 * 4 * 8, dtype=np.float32) % 17.0).reshape(128, 4, 8))
    w.add_tensor("blk.0.ffn_down_exps.weight",
                 ((np.arange(128 * 2 * 8, dtype=np.float32) + 5.0) % 29.0).reshape(128, 2, 8))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_fields(path: Path) -> dict[str, tuple[str, object]]:
    from gguf import GGUFReader
    r = GGUFReader(path)
    return {k: (f.types[0].name, f.contents()) for k, f in r.fields.items()}


def member_ne2(path: Path) -> dict[str, list[int]]:
    """<logical with .weight> -> member ne[2] values, from the output's names."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    out: dict[str, list[int]] = {}
    for t in r.tensors:
        name = t.name
        if name.endswith(".weight") and ".g" in name:
            logical, _, tail = name[:-len(".weight")].rpartition(".g")
            if tail.isdigit():
                out.setdefault(logical + ".weight", []).append(int(t.shape[2]))
    return {k: sorted(v) for k, v in out.items()}


def re_add_group_key(v1_out: Path, dst: Path) -> None:
    """Copy every KV and tensor of v1_out verbatim, then add taillis.expert_groups.

    The copy preserves all payload bytes, so a later --verify failure can only
    be the v1 key-present gate — the negative names its mechanism.
    """
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    r = GGUFReader(v1_out)
    w = GGUFWriter(str(dst), arch="", use_temp_file=False)
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        main = f.types[0]
        if main == GGUFValueType.ARRAY:
            w.add_key_value(k, f.contents(), GGUFValueType.ARRAY, sub_type=f.types[-1])
        else:
            w.add_key_value(k, f.contents(), main)
    entries = []
    for t in r.tensors:
        if ".g" in t.name and t.name.endswith(".weight"):
            logical = t.name[:-len(".weight")].rsplit(".g", 1)[0]
            entries.append(f"{logical}.g{t.name[:-len('.weight')].rsplit('.g', 1)[1]}")
        w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)
    w.add_key_value(GROUP_KEY, entries, GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def swap_member_payloads(src: Path, logical: str, name_a: str, name_b: str, dst: Path) -> None:
    """Copy src to dst with two members' DATA regions exchanged, layout untouched.

    Only valid for EQUAL-size members ( asserted): the regions swap in place,
    so offsets, sizes and alignment padding all stay exactly as written.
    """
    from gguf import GGUFReader
    r = GGUFReader(src)
    offs: dict[str, int] = {}
    lens: dict[str, int] = {}
    for t in r.tensors:
        if t.name in (f"{logical}.{name_a}.weight", f"{logical}.{name_b}.weight"):
            offs[t.name] = t.data_offset
            lens[t.name] = t.data.nbytes
    a_name, b_name = f"{logical}.{name_a}.weight", f"{logical}.{name_b}.weight"
    assert set(offs) == {a_name, b_name}, f"members not found: {sorted(offs)}"
    oa, la = offs[a_name], lens[a_name]
    ob, lb = offs[b_name], lens[b_name]
    assert la == lb, f"unequal member sizes ({la} vs {lb}); cannot swap in place"
    assert oa < ob, "members not in ascending offset order"
    raw = bytearray(src.read_bytes())
    pa = bytes(raw[oa:oa + la])
    pb = bytes(raw[ob:ob + lb])
    assert pa != pb, "payloads identical; swap would prove nothing"
    raw[oa:oa + la] = pb
    raw[ob:ob + lb] = pa
    dst.write_bytes(bytes(raw))


def main() -> int:
    print(f"tool under test: {TOOL}")
    with tempfile.TemporaryDirectory(prefix="taillis-r2-test-") as td:
        tmp = Path(td)
        flat = tmp / "flat-fixture.gguf"
        build_fixture(flat)

        # ---- part 1: the v1 build ----------------------------------------
        print("part 1: v1 build (--format v1 --groups 3)")
        v1 = tmp / "v1-g3.gguf"
        cp = run_tool("--src", str(flat), "--out", str(v1), "--format", "v1",
                      "--groups", "3", "--provenance", "test_run=synthetic")
        check("v1 build exits 0", cp.returncode == 0, cp.stdout + cp.stderr)
        check("v1 stdout says 'format      : v1'", "format      : v1" in cp.stdout)
        f = read_fields(v1)
        check("format_version == 1 (u32)", f.get(FORMAT_KEY) == ("UINT32", 1), str(f.get(FORMAT_KEY)))
        check("taillis.expert_groups ABSENT", GROUP_KEY not in f)
        check("parent_sha256 present (R1)", PARENT_KEY in f)
        check("provenance stamped", "taillis.provenance.test_run" in f)
        m = member_ne2(v1)
        check("members cover [0,128) as 43+43+42",
              m == {"blk.0.ffn_gate_up_exps.weight": [42, 43, 43],
                    "blk.0.ffn_down_exps.weight": [42, 43, 43]}, str(m))
        from gguf import GGUFReader
        v1names = {t.name for t in GGUFReader(v1).tensors}
        check("no flat expert tensor survives",
              "blk.0.ffn_gate_up_exps.weight" not in v1names
              and "blk.0.ffn_down_exps.weight" not in v1names)
        check("non-expert tensors survive verbatim by NAME",
              {"tok_emb.weight", "blk.0.attn_q_a.weight"} <= v1names)
        run_tool("--verify", "--src", str(flat), "--out", str(v1), "--format", "v1")
        print("  [ok  ] --verify --format v1 exits 0 (gates 1/5 + R1)")

        # ---- part 1b: the DEFAULT format is v1 (layout.md §8.3/§8.4) ------
        # The only case that deliberately passes NO --format: the default is
        # the thing under test. Pre-flip (R2, default v0) this part FAILS.
        print("part 1b: default format (no --format) is v1")
        dflt = tmp / "default.gguf"
        cp = run_tool("--src", str(flat), "--out", str(dflt), "--groups", "3")
        check("default build exits 0", cp.returncode == 0, cp.stdout + cp.stderr)
        check("default stdout says 'format      : v1'", "format      : v1" in cp.stdout,
              cp.stdout[:400])
        f = read_fields(dflt)
        check("default build stamps format_version == 1 (u32)",
              f.get(FORMAT_KEY) == ("UINT32", 1), str(f.get(FORMAT_KEY)))
        check("default build emits NO group key", GROUP_KEY not in f)
        run_tool("--verify", "--src", str(flat), "--out", str(dflt))
        print("  [ok  ] --verify with no --format accepts the default build")

        # ---- part 2: the v0 build on the same fixture --------------------
        print("part 2: v0 build (--format v0, explicit) — today's semantics, unchanged")
        v0 = tmp / "v0-g3.gguf"
        cp = run_tool("--src", str(flat), "--out", str(v0), "--format", "v0", "--groups", "3")
        check("v0 build exits 0", cp.returncode == 0, cp.stdout + cp.stderr)
        f = read_fields(v0)
        entries = f.get(GROUP_KEY, ("ABSENT", None))[1]
        check("group key present, string array of 6 entries",
              f.get(GROUP_KEY, (None, None))[0] == "ARRAY" and isinstance(entries, list)
              and len(entries) == 6, str(f.get(GROUP_KEY))[:120])
        check("format_version == 0 (u32; was 1 pre-R2, reclaimed — layout.md §8.1)",
              f.get(FORMAT_KEY) == ("UINT32", 0), str(f.get(FORMAT_KEY)))
        run_tool("--verify", "--src", str(flat), "--out", str(v0), "--format", "v0")
        print("  [ok  ] --verify --format v0 exits 0")
        cp = run_tool("--verify", "--src", str(flat), "--out", str(v0), expect=1)
        check("--verify with no --format REJECTS a v0 build — the default moved, so an "
              "absent flag is no longer 'v0'", "expected 1 (v1)" in cp.stdout,
              cp.stdout[:300])

        # ---- part 3: every malformed fixture is refused as a SOURCE ------
        print("part 3: malformed fixtures refused as group_experts sources")
        fx = tmp / "malformed"
        cp = subprocess.run([sys.executable, str(MALFORMED), "--out-dir", str(fx)],
                            capture_output=True, text=True)
        check("make_malformed_grouped exits 0", cp.returncode == 0, cp.stdout + cp.stderr)
        for name in ("taillis-t21-bad-sum.gguf",
                     "taillis-t21-bad-flat-and-grouped.gguf",
                     "taillis-r2-bad-dup-k.gguf",
                     "taillis-r2-bad-flat-and-grouped.gguf",
                     "taillis-r2-bad-sum.gguf",
                     "taillis-r2-non-expert-member.gguf"):
            src = fx / name
            out = tmp / f"never-{name}"
            cp = run_tool("--src", str(src), "--out", str(out), "--format", "v1",
                          "--groups", "3", expect=None)
            check(f"{name} refused (exit != 0)", cp.returncode != 0, cp.stdout + cp.stderr)
            # SystemExit(str) prints the refusal to stderr, hence both streams
            check(f"{name} refusal names 'flat parent'",
                  "flat parent" in cp.stdout + cp.stderr, (cp.stdout + cp.stderr)[:200])
            check(f"{name} refusal wrote nothing", not out.exists())
        print("  [ok  ] all six refused; the v1 fixtures' derivation-level defects "
              "are AUBIER-reader fixtures (docs/R2-reader-request.md) — no v1 reader "
              "exists on this side; what is tested here is the writer refusing "
              "any non-flat source")

        # ---- part 4: --verify dispatch negatives -------------------------
        print("part 4: --verify dispatch negatives")
        poisoned = tmp / "v1-with-key.gguf"
        re_add_group_key(v1, poisoned)
        f = read_fields(poisoned)
        check("poisoned copy carries the key AND version 1",
              GROUP_KEY in f and f[FORMAT_KEY] == ("UINT32", 1))
        cp = run_tool("--verify", "--src", str(flat), "--out", str(poisoned),
                      "--format", "v1", expect=1)
        check("v1 output with the key re-added FAILS naming the key",
              "expert_groups present in a v1 output" in cp.stdout, cp.stdout[:300])
        cp = run_tool("--verify", "--src", str(flat), "--out", str(v0),
                      "--format", "v1", expect=1)
        check("v0 output verified as v1 FAILS", "expected 1 (v1)" in cp.stdout,
              cp.stdout[:300])
        cp = run_tool("--verify", "--src", str(flat), "--out", str(v1),
                      "--format", "v0", expect=1)
        # the v0 dispatch fails EARLY on the absent key — that IS the mechanism
        check("v1 output verified as v0 (explicit) FAILS on the absent key",
              f"{GROUP_KEY} absent from" in cp.stdout, cp.stdout[:300])

        # ---- part 4b: the payload-permutation negative (the §8 seam) ------
        # v1 order is DERIVED (ascending k) and gate 1 compares bytes along
        # that order: exchange g0/g1's payload slabs (equal sizes, 43/43) and
        # verify must fail naming the sha mismatch — not accept the file.
        print("part 4b: payload-swap negative (derived order vs byte gates)")
        swapped = tmp / "v1-swapped.gguf"
        swap_member_payloads(v1, "blk.0.ffn_gate_up_exps", "g0", "g1", swapped)
        cp = run_tool("--verify", "--src", str(flat), "--out", str(swapped),
                      "--format", "v1", expect=1)
        check("swapped member payloads FAIL gate 1 naming the sha mismatch",
              "payload sha" in cp.stdout, cp.stdout[:300])
        check("the swap is caught for BOTH swapped members",
              cp.stdout.count("payload sha") >= 2, cp.stdout[:400])

    print(f"all {COUNT} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
