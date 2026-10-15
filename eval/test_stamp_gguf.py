#!/usr/bin/env python3
"""Self-test for tools/stamp_gguf.py — a test that CAN fail.

Builds a tiny SYNTHETIC GGUF fixture with gguf-py (three tensors, one of them
Q8_0-quantized; ordinary string/u32/f32/array KV, plus one UNKNOWN
non-taillis KV), then drives tools/stamp_gguf.py as a subprocess and asserts:

  1. the fixture is what it claims: exactly 3 tensors with the expected names
     and dtypes (F16/F32/Q8_0, read back through GGUFReader, non-empty
     payloads), the expected KV present, no taillis.* keys yet;
  2. stamping one u32 + two strings: --show lists all three stamped keys,
     --verify passes (every tensor payload sha256 identical);
  3. re-stamping with the SAME arguments leaves the key SET unchanged
     (idempotence) and payloads still identical;
  4. stamping an OVERWRITTEN value: the new value wins in --show and in the
     parsed file, payloads still identical;
  5. the UNKNOWN non-taillis KV and the ordinary KV survive verbatim
     (same python value AND same GGUF value type);
  5b. NEGATIVE: flipping ONE payload byte in the stamped output makes
     --verify exit 1 and name the mechanism (payload sha mismatch) —
     the gate can actually fail;
  6. the refusals refuse: --out onto an existing path exits nonzero, and an
     --out outside the write scope is refused (nothing written).

Needs gguf-py (venvs/convert/bin/python3); re-execs itself under that venv
python when gguf is not importable here, exactly like eval/test_corpus.py
delegates to its venv.  No GPU, no model file, runs in a temp dir under /tmp.

Gate: this process's exit code. 0 = pass, nonzero = fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "stamp_gguf.py"

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
        sys.exit(f"FATAL: stamp_gguf {' '.join(argv[:6])}... exited {cp.returncode}, "
                 f"expected {expect}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}")
    return cp


def build_fixture(path: Path) -> None:
    from gguf import GGUFWriter, GGUFValueType, GGMLQuantizationType

    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    w.add_key_value("general.architecture", "r1test", GGUFValueType.STRING)
    w.add_key_value("general.name", "stamp self-test fixture", GGUFValueType.STRING)
    w.add_key_value("custom.vendor.fingerprint", "must-survive-verbatim", GGUFValueType.STRING)
    w.add_key_value("r1test.block_count", np.uint32(2), GGUFValueType.UINT32)
    w.add_key_value("r1test.embedding_scale", np.float32(1.5), GGUFValueType.FLOAT32)
    w.add_key_value("r1test.names", ["alpha", "beta"], GGUFValueType.ARRAY,
                    sub_type=GGUFValueType.STRING)
    # F16 2-D
    w.add_tensor("tok_emb.weight", np.arange(64, dtype=np.float16).reshape(8, 8) / 7.0)
    # F32 1-D
    w.add_tensor("out_norm.weight", np.linspace(0.5, 2.0, 16, dtype=np.float32))
    # Q8_0 2-D, raw blocks: 32 rows x 34 bytes (one block = 32 elems = 34 bytes)
    w.add_tensor("blk.0.ffn_exps.weight",
                 (np.arange(32 * 34, dtype=np.uint8) * 31 % 251).reshape(32, 34),
                 raw_dtype=GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_fields(path: Path) -> dict[str, tuple[str, object]]:
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {}
    for k, f in r.fields.items():
        out[k] = (f.types[0].name, f.contents())
    return out


def kv_set(path: Path) -> set[str]:
    return {k for k in read_fields(path) if not k.startswith("GGUF.")}


def main() -> int:
    print(f"tool under test: {TOOL}")
    tmp = Path(tempfile.mkdtemp(prefix="r1stamp-"))
    fixture = tmp / "fixture.gguf"
    build_fixture(fixture)
    print(f"fixture: {fixture} ({fixture.stat().st_size} bytes)")

    print("part 1: the fixture is what it claims")
    fields = read_fields(fixture)
    from gguf import GGUFReader
    rt = list(GGUFReader(fixture).tensors)
    check("3 tensors in fixture (read from the file)", len(rt) == 3,
          f"got {len(rt)}: {[t.name for t in rt]}")
    check("tensor names and dtypes as built",
          [(t.name, t.tensor_type.name) for t in rt] ==
          [("tok_emb.weight", "F16"), ("out_norm.weight", "F32"),
           ("blk.0.ffn_exps.weight", "Q8_0")],
          str([(t.name, t.tensor_type.name) for t in rt]))
    check("tensor payloads non-empty (a payload gate needs bytes to gate)",
          all(t.data.nbytes > 0 for t in rt))
    check("no taillis.* keys yet", not [k for k in fields if k.startswith("taillis.")])
    check("unknown non-taillis KV present",
          fields.get("custom.vendor.fingerprint") == ("STRING", "must-survive-verbatim"))

    print("part 2: stamp 1 u32 + 2 strings; --show lists all three; --verify passes")
    stamped = tmp / "stamped.gguf"
    args = ["--file", str(fixture),
            "--set-u32", "taillis.format_version=1",
            "--set-str", "taillis.provenance.converter=convert_hf_to_gguf.py@deadbeef",
            "--set-str", "taillis.provenance.source_checkpoint=mistralai/Mistral-Small-4-119B-2603@main",
            "--out", str(stamped)]
    cp = run_tool(*args)
    check("--out does not exist beforehand, now it does", stamped.exists())
    show = run_tool("--file", str(stamped), "--show").stdout
    check("--show lists format_version", "taillis.format_version: UINT32 1" in show)
    check("--show lists converter", "taillis.provenance.converter: STRING 'convert_hf_to_gguf.py@deadbeef'" in show)
    check("--show lists source_checkpoint", "taillis.provenance.source_checkpoint" in show)
    run_tool("--verify", "--src", str(fixture), "--out", str(stamped))
    print(f"  [ok  ] --verify exit 0 ({stamped.stat().st_size} bytes)")

    print("part 3: re-stamp with the SAME arguments -> same key set (idempotence)")
    inplace = tmp / "inplace.gguf"
    inplace.write_bytes(stamped.read_bytes())
    before = kv_set(inplace)
    run_tool("--file", str(inplace),
             "--set-u32", "taillis.format_version=1",
             "--set-str", "taillis.provenance.converter=convert_hf_to_gguf.py@deadbeef",
             "--set-str", "taillis.provenance.source_checkpoint=mistralai/Mistral-Small-4-119B-2603@main")
    after = kv_set(inplace)
    check("key set unchanged", before == after, f"{sorted(before ^ after)}")
    check("no .stamp.tmp left behind", not list(tmp.glob("*.tmp")))
    run_tool("--verify", "--src", str(stamped), "--out", str(inplace))

    print("part 4: overwrite -> new value wins, payloads still identical")
    run_tool("--file", str(inplace),
             "--set-str", "taillis.provenance.converter=some_other_converter@ff00ff")
    fields = read_fields(inplace)
    check("overwritten value wins",
          fields["taillis.provenance.converter"] == ("STRING", "some_other_converter@ff00ff"))
    check("other stamped key untouched",
          fields["taillis.provenance.source_checkpoint"] ==
          ("STRING", "mistralai/Mistral-Small-4-119B-2603@main"))
    run_tool("--verify", "--src", str(stamped), "--out", str(inplace))

    print("part 5: UNKNOWN non-taillis KV + ordinary KV survive verbatim")
    src_fields = read_fields(fixture)
    for key in ("custom.vendor.fingerprint", "general.name", "r1test.block_count",
                "r1test.embedding_scale", "r1test.names"):
        check(f"{key} verbatim (value+type)",
              fields.get(key) == src_fields.get(key),
              f"{fields.get(key)} != {src_fields.get(key)}")

    print("part 5b: a ONE-byte payload flip must make --verify FAIL (exit 1)")
    flipped = tmp / "flipped.gguf"
    flipped.write_bytes(stamped.read_bytes())
    rt = list(GGUFReader(flipped).tensors)
    t = rt[-1]                      # flip the last byte of the last tensor's payload
    off = t.data_offset + t.data.nbytes - 1
    with open(flipped, "r+b") as fh:
        fh.seek(off)
        old = fh.read(1)[0]
        fh.seek(off)
        fh.write(bytes([old ^ 0xFF]))
    cp = run_tool("--verify", "--src", str(fixture), "--out", str(flipped), expect=None)
    check("one-byte payload flip -> --verify exit 1", cp.returncode == 1,
          f"exit {cp.returncode}; stdout:\n{cp.stdout}")
    check("the failure names the mechanism (payload sha mismatch)",
          "payload sha mismatch" in cp.stdout)

    print("part 6: the refusals refuse")
    cp = run_tool("--file", str(fixture), "--set-str", "taillis.x=1",
                  "--out", str(stamped), expect=None)
    check("--out refuses an existing path", cp.returncode != 0)
    cp = run_tool("--file", str(fixture), "--set-str", "taillis.x=1",
                  "--out", str(REPO / "results" / "r1-must-not-exist.gguf"), expect=None)
    check("--out outside the write scope refused", cp.returncode != 0)
    check("refused write created nothing",
          not (REPO / "results" / "r1-must-not-exist.gguf").exists())
    cp = run_tool("--file", str(fixture), "--set-str", "general.name=hijacked",
                  "--out", str(tmp / "never.gguf"), expect=None)
    check("non-taillis namespace refused", cp.returncode != 0)
    check("refused stamp wrote nothing", not (tmp / "never.gguf").exists())

    print(f"\nALL {COUNT} CHECKS PASSED ({tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
