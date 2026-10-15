#!/usr/bin/env python3
"""Build a small N-block SLICE of our own F16 Mistral-Small-4 GGUF.

Why this exists (context, not decoration): llama.cpp's MoE matmul dispatch refuses
its fast MMF path for F16/BF16 when the ubatch exceeds 128 tokens
(`ggml-cuda/mmf.cu`, `ggml_cuda_should_use_mmf()`, `mul_mat_id` branch:
`else if (src0_ne[1] > 1024 && src1_ncols > 128) return false;`), and MMQ — MMF's
alternative — supports only quantized types. The gate is keyed on the REAL expert
row count (4096 here), so a generic small MoE will not do: its expert rows are
<= 1024 and it takes the other branch (`> 512`), which is a false green. This
script produces a probe artifact that reproduces the real shapes on a GPU small
enough to iterate on. It is NOT a shippable model.

What it writes
  * every non-block tensor of the source (`token_embd.weight`, `output.weight`,
    `output_norm.weight`), and
  * every tensor of blocks 0 .. N-1,
with tensor names byte-identical to the source (`blk.0....`, `blk.1....` — never
renumbered), all metadata key/value/type copied VERBATIM except
`<arch>.block_count`, and tensor data streamed in one pass (never the whole
source through memory).

Usage
    tools/make_slice_gguf.py [--layers N] [--out PATH] [--src PATH] [--chunk-mib M]
    tools/make_slice_gguf.py --verify [--layers N] [--out PATH] [--src PATH]

`--verify` is the load-bearing gate: it re-parses both files and diffs the
(name, ggml-shape, ggml-type) tuple of every tensor the slice is supposed to hold.

Prints COUNTS, SIZES and PATHS only.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.lazy import LazyChunkedTensor
except ImportError as e:  # pragma: no cover - environment error, not logic
    sys.exit(f"FATAL: gguf-py not importable ({e}); use venvs/convert/bin/python3")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO / "models" / "Mistral-Small-4-119B-2603-f16.gguf"

BLK_RE = re.compile(r"^blk\.(\d+)\.")
# Reserved intra-file keys the reader synthesises; they are not real metadata.
RESERVED_KV = ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count")


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(f) < 1024.0:
            return f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} PiB"


def resolve_src(src: Path) -> Path:
    """Follow the symlink: we must read the real file and must never write near it."""
    real = src.resolve(strict=True)
    if not real.is_file():
        raise SystemExit(f"FATAL: source is not a regular file: {real}")
    return real


def check_write_scope(out: Path) -> None:
    """Write scope is /tmp or the repo's (gitignored) models/ — nowhere else."""
    real = out.resolve()
    allowed = (Path("/tmp"), (REPO / "models").resolve())
    if not any(real == a or a in real.parents for a in allowed):
        raise SystemExit(
            f"FATAL: refusing to write {real}; write scope is /tmp or {REPO / 'models'}")


def select_tensors(reader: GGUFReader, layers: int) -> list:
    """Source file order: non-block tensors interleaved as they appear, then blocks 0..layers-1."""
    keep = []
    for t in reader.tensors:
        m = BLK_RE.match(t.name)
        if m is None:
            keep.append(t)
        elif int(m.group(1)) < layers:
            keep.append(t)
    return keep


def chunk_loaders(fd: int, offset: int, shape: tuple[int, ...], dtype: np.dtype,
                  chunk_bytes: int) -> list:
    """One loader per row-block of `shape`, pread()-ing straight out of the source."""
    itemsize = dtype.itemsize
    row_elems = 1
    for d in shape[1:]:
        row_elems *= d
    row_bytes = row_elems * itemsize
    rows_per_chunk = max(1, chunk_bytes // row_bytes)
    n_rows = shape[0]

    def make(off: int, nbytes: int, cshape: tuple[int, ...]):
        def load() -> np.ndarray:
            buf = os.pread(fd, nbytes, off)
            if len(buf) != nbytes:
                raise IOError(f"short read at {off}: got {len(buf)}, wanted {nbytes}")
            return np.frombuffer(buf, dtype=dtype).reshape(cshape)
        return load

    out = []
    for r0 in range(0, n_rows, rows_per_chunk):
        rows = min(n_rows, r0 + rows_per_chunk) - r0
        out.append(make(offset + r0 * row_bytes, rows * row_bytes, (rows,) + tuple(shape[1:])))
    return out


def build(src: Path, out: Path, layers: int, chunk_mib: int) -> None:
    real = resolve_src(src)
    check_write_scope(out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} already exists; remove it or pick another --out")

    reader = GGUFReader(real)
    arch = reader.fields["general.architecture"].contents()
    bc_key = f"{arch}.block_count"
    if bc_key not in reader.fields:
        raise SystemExit(f"FATAL: source has no {bc_key}")

    tensors = select_tensors(reader, layers)
    if not tensors:
        raise SystemExit("FATAL: no tensors selected")
    total = sum(t.n_bytes for t in tensors)

    print(f"source      : {real}")
    print(f"out         : {out}")
    print(f"layers      : {layers} ({bc_key} -> {layers})")
    print(f"tensors     : {len(tensors)} of {len(reader.tensors)}")
    print(f"data bytes  : {total} ({human(total)})")

    # arch="" keeps the writer from pre-adding general.architecture: every key comes
    # back verbatim from the source instead (add_string() no-ops on an empty value).
    writer = GGUFWriter(str(out), arch="", use_temp_file=False)
    writer.data_alignment = reader.alignment

    n_kv = 0
    for key, field in reader.fields.items():
        if key in RESERVED_KV:
            continue
        main = field.types[0]
        if main == GGUFValueType.ARRAY:
            writer.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=field.types[-1])
        else:
            writer.add_key_value(key, field.contents(), main)
        n_kv += 1
    # Overwrite in place: an existing key keeps its position, so kv order is unchanged.
    writer.add_key_value(bc_key, layers, GGUFValueType.UINT32)
    print(f"kv pairs    : {n_kv} (verbatim) + {bc_key} override")

    fd = os.open(real, os.O_RDONLY)
    chunk_bytes = chunk_mib * 1024 * 1024
    try:
        for t in tensors:
            dtype = t.data.dtype
            shape = t.data.shape
            writer.add_tensor(
                t.name,
                LazyChunkedTensor(
                    chunk_loaders(fd, t.data_offset, shape, dtype, chunk_bytes),
                    shape, dtype,
                ),
                raw_dtype=t.tensor_type,
            )
        print(f"tensor info : {len(writer.tensors[0])} entries written")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        os.close(fd)
    writer.close()

    print(f"wrote       : {out}  {out.stat().st_size} bytes ({human(out.stat().st_size)})")


def verify(src: Path, out: Path, layers: int) -> int:
    """Gate 1: every copied tensor must match the source in name AND shape AND type."""
    real = resolve_src(src)
    a, b = GGUFReader(real), GGUFReader(out)
    sa = {t.name: (tuple(int(x) for x in t.shape), t.tensor_type.name) for t in a.tensors}
    sb = {t.name: (tuple(int(x) for x in t.shape), t.tensor_type.name) for t in b.tensors}

    want = [t.name for t in select_tensors(a, layers)]
    expected = set(want)
    errors = []

    missing = expected - set(sb)
    extra = set(sb) - expected
    if missing:
        errors.append(f"MISSING {len(missing)}: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"UNEXPECTED {len(extra)}: {sorted(extra)[:5]}")
    for name in sorted(expected & set(sb)):
        if sa[name] != sb[name]:
            errors.append(f"MISMATCH {name}: src {sa[name]} vs slice {sb[name]}")

    arch = a.fields["general.architecture"].contents()
    bc = b.fields[f"{arch}.block_count"].contents()
    ec = b.fields[f"{arch}.expert_count"].contents()
    if bc != layers:
        errors.append(f"BLOCK_COUNT {bc} != {layers}")
    if ec != a.fields[f"{arch}.expert_count"].contents():
        errors.append(f"EXPERT_COUNT drift: src {a.fields[f'{arch}.expert_count'].contents()} vs slice {ec}")

    # Metadata verbatim, except the one key we own.
    bc_key = f"{arch}.block_count"
    nkv_a = len([k for k in a.fields if k not in RESERVED_KV])
    nkv_b = len([k for k in b.fields if k not in RESERVED_KV])
    drift = [
        k for k in a.fields
        if k not in RESERVED_KV and k != bc_key
        and (k not in b.fields or a.fields[k].contents() != b.fields[k].contents())
    ]
    if nkv_a != nkv_b:
        errors.append(f"KV_COUNT src {nkv_a} vs slice {nkv_b}")
    for k in drift:
        errors.append(f"KV_DRIFT {k}")

    print(f"src tensors : {len(sa)}")
    print(f"slice       : {len(sb)} tensors, {human(Path(out).stat().st_size)}")
    print(f"checked     : {len(expected & set(sb))} (name + shape + type)")
    print(f"{arch}.block_count = {bc}   {arch}.expert_count = {ec}")
    if errors:
        for e in errors:
            print(f"FAIL {e}")
        return 1
    print("GATE1 OK: every expected tensor matches the source in name, shape and type; "
          "metadata verbatim")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=Path("/tmp/mistral-small-4-2L-f16.gguf"))
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--chunk-mib", type=int, default=64)
    p.add_argument("--verify", action="store_true",
                   help="diff the slice against the source instead of building it")
    a = p.parse_args()
    if a.layers < 1:
        raise SystemExit("FATAL: --layers must be >= 1")
    if a.verify:
        return verify(a.src, a.out, a.layers)
    build(a.src, a.out, a.layers, a.chunk_mib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
