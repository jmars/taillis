#!/usr/bin/env python3
"""Stamp taillis.* provenance keys into a GGUF (taillis/layout.md §7, R1).

Sets or overwrites keys in the `taillis.*` namespace on an EXISTING GGUF,
copying everything else verbatim: every KV (skipping the reader's synthetic
GGUF.* entries) and every tensor payload, streamed in chunks (never the whole
file in memory) via gguf-py's LazyChunkedTensor.  Adding KV shifts the
tensor-data block, so byte equality of PAYLOADS — not of offsets — is the right
gate, and that is what --verify checks.

Keys are stamped at BUILD time on future artifacts.  Existing measured
artifacts (the F16, arms A/B/S, -g1, -g3, the shipped quant) are documented in
FINDINGS/memory and are NEVER rewritten to add keys — layout.md §7.3.  This
tool exists for new derivations, which is why in-place mode is safe: it is
atomic, and a failure before the rename leaves the original untouched.

Usage
    tools/stamp_gguf.py --file F --set-u32 taillis.format_version=0 \
        --set-str taillis.provenance.converter=convert_hf_to_gguf.py@<commit> [--out PATH]

    (format_version: 0 = the taillis.expert_groups key is the declaration, the
     v0 semantics; 1 = the grouping is derived from the member tensor names —
     layout.md §8.1.  A FLAT converted artifact is 0, or absent, which means
     the same; stamp 1 only on a v1-grouped artifact, which group_experts.py
     --format v1 writes itself.)
    tools/stamp_gguf.py --file F --show
    tools/stamp_gguf.py --verify --src SRC --out OUT

Behaviour
  --show     prints every taillis.* key (name, type, value) and writes nothing;
             exit 0.
  (default)  IN-PLACE and ATOMIC: writes "<file>.stamp.tmp" in the SAME
             directory, fsync, then os.replace onto the target.  --out writes
             to a separate path instead and REFUSES if it already exists.
             Every existing KV is copied verbatim (skip the synthetic keys),
             then the requested keys are set or OVERWRITTEN, so re-stamping
             with the same arguments yields the same key SET (idempotent).
  --verify   re-parses src and out; asserts (a) the tensor set is identical
             (names in order, ggml shapes, dtypes, per-tensor nbytes) and
             (b) EVERY tensor payload sha256 is identical.  Prints COUNTS,
             SIZES, PATHS only.  Exit code is the gate.

Keys must live in the taillis.* namespace — the stamper refuses to touch
anything else (semantics-bearing keys like general.* are not its business).
Values are parsed per --set-<type>; K=V is split on the FIRST '=' (values may
contain '=').

Write scope follows group_experts.py's allowlist shape, extended by this repo's
own models/ — the convert_f16.sh stamp step targets it on future builds:
    /var/lib/taillis/models, /tmp, <repo>/models
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.lazy import LazyChunkedTensor
except ImportError as e:  # pragma: no cover - environment error, not logic
    sys.exit(f"FATAL: gguf-py not importable ({e}); use venvs/convert/bin/python3")

REPO = Path(__file__).resolve().parent.parent
NAMESPACE = "taillis."
# Reserved intra-file keys the reader synthesises; they are not real metadata.
RESERVED_KV = ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count")
CHUNK_BYTES = 8 * 1024 * 1024


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(f) < 1024.0:
            return f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} PiB"


def check_write_scope(target: Path) -> None:
    """Write-scope allowlist, same shape as group_experts.py's check_write_scope,
    extended by this repo's own models/ — for BUILD OUTPUTS of future builds
    (the convert_f16.sh stamp step targets them).  The measured fixtures that
    already live there (mistral-small-4-2L-f16/q8_0.gguf) are never-rewrite
    artifacts (layout.md §7.3); the tool cannot tell them from a fresh build,
    so the operator must."""
    real = target.resolve()
    allowed = (Path("/var/lib/taillis/models"), Path("/tmp"), REPO / "models")
    if not any(real == a or a in real.parents for a in allowed):
        raise SystemExit(
            f"FATAL: refusing to write {real}; write scope is "
            f"/var/lib/taillis/models, /tmp or {REPO / 'models'}")


def chunk_loaders(fd: int, offset: int, nbytes: int) -> list:
    """One pread() loader per CHUNK_BYTES slab, straight out of the source file."""
    def make(off: int, n: int):
        def load() -> np.ndarray:
            buf = os.pread(fd, n, off)
            if len(buf) != n:
                raise IOError(f"short read at {off}: got {len(buf)}, wanted {n}")
            return np.frombuffer(buf, dtype=np.uint8)
        return load
    return [make(offset + o, min(CHUNK_BYTES, nbytes - o))
            for o in range(0, nbytes, CHUNK_BYTES)]


def sha256_payload(fd: int, offset: int, nbytes: int) -> str:
    """sha256 over [offset, offset+nbytes) read chunk-wise (never materialised whole)."""
    h = hashlib.sha256()
    off = 0
    while off < nbytes:
        n = min(CHUNK_BYTES, nbytes - off)
        buf = os.pread(fd, n, offset + off)
        if len(buf) != n:
            raise IOError(f"short read at {offset + off}: got {len(buf)}, wanted {n}")
        h.update(buf)
        off += n
    return h.hexdigest()


def parse_kv(arg: str, vtype: GGUFValueType) -> tuple[str, object]:
    """'K=V' -> (key, typed value); split on the FIRST '=' so values may contain '='."""
    key, sep, raw = arg.partition("=")
    if not sep or not key:
        raise SystemExit(f"FATAL: --set-{vtype.name.lower()} expects K=V, got '{arg}'")
    if not key.startswith(NAMESPACE):
        raise SystemExit(f"FATAL: '{key}' is outside the {NAMESPACE}* namespace")
    if vtype == GGUFValueType.STRING:
        return key, raw
    if vtype == GGUFValueType.BOOL:
        low = raw.strip().lower()
        if low in ("true", "1"):
            return key, True
        if low in ("false", "0"):
            return key, False
        raise SystemExit(f"FATAL: bad bool '{raw}' for {key} (true/false/1/0)")
    if vtype in (GGUFValueType.UINT32, GGUFValueType.INT32):
        try:
            n = int(raw, 0)
        except ValueError:
            raise SystemExit(f"FATAL: bad integer '{raw}' for {key}") from None
        lo, hi = (-2**31, 2**31 - 1) if vtype == GGUFValueType.INT32 else (0, 2**32 - 1)
        if not lo <= n <= hi:
            raise SystemExit(f"FATAL: {n} out of {vtype.name} range for {key}")
        return key, n
    if vtype == GGUFValueType.FLOAT32:
        try:
            return key, float(raw)
        except ValueError:
            raise SystemExit(f"FATAL: bad float '{raw}' for {key}") from None
    raise SystemExit(f"FATAL: unsupported value type {vtype.name}")  # pragma: no cover


def stamp(src_file: Path, sets: list[tuple[str, object, GGUFValueType]], out: Path | None) -> None:
    real = src_file.resolve(strict=True)
    target = out.resolve() if out is not None else real
    check_write_scope(target)
    if out is not None and target.exists():
        raise SystemExit(f"FATAL: {target} already exists; --out refuses to overwrite")
    # atomic in both modes: build the new file beside the target, rename at the
    # end.  The tmp name is process+time unique and reserved O_EXCL, so two
    # concurrent stamps of one target cannot interleave into one tmp.
    import time
    names = [k for k, _, _ in sets]
    dupes = {k for k in names if names.count(k) > 1}
    if dupes:
        raise SystemExit(f"FATAL: key(s) given twice in one invocation: {sorted(dupes)}")
    owned = {k for k, _, _ in sets}
    while True:
        tmp = target.with_name(f"{target.name}.stamp.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            os.close(os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            continue
    # ANY failure between here and the successful os.replace removes the tmp
    # and leaves the target untouched.  writer.close() is deliberately NOT
    # called on this path: it can itself raise (flush error), which would skip
    # the unlink — the tmp's open handle is dropped with the process instead.
    fd = -1
    try:
        reader = GGUFReader(real)
        writer = GGUFWriter(str(tmp), arch="", use_temp_file=False)
        writer.data_alignment = reader.alignment

        n_kv = 0
        for key, field in reader.fields.items():
            if key in RESERVED_KV or key in owned:
                continue  # owned keys are re-added below (set or overwrite)
            main = field.types[0]
            if main == GGUFValueType.ARRAY:
                writer.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=field.types[-1])
            else:
                writer.add_key_value(key, field.contents(), main)
            n_kv += 1

        fd = os.open(real, os.O_RDONLY)
        n_tensors = 0
        for t in reader.tensors:
            # dtype must be t.data's own: for F32/F16 tensors data.shape is in
            # ELEMENTS, so declaring uint8 there undercounts nbytes by 4x/2x
            # (same reasoning as group_experts.py).
            writer.add_tensor(t.name, LazyChunkedTensor(
                chunk_loaders(fd, t.data_offset, t.data.nbytes),
                tuple(int(x) for x in t.data.shape), t.data.dtype,
            ), raw_dtype=t.tensor_type)
            n_tensors += 1
        for key, val, vtype in sets:
            writer.add_key_value(key, val, vtype)
        # the chunk loaders are lazy — they run inside write_tensors_to_file(),
        # so the source fd must still be open here (closed just below)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        os.close(fd)
        fd = -1
        writer.close()
        # fsync the tmp, then the atomic rename; the original is untouched until here
        fdi = os.open(tmp, os.O_RDONLY)
        os.fsync(fdi)
        os.close(fdi)
        os.replace(tmp, target)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    try:  # durability of the rename itself; best effort, no gate
        dfd = os.open(target.parent, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    except OSError:
        pass

    total = target.stat().st_size
    print(f"file        : {real}" if out is None else f"src         : {real}")
    print(f"out         : {target}  {total} bytes ({human(total)})")
    print(f"kv          : {n_kv} verbatim + {len(sets)} stamped")
    for key, val, vtype in sets:
        shown = val if not isinstance(val, str) or len(val) <= 72 else val[:69] + "..."
        print(f"  {vtype.name:<7} {key} = {shown}")
    print(f"tensors     : {n_tensors} copied verbatim (payloads streamed, offsets shifted)")


def show(path: Path) -> int:
    reader = GGUFReader(path.resolve(strict=True))
    keys = [k for k in reader.fields if k not in RESERVED_KV and k.startswith(NAMESPACE)]
    print(f"file        : {path.resolve()}")
    print(f"kv          : {len(reader.fields) - len(RESERVED_KV)} total, {len(keys)} taillis.*")
    if not keys:
        print("no taillis.* keys (format_version absent = 0 = pre-R1 semantics)")
        return 0
    for k in sorted(keys):
        f = reader.fields[k]
        if f.types[0] == GGUFValueType.ARRAY:
            vals = list(f.contents())
            shown = ", ".join(repr(v) for v in vals[:4])
            if len(vals) > 4:
                shown += f", ... +{len(vals) - 4} more"
            print(f"{k}: ARRAY[{f.types[-1].name}] n={len(vals)} [{shown}]")
        else:
            v = f.contents()
            s = repr(v)
            if len(s) > 100:
                s = s[:100] + "..."
            print(f"{k}: {f.types[0].name} {s}")
    return 0


def verify(src: Path, out: Path) -> int:
    """Tensor-set + payload-sha gate.  Byte equality of PAYLOADS (not offsets) is
    the right gate: adding KV shifts the tensor-data block."""
    real, stamped = src.resolve(strict=True), out.resolve(strict=True)
    a, b = GGUFReader(real), GGUFReader(stamped)
    errors: list[str] = []

    ta, tb = list(a.tensors), list(b.tensors)
    if len(ta) != len(tb):
        errors.append(f"tensor count {len(tb)} != source {len(ta)}")
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x.name != y.name:
            errors.append(f"tensor[{i}] name {y.name!r} != source {x.name!r}")
            continue
        if tuple(int(v) for v in x.shape) != tuple(int(v) for v in y.shape):
            errors.append(f"{x.name}: ggml shape {tuple(y.shape)} != source {tuple(x.shape)}")
        if x.tensor_type != y.tensor_type:
            errors.append(f"{x.name}: dtype {y.tensor_type.name} != source {x.tensor_type.name}")
        if x.data.nbytes != y.data.nbytes:
            errors.append(f"{x.name}: nbytes {y.data.nbytes} != source {x.data.nbytes}")

    fd_a = os.open(real, os.O_RDONLY)
    fd_b = os.open(stamped, os.O_RDONLY)
    n_ok = 0
    total = 0
    try:
        for x, y in zip(ta, tb):
            if x.name != y.name or x.data.nbytes != y.data.nbytes:
                continue  # already reported; do not hash mismatched slabs
            total += x.data.nbytes
            if sha256_payload(fd_a, x.data_offset, x.data.nbytes) == \
               sha256_payload(fd_b, y.data_offset, y.data.nbytes):
                n_ok += 1
            else:
                errors.append(f"{x.name}: payload sha mismatch")
    finally:
        os.close(fd_a)
        os.close(fd_b)

    print(f"src         : {real}  {real.stat().st_size} bytes")
    print(f"out         : {stamped}  {stamped.stat().st_size} bytes")
    print(f"tensors     : {len(tb)} (source {len(ta)})")
    print(f"payload     : {n_ok}/{len(tb)} sha-identical, {total} bytes ({human(total)}) hashed")
    print(f"kv          : source {len(a.fields) - len(RESERVED_KV)}, "
          f"stamped {len(b.fields) - len(RESERVED_KV)} "
          f"(informational only -- KV is not gated here; the payloads are the gate)")
    if errors:
        for e in errors[:20]:
            print(f"FAIL {e}")
        if len(errors) > 20:
            print(f"FAIL ... and {len(errors) - 20} more")
        return 1
    print(f"GATES OK ({stamped})")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--file", type=Path, default=None, help="GGUF to show or stamp")
    p.add_argument("--set-str", action="append", default=[], metavar="K=V",
                   help="set taillis.K to the STRING V (repeatable)")
    p.add_argument("--set-u32", action="append", default=[], metavar="K=N",
                   help="set taillis.K to the UINT32 N (repeatable)")
    p.add_argument("--set-i32", action="append", default=[], metavar="K=N",
                   help="set taillis.K to the INT32 N (repeatable)")
    p.add_argument("--set-f32", action="append", default=[], metavar="K=X",
                   help="set taillis.K to the FLOAT32 X (repeatable)")
    p.add_argument("--set-bool", action="append", default=[], metavar="K=true|false",
                   help="set taillis.K to the BOOL (repeatable)")
    p.add_argument("--out", type=Path, default=None,
                   help="write to this path instead of in-place; refuses if it exists")
    p.add_argument("--show", action="store_true", help="print taillis.* keys, write nothing")
    p.add_argument("--verify", action="store_true",
                   help="re-parse --src and --out and run the payload gates")
    p.add_argument("--src", type=Path, default=None, help="--verify: the unstamped source")
    args = p.parse_args()

    if args.verify:
        if args.src is None or args.out is None:
            raise SystemExit("FATAL: --verify needs --src and --out")
        sys.exit(verify(args.src, args.out))

    if args.file is None:
        raise SystemExit("FATAL: give --file (or --verify --src/--out)")

    if args.show:
        if any([args.set_str, args.set_u32, args.set_i32, args.set_f32, args.set_bool]):
            raise SystemExit("FATAL: --show writes nothing; do not combine it with --set-*")
        sys.exit(show(args.file))

    sets: list[tuple[str, object, GGUFValueType]] = []
    for flag, vtype in (("--set-str", GGUFValueType.STRING), ("--set-u32", GGUFValueType.UINT32),
                        ("--set-i32", GGUFValueType.INT32), ("--set-f32", GGUFValueType.FLOAT32),
                        ("--set-bool", GGUFValueType.BOOL)):
        for arg in getattr(args, flag.lstrip("-").replace("-", "_")):
            key, val = parse_kv(arg, vtype)
            sets.append((key, val, vtype))
    if not sets:
        raise SystemExit("FATAL: nothing to stamp; give at least one --set-* (or --show)")
    try:
        stamp(args.file, sets, args.out)
    except FileNotFoundError as e:
        raise SystemExit(f"FATAL: {e.filename} not found") from None


if __name__ == "__main__":
    main()
