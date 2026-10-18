#!/usr/bin/env python3
"""Write the OVERLAY file class (taillis/layout.md V4.3) from a base GGUF.

An OVERLAY is a GGUF that is NOT a model: only override tensors plus KV, read by
an overlay loader, never by `GGUFModel::load()`.  One override tensor per
`--override` argument:

    blk.<l>.<logical>.ov<k>.weight    ne = (ne0, ne1, n_ov), one band of experts

`.ov<k>` deliberately does not match `<logical>.g<digits>`, so §8.2's
name-derived v1 grouping cannot mistake an override for a grouping declaration.

THIS TOOL IMPLEMENTS THE IDENTITY PATH AND THE VERIFY PATH ONLY (unit S1).  The
F16/LoRA merge path is S1b: `--delta` is accepted and REFUSED with a named
failure, so the CLI shape does not have to change when S1b lands.

TWO KINDS OF OVERRIDE, AND THE KIND IS DERIVED FROM THE BASE TENSOR'S RANK
(layout.md V4.3 clause 3 — never declared in the file, never inferred from the
override tensor):

  EXPERT-SLICE  the base tensor has an expert axis (ne[2]) — `ffn_gate_up_exps`,
                `ffn_down_exps`.  `ne[2] == n_ov == len(ids)`, all ids inside ONE
                base segment, and the tensor carries its id key
                `taillis.overlay_experts.ov<k>`.
  WHOLE-TENSOR  the base tensor has NO expert axis (rank <= 2) — the router
                `ffn_gate_inp`, ne = (4096,128).  It replaces the whole tensor,
                needs NO id key, and its shape AND type must equal the base
                tensor's exactly.

Usage
    tools/repack_overlay.py --base BASE.gguf --out OVERLAY.gguf --identity \
        --override "L:LOGICAL:ID,ID,ID" [--override ...] [--manifest PATH] \
        [--imatrix-sha256 HEX] [--quant-flags STR]
    tools/repack_overlay.py --base BASE.gguf --out OVERLAY.gguf --identity \
        --override "L:LOGICAL"          # whole-tensor kind (no ids)
    tools/repack_overlay.py --verify --base BASE.gguf --overlay OVERLAY.gguf
    tools/repack_overlay.py --show OVERLAY.gguf

THE ID KEY IS PER-TENSOR, AND `k` IS GLOBALLY UNIQUE IN THE FILE.  layout.md V4.4
(withdrew the single ARRAY key of the first draft — aubier's objection, and they
are right): a length check cannot catch an equal-count SWAP of two entries, and if
both entries sit in the same base group the group and type checks pass too, so a
loader would install one expert's bytes at another's slot with every check green.
So: ONE SCALAR STRING PER OVERRIDE TENSOR, keyed by the `k` token already in the
tensor name — `taillis.overlay_experts.ov3 = "17,42,97"` for
`blk.<l>.<logical>.ov3.weight` — with `k` assigned ov0..ovN ACROSS THE WHOLE FILE
(never restarting per layer or logical, or two tensors could both be ov3 and the
key would be ambiguous).  There is NO positional alignment anywhere in this tool.
BOTH DIRECTIONS are checked: a tensor with no key FAILS, and a key with no tensor
FAILS.  A whole-tensor override that carries an id key is a FAIL.

THE ROW ARITHMETIC (the thing this unit exists to get right).  The quant block
axis is ne[0] (layout.md §1, MEASURED), so the ne[2] axis is the slowest one and
expert `e` is ONE CONTIGUOUS SLAB of a tensor, at

    offset = tensor.data_offset + (e - seg.lo) * stride
    stride = tensor_nbytes / ne[2]          (per SEGMENT: a grouped member's
                                             own nbytes and own ne[2])

where `seg` is the base segment that contains `e` — the flat parent itself
(`lo = 0`, `ne[2]` = the whole expert axis) or, on a grouped base, ONE member
`<logical>.g<k>.weight` with `[lo, hi)` its cumulative range in the expert axis
(layout.md §8.2).  Segment boundaries therefore come from the BASE (its v0
`taillis.expert_groups` array, or its v1 member names in ascending k), never
from an assumption about how evenly the axis was split — an override whose ids
straddle a group boundary is a NAMED FAILURE, because a group boundary read as
an expert boundary is exactly the corruption this gate exists to catch.

gguf-py shape trap (FINDINGS F3): `ReaderTensor.shape` is in GGML ne order
(ne0, ne1, ne2) while `ReaderTensor.data.shape` is NUMPY order (ne2, ne1,
row_bytes for a quantized type).  They are never interchanged here: ne[2] / ne0
/ ne1 come from `t.shape`, the per-expert byte stride and the written byte shape
come from `t.data`.

Data is streamed in CHUNK_BYTES slabs (never the whole 35 GB base in memory) via
gguf-py's LazyChunkedTensor, one loader per slab, per expert: an override's
experts need not be contiguous, so the chunk list is built expert by expert.

BUILD refuses, with a named failure:
  * `--override` ids not a strictly ascending, distinct list, or outside
    [0,128) / outside the base's expert axis;
  * ids not all inside ONE base segment (the group-boundary case);
  * a `(layer, logical)` with no base tensor, with both a flat tensor and group
    members (ambiguous), or whose members do not tile [0, total) contiguously;
  * an id list on a WHOLE-TENSOR override, or no id list on an EXPERT-SLICE one
    (the kind is the base tensor's rank; the two must agree);
  * a base that is itself an overlay (`.ov<k>` tensors), or a `--delta` request.

--verify (the gate; EXIT CODE IS THE GATE) checks, per `.ov<k>` tensor, bound to
its id key BY THE `k` TOKEN (never by position):
  * G9: no `.ov<k>` tensor in the base; no `.g<digits>` tensor in the overlay;
    no group key; the overlay carries no non-override tensor at all;
  * `k` GLOBALLY UNIQUE across the file (two tensors with one k are one address);
  * one scalar `taillis.overlay_experts.ov<k>` STRING key per EXPERT-SLICE
    tensor, and none for a WHOLE-TENSOR one: a tensor with no key FAILS, a key
    with no tensor FAILS, a stray key FAILS;
  * each entry an ascending, distinct list of ids in [0,128) and inside the
    base's expert axis; `ne[2] == len(entry)`; ne[0]/ne[1] equal the base
    segment's; every id of one entry falls in exactly ONE base segment, and the
    tensor's type EQUALS that segment's type (V4.3 clause 5);
  * a WHOLE-TENSOR override's shape and type EQUAL the base tensor's exactly;
  * THE PAYLOAD IS BYTE-COMPARED against the base — the concatenation of those
    experts' slabs in ascending id order, or the whole base tensor — a real byte
    comparison, counted per expert, naming the expert and the byte offset of the
    first difference (always run: it decides G8.3's direction);
  * G3: `taillis.parent_sha256` == sha256 of the whole base file bytes;
  * G8.3 both directions: bytes equal ⇒ `overlay_identity` MUST be true; bytes
    differ ⇒ it must be absent/false.  On a non-identity overlay the
    quantisation provenance (`taillis.provenance.imatrix_sha256` and
    `taillis.provenance.quantize_flags`) is REQUIRED — V4.4 makes that pair
    mandatory so a variant's quality is never confounded with a requantisation.
    On an IDENTITY cut nothing was quantised, so the pair is not required there
    and is NOT invented (see the docstring note in verify()).
  * `taillis.provenance.overlay_tool` PRESENT (V4.4 defines the name);
  * the overlay carries NO MODEL KV: `general.architecture` or any `<arch>.*`
    key of the base's architecture is a FAIL (V4.3 clause 5 — it is not a model,
    and a second copy of a fact the base owns is exactly the redundancy R1/R2
    removed);
  * `taillis.overlay_target` / `taillis.overlay_group` ABSENT (V4.4: derived
    facts; their presence is FAIL), and `taillis.format_version` ABSENT.
    **That last one is DELIBERATE AND STAYS**: an overlay declares no grouping,
    and §7.1 defines `format_version` normatively AS the grouping encoding, so a
    value there is meaningless at best and a false claim at worst.  Do not
    "fix" it by relaxing the check — if an overlay ever needs a version scalar
    that is a new key with new semantics, not this one.

Base files are READ-ONLY inputs.  Write scope is /var/lib/taillis/models and
/tmp; an existing `--out` is never overwritten.  `--manifest PATH` is a SIDE
CONVENIENCE for humans only, marked non-authoritative: the authority is the
file (`taillis.overlay_experts` in tensor-table order).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.lazy import LazyChunkedTensor
except ImportError as e:  # pragma: no cover - environment error, not logic
    sys.exit(f"FATAL: gguf-py not importable ({e}); use venvs/convert/bin/python3")

REPO = Path(__file__).resolve().parent.parent
GROUP_KEY = "taillis.expert_groups"
FORMAT_KEY = "taillis.format_version"
PARENT_KEY = "taillis.parent_sha256"
# V4.4: ONE SCALAR STRING PER OVERRIDE TENSOR, keyed by the `k` token in the
# tensor name (`taillis.overlay_experts.ov3`).  The single ARRAY key of the first
# draft is WITHDRAWN: a length check cannot catch a same-count swap of two
# entries, so there is no positional alignment in this tool at all.
OVERLAY_EXPERTS_PREFIX = "taillis.overlay_experts.ov"
OVERLAY_IDENTITY_KEY = "taillis.overlay_identity"
OVERLAY_TOOL_KEY = "taillis.provenance.overlay_tool"
IMATRIX_KEY = "taillis.provenance.imatrix_sha256"
QUANT_FLAGS_KEY = "taillis.provenance.quantize_flags"
QUANT_PROV_KEYS = (IMATRIX_KEY, QUANT_FLAGS_KEY)
FORBIDDEN_KEYS = ("taillis.overlay_target", "taillis.overlay_group")
KIND_SLICE = "expert-slice"
KIND_WHOLE = "whole-tensor"
MODEL_ARCH_KEY = "general.architecture"
# V4.3 clause 3 / V4.5 G8 state the literal expert-id space; the base's own
# expert_count is checked too, and for the MS4 base the two coincide (MEASURED 128).
EXPERT_ID_SPACE = 128
OV_RE = re.compile(r"^blk\.(\d+)\.(.+)\.ov(\d+)\.weight$")
MEMBER_FILE_RE = re.compile(r"^blk\.(\d+)\.(.+)\.g(\d+)\.weight$")
MEMBER_KEY_RE = re.compile(r"^blk\.(\d+)\.(.+)\.g(\d+)$")   # key entries carry NO .weight
FLAT_RE = re.compile(r"^blk\.(\d+)\.(.+)\.weight$")
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


def check_write_scope(out: Path) -> None:
    """Write scope per the unit brief: /var/lib/taillis/models or /tmp — nowhere else."""
    real = out.resolve()
    allowed = (Path("/var/lib/taillis/models"), Path("/tmp"))
    if not any(real == a or a in real.parents for a in allowed):
        raise SystemExit(
            f"FATAL: refusing to write {real}; write scope is /var/lib/taillis/models or /tmp")


def sha256_file(path: Path) -> str:
    """sha256 over the WHOLE FILE BYTES (header + KV + tensor data), chunk-wise."""
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            buf = f.read(CHUNK_BYTES)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def tool_version() -> str:
    """repack_overlay.py@<taillis HEAD>[+dirty]; '@unknown' if git is unavailable.

    Same K3 discipline as group_experts.py: the recorded pointer must name a tree
    that actually CONTAINS the writer, so local modification/untracked is +dirty.
    NOT a defined name in layout.md §7.1 yet — reported as a spec gap by S1.
    """
    name = "repack_overlay.py"
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        commit = head.stdout.strip() if head.returncode == 0 else ""
        tool = str(Path(__file__).resolve().relative_to(REPO))
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", tool],
                            capture_output=True, text=True, timeout=10)
        dirty = st.returncode != 0 or bool(st.stdout.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        commit, dirty = "", True
    if not commit:
        return f"{name}@unknown"
    return f"{name}@{commit}{' +dirty' if dirty else ''}"


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


def ne2_of(tensor, what: str) -> int:
    """ne[2] of an expert-indexed tensor, or a NAMED failure.

    A tensor with no ne[2] axis is not expert-indexed: a WHOLE-TENSOR override
    (the router `blk.<l>.ffn_gate_inp`, ne = (4096,128) F32) has no expert slab,
    and the V4.3 overlay class as settled — ne = (ne0, ne1, n_ov) with an expert
    id list whose length must equal ne[2] — cannot express one.  Found by S1,
    reported as a spec gap; the tool refuses rather than guessing an axis.
    """
    ne = tuple(int(x) for x in tensor.shape)      # ggml ne order (FINDINGS F3)
    if len(ne) != 3:
        raise SystemExit(
            f"FATAL: {what}: ne={ne} (ggml ne order) has no ne[2] expert axis, so it is not "
            f"an expert-indexed tensor. A whole-tensor override (e.g. ffn_gate_inp) has no "
            f"expert slab and the V4.3 overlay class cannot express one — that is a format "
            f"gap, not a tool limit")
    return ne[2]


class Segment:
    """One contiguous run of the expert axis in the BASE, and its row arithmetic.

    `lo`/`hi` are the segment's cumulative range in the expert axis; `off` is the
    tensor's absolute data offset in the base file; `stride` is its bytes PER
    EXPERT (its own nbytes / its own ne[2]).
    """

    __slots__ = ("name", "tensor", "lo", "hi", "off", "stride", "ne01", "tail", "dtype", "raw")

    def __init__(self, tensor, lo: int, hi: int):
        ne = tuple(int(x) for x in tensor.shape)          # ggml ne order
        shape = tuple(int(x) for x in tensor.data.shape)   # numpy order
        ne2 = ne2_of(tensor, tensor.name)
        if ne2 != hi - lo:
            raise SystemExit(f"FATAL: {tensor.name}: ne[2]={ne2} != segment width {hi - lo}")
        if len(shape) != 3 or shape[0] != ne2:
            raise SystemExit(f"FATAL: {tensor.name}: ne[2]={ne2} inconsistent with "
                             f"byte shape {shape} (FINDINGS F3: ne order vs numpy order)")
        if tensor.data.nbytes % ne2 != 0:
            raise SystemExit(f"FATAL: {tensor.name}: nbytes {tensor.data.nbytes} "
                             f"not divisible by ne[2]={ne2}")
        self.name = tensor.name
        self.tensor = tensor
        self.lo = lo
        self.hi = hi
        self.off = tensor.data_offset
        self.stride = tensor.data.nbytes // ne2
        self.ne01 = (ne[0], ne[1])
        self.tail = shape[1:]
        self.dtype = tensor.data.dtype
        self.raw = tensor.tensor_type if self.dtype == np.uint8 else None

    def slab_offset(self, expert: int) -> int:
        """THE ROW ARITHMETIC: base_bytes + data_offset + (e - lo) * stride."""
        return self.off + (expert - self.lo) * self.stride


def base_segments(reader, layer: int, logical: str, expert_count: int) -> list[Segment]:
    """The base's segments for blk.<layer>.<logical>, from the BASE's own declaration.

    flat parent   -> one segment covering the whole expert axis;
    grouped v0    -> `taillis.expert_groups` entries in ARRAY order (§8.2);
    grouped v1    -> member names `<logical>.g<k>.weight` in ASCENDING k (§8.2.3).
    Ranges are the cumulative ne[2] of the members — never an assumed split.
    """
    prefix = f"blk.{layer}.{logical}"
    by_name = {t.name: t for t in reader.tensors}
    flat = by_name.get(prefix + ".weight")
    members: list[tuple[int, str]] = []

    if GROUP_KEY in reader.fields:
        f = reader.fields[GROUP_KEY]
        if f.types[0] != GGUFValueType.ARRAY or f.types[-1] != GGUFValueType.STRING:
            raise SystemExit(f"FATAL: {GROUP_KEY} is not a string array ({f.types})")
        for entry in f.contents():
            m = MEMBER_KEY_RE.match(entry)
            if m and m.group(1) == str(layer) and m.group(2) == logical:
                members.append((int(m.group(3)), entry + ".weight"))
        order = "array order"
    else:
        for name in by_name:
            m = MEMBER_FILE_RE.match(name)
            if m and m.group(1) == str(layer) and m.group(2) == logical:
                members.append((int(m.group(3)), name))
        members.sort()
        order = "ascending k"

    if members and flat is not None:
        raise SystemExit(f"FATAL: base carries BOTH '{prefix}.weight' and {len(members)} "
                         f"'{prefix}.g<k>.weight' members — ambiguous grouping, not a base "
                         f"this tool can derive segments from")
    if members:
        dups = sorted({k for k, _ in members if [x for x, _ in members].count(k) > 1})
        if dups:
            raise SystemExit(f"FATAL: base declares duplicate member index k={dups} for "
                             f"'{prefix}' ({order})")
        segs: list[Segment] = []
        lo = 0
        for _, name in members:
            t = by_name.get(name)
            if t is None:
                raise SystemExit(f"FATAL: {GROUP_KEY} names '{name}' which is not in the base")
            ne2 = ne2_of(t, name)
            segs.append(Segment(t, lo, lo + ne2))
            lo += ne2
        if lo != expert_count:
            raise SystemExit(f"FATAL: '{prefix}': members tile [0,{lo}) but the base declares "
                             f"expert_count={expert_count} (layout.md §2 rule 1)")
        return segs
    if flat is not None:
        ne2 = ne2_of(flat, prefix + ".weight")
        if ne2 != expert_count:
            raise SystemExit(f"FATAL: '{prefix}.weight' has ne[2]={ne2} != "
                             f"expert_count={expert_count}")
        return [Segment(flat, 0, ne2)]
    raise SystemExit(f"FATAL: base has no tensor '{prefix}.weight' nor any "
                     f"'{prefix}.g<k>.weight' member")


def base_kind(reader, layer: int, logical: str, expert_count: int):
    """(KIND, base) for one override — DERIVED from the BASE tensor's RANK (V4.3 cl.3).

    rank 3 (has ne[2]) -> KIND_SLICE  + the list of base segments to cut from;
    rank <= 2          -> KIND_WHOLE  + the base tensor itself (replaced whole).

    The kind is never declared in the file and never inferred from the override;
    a base with neither shape fails here through base_segments.
    """
    prefix = f"blk.{layer}.{logical}"
    by_name = {t.name: t for t in reader.tensors}
    flat = by_name.get(prefix + ".weight")
    if flat is not None and len(tuple(flat.shape)) <= 2:
        return KIND_WHOLE, flat
    return KIND_SLICE, base_segments(reader, layer, logical, expert_count)


def def_arch(reader) -> str:
    f = reader.fields.get(MODEL_ARCH_KEY)
    return str(f.contents()) if f is not None else ""


def parse_override(spec: str) -> tuple[int, str, list[int] | None]:
    """'L:LOGICAL:ID,ID' -> (layer, logical, ids); 'L:LOGICAL' -> (layer, logical, None).

    The ids are absent for a WHOLE-TENSOR override only; whether that is allowed
    is decided against the BASE tensor's rank by build(), never here.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise SystemExit(f"FATAL: --override expects 'L:LOGICAL:ID,ID,ID' (expert-slice) or "
                         f"'L:LOGICAL' (whole-tensor), got '{spec}'")
    lstr, logical = parts[0], parts[1]
    idstr = parts[2] if len(parts) == 3 else None
    try:
        layer = int(lstr)
    except ValueError:
        raise SystemExit(f"FATAL: --override '{spec}': layer '{lstr}' is not an integer")
    if layer < 0:
        raise SystemExit(f"FATAL: --override '{spec}': layer must be >= 0")
    if not logical or logical.startswith(".") or logical.endswith("."):
        raise SystemExit(f"FATAL: --override '{spec}': bad logical '{logical}'")
    if re.search(r"\.ov\d+$", logical) or re.search(r"\.g\d+$", logical):
        raise SystemExit(f"FATAL: --override '{spec}': logical '{logical}' already carries an "
                         f"override/group suffix; name the BASE logical (V4.3 clause 1)")
    if idstr is None:
        return layer, logical, None
    try:
        ids = [int(x) for x in idstr.split(",")]
    except ValueError:
        raise SystemExit(f"FATAL: --override '{spec}': ids '{idstr}' are not integers")
    if not ids:
        raise SystemExit(f"FATAL: --override '{spec}': empty id list")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"FATAL: --override '{spec}': duplicate ids {sorted(ids)} (an id may "
                         f"appear once: it names one base slab)")
    if ids != sorted(ids):
        raise SystemExit(f"FATAL: --override '{spec}': ids {ids} are not ascending; the "
                         f"declared list is ASCENDING (V4.3 clause 6) — pass {sorted(ids)}")
    return layer, logical, ids


def place_ids(segs: list[Segment], ids: list[int], what: str) -> Segment:
    """The ONE base segment holding every id of an override (else a NAMED failure)."""
    hits = [s for s in segs if all(s.lo <= e < s.hi for e in ids)]
    if len(hits) == 1:
        return hits[0]
    per = {}
    for e in ids:
        for s in segs:
            if s.lo <= e < s.hi:
                per.setdefault((s.lo, s.hi), []).append(e)
    tiling = " ".join(f"[{s.lo},{s.hi})" for s in segs)
    if len(per) > 1:
        parts = "; ".join(f"experts {v} in [{lo},{hi})" for (lo, hi), v in sorted(per.items()))
        raise SystemExit(
            f"FATAL: {what}: ids straddle base group segments {tiling} — {parts}. A group "
            f"boundary is not an expert boundary (layout.md V4.3 clause 3); split this "
            f"override into one per segment")
    raise SystemExit(f"FATAL: {what}: ids {ids} fall in no base segment of {tiling}")


def expert_count_of(reader) -> int:
    arch = reader.fields.get("general.architecture")
    if arch is None:
        raise SystemExit("FATAL: base carries no general.architecture; cannot read expert_count")
    a = arch.contents()
    key = f"{a}.expert_count"
    if key not in reader.fields:
        raise SystemExit(f"FATAL: base carries no {key}")
    return int(reader.fields[key].contents())


def check_ids_space(ids: list[int], expert_count: int) -> None:
    bad = [e for e in ids if not (0 <= e < EXPERT_ID_SPACE) or e >= expert_count]
    if bad:
        raise SystemExit(f"FATAL: expert ids {bad} outside [0,{min(EXPERT_ID_SPACE, expert_count)})"
                         f" (V4.3 clause 3: ids lie in [0,{EXPERT_ID_SPACE}) and inside the "
                         f"base's expert axis of {expert_count})")


def build(base: Path, out: Path, raw_overrides: list[str], identity: bool,
          delta: Path | None, manifest: Path | None,
          imatrix: str | None, quant_flags: str | None) -> int:
    real = need(base, "--base")
    out = out.resolve()
    check_write_scope(out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} already exists; remove it or pick another --out")
    if out == real:
        raise SystemExit("FATAL: --out is the base file; an overlay is a NEW file, the base is a "
                         "READ-ONLY input")
    if delta is not None:
        raise SystemExit("FATAL: --delta is NOT implemented in S1 (the F16/LoRA merge path is "
                         "unit S1b); S1 writes IDENTITY overlays only")
    if not identity:
        raise SystemExit("FATAL: S1 writes identity overlays only — pass --identity (the "
                         "F16/LoRA merge path is unit S1b)")
    if not raw_overrides:
        raise SystemExit("FATAL: give at least one --override "
                         "'L:LOGICAL:ID,ID' (expert-slice) or 'L:LOGICAL' (whole-tensor)")

    reader = open_gguf(real, "base")
    ov_in_base = [t.name for t in reader.tensors if OV_RE.match(t.name)]
    if ov_in_base:
        raise SystemExit(f"FATAL: {real} carries {len(ov_in_base)} .ov<k> tensor(s) "
                         f"(e.g. {ov_in_base[0]}); it is an overlay, not a base "
                         f"(layout.md V4.5 G9)")
    expert_count = expert_count_of(reader)
    plan = [parse_override(s) for s in raw_overrides]

    # Resolve EVERY override BEFORE anything is written: the kind (from the base
    # tensor's rank), the segment and the id space are all decided here, so a
    # malformed plan leaves no partial file behind.  `k` is assigned 0..N over the
    # WHOLE file (V4.3 clause 6) — never restarted per layer or per logical.
    resolved: list[tuple[int, int, str, list[int] | None, str, object]] = []
    for k, (layer, logical, ids) in enumerate(plan):
        what = f"blk.{layer}.{logical} ov{k}"
        kind, obj = base_kind(reader, layer, logical, expert_count)
        if kind == KIND_SLICE:
            if ids is None:
                raise SystemExit(
                    f"FATAL: {what}: no expert ids given, but the base tensor of '{logical}' HAS "
                    f"an expert axis (ne[2]) — an EXPERT-SLICE override must declare them: "
                    f"'L:LOGICAL:ID,ID' (V4.3 clause 3)")
            check_ids_space(ids, expert_count)
            resolved.append((k, layer, logical, ids, KIND_SLICE,
                             place_ids(obj, ids, what)))
        else:
            if ids is not None:
                raise SystemExit(
                    f"FATAL: {what}: ids {ids} given, but the base tensor '{obj.name}' has NO "
                    f"expert axis (ne={tuple(int(x) for x in obj.shape)}) — it is a WHOLE-TENSOR "
                    f"override and declares no ids: 'L:LOGICAL' (V4.3 clause 3)")
            resolved.append((k, layer, logical, None, KIND_WHOLE, obj))

    parent_sha = sha256_file(real)
    tool = tool_version()

    writer = GGUFWriter(str(out), arch="", use_temp_file=False)
    writer.data_alignment = reader.alignment

    fd = os.open(real, os.O_RDONLY)
    n_bytes = 0
    try:
        for k, layer, logical, ids, kind, obj in resolved:
            name = f"blk.{layer}.{logical}.ov{k}.weight"
            if kind == KIND_SLICE:
                seg = obj
                chunks: list = []
                for e in ids:
                    chunks += chunk_loaders(fd, seg.slab_offset(e), seg.stride)
                writer.add_tensor(name, LazyChunkedTensor(
                    chunks, (len(ids),) + seg.tail, seg.dtype,
                ), raw_dtype=seg.raw)
                entry = ",".join(str(e) for e in ids)
                writer.add_key_value(f"{OVERLAY_EXPERTS_PREFIX}{k}", entry,
                                     GGUFValueType.STRING)
                n_bytes += len(ids) * seg.stride
                print(f"override {k}  : {name}  ne={seg.ne01 + (len(ids),)}  "
                      f"type={seg.tensor.tensor_type.name}  [{KIND_SLICE}]")
                print(f"  base seg : {seg.name}  [{seg.lo},{seg.hi})  "
                      f"data_offset={seg.off}  stride={seg.stride} "
                      f"({seg.stride * len(ids)} bytes for {len(ids)} expert(s))")
                print(f"  id key   : {OVERLAY_EXPERTS_PREFIX}{k} = \"{entry}\"")
                for e in ids:
                    print(f"    e={e:3d}  base_bytes + {seg.off} + ({e} - {seg.lo}) * "
                          f"{seg.stride} = {seg.slab_offset(e)}")
            else:
                t = obj
                # numpy-order shape + the tensor's own dtype: gguf-py reverses the
                # shape on write, so this reproduces the base tensor's ne exactly
                # (FINDINGS F3 — the two orders are never interchanged).
                shape = tuple(int(x) for x in t.data.shape)
                writer.add_tensor(name, LazyChunkedTensor(
                    chunk_loaders(fd, t.data_offset, t.data.nbytes), shape, t.data.dtype,
                ), raw_dtype=t.tensor_type if t.data.dtype == np.uint8 else None)
                n_bytes += t.data.nbytes
                print(f"override {k}  : {name}  ne={tuple(int(x) for x in t.shape)}  "
                      f"type={t.tensor_type.name}  [{KIND_WHOLE}]")
                print(f"  base tensor : {t.name}  data_offset={t.data_offset}  "
                      f"nbytes={t.data.nbytes} (replaced WHOLE; no id key)")

        writer.add_key_value(PARENT_KEY, parent_sha, GGUFValueType.STRING)
        writer.add_key_value(OVERLAY_IDENTITY_KEY, True, GGUFValueType.BOOL)
        writer.add_key_value(OVERLAY_TOOL_KEY, tool, GGUFValueType.STRING)
        for key, val in ((IMATRIX_KEY, imatrix), (QUANT_FLAGS_KEY, quant_flags)):
            if val is not None:
                writer.add_key_value(key, val, GGUFValueType.STRING)
        # NO taillis.format_version, no group key, and NO model KV at all: an
        # overlay declares no grouping and is not a model (V4.3 clause 5, §8.1).
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        os.close(fd)
    writer.close()

    total = out.stat().st_size
    print(f"base        : {real}")
    print(f"out         : {out}  {total} bytes ({human(total)}) — "
          f"{len(resolved)} override tensor(s), {human(n_bytes)} payload")
    print(f"identity    : True (taillis.overlay_identity; every payload is a VERBATIM cut)")
    print(f"kv pairs    : {PARENT_KEY} = {parent_sha}")
    print(f"              {OVERLAY_IDENTITY_KEY} = True")
    print(f"              {OVERLAY_TOOL_KEY} = {tool}")
    print(f"              {IMATRIX_KEY}: "
          f"{imatrix if imatrix is not None else 'ABSENT (nothing requantised here)'}")
    print(f"              {QUANT_FLAGS_KEY}: "
          f"{quant_flags if quant_flags is not None else 'ABSENT (nothing requantised here)'}")
    print(f"              {FORMAT_KEY}: ABSENT by design (an overlay declares no grouping)")
    print(f"              no model KV (V4.3 clause 5: the loader takes it from the base)")

    if manifest is not None:
        mpath = manifest.resolve()
        check_write_scope(mpath)
        if mpath.exists():
            raise SystemExit(f"FATAL: {mpath} already exists")
        doc = {
            "_authority": "NON-AUTHORITATIVE side manifest. The file is the authority: one scalar "
                          "taillis.overlay_experts.ov<k> key per override tensor, keyed by the k "
                          "token in the name (layout.md V4.3 clause 6 / V4.4).",
            "base": str(real),
            "base_sha256": parent_sha,
            "identity": True,
            "overrides": [],
        }
        for k, layer, logical, ids, kind, obj in resolved:
            item = {
                "tensor": f"blk.{layer}.{logical}.ov{k}.weight",
                "layer": layer, "logical": logical, "k": k, "kind": kind,
                "type": obj.tensor.tensor_type.name if kind == KIND_SLICE
                        else obj.tensor_type.name,
            }
            if kind == KIND_SLICE:
                item.update({"experts": ids, "group": obj.name,
                             "key": f"{OVERLAY_EXPERTS_PREFIX}{k}"})
            doc["overrides"].append(item)
        mpath.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"manifest    : {mpath} (NON-AUTHORITATIVE convenience)")
    return 0


def compare(fd_a: int, off_a: int, fd_b: int, off_b: int, n: int) -> tuple[int, int]:
    """Byte-compare [off_b, off_b+n) of B against [off_a, ...) of A, chunk-wise."""
    diffs = 0
    first = -1
    done = 0
    while done < n:
        m = min(CHUNK_BYTES, n - done)
        x = np.frombuffer(os.pread(fd_a, m, off_a + done), dtype=np.uint8)
        y = np.frombuffer(os.pread(fd_b, m, off_b + done), dtype=np.uint8)
        if len(x) != m or len(y) != m:
            return (n, first if first >= 0 else done)
        bad = np.flatnonzero(x != y)
        if bad.size:
            if first < 0:
                first = done + int(bad[0])
            diffs += int(bad.size)
        done += m
    return diffs, first


def open_gguf(path: Path, what: str):
    try:
        return GGUFReader(path)
    except Exception as e:
        raise SystemExit(f"FATAL: {what} {path} does not parse as a GGUF "
                         f"({type(e).__name__}: {e})")


def need(path: Path, what: str) -> Path:
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        raise SystemExit(f"FATAL: {what} {path} does not exist")


def parse_ids(entry: str) -> tuple[list[int] | None, str]:
    """A `taillis.overlay_experts.ov<k>` value -> (ids, "") or (None, why)."""
    try:
        ids = [int(x) for x in entry.split(",")]
    except ValueError:
        return None, "is not a comma-separated id list"
    if not ids:
        return None, "is empty"
    if len(set(ids)) != len(ids):
        return None, f"has duplicate ids {sorted(ids)}"
    if ids != sorted(ids):
        return None, f"ids {ids} are not ASCENDING (V4.3 clause 6)"
    return ids, ""


def verify(base: Path, overlay: Path) -> int:
    real = need(base, "--base")
    ov = need(overlay, "--overlay")
    a = open_gguf(real, "base")
    b = open_gguf(ov, "overlay")
    errors: list[str] = []

    # ---- G9: symmetry refusals ------------------------------------------
    ov_in_base = [t.name for t in a.tensors if OV_RE.match(t.name)]
    if ov_in_base:
        errors.append(f"G9: overlay tensors in a base: {len(ov_in_base)} "
                      f"(e.g. {ov_in_base[0]}) — {real} is an overlay, not a base")
    grp_in_ov = [t.name for t in b.tensors if MEMBER_FILE_RE.match(t.name)]
    if grp_in_ov:
        errors.append(f"G9: grouping declaration (.g<digits>) in an overlay: {len(grp_in_ov)} "
                      f"(e.g. {grp_in_ov[0]}) — an overlay declares no grouping (V4.3 cl.2)")
    if GROUP_KEY in b.fields:
        errors.append(f"G9: {GROUP_KEY} present in an overlay — an overlay declares no grouping")
    if ov_in_base:
        return fail(errors)   # nothing below is meaningful against a non-base

    # ---- an overlay declares no grouping, and carries NO model KV --------
    # The format_version check below is DELIBERATE (S1 decision, kept): an overlay
    # declares no grouping and §7.1 defines this scalar normatively AS the grouping
    # encoding, so a value here is meaningless at best and a false claim at worst.
    if FORMAT_KEY in b.fields:
        errors.append(f"{FORMAT_KEY} = {b.fields[FORMAT_KEY].contents()} in an overlay — an "
                      f"overlay declares no grouping, so the grouping scalar is meaningless "
                      f"(V4.3/V4.4: do not stamp it; this check is deliberate, not a bug)")
    for key in FORBIDDEN_KEYS:
        if key in b.fields:
            errors.append(f"{key} present — derived facts must not be declared (V4.4)")
    base_arch = def_arch(a)
    for key in b.fields:
        if key in RESERVED_KV:
            continue
        if key == MODEL_ARCH_KEY:
            errors.append(f"{MODEL_ARCH_KEY} present in an overlay — an overlay carries NO model "
                          f"KV (V4.3 clause 5): it is not a model, and a second copy of a fact "
                          f"the base owns is exactly the redundancy R1/R2 removed")
        elif base_arch and key.startswith(base_arch + "."):
            errors.append(f"'{key}' present in an overlay — <arch>.* model KV belongs to the base "
                          f"(V4.3 clause 5: no block_count, no expert count, no model KV)")

    # ---- the overlay's tensor surface -----------------------------------
    ov_tensors = [t for t in b.tensors if OV_RE.match(t.name)]
    stray = [t.name for t in b.tensors if not OV_RE.match(t.name)]
    if stray:
        more = f" (+{len(stray) - 4} more)" if len(stray) > 4 else ""
        errors.append(f"overlay carries {len(stray)} non-override tensor(s): "
                      f"{sorted(stray)[:4]}{more} — an overlay is not a model (V4.3)")
    if not ov_tensors:
        errors.append(f"no .ov<k> tensor in {ov} — an empty overlay overrides nothing")

    # ---- k is GLOBALLY UNIQUE across the file (V4.3 clause 6) ------------
    by_k: dict[int, object] = {}
    for t in ov_tensors:
        k = int(OV_RE.match(t.name).group(3))
        if k in by_k:
            errors.append(f"duplicate override index k={k}: '{by_k[k].name}' and '{t.name}' — k "
                          f"is GLOBALLY UNIQUE within an overlay file (V4.3 clause 6), because "
                          f"{OVERLAY_EXPERTS_PREFIX}{k} addresses exactly one tensor")
        else:
            by_k[k] = t

    # ---- the declaration: ONE SCALAR KEY PER EXPERT-SLICE TENSOR ---------
    keys: dict[int, str] = {}
    for key, fld in b.fields.items():
        if not key.startswith(OVERLAY_EXPERTS_PREFIX):
            continue
        tail = key[len(OVERLAY_EXPERTS_PREFIX):]
        if not tail.isdigit():
            errors.append(f"{key}: the suffix after '{OVERLAY_EXPERTS_PREFIX}' must be the k token "
                          f"of a tensor name (V4.4)")
            continue
        if fld.types[0] != GGUFValueType.STRING:
            errors.append(f"{key} is {fld.types[0].name}, not a STRING (V4.4: one scalar string "
                          f"per override tensor)")
            continue
        keys[int(tail)] = str(fld.contents())
    for k in sorted(keys):
        if k not in by_k:
            errors.append(f"{OVERLAY_EXPERTS_PREFIX}{k} = \"{keys[k]}\" has no matching .ov{k} "
                          f"tensor — a key with no tensor is a named failure (V4.4); there is no "
                          f"positional alignment to fall back on")

    # the identity flag, and the exporter's own provenance (V4.4 names it)
    iflag = b.fields.get(OVERLAY_IDENTITY_KEY)
    if iflag is not None and iflag.types[0] != GGUFValueType.BOOL:
        errors.append(f"{OVERLAY_IDENTITY_KEY} is {iflag.types[0].name}, not BOOL")
    declared_identity = (iflag is not None and bool(iflag.contents()))
    if OVERLAY_TOOL_KEY not in b.fields:
        errors.append(f"{OVERLAY_TOOL_KEY} absent — V4.4 defines it and requires it on an overlay")
    elif not str(b.fields[OVERLAY_TOOL_KEY].contents()).startswith("repack_overlay.py@"):
        errors.append(f"{OVERLAY_TOOL_KEY} = {b.fields[OVERLAY_TOOL_KEY].contents()!r}, expected "
                      f"repack_overlay.py@<commit>[ +dirty]")

    expert_count = expert_count_of(a)

    # ---- per tensor: kind from the base's rank, then THE ROW ARITHMETIC ---
    fd_a = os.open(real, os.O_RDONLY)
    fd_b = os.open(ov, os.O_RDONLY)
    n_tensors_ok = 0
    n_cmp_units = 0          # experts for an expert-slice, one unit for a whole tensor
    n_ok_units = 0
    n_bytes_cmp = 0
    n_slice = 0
    n_whole = 0
    n_experts = 0
    all_equal = True
    seg_cache: dict[tuple[int, str], list[Segment]] = {}
    try:
        for k in sorted(by_k):
            t = by_k[k]
            name = t.name
            m = OV_RE.match(name)
            layer, logical = int(m.group(1)), m.group(2)
            try:
                kind, obj = base_kind(a, layer, logical, expert_count)
            except SystemExit as e:
                errors.append(str(e).removeprefix("FATAL: "))
                continue
            entry = keys.get(k)
            ne = tuple(int(x) for x in t.shape)
            if kind == KIND_SLICE:
                n_slice += 1
                if entry is None:
                    errors.append(f"{name}: no {OVERLAY_EXPERTS_PREFIX}{k} key — the base tensor of "
                                  f"'{logical}' HAS an expert axis (ne[2]), so this is an "
                                  f"EXPERT-SLICE override and its id key is REQUIRED "
                                  f"(V4.3 clause 3, V4.4)")
                    continue
                ids, why = parse_ids(entry)
                if ids is None:
                    errors.append(f"{name}: {OVERLAY_EXPERTS_PREFIX}{k} = \"{entry}\" {why}")
                    continue
                bad = [e for e in ids if not (0 <= e < EXPERT_ID_SPACE) or e >= expert_count]
                if bad:
                    errors.append(f"{name}: ids {bad} outside "
                                  f"[0,{min(EXPERT_ID_SPACE, expert_count)}) (V4.3 clause 3, "
                                  f"V4.5 G8)")
                    continue
                n_experts += len(ids)
                try:
                    if (layer, logical) not in seg_cache:
                        seg_cache[(layer, logical)] = base_segments(a, layer, logical,
                                                                    expert_count)
                    seg = place_ids(seg_cache[(layer, logical)], ids, name)
                except SystemExit as e:
                    errors.append(str(e).removeprefix("FATAL: "))
                    continue
                if len(ne) != 3:
                    errors.append(f"{name}: shape {ne} (ggml ne order) has no ne[2]")
                    continue
                if ne[2] != len(ids):
                    errors.append(f"{name}: ne[2]={ne[2]} != declared expert count {len(ids)} "
                                  f"(\"{entry}\") (V4.3 clause 3)")
                if (ne[0], ne[1]) != seg.ne01:
                    errors.append(f"{name}: ne[0..1]={(ne[0], ne[1])} != base segment {seg.name} "
                                  f"{seg.ne01}")
                    continue
                if t.tensor_type != seg.tensor.tensor_type:
                    errors.append(f"{name}: type {t.tensor_type.name} != base segment {seg.name} "
                                  f"type {seg.tensor.tensor_type.name} (V4.3 clause 5)")
                    continue
                stride = seg.stride
                if t.data.nbytes != stride * len(ids):
                    errors.append(f"{name}: nbytes {t.data.nbytes} != {stride} * {len(ids)} "
                                  f"(base stride * declared experts)")
                    continue
                t_diffs = 0
                for j, e in enumerate(ids):
                    d, first = compare(fd_a, seg.slab_offset(e), fd_b,
                                       t.data_offset + j * stride, stride)
                    n_bytes_cmp += stride
                    n_cmp_units += 1
                    if d:
                        t_diffs += d
                        errors.append(
                            f"{name}: expert {e} payload DIFFERS from base slab at "
                            f"{seg.name} e={e}: {d} byte(s), first at overlay byte "
                            f"{j * stride + first} (expert-local byte {first})")
                    else:
                        n_ok_units += 1
                if t_diffs == 0:
                    n_tensors_ok += 1
                else:
                    all_equal = False
                print(f"expert band : {name}  {OVERLAY_EXPERTS_PREFIX}{k}=\"{entry}\"  "
                      f"type={t.tensor_type.name}  base seg={seg.name} [{seg.lo},{seg.hi})  "
                      f"stride={stride}  diffs={t_diffs}")
            else:
                n_whole += 1
                if entry is not None:
                    errors.append(f"{name}: stray {OVERLAY_EXPERTS_PREFIX}{k} = \"{entry}\" — the "
                                  f"base tensor '{obj.name}' has NO expert axis (rank "
                                  f"{len(tuple(obj.shape))}), so this is a WHOLE-TENSOR override "
                                  f"and declares no ids (V4.3 clause 3)")
                    continue
                bne = tuple(int(x) for x in obj.shape)
                if ne != bne:
                    errors.append(f"{name}: ne={ne} != base tensor '{obj.name}' ne={bne} — a "
                                  f"WHOLE-TENSOR override's shape must EQUAL the base's "
                                  f"(V4.3 clause 5)")
                    continue
                if t.tensor_type != obj.tensor_type:
                    errors.append(f"{name}: type {t.tensor_type.name} != base tensor "
                                  f"'{obj.name}' type {obj.tensor_type.name} (V4.3 clause 5)")
                    continue
                if t.data.nbytes != obj.data.nbytes:
                    errors.append(f"{name}: nbytes {t.data.nbytes} != base tensor "
                                  f"'{obj.name}' nbytes {obj.data.nbytes}")
                    continue
                d, first = compare(fd_a, obj.data_offset, fd_b, t.data_offset, obj.data.nbytes)
                n_bytes_cmp += obj.data.nbytes
                n_cmp_units += 1
                if d:
                    all_equal = False
                    errors.append(f"{name}: payload DIFFERS from base tensor '{obj.name}': "
                                  f"{d} byte(s), first at overlay byte {first}")
                else:
                    n_ok_units += 1
                    n_tensors_ok += 1
                print(f"whole tensor: {name}  ne={ne}  type={t.tensor_type.name}  "
                      f"base={obj.name}  nbytes={obj.data.nbytes}  diffs={d}")

        # ---- G3: the declared comparator --------------------------------
        if PARENT_KEY not in b.fields:
            errors.append(f"G3: {PARENT_KEY} absent from the overlay (V4.3 clause 4)")
        else:
            pf = b.fields[PARENT_KEY]
            if pf.types[0] != GGUFValueType.STRING:
                errors.append(f"G3: {PARENT_KEY} is {pf.types[0].name}, not STRING")
            else:
                want = sha256_file(real)
                got = str(pf.contents())
                if got != want:
                    errors.append(f"G3: {PARENT_KEY} {got} != sha256({real}) {want}")
                else:
                    print(f"G3          : {PARENT_KEY} == sha256(base) = {want}")

        # ---- G8.3: both directions --------------------------------------
        complete = n_cmp_units > 0 and n_ok_units == n_cmp_units
        if complete and not declared_identity:
            errors.append(f"G8.3: every override payload EQUALS its base but "
                          f"{OVERLAY_IDENTITY_KEY} is absent/false — an identity overlay is "
                          f"legal only when it declares itself (V4.5 G8.3)")
        if not all_equal and declared_identity:
            errors.append(f"G8.3: {OVERLAY_IDENTITY_KEY}=true but a payload DIFFERS from its base "
                          f"(see the diffs above)")
        # V4.4 makes the quantisation provenance MANDATORY, so that a variant's
        # measured quality is never confounded with a requantisation.  On an
        # IDENTITY cut nothing is quantised and the honest value does not exist —
        # requiring a sha there would force a FALSE declaration, so it is only
        # demanded where a requantisation could have happened, and absence on an
        # identity overlay is stated rather than invented.
        if not declared_identity:
            for key in QUANT_PROV_KEYS:
                if key not in b.fields:
                    errors.append(f"{key} absent — V4.4 makes the quantisation provenance "
                                  f"MANDATORY on an overlay that is not an identity cut (a "
                                  f"variant's quality must not be confounded with a "
                                  f"requantisation)")
    finally:
        os.close(fd_a)
        os.close(fd_b)

    print(f"tensors     : {len(ov_tensors)} .ov<k> ({n_slice} expert-slice, {n_whole} "
          f"whole-tensor), {n_experts} expert slab(s), {human(n_bytes_cmp)} compared byte-for-byte")
    print(f"payload     : {n_tensors_ok}/{len(ov_tensors)} override tensors byte-identical "
          f"({n_ok_units}/{n_cmp_units} compared unit(s): expert slabs of an expert-slice, or "
          f"one whole tensor)")
    if seg_cache:
        print("segments    : " + ", ".join(
            f"blk.{l}.{lg}=" + " ".join(f"[{s.lo},{s.hi})" for s in segs)
            for (l, lg), segs in sorted(seg_cache.items())))
    print(f"identity    : {OVERLAY_IDENTITY_KEY} "
          f"{'declared' if declared_identity else 'absent/false'}, payloads "
          f"{'all equal' if complete else ('DIFFER' if not all_equal else 'not all compared')}")
    print(f"quant prov  : " + (", ".join(
        f"{k}={'present' if k in b.fields else 'ABSENT'}" for k in QUANT_PROV_KEYS))
        + (" (not required: identity cut quantises nothing)" if declared_identity else ""))
    if errors:
        return fail(errors)
    print(f"GATES OK ({ov})")
    return 0


def fail(errors: list[str]) -> int:
    for e in errors[:20]:
        print(f"FAIL {e}")
    if len(errors) > 20:
        print(f"FAIL ... and {len(errors) - 20} more")
    return 1


def show(path: Path) -> int:
    """Print an overlay's taillis.* keys and its tensor table (read-only)."""
    real = need(path, "--overlay")
    r = open_gguf(real, "overlay")
    print(f"file        : {real}  {real.stat().st_size} bytes ({human(real.stat().st_size)})")
    print(f"tensors     : {len(r.tensors)}")
    for t in r.tensors:
        print(f"  {t.name}  ne={tuple(int(x) for x in t.shape)}  {t.tensor_type.name}  "
              f"nbytes={t.data.nbytes}")
    for key, f in r.fields.items():
        if key in RESERVED_KV or not key.startswith("taillis"):
            continue
        c = f.contents()
        if isinstance(c, list):
            c = f"[{len(c)} entries] {c}"
        print(f"  {key} = {c} ({f.types[0].name})")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", type=Path, default=None, help="the READ-ONLY base GGUF")
    p.add_argument("--out", type=Path, default=None, help="the overlay to write (must not exist)")
    p.add_argument("--overlay", type=Path, default=None, help="the overlay to --verify")
    p.add_argument("--override", action="append", default=[], metavar="L:LOGICAL[:ID,ID]",
                   help="one override (repeatable): 'L:LOGICAL:ID,ID' cuts those experts of ONE "
                        "base segment into an EXPERT-SLICE .ov<k>; 'L:LOGICAL' replaces a "
                        "WHOLE-TENSOR (no expert axis, e.g. ffn_gate_inp)")
    p.add_argument("--identity", action="store_true",
                   help="mark the overlay taillis.overlay_identity=true (the only mode S1 "
                        "writes: every payload is a verbatim cut of the base slab)")
    p.add_argument("--delta", type=Path, default=None,
                   help="S1b ONLY: an F16 delta to merge — NOT IMPLEMENTED in S1")
    p.add_argument("--manifest", type=Path, default=None,
                   help="optional NON-AUTHORITATIVE side manifest.json (a shelf label)")
    p.add_argument("--imatrix-sha256", type=str, default=None, metavar="HEX",
                   help="stamp taillis.provenance.imatrix_sha256 (V4.4's mandatory pair for a "
                        "REQUANTISED overlay; on an identity cut it may only describe the BASE's "
                        "own quantisation — the bytes are copied verbatim)")
    p.add_argument("--quant-flags", type=str, default=None, metavar="STR",
                   help="stamp taillis.provenance.quantize_flags (the other half of V4.4's pair)")
    p.add_argument("--verify", action="store_true",
                   help="run the byte-level gates of an overlay against its base")
    p.add_argument("--show", action="store_true", help="print an overlay's keys + tensor table")
    args = p.parse_args()

    if args.show:
        if args.overlay is None:
            raise SystemExit("FATAL: --show needs --overlay")
        sys.exit(show(args.overlay))
    if args.verify:
        if args.base is None or args.overlay is None:
            raise SystemExit("FATAL: --verify needs --base and --overlay")
        sys.exit(verify(args.base, args.overlay))
    if args.base is None or args.out is None:
        raise SystemExit("FATAL: give --base and --out (or --verify)")
    sys.exit(build(args.base, args.out, args.override, args.identity, args.delta, args.manifest,
                   args.imatrix_sha256, args.quant_flags))


if __name__ == "__main__":
    main()
