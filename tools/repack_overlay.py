#!/usr/bin/env python3
"""Write the OVERLAY file class (taillis/layout.md V4.3) from a base GGUF.

An OVERLAY is a GGUF that is NOT a model: only override tensors plus KV, read by
an overlay loader, never by `GGUFModel::load()`.  One override tensor per
`--override` argument:

    blk.<l>.<logical>.ov<k>.weight    ne = (ne0, ne1, n_ov), one band of experts

`.ov<k>` deliberately does not match `<logical>.g<digits>`, so §8.2's
name-derived v1 grouping cannot mistake an override for a grouping declaration.

TWO PAYLOAD PATHS.  The IDENTITY path (unit S1) copies the base's own slab
bytes verbatim.  The QUANTISED path (unit S1b, `--requantise`) cuts the same
rows out of the >=16-bit source, optionally merges an F16 delta, and RE-
QUANTISES them with the base's OWN imatrix and the base's OWN quant spec
(aubier `docs/expert-swap.md` §5.3: *merge at F16, then re-quantize with the
SAME imatrix and the SAME quant spec*; never merge a float delta into a low-bit
base).  The quantiser is llama-quantize itself, run over a THROWAWAY GGUF that
holds only the override rows — never a 221 GiB intermediate, and never a
requantisation of a 2-bit file (llama.cpp's own --allow-requantize warning).

`--verify` and the quantised path share ONE implementation of cut+merge+quantise
(`requantised_payload`), so `--verify --f16 ...` REPRODUCES a non-identity
payload from its declared provenance and compares it byte for byte.  That is the
only check that can catch a corrupted byte in an overlay whose bytes are
SUPPOSED to differ from its base.

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
    # the DIFFERENT-TYPE regime (S7): the override is produced at a type the base
    # does not use, and NO imatrix is asked for or stamped (q5_k needs none)
    tools/repack_overlay.py --base BASE.gguf --f16 F16.gguf --out OVERLAY.gguf \
        --requantise --target-type Q5_K --override "L:LOGICAL:ID,ID" \
        --llama-quantize BIN [--delta-synth SEED:RELRMS]
    tools/repack_overlay.py --verify --base BASE.gguf --overlay OVERLAY.gguf
    tools/repack_overlay.py --show --overlay OVERLAY.gguf

    # the S1b merge path: quantised from the F16 with the BASE's own spec
    tools/repack_overlay.py --base BASE.gguf --f16 F16.gguf --out OVERLAY.gguf \
        --requantise --override "L:LOGICAL:ID,ID" \
        --imatrix IMATRIX.gguf --type-file TYPES.txt --llama-quantize BIN \
        [--quant-type IQ2_XXS] [--delta DELTA.gguf | --delta-synth SEED:RELRMS]
    # the ADAPTER path (unit LORA): the same merge, with the delta DERIVED from a
    # PEFT adapter instead of handed in precomputed
    tools/repack_overlay.py --base BASE.gguf --f16 F16.gguf --out OVERLAY.gguf \
        --requantise --override "L:LOGICAL:ID,ID" \
        --imatrix IMATRIX.gguf --type-file TYPES.txt --llama-quantize BIN \
        --lora ADAPTER.safetensors [--lora-config adapter_config.json]
    # and the reproduction gate (the only check that can see a corrupted byte in
    # an overlay whose bytes are supposed to differ from the base)
    tools/repack_overlay.py --verify --base BASE.gguf --overlay OVERLAY.gguf \
        --f16 F16.gguf --imatrix IMATRIX.gguf --type-file TYPES.txt \
        --llama-quantize BIN [--delta ... | --delta-synth SEED:RELRMS | --lora ...]

THE ADAPTER PATH DOES NOT FORK THE QUANTISE PATH, ON PURPOSE.  `--lora` produces
exactly the array `--delta` used to supply — a per-expert (n_ov, ne1, ne0) delta —
and hands it to the SAME cut/merge/re-quantise implementation, so there is still
exactly ONE place where an override's bytes are made (and `--verify --f16 --lora`
re-derives the payload from the adapter itself).  The delta is built in F32 and
merged by one F16 store, because an adapter is a rank-r FACTORISATION: rounding
B@A down to F16 first would throw away precision that costs nothing to keep.

THE ADAPTER FORMAT, and the one thing it must never do.  Accepted: PEFT
safetensors, one 2-D `lora_A.weight` (r, in) + `lora_B.weight` (out, r) pair per
(expert, projection), module paths `layers.<l>.experts.<e>.{w1,w2,w3}` (layout.md
§6) or their HF aliases `{gate_proj,up_proj,down_proj}`, optionally under
`base_model.model.` and one model prefix.  W = W + (alpha/rank) B@A per expert,
w1|w3 placed into the FUSED `ffn_gate_up_exps` out axis (which half is which was
MEASURED against the checkpoint's own weights, not assumed — see the mapping block
below).  A pair with NO expert index is REFUSED by name: it is the shape
llama-export-lora ABORTS on, and applying one delta to 128 experts is the silent
mis-weighting this project exists to prevent.  rsLoRA/DoRA/per-module alpha-or-rank
are refused too (each is a different scale), as is an unrecognised module name.
Recognised DENSE modules (attention, router, shared experts) cannot be expressed by
an expert-slice override at all; they are counted and DECLARED in the merge_delta
provenance, never silently dropped.

THE TARGET TYPE, AND THE IMATRIX RULE THAT FOLLOWS FROM IT (S7).  The override's
type is the tensor's own ggml type field (layout.md V4.3 clause 5) — there is no
second declaration and nothing to parse.  `--target-type` asks for a type; absent, the
base segment's own type is the target (the SAME-TYPE regime, aubier's S1-S5), and a
different value is the DIFFERENT-TYPE regime (S7), where the store must give those
experts their own segment because a slice carries one type.  The type must be
EXPRESSIBLE for every overridden tensor (its block size must divide ne[0]); if it is
not, the build STOPS rather than substituting a type that is.

`--imatrix` follows llama-quantize's OWN requirement line
(`tensor_requires_imatrix`, llama-quant.cpp:856), not a policy of ours: REQUIRED for
`IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, IQ1_S` and for `Q2_K` inside a `Q2_K_S` run,
and REFUSED for every other type.  A refused imatrix is not pedantry: `q5_k` never
reads one, so stamping `taillis.provenance.imatrix_sha256` (and an `expert_crop`) on
such a build would be a false provenance claim — the same lie as a missing one,
pointed the other way.  That is also what makes an S7 variant over the shipped
`IQ2_XXS`/`IQ2_S` base buildable at all today.

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
  * a base that is itself an overlay (`.ov<k>` tensors), or a `--delta` request;
  * on `--requantise`: a whole-tensor override (it has no expert axis to crop an
    imatrix to and its base type is not quantised), or a hand-written
    `--quant-flags`/`--imatrix-sha256` (both are DERIVED from the files used).

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
  * **EVERY override tensor's PAYLOAD HASHES TO ITS `taillis.overlay_digest.<ovk>`**
    (V4.4, settled 2026-10-19; REQUIRED on every overlay, one scalar sha256 per
    tensor keyed by `k`, recomputed here from the file's own bytes).  This is the
    only CONTENT check that exists for a non-identity overlay: its payload is
    supposed to differ from its base's, so no base comparison and no structural
    check can see a corrupted byte.  Missing key ⇒ FAIL; mismatch ⇒ FAIL naming
    the tensor and both hashes.  No `--f16`, no quantiser.
  * `taillis.provenance.expert_crop`, when present, is parsed as `ov<k>=<ids>`
    and must EQUAL that tensor's `overlay_experts` entry: the crop is the build
    fact (what the imatrix was cropped to) and the id key is the format fact
    (which experts the payload holds); a payload quantised with a different crop
    than it declares would otherwise pass every other check with wrong weights;
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
    On an IDENTITY overlay the pair is PERMITTED, never demanded and never
    forbidden (V4.4, settled 2026-10-19): "identity" is a statement about bytes,
    "was quantised" a statement about provenance, and a re-quantised slice that
    reproduces the base's bytes is both.
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
    from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
    from gguf.lazy import LazyChunkedTensor
    from gguf.quants import GGML_QUANT_SIZES
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
# V4.4 (settled 2026-10-19): sha256 of the override tensor's PAYLOAD, one scalar
# string per tensor, keyed by the SAME `k` token as its id key.  REQUIRED on every
# overlay.  This is the only CONTENT check possible on a non-identity overlay: its
# payload is supposed to differ from the base's, so without a digest nothing in the
# file can tell a correct override from a corrupted one.
OVERLAY_DIGEST_PREFIX = "taillis.overlay_digest.ov"
# V4.4 (settled 2026-10-19): the build fact behind "re-quantise with the SAME
# imatrix" — llama-quantize refuses a full imatrix for a slice, so the matrix is
# cropped to the override's experts.  Stamped as `ov<k>=<ids>;...` so the crop the
# build ACTUALLY used can be compared with the ids the file declares.
EXPERT_CROP_KEY = "taillis.provenance.expert_crop"
OVERLAY_IDENTITY_KEY = "taillis.overlay_identity"
OVERLAY_TOOL_KEY = "taillis.provenance.overlay_tool"
IMATRIX_KEY = "taillis.provenance.imatrix_sha256"
QUANT_FLAGS_KEY = "taillis.provenance.quantize_flags"
QUANT_PROV_KEYS = (IMATRIX_KEY, QUANT_FLAGS_KEY)
# MEASURED, llama-quant.cpp:856 `tensor_requires_imatrix`: the types whose
# quantisation NEEDS an importance matrix, and the one type that needs it only as
# part of a Q2_K_S file.  The requirement here mirrors the quantiser's OWN line —
# not a policy of ours — so `--imatrix` is demanded exactly where llama-quantize
# would refuse without it, and REFUSED where it could not read it: a stamp naming
# a matrix nothing read is the same lie as a missing one, pointed the other way.
IMATRIX_TYPES = frozenset({"IQ3_XXS", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_M", "IQ1_S"})
# §7.1 names reused verbatim by the quantised path (never invented here).
TYPE_FILE_KEY = "taillis.provenance.type_file_sha256"
QUANT_TOOL_KEY = "taillis.provenance.quantize_tool"
INPUT_KEY = "taillis.provenance.input_sha256"
# NOT yet a §7.1 name: the merged delta a quantised payload was built from.  It is
# the fact that makes a variant's quality attributable (the K3 reason), so it is
# stamped and reported as a proposed defined name rather than left unrecorded.
MERGE_DELTA_KEY = "taillis.provenance.merge_delta"
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


# ---------------------------------------------------------------------------
# THE QUANTISED PAYLOAD PATH (unit S1b) — aubier expert-swap.md §5.3
#
#   merge at F16  ->  re-quantise with the SAME imatrix and the SAME quant spec
#
# The quantiser is llama-quantize itself over a THROWAWAY GGUF holding only the
# override rows.  Nothing here writes a whole-model intermediate, and nothing
# ever reads a low-bit base back into the merge: the base supplies the SPEC (its
# per-segment type, its imatrix) and the F16 supplies the VALUES.
# ---------------------------------------------------------------------------


def quantize_runner(binary: Path) -> tuple[str, dict]:
    """(absolute path, env) for llama-quantize; its libraries sit next to it.

    MEASURED: every binary in that build tree carries an absolute RUNPATH into a
    directory that no longer exists, so LD_LIBRARY_PATH must name its own bin/.
    Prepending rather than replacing keeps a caller's ROCm/venv path intact.
    """
    real = need(binary, "--llama-quantize")
    env = dict(os.environ)
    libs = str(real.parent)
    env["LD_LIBRARY_PATH"] = libs + (":" + env["LD_LIBRARY_PATH"]
                                     if env.get("LD_LIBRARY_PATH") else "")
    return str(real), env


def base_quantize_provenance(reader) -> dict:
    """What the BASE itself declares about the quantisation that made it.

    Read from the file, never assumed: llama-quantize writes `quantize.imatrix.*`
    into every IQ output.  These are the only facts that let this tool CHECK that
    the imatrix it is handed is the base's own (§5.3) instead of merely claiming
    it in the stamped provenance.
    """
    out = {}
    for key, field in reader.fields.items():
        if key.startswith("quantize.imatrix."):
            out[key] = field.contents()
    return out


def check_imatrix_against_base(reader, imatrix_path: Path) -> str:
    """Refuse a mismatch the base can prove; WARN where only the path moved.

    A different `entries_count` is a different imatrix (the count is the number of
    tensors the matrix covers) — that is a NAMED FAILURE, because §5.3's whole
    point is that the REQUANTISATION and the base share one matrix.  A different
    recorded PATH is only a warning: paths move between hosts, and the honest
    checkable facts are the count and the sha this tool stamps.
    """
    prov = base_quantize_provenance(reader)
    want = prov.get("quantize.imatrix.entries_count")
    if want is None:
        print(f"imatrix     : base declares no quantize.imatrix.entries_count — the "
              f"identity of the base's own matrix is NOT checkable from the file; the "
              f"sha stamped below is a CLAIM, and nothing here verifies it")
        return "unchecked"
    our = open_gguf(imatrix_path, "imatrix")
    have = len(our.tensors) // 4          # .in_sum2/.counts/.in_sum/.stats per tensor
    if int(want) != have:
        raise SystemExit(
            f"FATAL: the imatrix {imatrix_path} covers {have} tensor(s) but the base "
            f"declares quantize.imatrix.entries_count={want} — this is NOT the base's own "
            f"imatrix, so a requantisation with it would break §5.3 ('the SAME imatrix') "
            f"and confound any later quality measurement with a requantisation")
    rec = str(prov.get("quantize.imatrix.file", ""))
    if rec and Path(rec).name != Path(imatrix_path).name:
        print(f"imatrix     : WARN base records quantize.imatrix.file={rec}, this run uses "
              f"{imatrix_path} — same entries_count ({have}), different path. Accepting on "
              f"the count; the stamped sha256 is this file's own.")
    return f"entries_count={have} matches the base's declaration"


def f16_tensor_for(reader, layer: int, logical: str, ne01: tuple[int, int]):
    """The >=16-bit source's tensor for one base logical — the merge reference.

    Refuses when ne[0..1] differ from the base segment's: that means the F16 is
    NOT the source the base was quantised from, and merging into it would produce
    an override whose rows do not correspond to the experts it declares.
    """
    name = f"blk.{layer}.{logical}.weight"
    by_name = {t.name: t for t in reader.tensors}
    t = by_name.get(name)
    if t is None:
        raise SystemExit(f"FATAL: the >=16-bit source has no '{name}' — it cannot be the source "
                         f"of this base")
    ne = tuple(int(x) for x in t.shape)
    if len(ne) != 3:
        raise SystemExit(f"FATAL: {name}: ne={ne} is not expert-indexed; a quantised override "
                         f"needs an expert axis to cut")
    if (ne[0], ne[1]) != tuple(ne01):
        raise SystemExit(f"FATAL: {name}: ne[0..1]={(ne[0], ne[1])} but the base segment's is "
                         f"{tuple(ne01)} — this >=16-bit file is NOT the source of that base, "
                         f"so its rows are not the rows the base was quantised from")
    return t


def cut_rows(t, ids: list[int], what: str) -> np.ndarray:
    """Those experts' rows out of the >=16-bit source, as F16, contiguous."""
    ne2 = int(t.shape[2])
    bad = [e for e in ids if not (0 <= e < ne2)]
    if bad:
        raise SystemExit(f"FATAL: {what}: ids {bad} outside the source's expert axis [0,{ne2})")
    return np.ascontiguousarray(t.data[list(ids)])


def build_delta(name: str, shape: tuple, rows: np.ndarray, delta_path: Path | None,
                delta_synth: str | None) -> tuple[np.ndarray | None, str]:
    """The F16 delta to merge, and the provenance string describing it.

    `--delta FILE` reads a tensor named exactly as the override tensor
    (`blk.<l>.<logical>.ov<k>.weight`, numpy-order shape) — the shape of a real
    converted adapter's slice, so a trained variant needs no new code path.

    `--delta-synth SEED:RELRMS` is a DETERMINISTIC synthetic delta, N(0,(RELRMS *
    rms(rows))^2), drawn with numpy's PCG64 default RNG.  It exists because no
    trained adapter exists in this project yet; it is labelled synthetic in the
    file (`merge_delta=synthetic:...`) so it can never be mistaken for one.
    """
    if delta_path is not None and delta_synth is not None:
        raise SystemExit("FATAL: give --delta or --delta-synth, not both")
    if delta_path is None and delta_synth is None:
        return None, ""
    if delta_path is not None:
        r = open_gguf(delta_path, "delta")
        by_name = {t.name: t for t in r.tensors}
        t = by_name.get(name)
        if t is None:
            raise SystemExit(f"FATAL: delta {delta_path} has no tensor '{name}' — a delta names "
                             f"the override tensor it belongs to")
        got = tuple(int(x) for x in t.data.shape)     # numpy order (FINDINGS F3)
        if got != tuple(shape):
            raise SystemExit(f"FATAL: delta tensor '{name}' has byte shape {got}, expected "
                             f"{tuple(shape)} (numpy order: (n_ov, ne1, ne0))")
        return np.ascontiguousarray(t.data).astype(np.float16), \
            f"sha256:{sha256_file(delta_path)}"
    try:
        seed_s, scale_s = delta_synth.split(":")
        seed, scale = int(seed_s), float(scale_s)
    except ValueError:
        raise SystemExit(f"FATAL: --delta-synth expects SEED:RELRMS (e.g. 20261019:0.02), got "
                         f"'{delta_synth}'")
    if not (0.0 < scale < 1.0):
        raise SystemExit(f"FATAL: --delta-synth RELRMS {scale} is not in (0,1) — it is a "
                         f"fraction of the slice's own RMS")
    rms = float(np.sqrt(np.mean(np.square(rows.astype(np.float32)))))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape, dtype=np.float32)
    delta = (noise * np.float32(scale * rms)).astype(np.float16)
    return delta, f"synthetic:seed={seed},relrms={scale},slice_rms={rms:.8g}"


def merged_f16(rows: np.ndarray, delta: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """MERGE AT F16 (§5.3): accumulate in F32, store F16 (round-to-nearest-even).

    The convention is stated because it is a real choice: merging two F16 arrays
    elementwise in F16 arithmetic rounds twice, and a 221 GiB F32 reference does
    not exist here.  F32 accumulate + one F16 store is the widest reference this
    tree can hold, and it is applied identically on every run.
    """
    if delta is None:
        return rows, {"merged": False}
    acc = rows.astype(np.float32) + delta.astype(np.float32)
    out = acc.astype(np.float16)
    d = delta.astype(np.float32)
    return out, {
        "merged": True,
        "delta_rms": float(np.sqrt(np.mean(np.square(d)))),
        "delta_abs_max": float(np.max(np.abs(d))),
        "elems_changed": int(np.count_nonzero(out != rows)),
    }


# ---------------------------------------------------------------------------
# THE ADAPTER PATH (unit LORA): a PEFT safetensors adapter -> the F16 delta
# ---------------------------------------------------------------------------
# MEASURED (handoff-taillis-xcheck-result, against llama.cpp's own llama-export-lora
# on six cases): W' = W + (alpha/rank) * B @ A, with rank = lora_b's rank axis, and
# an ABSENT alpha meaning scale 1.0 (the user scale alone, llama-graph.cpp:1625).
# A is stored TRANSPOSED in GGUF (lora_a ne=(in, r) = numpy (r, in)) — which is
# exactly PEFT's `lora_A.weight`, so the two conventions meet with no transpose here.
# The scale is also PEFT's own: alpha defaults to r, so "no alpha" is scale 1.0 in
# both readings.
#
# THE MAPPING, and where it was MEASURED rather than assumed.  The GGUF fuses the
# source's per-expert w1 (gate) and w3 (up) into one `ffn_gate_up_exps` tensor and
# keeps w2 as `ffn_down_exps` (layout.md §6).  WHICH HALF of the fused out axis is
# which was measured on this host against the checkpoint's own FP8 weights
# (results/lora-gates/order_probe.py — re-runnable; its first home was wiped by a
# machine reboot and the probe was re-written and re-measured, same numbers): expert
# 0 of blk.0.ffn_gate_up_exps has rows [0,2048) equal to layers.0.experts.0.w1
# ELEMENTWISE (corr 1.00000000, max |diff| 3.0e-8, i.e. the F16 store) and rows
# [2048,4096) equal to w3 (same), while the crossed pairs give corr 0.0049 and a
# wrong-expert control gives 0.0027.  So, in the override tensor's own numpy
# (ne1=out, ne0=in) space:
#     w1 / gate_proj -> ffn_gate_up_exps rows [0, ne1/2)
#     w3 / up_proj   -> ffn_gate_up_exps rows [ne1/2, ne1)
#     w2 / down_proj -> ffn_down_exps, the whole out axis
# A per-expert delta is then numpy (out, in) = the override's own (ne1, ne0) slice,
# which is the space `--delta` already consumes, so no new payload path exists.
#
# THE HAZARD IS THE DESIGN CONSTRAINT.  A pair with NO expert index is a SHARED 2-D
# update; against a 3-D expert tensor llama-export-lora ABORTS (GGML_ASSERT(
# ggml_can_repeat), exit 134, MEASURED twice) and the runtime kernel would broadcast
# one delta to all 128 experts.  That shape is REFUSED by name here, never applied.
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
# PEFT prefixes the module path with `base_model.model.`.  The two model prefixes are
# the two checkpoint trees this project has on disk (layout.md §6's SOURCE names
# `layers.<l>.experts.<e>.w1`; the HF-format tree names
# `language_model.model.layers.<l>.mlp.experts.<e>.gate_proj`).  The list is FINITE:
# anything outside it is a named refusal, never a guessed strip.
LORA_PREFIXES = ("base_model.model.",)
LORA_MODEL_PREFIXES = ("language_model.model.", "model.", "")
LORA_MODULE_RE = re.compile(
    r"^layers\.(\d+)\.(?:mlp\.)?experts\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)$")
LORA_SHARED_RE = re.compile(
    r"^layers\.(\d+)\.(?:mlp\.)?experts\.(gate_up_proj|w1|w2|w3|gate_proj|up_proj|down_proj)$")
# projection -> (GGUF logical, block index in the fused out axis)
LORA_PROJ = {
    "w1": ("ffn_gate_up_exps", 0), "gate_proj": ("ffn_gate_up_exps", 0),
    "w3": ("ffn_gate_up_exps", 1), "up_proj": ("ffn_gate_up_exps", 1),
    "w2": ("ffn_down_exps", 0), "down_proj": ("ffn_down_exps", 0),
}
LORA_FUSED = "ffn_gate_up_exps"
# Dense module paths this architecture HAS but this overlay path CANNOT express: an
# expert-slice override is the only override kind that carries a merged delta (a
# whole-tensor one is an identity cut), so a dense-module LoRA has nowhere to land.
# A RECOGNISED dense name is counted and declared (never applied silently); an
# unrecognised one is a named refusal, so a new architecture fails loudly.
LORA_DENSE_RE = re.compile(
    r"^layers\.(\d+)\.(?:"
    r"attention\.(?:wq_a|wq_b|wkv_a_with_mqa|wkv_b|wo)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj|kv_a_proj_with_mqa|kv_b_proj"
    r"|q_a_proj|q_b_proj)"
    r"|mlp\.gate|mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)"
    r"|ffn_gate_inp"
    r")$")
LORA_CONFIG_REFS = ("rank_pattern", "alpha_pattern")


def lora_scale(alpha: float | None, rank: int) -> float:
    """`alpha/rank` when the adapter declares alpha, else 1.0 — MEASURED, not read.

    The two competing conventions this rules out are alpha/sqrt(rank) (rsLoRA, which
    is refused at load rather than silently mis-scaled) and pre-scaled alpha.
    """
    if alpha is None:
        return 1.0
    return float(alpha) / float(rank)


def _strip_lora_prefix(module: str) -> str | None:
    for p in LORA_PREFIXES:
        if module.startswith(p):
            module = module[len(p):]
            break
    for p in LORA_MODEL_PREFIXES:
        if p and module.startswith(p):
            return module[len(p):]
    return module if "" in LORA_MODEL_PREFIXES else None


class LoraAdapter:
    """A PEFT adapter, read ONCE: one 2-D (A, B) pair per (layer, expert, logical, block).

    `pairs[(layer, expert, logical, block)] = (A float32 (r, in), B float32 (out, r))`
    — PEFT's own layout, which is also ggml's lora_a/lora_b up to the transpose
    GGUF already applies to a numpy-order tensor.  Nothing is scaled at load: the
    scale is applied once, in `delta_for`.
    """

    def __init__(self, path: Path, cfg_path: Path, pairs: dict, alpha: float | None,
                 rank: int, dense: list[str]) -> None:
        self.path = path
        self.cfg_path = cfg_path
        self.pairs = pairs
        self.alpha = alpha
        self.rank = rank
        self.scale = lora_scale(alpha, rank)
        self.dense = dense
        self.sha = sha256_file(path)
        self.cfg_sha = sha256_file(cfg_path)

    def bounds(self, logical: str, ne1: int, what: str) -> list[tuple[int, int]]:
        """The (lo, hi) row blocks of the fused out axis for one logical."""
        if logical == LORA_FUSED:
            if ne1 % 2 or ne1 == 0:
                raise SystemExit(
                    f"FATAL: [lora:shape-mismatch] {what}: the base tensor's ne1={ne1} is not "
                    f"even, so it is not a w1|w3 fusion and an adapter pair cannot be placed")
            return [(0, ne1 // 2), (ne1 // 2, ne1)]
        return [(0, ne1)]

    def pair(self, layer: int, expert: int, logical: str, block: int, lo: int, hi: int,
             ne0: int, what: str) -> tuple[np.ndarray, np.ndarray]:
        key = (layer, expert, logical, block)
        got = self.pairs.get(key)
        if got is None:
            fused = (f" A FUSED logical needs BOTH halves for every expert: '{logical}' is "
                     f"w1|w3, so a gate-only (or up-only) adapter cannot be merged into it — "
                     f"half a fused tensor is not an overridable unit here."
                     if logical == LORA_FUSED else "")
            raise SystemExit(
                f"FATAL: [lora:uncovered-expert] {what}: the adapter has NO pair for layer "
                f"{layer} expert {expert} of '{logical}' block {block} — an override DECLARES "
                f"these experts covered, so leaving one at its base value would be a variant "
                f"that silently is not the adapter.{fused} Override the experts the adapter "
                f"covers, or train/convert a pair for this one")
        a, b = got
        if a.shape != (self.rank, ne0):
            raise SystemExit(
                f"FATAL: [lora:shape-mismatch] {what}: layer {layer} expert {expert} "
                f"'{logical}' block {block}: lora_A is {a.shape} but the base tensor's in axis "
                f"is ne0={ne0} and the rank is {self.rank} — expected {(self.rank, ne0)} "
                f"(PEFT lora_A.weight is (r, in); ggml lora_a ne=(in, r) is its transpose)")
        if b.shape != (hi - lo, self.rank):
            raise SystemExit(
                f"FATAL: [lora:shape-mismatch] {what}: layer {layer} expert {expert} "
                f"'{logical}' block {block}: lora_B is {b.shape} but rows [{lo},{hi}) of the "
                f"base tensor's out axis need {(hi - lo, self.rank)} (PEFT lora_B.weight is "
                f"(out, r))")
        return a, b

    def check_for(self, layer: int, logical: str, ids: list[int], ne01, what: str) -> None:
        """Shape/coverage checks for ONE override — no arithmetic, used before any write."""
        ne0, ne1 = int(ne01[0]), int(ne01[1])
        for block, (lo, hi) in enumerate(self.bounds(logical, ne1, what)):
            for e in ids:
                self.pair(layer, e, logical, block, lo, hi, ne0, what)

    def delta_for(self, layer: int, logical: str, ids: list[int], ne01, what: str) -> np.ndarray:
        """The (len(ids), ne1, ne0) float32 delta — `alpha/rank * B @ A` per block.

        FLOAT32, not F16: the arithmetic reference is as wide as this tree can hold, so
        the only rounding is `merged_f16`'s single F16 store.  `--delta` arrives already
        F16 because a GGUF holds F16; an adapter is a rank-r FACTORISATION, and
        rounding the product down before the merge would throw away precision that
        costs nothing to keep.
        """
        ne0, ne1 = int(ne01[0]), int(ne01[1])
        out = np.zeros((len(ids), ne1, ne0), dtype=np.float32)
        for j, e in enumerate(ids):
            for block, (lo, hi) in enumerate(self.bounds(logical, ne1, what)):
                a, b = self.pair(layer, e, logical, block, lo, hi, ne0, what)
                out[j, lo:hi] = np.float32(self.scale) * (b @ a)
        return out

    def map_note(self, ne1_by_logical: dict[str, int]) -> str:
        """The row-block placement, STAMPED: a wrong half is invisible once merged."""
        parts = []
        for logical in sorted(ne1_by_logical):
            ne1 = ne1_by_logical[logical]
            if logical == LORA_FUSED:
                h = ne1 // 2
                parts.append(f"{logical}[w1:0:{h},w3:{h}:{ne1}]")
            else:
                parts.append(f"{logical}[w2:0:{ne1}]")
        return ";".join(parts)


def load_lora(path: Path, config_path: Path | None) -> LoraAdapter:
    """Read a PEFT safetensors adapter.  EVERY refusal here is NAMED and pre-write.

    Accepted: `<module>.lora_A.weight` (r, in) + `<module>.lora_B.weight` (out, r), one
    pair per (expert, projection), module paths as in layout.md §6 or their HF aliases.
    An adapter is a FORMAT CONTRACT, so what cannot be honoured is refused rather than
    approximated: rsLoRA/DoRA/per-module rank or alpha (a different scale), a bias or
    a >2-D tensor (not a weight update this format can carry), a shared pair (the
    abort shape), a duplicated or half pair (an ambiguous or missing factor).
    """
    real = need(path, "--lora")
    if config_path is not None:
        cfg_path = need(config_path, "--lora-config")
    else:
        cand = real.parent / "adapter_config.json"
        if not cand.is_file():
            raise SystemExit(
                f"FATAL: [lora:config-missing] no {cand.name} beside {real.name} and no "
                f"--lora-config given. The config is where alpha comes from, and alpha IS the "
                f"merge scale (MEASURED: alpha/rank, or 1.0 when absent) — guessing it would "
                f"silently rescale every delta in the file")
        cfg_path = cand
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"FATAL: [lora:config-missing] cannot read {cfg_path}: {e}")

    if cfg.get("peft_type") not in (None, "LORA"):
        raise SystemExit(f"FATAL: [lora:unsupported-config] peft_type="
                         f"{cfg.get('peft_type')!r} is not LORA")
    if cfg.get("use_rslora"):
        raise SystemExit(
            "FATAL: [lora:unsupported-config] use_rslora=true — rsLoRA's scale is "
            "alpha/sqrt(rank), and the merge scale llama.cpp applies (and this tool "
            "reproduces) is alpha/rank. Applying one to an adapter trained with the other "
            "silently rescales every delta by sqrt(rank)")
    if cfg.get("use_dora"):
        raise SystemExit(
            "FATAL: [lora:unsupported-config] use_dora=true — DoRA's update is not B@A; it "
            "scales by the base weight's column norms, which a plain rank-r product cannot "
            "reproduce")
    for key in LORA_CONFIG_REFS:
        if cfg.get(key):
            raise SystemExit(
                f"FATAL: [lora:unsupported-config] {key} is non-empty — a per-module rank or "
                f"alpha is not one scale; folding it into a single alpha/rank would silently "
                f"mis-scale the modules it names")
    alpha = cfg.get("lora_alpha")
    if alpha is not None:
        if float(alpha) == 0.0:
            raise SystemExit(
                "FATAL: [lora:unsupported-config] lora_alpha=0 — llama.cpp's scale test is "
                "`alpha ? ... : user_scale`, so 0 means ABSENT there (scale 1.0) while PEFT "
                "would compute 0 (a no-op). Refusing to pick one for you")
        alpha = float(alpha)
    cfg_rank = cfg.get("r", cfg.get("rank"))

    try:
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover - environment error, not logic
        raise SystemExit(f"FATAL: [lora:tools] --lora needs safetensors ({e}); use "
                         f"venvs/convert/bin/python3")

    pairs: dict[tuple[int, int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    half_a: dict[tuple[int, int, str, int], np.ndarray] = {}
    half_b: dict[tuple[int, int, str, int], np.ndarray] = {}
    dense: list[str] = []
    seen_dense: set[str] = set()
    rank = -1

    def classify(key: str) -> tuple[str, str]:
        if key.endswith(LORA_A_SUFFIX):
            return key[:-len(LORA_A_SUFFIX)], "A"
        if key.endswith(LORA_B_SUFFIX):
            return key[:-len(LORA_B_SUFFIX)], "B"
        raise SystemExit(
            f"FATAL: [lora:unmappable-module] {real.name}: '{key}' is neither "
            f"<module>{LORA_A_SUFFIX} nor <module>{LORA_B_SUFFIX} — a tensor this format "
            f"cannot place (a DoRA magnitude vector and an embedding adapter both land here; "
            f"a bias delta is not a weight update at all)")

    with safe_open(str(real), framework="pt") as f:
        for key in f.keys():
            module, which = classify(key)
            bare = _strip_lora_prefix(module)
            m = LORA_MODULE_RE.match(bare) if bare else None
            if m is None:
                if bare and LORA_SHARED_RE.match(bare):
                    raise SystemExit(
                        f"FATAL: [lora:shared-pair-on-expert-tensor] {real.name}: '{key}' "
                        f"targets '{bare}', which has NO expert index. The base tensor "
                        f"'ffn_gate_up_exps'/'ffn_down_exps' is 3-D (one weight per expert), so "
                        f"a 2-D pair here is ONE SHARED update: llama-export-lora ABORTS on "
                        f"exactly this shape (GGML_ASSERT(ggml_can_repeat), exit 134, MEASURED) "
                        f"and the runtime kernel would broadcast it to all 128 experts. Emit one "
                        f"pair per expert (e.g. '...experts.<e>.w1' / '.w3' / '.w2') instead — "
                        f"this tool will not apply one delta to 128 experts")
                if bare and LORA_DENSE_RE.match(bare):
                    if bare not in seen_dense:
                        seen_dense.add(bare)
                        dense.append(bare)
                    continue
                raise SystemExit(
                    f"FATAL: [lora:unmappable-module] {real.name}: '{key}' -> '{bare}' matches "
                    f"no module this tool can place. Accepted: "
                    f"'layers.<l>.experts.<e>.(w1|w2|w3) or (.gate_proj|.up_proj|.down_proj)', "
                    f"optionally under 'base_model.model.' and one of "
                    f"{[p for p in LORA_MODEL_PREFIXES if p]}. An unmappable name is refused "
                    f"rather than skipped: a skipped pair is a variant that silently is not "
                    f"the adapter")
            layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            logical, block = LORA_PROJ[proj]
            arr = f.get_tensor(key).to(dtype=_torch_f32()).numpy()
            if arr.ndim != 2:
                raise SystemExit(
                    f"FATAL: [lora:unsupported-tensor] {real.name}: '{key}' is {arr.ndim}-D "
                    f"{tuple(arr.shape)} — this path carries ONE 2-D pair per (expert, "
                    f"projection). A batched pair cannot say which half of a fused "
                    f"gate_up axis its rows mean, and a broadcast pair is the abort shape")
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            k = (layer, expert, logical, block)
            r_here = arr.shape[0] if which == "A" else arr.shape[1]
            if rank < 0:
                rank = r_here
            elif r_here != rank:
                raise SystemExit(
                    f"FATAL: [lora:rank-mismatch] {real.name}: '{key}' has rank {r_here} but "
                    f"another pair in the same adapter has {rank} — one adapter, one scale "
                    f"(alpha/rank), so a mixed-rank adapter cannot be merged with a single "
                    f"scale")
            if which == "A":
                mine, other = half_a, half_b
            else:
                mine, other = half_b, half_a
            if k in mine or k in pairs:
                raise SystemExit(
                    f"FATAL: [lora:duplicate-pair] {real.name}: '{key}' is a SECOND lora_{which} "
                    f"for layer {layer} expert {expert} '{logical}' block {block} — two names "
                    f"map to one (expert, projection) (e.g. 'w1' and 'gate_proj'), so which "
                    f"factor wins would be arbitrary")
            if k in other:
                b_arr = other.pop(k)
                pairs[k] = (arr, b_arr) if which == "A" else (b_arr, arr)
            else:
                mine[k] = arr

    if half_a or half_b:
        missing = sorted(set(half_a) | set(half_b))
        k0 = missing[0]
        raise SystemExit(
            f"FATAL: [lora:incomplete-pair] {real.name}: {len(missing)} pair(s) have only one "
            f"factor (e.g. layer {k0[0]} expert {k0[1]} '{k0[2]}' block {k0[3]}) — a LoRA "
            f"update needs BOTH A and B; merging one alone would be a different update")
    if not pairs:
        raise SystemExit(f"FATAL: [lora:unmappable-module] {real.name} carries no usable "
                         f"expert pair ({len(dense)} dense module(s) and nothing else)")
    if rank < 1:
        # A (0, in) lora_A is constructible in safetensors, and `lora_scale` divides by the
        # rank — so without this it is a bare ZeroDivisionError instead of a named refusal.
        raise SystemExit(
            f"FATAL: [lora:rank-mismatch] {real.name}: the tensors' rank is {rank} — the merge "
            f"scale is alpha/rank, so rank 0 is not a degenerate adapter, it is an undefined "
            f"one (and a rank-0 update is a no-op that would still be declared as a merge)")
    if cfg_rank is not None and int(cfg_rank) != rank:
        raise SystemExit(
            f"FATAL: [lora:rank-mismatch] the config declares r={cfg_rank} but the tensors' "
            f"rank is {rank} — the scale is alpha/rank, so the two cannot disagree")

    return LoraAdapter(real, cfg_path, pairs, alpha, rank, dense)


def _torch_f32():
    import torch
    return torch.float32


def lora_plan_report(ad: LoraAdapter, resolved: list) -> str:
    """Refuse every adapter/override disagreement BEFORE anything is written, and
    return the ONE provenance note the whole file carries.

    Two aggregates can only be made here, and both are DECLARED rather than silent:
    the pairs this build APPLIES, and the pairs it does not — a recognised dense
    module (this overlay path has nowhere to put a dense delta) and an expert pair
    that no override covers.  A partial application is legitimate (an 8-hot-expert
    overlay IS one), but it must be visible in the file and in the log, because
    "the adapter's variant" and "part of the adapter's variant" are different
    artifacts.
    """
    used: set[tuple[int, int, str, int]] = set()
    ne1_by_logical: dict[str, int] = {}
    for k, layer, logical, ids, kind, seg in resolved:
        what = f"blk.{layer}.{logical} ov{k}"
        ad.check_for(layer, logical, ids, seg.ne01, what)
        ne1 = int(seg.ne01[1])
        ne1_by_logical[logical] = ne1
        for e in ids:
            for block in range(len(ad.bounds(logical, ne1, what))):
                used.add((layer, e, logical, block))
    unused = sorted(set(ad.pairs) - used)
    if unused:
        per_logical: dict[str, int] = {}
        for (_, _, logical, _) in unused:
            per_logical[logical] = per_logical.get(logical, 0) + 1
        print(f"WARNING     : {len(unused)} adapter pair(s) are NOT applied "
              f"({', '.join(f'{v} {k}' for k, v in sorted(per_logical.items()))}) — no "
              f"--override covers those (layer, expert). The overlay is PART of this adapter, "
              f"not all of it; declare the remaining experts to apply them")
    if ad.dense:
        print(f"WARNING     : {len(ad.dense)} recognised DENSE module(s) are NOT applied "
              f"({', '.join(sorted(ad.dense)[:3])}"
              f"{' ...' if len(ad.dense) > 3 else ''}) — an expert-slice override is the only "
              f"override kind that carries a merged delta, so this format has nowhere to put a "
              f"dense-module LoRA. Their absence is declared below, never silent")
    alpha_note = "absent" if ad.alpha is None else f"{ad.alpha:g}"
    return (f"lora:sha256={ad.sha},config={ad.cfg_sha},alpha={alpha_note},rank={ad.rank},"
            f"scale={ad.scale:.10g},applied={len(used)},unapplied={len(unused)},"
            f"dense_skipped={len(ad.dense)},map={ad.map_note(ne1_by_logical)}")


def type_requires_imatrix(target_type: str, ftype: str) -> bool:
    """True iff llama-quantize needs an importance matrix for this target type.

    MIRRORS THE QUANTISER'S OWN LINE (`tensor_requires_imatrix`,
    llama-quant.cpp:856-875), not a policy invented here: `IQ3_XXS, IQ2_XXS,
    IQ2_XS, IQ2_S, IQ1_M, IQ1_S` always, and `Q2_K` only inside a `Q2_K_S` file
    (`ftype` is the file type, i.e. the positional ftype of the run).
    """
    t = target_type.upper()
    if t in IMATRIX_TYPES:
        return True
    return t == "Q2_K" and ftype.upper() in ("Q2_K_S", "MOSTLY_Q2_K_S")


def type_sizes(type_name: str) -> tuple[int, int]:
    """(block_size, type_size) of a ggml type by NAME; a NAMED failure if unknown."""
    try:
        return GGML_QUANT_SIZES[GGMLQuantizationType[type_name.upper()]]
    except KeyError:
        raise SystemExit(f"FATAL: '{type_name}' is not a ggml type this tool knows; the "
                         f"override's type IS the tensor's own ggml type (V4.3 clause 5), so an "
                         f"unnameable type cannot be built or checked")


def expert_stride(type_name: str, ne01: tuple[int, int]) -> int:
    """Bytes per expert for a tensor of this type and shape, from the type's own
    block size — the check that a type is EXPRESSIBLE for the tensor (V4.3
    clause 5: any type whose block size divides ne[0])."""
    block, size = type_sizes(type_name)
    ne0, ne1 = ne01
    if ne0 % block:
        raise SystemExit(
            f"FATAL: type {type_name} has a block size of {block} and ne[0]={ne0} is not "
            f"divisible by it, so this type cannot express this tensor at all (layout.md §1, "
            f"V4.3 clause 5)")
    return (ne0 // block) * size * ne1


def write_slice_gguf(path: Path, src_reader, tensor_name: str, rows: np.ndarray) -> None:
    """A THROWAWAY GGUF holding only the override rows, quantisable by llama-quantize.

    The KV block is copied VERBATIM from the >=16-bit source: llama-quantize loads
    a MODEL (hparams, then the tensor table), so the arch keys must be there.  The
    tensor is named exactly as the base logical, because the base's own
    `--tensor-type-file` patterns are anchored on that name — renaming it would
    silently send the slice to the type file's fall-through and produce the WRONG
    type with no error.
    """
    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    w.data_alignment = src_reader.alignment
    for key, field in src_reader.fields.items():
        if key in RESERVED_KV:
            continue
        main = field.types[0]
        if main == GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), main)
    w.add_tensor(tensor_name, rows)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_cropped_imatrix(path: Path, src_reader, tensor_name: str, ids: list[int],
                          ne0: int) -> None:
    """The base's OWN imatrix, cropped to the experts this override covers.

    WHY THIS IS NOT OPTIONAL (MEASURED from llama-quant.cpp:1284 and 784-786, and
    by the quantiser REFUSING our first attempt):

      * an imatrix entry for an expert tensor is per-(expert, column): its length
        must be exactly ne[0]*ne[2] (`it->second.size() == tensor->ne[0]*tensor->ne[2]`),
        and the weights a row uses are `imatrix + (row_global/nrows_per_expert) * ne[0]`
        — i.e. ONE ne[0] block per expert, shared by all of that expert's rows.
      * a 2-expert slice therefore has ne[0]*2 = 8192 weights, and handing the
        quantiser the full 524288-length matrix is a HARD ERROR ("imatrix size
        524288 is different from tensor size 8192"), not a warning.

    So "re-quantise with the SAME imatrix" (§5.3) has to mean: the same matrix,
    CROPPED to the same experts.  The crop is exact and lossless — expert e's
    block is bytes [e*ne0, (e+1)*ne0) and no row of expert e depends on any other
    expert's block.  `.stats` is NOT copied: it describes the full 128-expert
    tensor and would be a false declaration about a slice (the loader does not
    read it).
    """
    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    for key, field in src_reader.fields.items():
        if key in RESERVED_KV:
            continue
        main = field.types[0]
        if main == GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), main)
    by_name = {t.name: t for t in src_reader.tensors}
    counts = by_name.get(tensor_name + ".counts")
    if counts is None:
        raise SystemExit(f"FATAL: the imatrix has no '{tensor_name}.counts' entry, which the "
                         f"loader REQUIRES alongside in_sum2 (imatrix-loader.cpp)")
    n_expert = None
    for suffix in (".in_sum2", ".in_sum"):
        t = by_name.get(tensor_name + suffix)
        if t is None:
            if suffix == ".in_sum2":
                raise SystemExit(f"FATAL: the imatrix has no '{tensor_name}.in_sum2' entry")
            continue
        arr = np.ascontiguousarray(t.data, dtype=np.float32)      # numpy (n_expert, ne0)
        if arr.ndim != 2 or arr.shape[1] != ne0:
            raise SystemExit(f"FATAL: {t.name} has shape {tuple(arr.shape)}, expected "
                             f"(n_expert, {ne0})")
        n_expert = arr.shape[0]
        bad = [e for e in ids if not (0 <= e < n_expert)]
        if bad:
            raise SystemExit(f"FATAL: {t.name}: ids {bad} outside its {n_expert} expert rows")
        w.add_tensor(tensor_name + suffix, np.ascontiguousarray(arr[list(ids)]))
    carr = np.ascontiguousarray(counts.data, dtype=np.float32).reshape(-1)
    if carr.size == n_expert:
        w.add_tensor(tensor_name + ".counts", np.ascontiguousarray(carr[list(ids)]))
    else:
        w.add_tensor(tensor_name + ".counts", carr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def run_llama_quantize(binary: Path, slice_path: Path, out_path: Path, imatrix: Path | None,
                       type_file: Path | None, quant_type: str, nthread: int,
                       log_path: Path) -> str:
    """Quantise the slice at `quant_type`; return the llama-quantize tool string.

    A --dry-run pass runs FIRST: it parses the spec and reports the plan, so a
    malformed type/type-file fails in ~200 ms instead of after a real quantisation
    (this is the check that caught a name=type vs name type error in an earlier
    unit).  The PRODUCED type is then read back from the output file by the caller
    — the dry run is not trusted to prove the assignment.

    `imatrix` and `type_file` are OPTIONAL and their absence is not a fallback: an
    imatrix is passed only when the target type requires one (llama-quant.cpp:856),
    and a type file only when the caller has one.  Neither is ever synthesised.
    """
    argv0, env = quantize_runner(binary)
    base_cmd = [argv0]
    if imatrix is not None:
        base_cmd += ["--imatrix", str(imatrix)]
    if type_file is not None:
        base_cmd += ["--tensor-type-file", str(type_file)]
    with open(log_path, "wb") as log:
        for extra in (["--dry-run"], []):
            cmd = [argv0] + extra + base_cmd[1:] + [str(slice_path), str(out_path),
                                                   quant_type, str(nthread)]
            r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            if r.returncode != 0:
                tail = log_path.read_text(errors="replace").splitlines()[-6:]
                raise SystemExit(
                    f"FATAL: llama-quantize {'--dry-run ' if extra else ''}failed "
                    f"(exit {r.returncode}) on {slice_path}; last lines:\n  "
                    + "\n  ".join(tail))
    tool = "llama-quantize@unknown"
    for line in log_path.read_text(errors="replace").splitlines()[:4]:
        m = re.search(r"commit ([0-9a-f]{7,})", line)
        if m:
            tool = f"llama-quantize@{m.group(1)}"
            break
    return tool


def read_quantised(path: Path, want_type: str, want_nbytes: int, what: str):
    """(bytes, numpy shape, ggml type) of the quantised slice, with the type CHECKED.

    A produced type that is not the REQUESTED one is a NAMED FAILURE and not a
    warning: it means the quantiser's spec did not match this slice, and an overlay
    of an unexpected type is refused at load by V4.3 clause 5 — or worse, by a
    stride CHECK at the first rotation far too late (expert-swap.md §5.3).

    `want_type` is the target type the build asked for, which on the SAME-TYPE
    regime is the base segment's own type and on the DIFFERENT-TYPE regime (S7)
    deliberately is not: the type is the tensor's own ggml type either way, and
    nothing about the encoding changes (V4.3 clause 5).
    """
    r = open_gguf(path, "quantised slice")
    if len(r.tensors) != 1:
        raise SystemExit(f"FATAL: {what}: the quantiser wrote {len(r.tensors)} tensors into "
                         f"{path}; expected exactly 1")
    t = r.tensors[0]
    if t.tensor_type.name != want_type:
        raise SystemExit(
            f"FATAL: {what}: the quantiser produced {t.tensor_type.name} but {want_type} was "
            f"requested — the spec handed to llama-quantize did NOT hold for this slice (its "
            f"tensor name did not match the type file, or the positional ftype was not the "
            f"target), so these bytes are not the type the overlay would declare")
    if t.data.nbytes != want_nbytes:
        raise SystemExit(f"FATAL: {what}: quantised nbytes {t.data.nbytes} != "
                         f"{want_nbytes} (type stride * experts)")
    return np.ascontiguousarray(t.data), tuple(int(x) for x in t.data.shape), t.tensor_type


def requantised_payload(*, base_name: str, layer: int, logical: str, ids: list[int],
                        seg, f16_reader, imatrix_reader, imatrix: Path | None,
                        type_file: Path | None,
                        quant_type: str | None = None, target_type: str | None = None,
                        nthread: int, quantize_bin: Path, workdir: Path,
                        delta_path: Path | None, delta_synth: str | None,
                        ov_name: str, lora=None, lora_note: str = "") -> dict:
    """cut the F16 rows -> merge the delta at F16 -> re-quantise at the target type.

    ONE implementation, used by BUILD and by --verify's reproduction mode, so the
    gate and the writer cannot drift apart.

    `target_type` is the type the override must be produced at: the base segment's
    own type on the SAME-TYPE regime (the default, and aubier's S1-S5), or a
    DIFFERENT type on S7's regime.  `imatrix_reader`/`imatrix` are None whenever
    the target type does not require one (llama-quant.cpp:856), and then NO crop is
    written and none is declared: nothing was imatrix-weighted, so naming a matrix
    would be a false provenance claim.
    """
    what = f"{ov_name} ({base_name})"
    target = (target_type or seg.tensor.tensor_type.name).upper()
    # The POSITIONAL ftype handed to llama-quantize: the caller's --quant-type when
    # given, else the target itself.  It is also the `ftype` llama-quantize judges
    # `Q2_K` against (`tensor_requires_imatrix`'s Q2_K_S case), so it is recorded in
    # the flags and re-derived by --verify — never guessed.
    positional = (quant_type or target).upper()
    src = f16_tensor_for(f16_reader, layer, logical, seg.ne01)
    rows = cut_rows(src, ids, what)
    # element dims, NOT seg.tail: for a QUANTISED base segment, tensor.data.shape is
    # numpy order (ne2, ne1, row_bytes) — its last dim is a packed byte count, not
    # ne0 (FINDINGS F3, the same trap the S1 writer documents).  The F16 cut is in
    # elements: (n_ov, ne1, ne0).
    want_shape = (len(ids), seg.ne01[1], seg.ne01[0])
    if tuple(rows.shape) != want_shape:
        raise SystemExit(f"FATAL: {what}: cut shape {tuple(rows.shape)} != {want_shape}")
    if lora is not None:
        # The adapter path: alpha/rank * B @ A in F32, per expert and per fused block.
        delta = lora.delta_for(layer, logical, ids, seg.ne01, what)
        delta_note = lora_note
    else:
        delta, delta_note = build_delta(ov_name, want_shape, rows, delta_path, delta_synth)
    merged, merge_info = merged_f16(rows, delta)

    workdir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", ov_name)
    slice_path = workdir / f"{tag}.f16.gguf"
    quant_path = workdir / f"{tag}.{positional.lower()}.gguf"
    for p in (slice_path, quant_path):
        if p.exists():
            p.unlink()                      # our own throwaway, never a base
    write_slice_gguf(slice_path, f16_reader, f"blk.{layer}.{logical}.weight", merged)
    crop_path = None
    if imatrix_reader is not None:
        crop_path = workdir / f"{tag}.imatrix.gguf"
        if crop_path.exists():
            crop_path.unlink()
        write_cropped_imatrix(crop_path, imatrix_reader, f"blk.{layer}.{logical}.weight", ids,
                              seg.ne01[0])
    tool = run_llama_quantize(quantize_bin, slice_path, quant_path, crop_path, type_file,
                              positional, nthread, workdir / f"{tag}.quantize.log")
    # The stamped spec names the BASE matrix, not the throwaway crop: the crop is a
    # deterministic function of (that matrix, these ids), so it is reproducible
    # from the declaration, while a /tmp path would not be.  `--expert-crop` is OUR
    # clause, not a llama-quantize flag — llama-quantize has none, and cannot have
    # one, because the crop is what makes its length check pass.  With no imatrix in
    # hand neither clause is stamped, because neither is true.
    parts: list[str] = []
    if imatrix is not None:
        parts += ["--imatrix", str(imatrix), "--expert-crop", ",".join(str(e) for e in ids)]
    if type_file is not None:
        parts += ["--tensor-type-file", str(type_file)]
    parts += [positional, str(nthread)]
    flags = " ".join(parts)
    same_type = target == seg.tensor.tensor_type.name.upper()
    want_nbytes = (seg.stride if same_type else expert_stride(target, seg.ne01)) * len(ids)
    payload, out_shape, out_type = read_quantised(quant_path, target, want_nbytes, what)
    return {"payload": payload, "shape": out_shape, "type": out_type, "flags": flags,
            "tool": tool, "target": target, "same_type": same_type,
            "delta_note": delta_note, "merge": merge_info, "slice": slice_path,
            "quantised": quant_path}


def one_delta_source(lora: Path | None, delta: Path | None,
                     delta_synth: str | None) -> None:
    """Exactly ONE delta source, on BUILD and on --verify alike.

    Two sources would mean one is silently ignored — and `requantised_payload` prefers
    the adapter, so `--verify --lora ... --delta ...` would reproduce from the adapter
    while the caller believed the delta was in play.  One rule, one wording, both paths.
    """
    sources = [n for n, v in (("--lora", lora), ("--delta", delta),
                              ("--delta-synth", delta_synth)) if v is not None]
    if len(sources) > 1:
        raise SystemExit(
            f"FATAL: [lora:delta] {', '.join(sources)} are all delta SOURCES — give exactly "
            f"one. Two would mean one is silently ignored while the file declares a merge it "
            f"did not make")


def build(base: Path, out: Path, raw_overrides: list[str], identity: bool,
          delta: Path | None, manifest: Path | None,
          imatrix: str | None, quant_flags: str | None,
          f16: Path | None = None, requantise: bool = False,
          imatrix_path: Path | None = None, type_file: Path | None = None,
          quant_type: str | None = None, nthread: int = 8,
          quantize_bin: Path | None = None, workdir: Path | None = None,
          delta_synth: str | None = None, input_sha: str | None = None,
          lora: Path | None = None, lora_config: Path | None = None,
          target_type: str | None = None) -> int:
    real = need(base, "--base")
    out = out.resolve()
    check_write_scope(out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} already exists; remove it or pick another --out")
    if out == real:
        raise SystemExit("FATAL: --out is the base file; an overlay is a NEW file, the base is a "
                         "READ-ONLY input")
    if identity == requantise:
        raise SystemExit("FATAL: pass exactly one of --identity (verbatim cut, unit S1) or "
                         "--requantise (merge at F16 then re-quantise with the base's own "
                         "spec, unit S1b) — they are different payload paths and the file must "
                         "declare which one produced it")
    one_delta_source(lora, delta, delta_synth)
    if requantise:
        missing = [n for n, v in (("--f16", f16), ("--llama-quantize", quantize_bin)) if v is None]
        if missing:
            raise SystemExit(
                f"FATAL: --requantise needs {', '.join(missing)} — the >=16-bit source is the "
                f"only place the merge may happen, and the quantiser is what makes the payload. "
                f"(--imatrix and --type-file are decided against the TARGET type below: "
                f"llama-quantize requires an imatrix only for "
                f"{', '.join(sorted(IMATRIX_TYPES))} and for Q2_K inside a Q2_K_S file, so "
                f"demanding one unconditionally demands a file the quantiser may not read)")
        if quant_flags is not None or imatrix is not None:
            raise SystemExit(
                "FATAL: --quant-flags / --imatrix-sha256 are hand-written CLAIMS. On the "
                "--requantise path both are DERIVED from the files actually used, so declaring "
                "them is a second thing that can disagree with the first (V4.4)")
        if target_type is not None:
            target_type = target_type.upper()
            type_sizes(target_type)          # named failure if it is not a ggml type
    else:
        if delta is not None or delta_synth is not None or lora is not None:
            raise SystemExit("FATAL: a delta is merged into a QUANTISED payload; it has no "
                             "meaning on the --identity path, which copies the base's bytes "
                             "verbatim and must produce exactly those bytes (§5.3)")
        if f16 is not None or imatrix_path is not None or type_file is not None:
            raise SystemExit("FATAL: --f16 / --imatrix / --type-file belong to --requantise; an "
                             "identity overlay reproduces the base's own bytes and quantises "
                             "nothing")
        if target_type is not None:
            raise SystemExit("FATAL: --target-type belongs to --requantise; an identity overlay "
                             "copies the base's own bytes at the base's own type, so asking for a "
                             "different one is a contradiction (V4.3 clause 5)")
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

    if requantise:
        whole = [f"blk.{l}.{lg}" for _, l, lg, ids, kind, _ in resolved if kind == KIND_WHOLE]
        if whole:
            raise SystemExit(
                f"FATAL: --requantise was given {', '.join(whole)}, which is a WHOLE-TENSOR "
                f"override. It has no expert axis to cut, its base type is F32/F16 (not a "
                f"quantised type), and re-quantising it with the same spec is a byte copy — "
                f"so §5.3 has nothing to say about it. Use --identity for those tensors and a "
                f"separate overlay for the quantised experts")

    # THE TARGET TYPE and the IMATRIX DECISION, both resolved per override BEFORE
    # the base is hashed and before anything is written.
    #
    # The target type is the base segment's OWN type (the SAME-TYPE regime, aubier's
    # S1-S5) unless --target-type asks for a DIFFERENT one (S7).  Expressibility is
    # checked here and FAILS LOUDLY: a type whose block size does not divide ne[0]
    # cannot express the tensor at all (layout.md §1, V4.3 clause 5), and
    # substituting a type that can would be a silent lie about what the overlay is.
    targets: dict[int, str] = {}
    imatrix_needed = False
    if requantise:
        for k, layer, logical, ids, kind, obj in resolved:
            t = (target_type or obj.tensor.tensor_type.name).upper()
            try:
                expert_stride(t, obj.ne01)
            except SystemExit as e:
                raise SystemExit(f"{e}\n  (override blk.{layer}.{logical}.ov{k}: "
                                 f"ne={obj.ne01 + (len(ids),)}; STOPPING rather than "
                                 f"substituting a type — V4.3 clause 5 makes the tensor's "
                                 f"own type the whole declaration)")
            targets[k] = t
        # `ftype` in llama-quantize's check is the POSITIONAL of the run, which is
        # --quant-type when given and the target otherwise (requantised_payload builds
        # the same value) — so a Q2_K target is judged as a Q2_K run, and only a
        # Q2_K_S positional makes it imatrix-requiring, exactly as the quantiser does.
        def _positional(t: str) -> str:
            return (quant_type or t).upper()

        needs_im = sorted({t for t in targets.values()
                           if type_requires_imatrix(t, _positional(t))})
        cannot = sorted({t for t in targets.values()
                         if not type_requires_imatrix(t, _positional(t))})
        if needs_im and imatrix_path is None:
            raise SystemExit(
                f"FATAL: target type(s) {', '.join(needs_im)} REQUIRE an importance matrix "
                f"(llama-quant.cpp:856 tensor_requires_imatrix) and no --imatrix was given. "
                f"llama-quantize refuses these types without one, and quantising them without "
                f"it is what the quantiser's own gate exists to stop")
        if imatrix_path is not None and not needs_im:
            raise SystemExit(
                f"FATAL: --imatrix was given but the target type(s) {', '.join(cannot)} cannot "
                f"use one — llama-quantize requires an importance matrix only for "
                f"{', '.join(sorted(IMATRIX_TYPES))} and for Q2_K inside a Q2_K_S file "
                f"(llama-quant.cpp:856). Stamping an imatrix sha256 (and a crop) on a build "
                f"that never read it is a FALSE PROVENANCE claim — the same lie as a missing "
                f"one, pointed the other way — so it is refused, not warned about")
        imatrix_needed = bool(needs_im)

    # The ADAPTER is read and CROSS-CHECKED against the whole override plan BEFORE the
    # base is hashed and before anything is written: an adapter whose shapes disagree
    # with the base, whose pairs miss a declared expert, or that cannot be placed at
    # all must leave nothing behind.  (The delta itself is built inside
    # requantised_payload, which --verify shares — one implementation, not two.)
    lora_ad = None
    lora_note = ""
    if lora is not None:
        lora_ad = load_lora(lora, lora_config)
        print(f"adapter     : {lora_ad.path.name}  rank={lora_ad.rank}  "
              f"alpha={'absent (scale 1.0)' if lora_ad.alpha is None else f'{lora_ad.alpha:g}'}"
              f"  scale={lora_ad.scale:.10g} (= alpha/rank, MEASURED)  "
              f"pairs={len(lora_ad.pairs)}  dense={len(lora_ad.dense)}")
        lora_note = lora_plan_report(lora_ad, resolved)
        print(f"  map       : {lora_ad.map_note({lg: int(seg.ne01[1]) for _, _, lg, _, _, seg in resolved})}")
        print(f"  sha256    : {lora_ad.sha} (adapter), {lora_ad.cfg_sha} (config)")

    parent_sha = sha256_file(real)
    tool = tool_version()

    # The quantised path's inputs are resolved and CROSS-CHECKED before anything is
    # written: the base's own declared provenance is compared with the imatrix in
    # hand (§5.3's "SAME imatrix" is only checkable from the file), and the >=16-bit
    # source's shapes are checked against the base's segments per override.
    payloads: dict[int, dict] = {}
    quant_prov: dict[str, str] = {}
    digests: dict[int, str] = {}
    crops: list[str] = []
    f16_reader = None
    if requantise:
        f16_reader = open_gguf(need(f16, "--f16"), "F16 source")
        imatrix_path = need(imatrix_path, "--imatrix") if imatrix_needed else None
        type_file = need(type_file, "--type-file") if type_file is not None else None
        imatrix_reader = None
        if imatrix_path is not None:
            print(f"imatrix chk : {check_imatrix_against_base(reader, imatrix_path)}")
            imatrix_reader = open_gguf(imatrix_path, "imatrix")
        if input_sha is None:
            input_sha = sha256_file(need(f16, "--f16"))
        else:
            print(f"input sha   : {input_sha} SUPPLIED with --input-sha256, not recomputed — "
                  f"hashing a 222 GiB source costs ~5 min per build; the value is the caller's "
                  f"measurement and any reader can re-derive it with sha256sum")
        quant_prov = {INPUT_KEY: input_sha}
        if imatrix_path is not None:
            quant_prov[IMATRIX_KEY] = sha256_file(imatrix_path)
        else:
            print(f"imatrix     : ABSENT BY RULE — no target type needs one "
                  f"(llama-quant.cpp:856), so {IMATRIX_KEY} and {EXPERT_CROP_KEY} are NOT "
                  f"stamped: nothing was imatrix-weighted and claiming otherwise would be "
                  f"false provenance")
        if type_file is not None:
            quant_prov[TYPE_FILE_KEY] = sha256_file(type_file)
        wd = workdir if workdir is not None else Path("/tmp") / f"repack-overlay-{os.getpid()}"
        check_write_scope((wd / "probe").resolve())
        for k, layer, logical, ids, kind, obj in resolved:
            pos = (quant_type or targets[k]).upper()
            uses_imatrix = type_requires_imatrix(targets[k], pos)
            rp = requantised_payload(
                base_name=real.name, layer=layer, logical=logical, ids=ids, seg=obj,
                f16_reader=f16_reader,
                imatrix_reader=(imatrix_reader if uses_imatrix else None),
                imatrix=(imatrix_path if uses_imatrix else None), type_file=type_file,
                quant_type=pos, target_type=targets[k], nthread=nthread,
                quantize_bin=quantize_bin, workdir=wd,
                delta_path=delta, delta_synth=delta_synth,
                ov_name=f"blk.{layer}.{logical}.ov{k}.weight",
                lora=lora_ad, lora_note=lora_note)
            payloads[k] = rp
            quant_prov[QUANT_FLAGS_KEY] = rp["flags"]
            quant_prov[QUANT_TOOL_KEY] = rp["tool"]
            if rp["delta_note"]:
                quant_prov.setdefault(MERGE_DELTA_KEY, rp["delta_note"])

    # On the quantised path `overlay_identity` is DERIVED, never declared: the
    # writer holds both the payload and the base slab, so it compares them and
    # stamps the flag only if every payload really equals its base.  A requantise
    # with no delta SHOULD reproduce the base exactly (that is the S1b gate) — and
    # when it does, the honest file says identity=true while still carrying the
    # quantisation provenance, because a quantiser really did run.  V4.4's
    # biconditional ("identity ⇒ nothing was quantised, so no provenance") is
    # FALSIFIED by this case; the flag keeps its byte-level meaning and the
    # provenance states the build.
    derived_identity = identity
    if requantise:
        derived_identity = True
        fd_cmp = os.open(real, os.O_RDONLY)
        try:
            for k, layer, logical, ids, kind, seg in resolved:
                if targets[k] != seg.tensor.tensor_type.name.upper():
                    # DIFFERENT-TYPE regime (S7): the payload has a different number
                    # of bytes per expert than the base slab, so it cannot equal it —
                    # the flag is false by construction, and the byte comparison that
                    # would prove it is not even defined here.
                    derived_identity = False
                    print(f"base cmp    : blk.{layer}.{logical}.ov{k} -> not comparable "
                          f"(overlay {targets[k]} vs base {seg.tensor.tensor_type.name}): the "
                          f"payload is {expert_stride(targets[k], seg.ne01) * len(ids)} byte(s) "
                          f"where the base slab set is {seg.stride * len(ids)} — DIFFERENT-TYPE "
                          f"by construction, so {OVERLAY_IDENTITY_KEY} is not stamped")
                    continue
                act = payloads[k]["payload"].reshape(-1)
                per = []
                for j, e in enumerate(ids):
                    ref = np.frombuffer(os.pread(fd_cmp, seg.stride, seg.slab_offset(e)),
                                        dtype=np.uint8)
                    got = act[j * seg.stride:(j + 1) * seg.stride]
                    per.append(int(np.count_nonzero(ref != got)))
                if any(per):
                    derived_identity = False
                print(f"base cmp    : blk.{layer}.{logical}.ov{k} -> "
                      f"{sum(1 for d in per if d)}/{len(ids)} expert slab(s) differ from the "
                      f"base ({'payloads reproduce the base exactly' if not any(per) else 'the delta changed them'})")
        finally:
            os.close(fd_cmp)

    writer = GGUFWriter(str(out), arch="", use_temp_file=False)
    writer.data_alignment = reader.alignment

    fd = os.open(real, os.O_RDONLY)
    n_bytes = 0
    try:
        for k, layer, logical, ids, kind, obj in resolved:
            name = f"blk.{layer}.{logical}.ov{k}.weight"
            if kind == KIND_SLICE:
                seg = obj
                entry = ",".join(str(e) for e in ids)
                if requantise:
                    rp = payloads[k]
                    # The tensor's OWN ggml type IS the declaration (V4.3 clause 5) —
                    # on the different-type regime that is deliberately not the base
                    # segment's type, and the tensor is written at the type the
                    # quantiser produced (read back, checked, never assumed).
                    writer.add_tensor(name, rp["payload"], raw_dtype=rp["type"])
                    mode = (f"REQUANTISED from F16 at {rp['target']}"
                            + ("" if rp["same_type"] else
                               f" (DIFFERENT from the base segment's "
                               f"{seg.tensor.tensor_type.name})"))
                else:
                    chunks: list = []
                    for e in ids:
                        chunks += chunk_loaders(fd, seg.slab_offset(e), seg.stride)
                    writer.add_tensor(name, LazyChunkedTensor(
                        chunks, (len(ids),) + seg.tail, seg.dtype,
                    ), raw_dtype=seg.raw)
                    mode = "VERBATIM cut"
                writer.add_key_value(f"{OVERLAY_EXPERTS_PREFIX}{k}", entry,
                                     GGUFValueType.STRING)
                # V4.4: the payload digest, REQUIRED on every overlay.  It is
                # computed from the bytes actually written — the re-quantised
                # array, or the base's own slabs in ascending id order — never from
                # a second read of the file after the fact.
                if requantise:
                    dig = digest_array(payloads[k]["payload"])
                else:
                    dig = digest_ranges(fd, [(seg.slab_offset(e), seg.stride) for e in ids])
                writer.add_key_value(f"{OVERLAY_DIGEST_PREFIX}{k}", dig, GGUFValueType.STRING)
                digests[k] = dig
                if requantise:
                    # An imatrix was actually cropped only on this path AND only when
                    # the target type requires one; a q5_k build reads no matrix, so it
                    # declares no crop (V4.4's pair is a claim about bytes that were
                    # actually weighted).
                    if "--expert-crop" in payloads[k]["flags"]:
                        crops.append(f"ov{k}={entry}")
                n_bytes += (payloads[k]["payload"].nbytes if requantise
                            else len(ids) * seg.stride)
                print(f"override {k}  : {name}  ne={seg.ne01 + (len(ids),)}  "
                      f"type={payloads[k]['target'] if requantise else seg.tensor.tensor_type.name}"
                      f"  [{KIND_SLICE}, {mode}]")
                print(f"  base seg : {seg.name}  [{seg.lo},{seg.hi})  "
                      f"data_offset={seg.off}  stride={seg.stride} "
                      f"({seg.stride * len(ids)} bytes for {len(ids)} expert(s))")
                print(f"  id key   : {OVERLAY_EXPERTS_PREFIX}{k} = \"{entry}\"")
                print(f"  digest   : {OVERLAY_DIGEST_PREFIX}{k} = {dig}  "
                      f"(sha256 of the {payloads[k]['payload'].nbytes if requantise else seg.stride * len(ids)} payload byte(s))")
                if requantise:
                    mi = payloads[k]["merge"]
                    print(f"  merge    : delta={payloads[k]['delta_note'] or 'NONE'}  "
                          f"elems_changed={mi.get('elems_changed', 0)}  "
                          f"delta_rms={mi.get('delta_rms', 0.0):.8g}")
                else:
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
                dig = digest_ranges(fd, [(t.data_offset, t.data.nbytes)])
                writer.add_key_value(f"{OVERLAY_DIGEST_PREFIX}{k}", dig, GGUFValueType.STRING)
                digests[k] = dig
                n_bytes += t.data.nbytes
                print(f"override {k}  : {name}  ne={tuple(int(x) for x in t.shape)}  "
                      f"type={t.tensor_type.name}  [{KIND_WHOLE}]")
                print(f"  digest   : {OVERLAY_DIGEST_PREFIX}{k} = {dig}  "
                      f"(sha256 of the {t.data.nbytes} payload byte(s))")
                print(f"  base tensor : {t.name}  data_offset={t.data_offset}  "
                      f"nbytes={t.data.nbytes} (replaced WHOLE; no id key)")

        writer.add_key_value(PARENT_KEY, parent_sha, GGUFValueType.STRING)
        # V4.4 (settled): the crop the SAME imatrix was taken with, per override
        # tensor.  It must equal that tensor's id key — the checker cross-checks
        # the two, because a payload quantised with a DIFFERENT crop than the ids
        # it declares would otherwise be undetectable.
        if crops:
            writer.add_key_value(EXPERT_CROP_KEY, ";".join(crops), GGUFValueType.STRING)
        if derived_identity:
            writer.add_key_value(OVERLAY_IDENTITY_KEY, True, GGUFValueType.BOOL)
        writer.add_key_value(OVERLAY_TOOL_KEY, tool, GGUFValueType.STRING)
        for key, val in quant_prov.items():
            writer.add_key_value(key, val, GGUFValueType.STRING)
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
    if requantise:
        ident = ("= true (DERIVED: every requantised payload was compared with its base slab "
                 "and equals it byte for byte)" if derived_identity else
                 "ABSENT (DERIVED: the payloads differ from their base — a delta was merged, or "
                 "the target type is not the base segment's)")
        print(f"identity    : {OVERLAY_IDENTITY_KEY} {ident}")
    else:
        print(f"identity    : True (taillis.overlay_identity; every payload is a VERBATIM cut)")
    print(f"kv pairs    : {PARENT_KEY} = {parent_sha}")
    if derived_identity:
        print(f"              {OVERLAY_IDENTITY_KEY} = True")
    print(f"              {OVERLAY_TOOL_KEY} = {tool}")
    for key, val in quant_prov.items():
        print(f"              {key} = {val}")
    if crops:
        print(f"              {EXPERT_CROP_KEY} = {';'.join(crops)}")
    print(f"              {OVERLAY_DIGEST_PREFIX}<k>: {len(digests)} payload digest(s) stamped "
          f"(V4.4: REQUIRED on every overlay)")
    if not requantise:
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
            "identity": not requantise,
            "payload_path": "requantise" if requantise else "identity",
            "overrides": [],
        }
        for k, layer, logical, ids, kind, obj in resolved:
            item = {
                "tensor": f"blk.{layer}.{logical}.ov{k}.weight",
                "layer": layer, "logical": logical, "k": k, "kind": kind,
                "type": (targets.get(k) if requantise else None)
                        or (obj.tensor.tensor_type.name if kind == KIND_SLICE
                            else obj.tensor_type.name),
            }
            if kind == KIND_SLICE:
                item.update({"experts": ids, "group": obj.name,
                             "key": f"{OVERLAY_EXPERTS_PREFIX}{k}"})
                if requantise:
                    item.update({"merge": payloads[k]["delta_note"] or None,
                                 "quantize_flags": payloads[k]["flags"]})
            doc["overrides"].append(item)
        mpath.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"manifest    : {mpath} (NON-AUTHORITATIVE convenience)")
    return 0


def digest_ranges(fd: int, ranges: list[tuple[int, int]]) -> str:
    """sha256 over the CONCATENATION of those byte ranges, in order, chunk-wise.

    The payload is hashed as STORED: a tensor's `data.nbytes` bytes at its
    `data_offset`, and for an expert band the per-expert slabs in ASCENDING id
    order.  GGUF alignment PADDING after a tensor is NOT payload and is not
    covered — a reader can change padding without changing what the tensor says.
    """
    h = hashlib.sha256()
    for off, n in ranges:
        done = 0
        while done < n:
            m = min(CHUNK_BYTES, n - done)
            buf = os.pread(fd, m, off + done)
            if len(buf) != m:
                raise IOError(f"short read of payload at {off + done}: got {len(buf)}, "
                              f"wanted {m}")
            h.update(buf)
            done += m
    return h.hexdigest()


def digest_array(arr) -> str:
    """sha256 over an in-memory payload (the re-quantised path holds the bytes)."""
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


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


def verify(base: Path, overlay: Path, f16: Path | None = None, imatrix_path: Path | None = None,
           type_file: Path | None = None, quant_type: str | None = None, nthread: int = 8,
           quantize_bin: Path | None = None, workdir: Path | None = None,
           delta: Path | None = None, delta_synth: str | None = None,
           lora: Path | None = None, lora_config: Path | None = None) -> int:
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

    # ---- the PAYLOAD DIGESTS: one per tensor, keyed by the same k (V4.4) --
    # REQUIRED on every overlay, and the only content check that can exist on a
    # non-identity one: its payload is SUPPOSED to differ from the base's, so a
    # corrupted byte is invisible to any base comparison and to every structural
    # check.  Recomputed here from the file's own bytes — no F16, no quantiser.
    digs: dict[int, str] = {}
    for key, fld in b.fields.items():
        if not key.startswith(OVERLAY_DIGEST_PREFIX):
            continue
        tail = key[len(OVERLAY_DIGEST_PREFIX):]
        if not tail.isdigit():
            errors.append(f"{key}: the suffix after '{OVERLAY_DIGEST_PREFIX}' must be the k token "
                          f"of a tensor name (V4.4)")
            continue
        if fld.types[0] != GGUFValueType.STRING:
            errors.append(f"{key} is {fld.types[0].name}, not a STRING (V4.4: one scalar sha256 "
                          f"per override tensor)")
            continue
        digs[int(tail)] = str(fld.contents())
    for k in sorted(digs):
        if k not in by_k:
            errors.append(f"{OVERLAY_DIGEST_PREFIX}{k} has no matching .ov{k} tensor — a key with "
                          f"no tensor is a named failure (V4.4), as for the id key")
    for k in sorted(by_k):
        if k not in digs:
            errors.append(f"{by_k[k].name}: no {OVERLAY_DIGEST_PREFIX}{k} key — V4.4 makes the "
                          f"payload digest REQUIRED on every overlay; without it a non-identity "
                          f"overlay cannot be audited at all (its bytes are supposed to differ "
                          f"from its base's)")

    # ---- the imatrix crop, cross-checked against the id keys (V4.4) ------
    # The crop is a BUILD fact (what the matrix was cropped to) and the id key is a
    # FORMAT fact (which experts the tensor holds).  They must agree: a payload
    # quantised with a different crop than the ids it declares would otherwise be
    # undetectable — every other check would pass and the weights would be wrong.
    crop_field = b.fields.get(EXPERT_CROP_KEY)
    if crop_field is not None:
        if crop_field.types[0] != GGUFValueType.STRING:
            errors.append(f"{EXPERT_CROP_KEY} is {crop_field.types[0].name}, not STRING (V4.4)")
        else:
            seen_crop: dict[int, str] = {}
            for item in str(crop_field.contents()).split(";"):
                m = re.match(r"^ov(\d+)=(.*)$", item.strip())
                if not m:
                    errors.append(f"{EXPERT_CROP_KEY} entry {item!r} is not 'ov<k>=<ids>' — the "
                                  f"crop is per override tensor and k is the address (V4.4)")
                    continue
                seen_crop[int(m.group(1))] = m.group(2)
            for k in sorted(seen_crop):
                if k not in by_k:
                    errors.append(f"{EXPERT_CROP_KEY} names ov{k}, which has no .ov{k} tensor")
                elif k in keys and seen_crop[k] != keys[k]:
                    errors.append(f"{EXPERT_CROP_KEY} ov{k} = \"{seen_crop[k]}\" != "
                                  f"{OVERLAY_EXPERTS_PREFIX}{k} = \"{keys[k]}\" — the imatrix "
                                  f"was cropped to DIFFERENT experts than this tensor declares, "
                                  f"so its weights do not belong to these ids")
            for k in sorted(keys):
                if k not in seen_crop:
                    print(f"crop        : WARN {OVERLAY_EXPERTS_PREFIX}{k} has no {EXPERT_CROP_KEY} "
                          f"entry — the crop used is undeclared (V4.4 records it as a build fact)")

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

    # ---- the REPRODUCTION inputs (optional), cross-checked against the file's
    # own declarations before they are used.  A payload that is SUPPOSED to differ
    # from its base cannot be audited against the base at all; the only check that
    # can see a corrupted byte in one is re-running the quantiser with the spec the
    # file declares and comparing.  The declarations are therefore the authority:
    # the imatrix/type-file handed in must hash to what the file says, or the
    # reproduction is not a reproduction (it would be a fresh build that agrees
    # with nothing).
    repro = f16 is not None
    f16_reader = None
    rd_wd = None
    # The declared spec's ftype is read here, once, because BOTH the reproduction
    # inputs and the imatrix-provenance gate below are decided by the file's own
    # types against llama-quantize's own requirement line (llama-quant.cpp:856).
    declared_flags = (str(b.fields[QUANT_FLAGS_KEY].contents())
                      if QUANT_FLAGS_KEY in b.fields else "")
    flag_toks = declared_flags.split()
    ftype = quant_type or (flag_toks[-2] if len(flag_toks) >= 2 else "IQ2_XXS")
    if repro:
        # WHICH INPUTS THE REPRODUCTION NEEDS IS DECIDED BY THE FILE'S OWN TYPES, not
        # by a fixed list: an imatrix is required exactly when a declared type
        # requires one (llama-quant.cpp:856), and a type file exactly when the file
        # declares one.  A q5_k overlay was built with neither, so demanding them
        # would make the only content gate unrunnable on it.
        needs_im = sorted({t.tensor_type.name for t in ov_tensors
                           if type_requires_imatrix(t.tensor_type.name, ftype)})
        no_im = sorted({t.tensor_type.name for t in ov_tensors
                        if not type_requires_imatrix(t.tensor_type.name, ftype)})
        if imatrix_path is not None and not needs_im:
            raise SystemExit(
                f"FATAL: --imatrix was given but this overlay's type(s) {', '.join(no_im) or '?'} "
                f"cannot use one — llama-quantize requires an importance matrix only for "
                f"{', '.join(sorted(IMATRIX_TYPES))} and for Q2_K inside a Q2_K_S file "
                f"(llama-quant.cpp:856). Passing one here would be reproducing with an input "
                f"the build never read")
        missing = [n for n, v in (("--llama-quantize", quantize_bin),) if v is None]
        if needs_im and imatrix_path is None:
            missing.append("--imatrix")
        if TYPE_FILE_KEY in b.fields and type_file is None:
            missing.append("--type-file")
        if missing:
            raise SystemExit(
                f"FATAL: --f16 asks for the REPRODUCTION gate, which needs "
                f"{', '.join(missing)} here — the payload is reproduced by re-running the "
                f"quantiser with the spec the file declares, and nothing is trusted "
                f"(this overlay's type(s) {', '.join(needs_im) or 'need no imatrix'} decide "
                f"whether an imatrix is an input at all)")
        f16_reader = open_gguf(need(f16, "--f16"), "F16 source")
        imatrix_reader = None
        if needs_im:
            imatrix_path = need(imatrix_path, "--imatrix")
            imatrix_reader = open_gguf(imatrix_path, "imatrix")
        if type_file is not None:
            type_file = need(type_file, "--type-file")
        rd_wd = workdir if workdir is not None else Path("/tmp") / f"verify-overlay-{os.getpid()}"
        check_write_scope((rd_wd / "probe").resolve())
        declared = {k: (str(b.fields[k].contents()) if k in b.fields else None)
                    for k in (IMATRIX_KEY, TYPE_FILE_KEY, QUANT_FLAGS_KEY)}
        held = [(IMATRIX_KEY, imatrix_path)] if needs_im else []
        if type_file is not None:
            held.append((TYPE_FILE_KEY, type_file))
        for key, path in held:
            if declared[key] is None:
                errors.append(f"reproduction: {key} is ABSENT from the overlay, so the spec "
                              f"that made these bytes is not declared and cannot be reproduced")
                repro = False
            elif sha256_file(path) != declared[key]:
                errors.append(f"reproduction: {path} sha256 {sha256_file(path)} != the "
                              f"declared {key} {declared[key]} — these are not the inputs that "
                              f"built this overlay")
                repro = False
        if quant_type is None:
            if len(flag_toks) >= 2:
                quant_type = flag_toks[-2]
            if len(flag_toks) >= 1 and flag_toks[-1].isdigit():
                nthread = int(flag_toks[-1])
        im_note = (imatrix_path.name if imatrix_path is not None else
                   f"NONE REQUIRED for {', '.join(no_im)} (llama-quant.cpp:856)")
        print(f"reproduce   : declared spec '{declared[QUANT_FLAGS_KEY]}' -> --imatrix "
              f"{im_note}, --tensor-type-file "
              f"{type_file.name if type_file else 'NONE'}, {quant_type}, nthread {nthread}")

    # ---- the ADAPTER for the reproduction, if this overlay was built from one ------
    # TWO DIRECTIONS, and the MISSING-INPUT one is the one that used to accuse an
    # innocent party: a lora-built overlay verified WITHOUT --lora re-derives with no
    # delta, cannot match, and used to be reported as "corrupted or built from other
    # inputs" — while the file correctly declares its merge and the CALLER omitted the
    # adapter.  So: the declared merge is read first, the omission is named as an
    # omission, and "corrupted" is kept for the case where the declared inputs WERE
    # supplied and still do not reproduce.
    one_delta_source(lora, delta, delta_synth)
    declared_delta = (str(b.fields[MERGE_DELTA_KEY].contents())
                      if MERGE_DELTA_KEY in b.fields else "")
    lora_ad = None
    if lora is not None and not repro:
        raise SystemExit(
            "FATAL: [lora:path] --lora is only meaningful WITH --f16: it re-derives the "
            "payload from the adapter, and a reproduction needs the whole declared spec "
            "(--imatrix/--type-file/--llama-quantize) to be a reproduction at all")
    if repro and declared_delta and lora is None and delta is None and delta_synth is None:
        need_arg = ("--lora ADAPTER --lora-config CONFIG" if declared_delta.startswith("lora:")
                    else "--delta FILE" if declared_delta.startswith("sha256:")
                    else "--delta-synth SEED:RELRMS")
        errors.append(
            f"reproduction: this overlay DECLARES a merge — {MERGE_DELTA_KEY} = "
            f"{declared_delta} — but no delta source was supplied, so the reproduction below "
            f"would merge NOTHING and could not match. Pass {need_arg}. This is a MISSING "
            f"INPUT, not evidence that the file is corrupt")
        repro = False
    if repro and lora is not None:
        lora_ad = load_lora(lora, lora_config)
        # The adapter handed in must be the one the file SAYS built it — otherwise every
        # reproduction below would be a fresh build agreeing with nothing (the same rule
        # the imatrix/type-file are held to).  Both halves of the identity are checked:
        # the adapter's bytes AND its config, because a different alpha is a different
        # build even when the rank (and therefore the tensors' shape) agrees.
        if not declared_delta.startswith("lora:"):
            errors.append(f"reproduction: --lora was given but the overlay declares "
                          f"{MERGE_DELTA_KEY}={declared_delta or 'ABSENT'} — this payload was "
                          f"not built from an adapter")
            repro = False
        elif (f"sha256={lora_ad.sha}," not in declared_delta
              or f"config={lora_ad.cfg_sha}," not in declared_delta):
            errors.append(f"reproduction: the adapter in hand (sha256 {lora_ad.sha}, config "
                          f"{lora_ad.cfg_sha}) is not the one declared — {MERGE_DELTA_KEY} = "
                          f"{declared_delta}. A different adapter or a different "
                          f"alpha/rank/config is a different build")
            repro = False
        else:
            print(f"reproduce   : adapter {lora_ad.path.name} rank={lora_ad.rank} "
                  f"alpha={lora_ad.alpha} scale={lora_ad.scale:.10g} — sha256 AND config "
                  f"match the declared merge_delta")

    # ---- per tensor: kind from the base's rank, then THE ROW ARITHMETIC ---
    n_repro = 0
    n_repro_units = 0
    fd_a = os.open(real, os.O_RDONLY)
    fd_b = os.open(ov, os.O_RDONLY)
    n_tensors_ok = 0
    n_cmp_units = 0          # experts for an expert-slice, one unit for a whole tensor
    n_ok_units = 0
    n_bytes_cmp = 0
    n_slice = 0
    n_whole = 0
    n_dig_ok = 0
    n_experts = 0
    n_diff_type = 0
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
                # ---- WHICH REGIME IS THIS: same-type or different-type ---------
                # V4.3 clause 5: the tensor's own ggml type IS the type, and there are
                # exactly two regimes.  SAME-TYPE (aubier's S1-S5) is a source-pointer
                # swap, so a type mismatch there is a NAMED FAILURE.  DIFFERENT-TYPE
                # (S7) is the point of the file: the store must give these experts
                # their own SEGMENT, because a slice carries one type.  The regime is
                # READ off the file, never declared by a second key.
                same_type = t.tensor_type == seg.tensor.tensor_type
                if same_type:
                    stride = seg.stride
                    if t.data.nbytes != stride * len(ids):
                        errors.append(f"{name}: nbytes {t.data.nbytes} != {stride} * {len(ids)} "
                                      f"(base stride * declared experts)")
                        continue
                else:
                    try:
                        stride = expert_stride(t.tensor_type.name, seg.ne01)
                    except SystemExit as e:
                        errors.append(f"{name}: {str(e).removeprefix('FATAL: ')}")
                        continue
                    n_diff_type += 1
                    print(f"  type      : DIFFERENT-TYPE regime (V4.3 clause 5) — override "
                          f"{t.tensor_type.name} vs base segment {seg.name} "
                          f"{seg.tensor.tensor_type.name}; a slice carries ONE type, so the "
                          f"engine must give these experts their own segment")
                    if t.data.nbytes != stride * len(ids):
                        errors.append(f"{name}: nbytes {t.data.nbytes} != {stride} * {len(ids)} "
                                      f"({t.tensor_type.name} stride * declared experts)")
                        continue
                # ---- THE PAYLOAD DIGEST (V4.4): the content check ----------
                got_dig = digest_ranges(fd_b, [(t.data_offset, t.data.nbytes)])
                want_dig = digs.get(k)
                if want_dig is not None:
                    if got_dig != want_dig:
                        errors.append(
                            f"{name}: PAYLOAD DIGEST MISMATCH — the file declares "
                            f"{OVERLAY_DIGEST_PREFIX}{k} = {want_dig} but the payload hashes to "
                            f"{got_dig} ({t.data.nbytes} byte(s) of {t.tensor_type.name}). The "
                            f"payload is not the one this overlay was built with: it was "
                            f"corrupted, truncated or replaced after the build. (This is the only "
                            f"check that can see it — a non-identity payload is SUPPOSED to "
                            f"differ from its base.)")
                    else:
                        n_dig_ok += 1
                print(f"  digest    : {OVERLAY_DIGEST_PREFIX}{k} "
                      f"{'OK' if got_dig == want_dig else 'MISMATCH'} {got_dig}")
                t_diffs = 0
                n_diff_units = 0
                # A difference from the base is an ERROR only on an overlay that
                # DECLARES itself identity.  On any other overlay the bytes are
                # supposed to differ (that is what a variant IS), so a difference is
                # counted and named, and the gate is G8.3's direction (below) plus
                # --f16's reproduction, not byte equality.
                if not same_type:
                    # No byte comparison EXISTS across two types: the payload holds a
                    # different number of bytes per expert.  The digest above is the
                    # content check, and the reproduction below is the stronger one.
                    all_equal = False
                    print(f"  base cmp  : NOT COMPARABLE — {t.tensor_type.name} payload "
                          f"({stride} B/expert) vs {seg.tensor.tensor_type.name} base slab "
                          f"({seg.stride} B/expert); a cross-type byte comparison is not "
                          f"defined. The payload digest is the content check here, and "
                          f"--f16 the reproduction")
                for j, e in enumerate(ids if same_type else []):
                    d, first = compare(fd_a, seg.slab_offset(e), fd_b,
                                       t.data_offset + j * stride, stride)
                    n_bytes_cmp += stride
                    n_cmp_units += 1
                    if d:
                        t_diffs += d
                        n_diff_units += 1
                        msg = (f"{name}: expert {e} payload DIFFERS from base slab at "
                               f"{seg.name} e={e}: {d} byte(s), first at overlay byte "
                               f"{j * stride + first} (expert-local byte {first})")
                        if declared_identity:
                            errors.append(msg + " — and the overlay declares "
                                          f"{OVERLAY_IDENTITY_KEY}=true, which is a false claim")
                        else:
                            print(f"  base diff : {msg}")
                    else:
                        n_ok_units += 1
                # A tensor that was NOT compared is not "byte-identical": on the
                # different-type regime nothing was compared at all, and a summary
                # that counts it as equal would say the opposite of the truth.
                if not same_type:
                    all_equal = False
                elif t_diffs == 0:
                    n_tensors_ok += 1
                else:
                    all_equal = False
                print(f"expert band : {name}  {OVERLAY_EXPERTS_PREFIX}{k}=\"{entry}\"  "
                      f"type={t.tensor_type.name}  base seg={seg.name} [{seg.lo},{seg.hi})  "
                      f"stride={stride}  "
                      + (f"diffs={t_diffs} bytes in {n_diff_units}/{len(ids)} expert(s)"
                         if same_type else f"diffs=n/a (different-type, {len(ids)} expert(s))"))

                # ---- REPRODUCTION: re-derive the payload and compare it -----
                if repro:
                    rp = requantised_payload(
                        base_name=real.name, layer=layer, logical=logical, ids=ids, seg=seg,
                        f16_reader=f16_reader, imatrix_reader=imatrix_reader,
                        imatrix=imatrix_path, type_file=type_file,
                        quant_type=quant_type, target_type=t.tensor_type.name,
                        nthread=nthread,
                        quantize_bin=quantize_bin, workdir=rd_wd, delta_path=delta,
                        delta_synth=delta_synth, ov_name=name, lora=lora_ad,
                        lora_note=(f"lora:sha256={lora_ad.sha},rank={lora_ad.rank},"
                                   f"scale={lora_ad.scale:.10g}"
                                   if lora_ad is not None else ""))
                    exp = np.ascontiguousarray(rp["payload"]).reshape(-1)
                    act = np.frombuffer(os.pread(fd_b, exp.size, t.data_offset), dtype=np.uint8)
                    if act.size != exp.size:
                        errors.append(f"{name}: reproduction read {act.size} bytes, expected "
                                      f"{exp.size}")
                        continue
                    n_repro_units += len(ids)
                    t_bad = 0
                    for j, e in enumerate(ids):
                        lo = j * stride
                        x = exp[lo:lo + stride]
                        y = act[lo:lo + stride]
                        bad = np.flatnonzero(x != y)
                        if bad.size:
                            t_bad += int(bad.size)
                            errors.append(
                                f"{name}: expert {e} does NOT REPRODUCE from the declared spec: "
                                f"{bad.size} byte(s) differ, first at overlay byte "
                                f"{lo + int(bad[0])} (expert-local byte {int(bad[0])}) — the "
                                f"payload is not what the declared imatrix/type-file/flags "
                                f"produce, so it was corrupted or built from other inputs")
                    if t_bad == 0:
                        n_repro += 1
                    print(f"  reproduce : {name}  "
                          f"{'BYTE-IDENTICAL to the re-quantised F16 slice' if t_bad == 0 else str(t_bad) + ' byte(s) DIFFER'}"
                          f"  (merge: {rp['delta_note'] or 'none'})")
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
                got_dig = digest_ranges(fd_b, [(t.data_offset, t.data.nbytes)])
                want_dig = digs.get(k)
                if want_dig is not None:
                    if got_dig != want_dig:
                        errors.append(
                            f"{name}: PAYLOAD DIGEST MISMATCH — declared "
                            f"{OVERLAY_DIGEST_PREFIX}{k} = {want_dig} but the payload hashes to "
                            f"{got_dig} ({t.data.nbytes} byte(s) of {t.tensor_type.name})")
                    else:
                        n_dig_ok += 1
                print(f"  digest    : {OVERLAY_DIGEST_PREFIX}{k} "
                      f"{'OK' if got_dig == want_dig else 'MISMATCH'} {got_dig}")
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
        # `complete` means EVERY override was compared AND found equal.  A
        # different-type tensor is never compared (no cross-type byte comparison
        # exists), so a file containing one can never be "complete" — it cannot be an
        # identity overlay, which is exactly right.
        complete = n_cmp_units > 0 and n_ok_units == n_cmp_units and n_diff_type == 0
        if complete and not declared_identity:
            errors.append(f"G8.3: every override payload EQUALS its base but "
                          f"{OVERLAY_IDENTITY_KEY} is absent/false — an identity overlay is "
                          f"legal only when it declares itself (V4.5 G8.3)")
        if not all_equal and declared_identity:
            errors.append(f"G8.3: {OVERLAY_IDENTITY_KEY}=true but a payload DIFFERS from its base "
                          f"(see the diffs above)")
        # V4.4 (settled 2026-10-19): the quantisation provenance is MANDATORY on a
        # non-identity overlay (or a variant's measured quality is confounded with a
        # requantisation) and PERMITTED — not demanded, not forbidden — on an
        # identity one, because "identity" is a statement about BYTES and "was
        # quantised" is a statement about PROVENANCE.  This lane's own
        # MS4-overlay-L20-e8-requant.gguf is both: re-quantising the base's F16
        # slice at the base's own type returns the base's bytes exactly, and a
        # quantiser really ran.  So only the non-identity case is gated.
        #
        # THE IMATRIX HALF IS GATED BY THE TYPE, NOT BY THE FLAG: llama-quantize needs
        # a matrix only for IMATRIX_TYPES (and Q2_K in a Q2_K_S file), so a q5_k
        # overlay MUST NOT carry taillis.provenance.imatrix_sha256 or a crop — naming
        # a matrix nothing read is false provenance, the same lie as a missing one
        # pointed the other way.  Both directions are errors here.
        file_types = {t.tensor_type.name for t in ov_tensors}
        needs_im_any = sorted(t for t in file_types if type_requires_imatrix(t, ftype))
        if not declared_identity:
            if QUANT_FLAGS_KEY not in b.fields:
                errors.append(f"{QUANT_FLAGS_KEY} absent — V4.4 makes the quantisation provenance "
                              f"MANDATORY on an overlay that is not an identity cut (a variant's "
                              f"quality must not be confounded with a requantisation)")
            if needs_im_any and IMATRIX_KEY not in b.fields:
                errors.append(f"{IMATRIX_KEY} absent — this overlay's type(s) "
                              f"{', '.join(needs_im_any)} REQUIRE an importance matrix "
                              f"(llama-quant.cpp:856), so the matrix that weighted these bytes "
                              f"is a fact the file must carry")
        if IMATRIX_KEY in b.fields and not needs_im_any:
            errors.append(f"{IMATRIX_KEY} present but this overlay's type(s) "
                          f"{', '.join(sorted(file_types))} cannot use an importance matrix "
                          f"(llama-quant.cpp:856) — a FALSE PROVENANCE claim: nothing was "
                          f"imatrix-weighted, so the stamp names a matrix no quantiser read")
        if EXPERT_CROP_KEY in b.fields and IMATRIX_KEY not in b.fields:
            errors.append(f"{EXPERT_CROP_KEY} present without {IMATRIX_KEY} — the crop is a "
                          f"statement about the matrix that was cropped, and there is no such "
                          f"statement here (false provenance, V4.4)")
    finally:
        os.close(fd_a)
        os.close(fd_b)

    print(f"tensors     : {len(ov_tensors)} .ov<k> ({n_slice} expert-slice, {n_whole} "
          f"whole-tensor), {n_experts} expert slab(s), {human(n_bytes_cmp)} compared byte-for-byte")
    print(f"payload     : {n_tensors_ok}/{len(ov_tensors)} override tensors byte-identical "
          f"({n_ok_units}/{n_cmp_units} compared unit(s): expert slabs of an expert-slice, or "
          f"one whole tensor)"
          + (f" — {n_diff_type} tensor(s) NOT COMPARED: different-type regime, no cross-type "
             f"byte comparison exists (V4.3 clause 5)" if n_diff_type else ""))
    if seg_cache:
        print("segments    : " + ", ".join(
            f"blk.{l}.{lg}=" + " ".join(f"[{s.lo},{s.hi})" for s in segs)
            for (l, lg), segs in sorted(seg_cache.items())))
    print(f"identity    : {OVERLAY_IDENTITY_KEY} "
          f"{'declared' if declared_identity else 'absent/false'}, payloads "
          f"{'all equal' if complete else ('DIFFER' if not all_equal else 'not all compared')}")
    print(f"payload dig : {n_dig_ok}/{len(by_k)} tensor(s) hash to their declared "
          f"{OVERLAY_DIGEST_PREFIX}<k> (V4.4; REQUIRED on every overlay — this is the only "
          f"content check on a non-identity payload)")
    print(f"quant prov  : " + (", ".join(
        f"{k}={'present' if k in b.fields else 'ABSENT'}"
        for k in QUANT_PROV_KEYS + (TYPE_FILE_KEY, QUANT_TOOL_KEY, MERGE_DELTA_KEY)))
        + (" (PERMITTED on a declared-identity overlay — V4.4 requires the pair only on a "
           "non-identity one, and this file carries it because a quantiser really ran; the "
           "payload happens to reproduce the base byte for byte)"
           if (declared_identity and any(k in b.fields for k in QUANT_PROV_KEYS)) else
           (" (OPTIONAL on an identity overlay, and absent here: nothing was quantised, so no "
            "true value exists to declare — V4.4 settled that this is permitted, not required)"
            if declared_identity else "")))
    if f16 is not None:
        print(f"reproduced  : {n_repro}/{n_slice} expert-slice tensor(s) BYTE-IDENTICAL to a fresh "
              f"re-quantisation from {Path(f16).name} with the declared spec "
              f"({n_repro_units} expert slab(s) compared)")
    elif not declared_identity:
        print(f"reproduced  : NOT CHECKED — a non-identity payload is SUPPOSED to differ from the "
              f"base, so nothing in this run can see a corrupted byte in it. Pass --f16 with the "
              f"declared imatrix/type-file/quantiser to reproduce the payload and compare it.")
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
                   help="write IDENTITY payloads: every override is a verbatim cut of the "
                        "base's own slab (unit S1). Exactly one of --identity / --requantise")
    p.add_argument("--requantise", action="store_true",
                   help="write QUANTISED payloads: cut the rows out of --f16, merge the delta "
                        "at F16, re-quantise with the BASE's own --imatrix and --type-file "
                        "(unit S1b, aubier expert-swap.md §5.3)")
    p.add_argument("--f16", type=Path, default=None,
                   help="the >=16-bit source the base was quantised from (REQUIRED with "
                        "--requantise, and by --verify for the reproduction gate)")
    p.add_argument("--imatrix", "--imatrix-path", dest="imatrix_path", type=Path, default=None,
                   help="the base's own imatrix GGUF. REQUIRED only when the TARGET type requires "
                        "one — llama-quantize's own line (llama-quant.cpp:856): "
                        "IQ3_XXS/IQ2_XXS/IQ2_XS/IQ2_S/IQ1_M/IQ1_S, and Q2_K inside a Q2_K_S file. "
                        "REFUSED for a type that cannot use it (a stamp naming a matrix nothing "
                        "read is false provenance)")
    p.add_argument("--type-file", type=Path, default=None,
                   help="the base's own --tensor-type-file. OPTIONAL: the positional --quant-type "
                        "decides the type when it is absent, and when it is present the produced "
                        "type is checked against --target-type")
    p.add_argument("--target-type", type=str, default=None,
                   help="the ggml type the override must be PRODUCED at (e.g. Q5_K). Default: the "
                        "base segment's own type — the SAME-TYPE regime (aubier S1-S5). A "
                        "different value is the DIFFERENT-TYPE regime (S7): the tensor's own "
                        "type IS the declaration (layout.md V4.3 clause 5), and the type must be "
                        "expressible for every overridden tensor or the build STOPS")
    p.add_argument("--quant-type", type=str, default=None,
                   help="the positional ftype for the quantiser run. Default: the TARGET type "
                        "(--target-type, or the base segment's own), which is what makes a "
                        "type-only build need no second declaration. The type-file still "
                        "overrides it per tensor. It must be a QUANTISED type or llama-quantize "
                        "ignores every override — MEASURED in an earlier unit. On --verify it is "
                        "read from the file's own declared quantize_flags instead")
    p.add_argument("--nthread", type=int, default=8,
                   help="threads for the quantiser run (default 8)")
    p.add_argument("--llama-quantize", type=Path, default=None,
                   help="the llama-quantize binary to run (REQUIRED with --requantise). Its "
                        "LD_LIBRARY_PATH is taken from its own directory")
    p.add_argument("--input-sha256", type=str, default=None, metavar="HEX",
                   help="the >=16-bit source's sha256 (stamped as provenance.input_sha256). "
                        "SUPPLYING it skips re-hashing a 222 GiB source (~5 min); the tool then "
                        "RECORDS the caller's measurement instead of making its own")
    p.add_argument("--workdir", type=Path, default=None,
                   help="where the throwaway slice/quantised GGUFs and their logs go "
                        "(default /tmp/repack-overlay-<pid>)")
    p.add_argument("--delta", type=Path, default=None,
                   help="an F16 delta GGUF to merge AT F16 before re-quantising; its tensors "
                        "are named exactly as the override tensors, so a converted adapter's "
                        "slice needs no new code path")
    p.add_argument("--delta-synth", type=str, default=None, metavar="SEED:RELRMS",
                   help="a DETERMINISTIC SYNTHETIC F16 delta, N(0,(RELRMS*rms(slice))^2), for "
                        "artifacts built before any adapter exists. Stamped as "
                        "merge_delta=synthetic:... so it can never be read as a trained one")
    p.add_argument("--lora", type=Path, default=None, metavar="ADAPTER.safetensors",
                   help="a PEFT safetensors ADAPTER to merge (unit LORA): one 2-D "
                        "lora_A/lora_B pair per (layer, expert, w1|w2|w3), merged as "
                        "alpha/rank * B@A per expert into the F16 slice before the EXISTING "
                        "re-quantisation. Requires --requantise; exclusive with --delta/"
                        "--delta-synth")
    p.add_argument("--lora-config", type=Path, default=None, metavar="adapter_config.json",
                   help="the adapter's PEFT config, where alpha comes from (default: "
                        "adapter_config.json beside --lora; REQUIRED to exist — alpha IS the "
                        "merge scale, so it is never guessed)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="optional NON-AUTHORITATIVE side manifest.json (a shelf label)")
    p.add_argument("--imatrix-sha256", type=str, default=None, metavar="HEX",
                   help="stamp taillis.provenance.imatrix_sha256 BY HAND. Only meaningful where "
                        "nothing can derive it; on --requantise it is refused (derived from the "
                        "file actually used). V4.4 requires the pair on a non-identity overlay and "
                        "PERMITS it on an identity one")
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
        sys.exit(verify(args.base, args.overlay, f16=args.f16,
                        imatrix_path=args.imatrix_path, type_file=args.type_file,
                        quant_type=args.quant_type,
                        nthread=args.nthread, quantize_bin=args.llama_quantize,
                        workdir=args.workdir, delta=args.delta,
                        delta_synth=args.delta_synth,
                        lora=args.lora, lora_config=args.lora_config))
    if args.base is None or args.out is None:
        raise SystemExit("FATAL: give --base and --out (or --verify)")
    sys.exit(build(args.base, args.out, args.override, args.identity, args.delta, args.manifest,
                   args.imatrix_sha256, args.quant_flags, f16=args.f16,
                   requantise=args.requantise, imatrix_path=args.imatrix_path,
                   type_file=args.type_file, quant_type=args.quant_type,
                   nthread=args.nthread, quantize_bin=args.llama_quantize,
                   workdir=args.workdir, delta_synth=args.delta_synth,
                   input_sha=args.input_sha256, lora=args.lora,
                   lora_config=args.lora_config, target_type=args.target_type))


if __name__ == "__main__":
    main()
