#!/usr/bin/env python3
"""Write the GROUPED-expert GGUF format (taillis/layout.md §2) from a flat one.

One logical expert tensor (`blk.N.ffn_gate_up_exps.weight`,
`blk.N.ffn_down_exps.weight`; expert axis ne[2] = 128) becomes G member tensors
`<logical>.g<k>.weight`, each covering a contiguous range of the expert axis in
order, plus ONE string-array metadata key:

    taillis.expert_groups = [ "<logical>.g0", "<logical>.g1", ... ]

Entries carry NO `.weight` suffix — that is what aubier's reader
(`engine/gguf_model.h`, TAILLIS_EXPERT_GROUPS_KEY) parses; it derives the ranges
as the cumulative ne[2] of the members in array order.  Every other tensor is
copied verbatim and is NOT named in the key.  All other metadata is copied
verbatim; the group key is appended after them.

The split is along ne[2] only.  The quantization block dimension is ne[0]
(layout.md §1, MEASURED), so a member's payload is the contiguous byte slab
    bytes[lo*stride : hi*stride],  stride = parent_nbytes / ne[2]
and grouping never touches a quantization block.  For T2.1 every member carries
the SAME type as the flat parent — the FORMAT is under test here, not the
allocation policy (that is T2.2, out of scope).  G=1 is the trivial partition
and must be byte-inert; that is checked by --verify.

Data is streamed from the source in chunks (never the whole file in memory) via
gguf-py's LazyChunkedTensor.

Usage
    tools/group_experts.py --src FLAT.gguf --out GROUPED.gguf [--groups G] [--spec "a,b,..."]
                           [--format {v0,v1}] [--provenance K=V ...]
    tools/group_experts.py --verify --src FLAT.gguf --out GROUPED.gguf [--format {v0,v1}]
    tools/group_experts.py --show --out ANY.gguf

`--groups G` (default 1) splits ne[2] as evenly as possible (128 = 43+43+42 for
G=3); `--spec` names the member sizes explicitly (must sum to ne[2]).
`--provenance K=V` (repeatable) adds the build fact taillis.provenance.K=V.

`--format` (layout.md §8): v0 (DEFAULT — today's semantics) declares the grouping
through the taillis.expert_groups string array and stamps format_version = 0;
v1 (OPT-IN, aubier cannot read it yet — layout.md §8.4) DELETES the array and
stamps format_version = 1: membership and order are derived from the member
tensor names (<logical>.g<k>.weight, sorted by ascending k).  A v1 file carrying
the group key is malformed, and so is a v0 file carrying version 1.
NOTE the stamp history: the pre-R2 writer stamped version 1 into v0 files under
the provisional §7.1 meaning; no artifact was ever built that way (MEASURED
2026-10-15) and v0 now stamps 0 — layout.md §8.1.

At write time the R1 keys (taillis/layout.md §7) are stamped too:
    taillis.format_version = 0 (v0) or 1 (v1), u32
    taillis.parent_sha256  = sha256 over the WHOLE FILE BYTES of --src
                             (computed once, chunk-wise; if --src is modified
                             after this, that is a caller error — no re-hash)
    taillis.provenance.grouping_tool = group_experts.py@<taillis HEAD>
                                       ("unknown" if git is unavailable; "+dirty"
                                       suffix when the tool file is untracked or
                                       modified at that HEAD — layout.md §7.1)
Existing taillis.* keys in the source that we do not stamp are copied verbatim,
so an already-stamped parent's provenance (source_checkpoint, converter, ...)
survives grouping.

An already-grouped SOURCE is refused two ways: it carries the group key (a
key-carrying source is grouped, member-shaped names or not — otherwise a v1
output could inherit a stale key and be §8.1-malformed), or its tensor table
holds member-shaped names (<anything>.g<k>.weight; layout.md §8.3).  Both hold
for v0 and v1 sources alike, and the name scan also refuses member-shaped
names outside the expert surface (a file carrying those is not a flat parent).

`--verify` re-parses both files and checks, with real byte comparisons:
  gate 1  every tensor's payload byte-identical to the flat parent's
          (members compared against the corresponding parent byte slab);
  gate 2  the declaration is well-formed — v0: the key parses, members exist,
          logicals are absent flat, indices are 0..G-1 in array order, ranges
          ordered+contiguous covering [0, ne[2]) exactly, member types equal
          the parent type; v1: the group key is ABSENT, version == 1, and the
          expected membership is DERIVED from the output's member names
          (duplicate k rejected, ascending k = expert order, every member
          claimed by a flat expert tensor of the source);
  gate 5  per logical tensor, sum(member nbytes) == parent nbytes;
  gate R1 format_version present (0 for v0, 1 for v1), parent_sha256 present
          and equal to the src file's sha256, grouping_tool present.
The --format of --verify must MATCH the output: a v0 output verified as v1, or
a v1 output verified as v0, fails.  Prints COUNTS, SIZES and PATHS only.
Exit code is the gate.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_SRC = Path("~/futaie/aubier/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf")
GROUP_KEY = "taillis.expert_groups"
FORMAT_KEY = "taillis.format_version"
PARENT_KEY = "taillis.parent_sha256"
PROVENANCE_PREFIX = "taillis.provenance."
GROUPING_TOOL_KEY = PROVENANCE_PREFIX + "grouping_tool"
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:gate_up|down)_exps\.weight$")
MEMBER_RE = re.compile(r"^(.+)\.g(\d+)$")   # key entries carry NO ".weight" suffix
MEMBER_FILE_RE = re.compile(r"^(.+)\.g(\d+)\.weight$")   # member names in the FILE do
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


def partition(ne2: int, groups: int, spec: str | None) -> list[int]:
    """Member sizes along ne[2]: ordered, contiguous, summing to ne2 exactly."""
    if spec is not None:
        sizes = [int(s) for s in spec.split(",")]
        if not sizes or any(s <= 0 for s in sizes):
            raise SystemExit(f"FATAL: bad --spec '{spec}' (need positive sizes)")
        if sum(sizes) != ne2:
            raise SystemExit(f"FATAL: --spec sums to {sum(sizes)}, ne[2] is {ne2}")
        return sizes
    if groups < 1:
        raise SystemExit(f"FATAL: --groups must be >= 1, got {groups}")
    base, rem = divmod(ne2, groups)
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    if sizes[-1] == 0:
        raise SystemExit(f"FATAL: --groups {groups} exceeds ne[2]={ne2} (empty member)")
    return sizes


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


def taillis_head() -> str:
    """group_experts.py@<taillis HEAD>[+dirty]; '@unknown' if git is unavailable.

    The +dirty suffix (layout.md §7.1) is load-bearing: this tool file is the
    writer a grouped artifact's provenance points at, so the recorded commit
    must actually CONTAIN it.  Untracked or locally modified => the commit does
    not, and @<HEAD> alone would name a tree lacking its writer (the K3 gap).
    """
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        commit = head.stdout.strip() if head.returncode == 0 else ""
        tool = str(Path(__file__).resolve().relative_to(REPO))
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", tool],
                            capture_output=True, text=True, timeout=10)
        dirty = st.returncode != 0 or bool(st.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        commit, dirty = "", True
    if not commit:
        return "group_experts.py@unknown"
    return f"group_experts.py@{commit}{' +dirty' if dirty else ''}"


def parse_provenance(pairs: list[str]) -> list[tuple[str, str]]:
    """'K=V' list -> [(full key, value)]; full key = taillis.provenance.<K>."""
    out: list[tuple[str, str]] = []
    for arg in pairs:
        k, sep, v = arg.partition("=")
        if not sep or not k:
            raise SystemExit(f"FATAL: --provenance expects K=V, got '{arg}'")
        out.append((PROVENANCE_PREFIX + k, v))
    return out


def build(src: Path, out: Path, groups: int, spec: str | None,
          provenance: list[tuple[str, str]], fmt: str) -> None:
    real = src.resolve(strict=True)
    out = out.resolve()
    check_write_scope(out)
    if out.exists():
        raise SystemExit(f"FATAL: {out} already exists; remove it or pick another --out")

    # R1 stamp: computed ONCE over the src's whole file bytes; a caller that
    # modifies --src after this point has invalidated its own parent_sha256.
    parent_sha = sha256_file(real)
    grouping_tool = taillis_head()
    # GROUP_KEY is in the skip set unconditionally: build() emits its own (v0)
    # or none (v1), so a source's key must never ride along verbatim — a v1
    # output with the key would be §8.1-malformed.  (The refusal below makes
    # this unreachable today; it is here so the invariant does not depend on
    # the refusal surviving refactors.)
    stamp_keys = {FORMAT_KEY, PARENT_KEY, GROUPING_TOOL_KEY, GROUP_KEY} | {k for k, _ in provenance}

    try:
        reader = GGUFReader(real)
    except Exception as e:
        # e.g. duplicated tensor names in the table — whatever the cause, a
        # source gguf-py cannot parse is not a flat parent (refused loudly)
        raise SystemExit(f"FATAL: {real} does not parse as a flat parent "
                         f"({type(e).__name__}: {e})")
    # A flat parent declares NO grouping: neither the key (a key-carrying
    # source is grouped, member-shaped names or not — refusing it here is what
    # keeps a v1 output from inheriting a stale key and turning §8.1-malformed)
    # nor member-shaped tensor names (a name-carrying source would have its
    # members copied verbatim under an empty/absent declaration; a member-shaped
    # name outside the expert surface is equally "not a flat parent").
    if GROUP_KEY in reader.fields:
        raise SystemExit(
            f"FATAL: {real} carries {GROUP_KEY}; a flat parent does not declare "
            f"a grouping — derive from the flat parent instead (layout.md §8.3)")
    member_srcs = [t.name for t in reader.tensors if MEMBER_FILE_RE.match(t.name)]
    if member_srcs:
        raise SystemExit(
            f"FATAL: {real} already carries {len(member_srcs)} member-shaped tensors "
            f"(e.g. {member_srcs[0]}); derive from the flat parent instead "
            f"(taillis.parent_sha256 names it, layout.md §7.2/§8)")
    arch = reader.fields["general.architecture"].contents()
    expert_count = int(reader.fields[f"{arch}.expert_count"].contents())

    n_expert = 0
    n_plain = 0
    expert_bytes = 0
    entries: list[str] = []
    writer = GGUFWriter(str(out), arch="", use_temp_file=False)
    writer.data_alignment = reader.alignment

    n_kv = 0
    for key, field in reader.fields.items():
        if key in RESERVED_KV or key in stamp_keys:
            continue  # keys we stamp are re-added below; never duplicated
        main = field.types[0]
        if main == GGUFValueType.ARRAY:
            writer.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=field.types[-1])
        else:
            writer.add_key_value(key, field.contents(), main)
        n_kv += 1

    fd = os.open(real, os.O_RDONLY)
    try:
        for t in reader.tensors:
            name = t.name
            m = EXPERT_RE.match(name)
            if not m:
                # every non-expert tensor is copied VERBATIM and not named in the key.
                # dtype must be t.data's own: for F32/F16 tensors data.shape is in
                # ELEMENTS, so declaring uint8 there undercounts nbytes by 4x/2x.
                writer.add_tensor(name, LazyChunkedTensor(
                    chunk_loaders(fd, t.data_offset, t.data.nbytes),
                    tuple(int(x) for x in t.data.shape), t.data.dtype,
                ), raw_dtype=t.tensor_type)
                n_plain += 1
                continue
            # MEASURED (gguf-py 0.19): t.shape is in ggml ne order (ne[0], ne[1], ne[2])
            # while t.data.shape is numpy order (ne2, ne1, row_bytes).
            ne = tuple(int(x) for x in t.shape)
            byte_shape = tuple(int(x) for x in t.data.shape)
            ne2 = ne[2]
            if ne2 != expert_count:
                raise SystemExit(f"FATAL: {name} has ne[2]={ne2} != {arch}.expert_count={expert_count}")
            if len(ne) != 3 or len(byte_shape) != 3 or byte_shape[0] != ne2:
                raise SystemExit(f"FATAL: {name} shape {ne} inconsistent with byte shape {byte_shape}")
            if t.data.nbytes % ne2 != 0:
                raise SystemExit(f"FATAL: {name} nbytes {t.data.nbytes} not divisible by ne[2]={ne2}")
            # stride: bytes of ONE expert slice (quant blocks live along ne[0], so the
            # ne[2] axis is the slowest one and each slice is one contiguous slab).
            stride = t.data.nbytes // ne2
            sizes = partition(ne2, groups, spec)
            logical = name[:-len(".weight")]
            lo = 0
            for k, s in enumerate(sizes):
                hi = lo + s
                mbytes = stride * s
                writer.add_tensor(f"{logical}.g{k}.weight", LazyChunkedTensor(
                    chunk_loaders(fd, t.data_offset + lo * stride, mbytes),
                    (hi - lo,) + byte_shape[1:], t.data.dtype,
                ), raw_dtype=t.tensor_type)
                entries.append(f"{logical}.g{k}")
                lo = hi
            assert lo == ne2
            n_expert += 1
            expert_bytes += t.data.nbytes

        # the chunk loaders are lazy — they run inside write_tensors_to_file(),
        # so the source fd must still be open here (closed in the finally below)
        version = 1 if fmt == "v1" else 0
        if fmt == "v0":
            writer.add_key_value(GROUP_KEY, entries, GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)
        writer.add_key_value(FORMAT_KEY, version, GGUFValueType.UINT32)
        writer.add_key_value(PARENT_KEY, parent_sha, GGUFValueType.STRING)
        writer.add_key_value(GROUPING_TOOL_KEY, grouping_tool, GGUFValueType.STRING)
        for k, v in provenance:
            writer.add_key_value(k, v, GGUFValueType.STRING)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        os.close(fd)

    writer.close()

    total = out.stat().st_size
    print(f"source      : {real}")
    print(f"out         : {out}  {total} bytes ({human(total)})")
    print(f"format      : {fmt} (layout.md §8)")
    sizes = partition(expert_count, groups, spec)
    print(f"partition   : {groups if spec is None else '--spec ' + spec} -> {sizes} (x {n_expert} expert tensors)")
    print(f"tensors     : {n_expert} expert logical -> {len(entries)} members; {n_plain} verbatim")
    print(f"expert data : {expert_bytes} bytes ({human(expert_bytes)}) re-partitioned, none copied as flat")
    if fmt == "v0":
        print(f"kv pairs    : {n_kv} (verbatim) + 1 ({GROUP_KEY}, {len(entries)} entries) + R1 stamp:")
    else:
        print(f"kv pairs    : {n_kv} (verbatim) + 0 (v1: no {GROUP_KEY} — layout.md §8.1) + R1 stamp:")
    print(f"  {FORMAT_KEY} = {version} (u32)")
    print(f"  {PARENT_KEY} = {parent_sha}")
    print(f"  {GROUPING_TOOL_KEY} = {grouping_tool}")
    for k, v in provenance:
        shown = v if len(v) <= 72 else v[:69] + "..."
        print(f"  {k} = {shown}")


def verify(src: Path, out: Path, provenance: list[tuple[str, str]], fmt: str) -> int:
    real = src.resolve(strict=True)
    grp = out.resolve()
    a, b = GGUFReader(real), GGUFReader(grp)
    arch = a.fields["general.architecture"].contents()
    expert_count = int(a.fields[f"{arch}.expert_count"].contents())
    errors: list[str] = []

    # keys our writer stamps with exact values: they are exempt from the
    # verbatim-KV gate.  Anything under taillis.provenance.* may also be
    # added (every provenance key is optional, layout.md §7.2) — the added-key
    # gate below only rejects additions OUTSIDE that namespace.  v1 owns no
    # group key: re-adding it is a foreign injection AND a format violation.
    owned = {FORMAT_KEY, PARENT_KEY, GROUPING_TOOL_KEY}
    if fmt == "v0":
        owned.add(GROUP_KEY)

    by_name_a = {t.name: t for t in a.tensors}
    by_name_b = {t.name: t for t in b.tensors}

    # ---- dispatch: format gate + membership -------------------------------
    # logicals: <logical> -> [(k, member-name WITH .weight)]; v0 in ARRAY order,
    # v1 sorted by ascending k (= the expert order, layout.md §8.2.3).
    logicals: dict[str, list[tuple[int, str]]] = {}
    entries: list[str] = []
    if fmt == "v0":
        # ---- gate 2 (v0): the key is the declaration, well-formed --------
        if GROUP_KEY not in b.fields:
            return fail([f"{GROUP_KEY} absent from {grp}"])
        bfield = b.fields[GROUP_KEY]
        if bfield.types[0] != GGUFValueType.ARRAY or bfield.types[-1] != GGUFValueType.STRING:
            errors.append(f"{GROUP_KEY} is not a string array ({bfield.types})")
        entries = list(bfield.contents())
        k_in_array_order = True
        for i, e in enumerate(entries):
            mm = MEMBER_RE.match(e)
            if not mm:
                errors.append(f"{GROUP_KEY}[{i}]='{e}' does not end in .g<digits>")
                continue
            logical, k = mm.group(1), int(mm.group(2))
            if f"{e}.weight" not in by_name_b:
                errors.append(f"{GROUP_KEY}[{i}]: member '{e}.weight' not in file")
                continue
            if logical + ".weight" in by_name_b:
                errors.append(f"'{logical}.weight' is grouped but ALSO present flat")
            logicals.setdefault(logical, []).append((k, f"{e}.weight"))
        if not logicals:
            errors.append(f"{GROUP_KEY} named no members")
        expect_version = 0
    else:
        # ---- gate 2 (v1): version dispatch + DERIVED membership ----------
        k_in_array_order = False
        if GROUP_KEY in b.fields:
            errors.append(f"{GROUP_KEY} present in a v1 output — malformed "
                          f"(layout.md §8.1: v1 derives the grouping from the member names)")
        expect_version = 1
        flat_experts_a = {n for n in by_name_a if EXPERT_RE.match(n)}
        stray: list[str] = []
        for n in by_name_b:
            mm = MEMBER_FILE_RE.match(n)
            if not mm:
                continue
            if mm.group(1) + ".weight" in flat_experts_a:
                logicals.setdefault(mm.group(1), []).append((int(mm.group(2)), n))
            else:
                stray.append(n)
        if stray:
            more = f" (+{len(stray) - 4} more)" if len(stray) > 4 else ""
            errors.append("member-shaped tensors whose logical is not a flat expert tensor "
                          f"of the source: {sorted(stray)[:4]}{more}")
        for logical in logicals:
            if logical + ".weight" in by_name_b:
                errors.append(f"'{logical}.weight' is grouped but ALSO present flat")
        if not logicals:
            errors.append("no member tensors found in the v1 output")

    n_expert_flat = sum(1 for n in by_name_a if EXPERT_RE.match(n))
    if len(logicals) != n_expert_flat:
        what = "key covers" if fmt == "v0" else "members cover"
        errors.append(f"{what} {len(logicals)} logical tensors, source has {n_expert_flat} expert tensors")

    # ---- gate R1 (version half): the dispatch scalar ----------------------
    f = b.fields.get(FORMAT_KEY)
    if f is None:
        errors.append(f"{FORMAT_KEY} absent (R1)")
    elif f.types[0] != GGUFValueType.UINT32:
        errors.append(f"{FORMAT_KEY} is {f.types[0].name}, not UINT32")
    elif int(f.contents()) != expect_version:
        errors.append(f"{FORMAT_KEY} = {int(f.contents())}, expected {expect_version} ({fmt})")

    # ---- per-logical checks + gates 1 and 5 (real byte comparisons) ------
    fd_a = os.open(real, os.O_RDONLY)
    fd_b = os.open(grp, os.O_RDONLY)
    n_member_cmp = 0
    n_plain_cmp = 0
    n_member_bytes_ok = 0
    try:
        for logical, members in sorted(logicals.items()):
            parent = by_name_a.get(logical + ".weight")
            if parent is None:
                errors.append(f"logical '{logical}.weight' not in source")
                continue
            ks = [k for k, _ in members]
            if k_in_array_order:
                if ks != list(range(len(members))):
                    errors.append(f"{logical}: member indices {ks} are not 0..G-1 in array order")
            else:
                # v1: k is an ordering token — duplicates are malformed, gaps are fine
                dups = sorted({k for k in ks if ks.count(k) > 1})
                if dups:
                    errors.append(f"{logical}: duplicate member index k={dups} (layout.md §8.2.3)")
                members = sorted(members)   # ascending k = the expert order
            ne2 = int(parent.shape[2])                    # ggml ne order: shape[2] == ne[2]
            if parent.data.nbytes % ne2 != 0:
                errors.append(f"{logical}: source nbytes {parent.data.nbytes} not divisible by ne[2]={ne2}")
                continue
            stride = parent.data.nbytes // ne2
            lo = 0
            sum_bytes = 0
            for k, e in members:
                mt = by_name_b[e]
                mne2 = int(mt.shape[2])
                hi = lo + mne2
                # T2.1: every member carries the parent's type (allocation policy is T2.2)
                if mt.tensor_type != parent.tensor_type:
                    errors.append(f"{e}: type {mt.tensor_type.name} != parent {parent.tensor_type.name}")
                # member ne[0]/ne[1] (ggml ne order: shape[:2]) must match parent
                if tuple(int(x) for x in mt.shape[:2]) != tuple(int(x) for x in parent.shape[:2]):
                    errors.append(f"{e}: shape {tuple(mt.shape)} inconsistent with parent {tuple(parent.shape)}")
                if not (0 <= lo < hi <= ne2):
                    errors.append(f"{e}: range [{lo},{hi}) out of [0,{ne2})")
                # gate 1 + 5: member bytes == the parent's byte slab, exactly
                want = sha256_payload(fd_a, parent.data_offset + lo * stride, mt.data.nbytes)
                got = sha256_payload(fd_b, mt.data_offset, mt.data.nbytes)
                if want != got:
                    errors.append(f"{e}: payload sha {got} != parent slab [{lo},{hi}) sha {want}")
                else:
                    n_member_cmp += 1
                sum_bytes += mt.data.nbytes
                lo = hi
            if lo != ne2:
                errors.append(f"{logical}: members cover [0,{lo}), not [0,{ne2})")
            if sum_bytes != parent.data.nbytes:
                errors.append(f"{logical}: member bytes {sum_bytes} != parent {parent.data.nbytes}")
            else:
                n_member_bytes_ok += 1

        # non-expert tensors must be verbatim (gate 1 for the other tensors)
        plain_a = {n: t for n, t in by_name_a.items() if not EXPERT_RE.match(n)}
        plain_b = {n: t for n, t in by_name_b.items()
                   if not EXPERT_RE.match(n) and not MEMBER_FILE_RE.match(n)}
        if set(plain_a) != set(plain_b):
            errors.append(f"non-expert tensor sets differ: {len(set(plain_a) ^ set(plain_b))} names")
        for n in sorted(set(plain_a) & set(plain_b)):
            ta, tb = plain_a[n], plain_b[n]
            if ta.data.nbytes != tb.data.nbytes:
                errors.append(f"{n}: nbytes {tb.data.nbytes} != source {ta.data.nbytes}")
                continue
            want = sha256_payload(fd_a, ta.data_offset, ta.data.nbytes)
            got = sha256_payload(fd_b, tb.data_offset, tb.data.nbytes)
            if want != got:
                errors.append(f"{n}: payload sha mismatch")
            else:
                n_plain_cmp += 1

        # ---- R1 gates: parent_sha256 == sha(src), grouping_tool ----
        # (format_version is checked per-format in the dispatch section above)
        if PARENT_KEY not in b.fields:
            errors.append(f"{PARENT_KEY} absent (R1)")
        else:
            want = sha256_file(real)
            got = b.fields[PARENT_KEY].contents()
            if got != want:
                errors.append(f"{PARENT_KEY} {got} != sha256({real}) {want}")
        if GROUPING_TOOL_KEY not in b.fields:
            errors.append(f"{GROUPING_TOOL_KEY} absent (R1)")
        elif not str(b.fields[GROUPING_TOOL_KEY].contents()).startswith("group_experts.py@"):
            errors.append(f"{GROUPING_TOOL_KEY} = "
                          f"{b.fields[GROUPING_TOOL_KEY].contents()!r}, expected group_experts.py@<commit>")
        for k, v in provenance:
            if b.fields.get(k, None) is None or b.fields[k].contents() != v:
                errors.append(f"{k} = {b.fields[k].contents() if k in b.fields else '<absent>'!r}, expected {v!r}")

        # metadata verbatim except the keys we own (group key + the R1 stamp)
        for key in a.fields:
            if key in RESERVED_KV or key in owned:
                continue
            if key not in b.fields or a.fields[key].contents() != b.fields[key].contents():
                errors.append(f"KV_DRIFT {key}")
        # the writer may add its stamp keys and anything under the optional
        # provenance namespace; anything else is a foreign injection
        added = set(b.fields) - set(a.fields)
        foreign = added - owned - {k for k in added if k.startswith(PROVENANCE_PREFIX)}
        if foreign:
            errors.append(f"keys added outside the stamp set: {sorted(foreign)}")
        dropped = set(a.fields) - set(b.fields)
        if dropped - owned:
            errors.append(f"source keys dropped: {sorted(dropped - owned)}")
    finally:
        os.close(fd_a)
        os.close(fd_b)

    print(f"logical     : {len(logicals)} expert tensors grouped "
          f"({sum(len(v) for v in logicals.values())} members)")
    print(f"gate 1      : {n_member_cmp}/{sum(len(v) for v in logicals.values())} member payloads "
          f"byte-identical to the parent slab; {n_plain_cmp}/{len(plain_a)} verbatim tensors identical")
    print(f"gate 5      : {n_member_bytes_ok}/{len(logicals)} logical tensors with "
          f"sum(member nbytes) == parent nbytes")
    if fmt == "v0":
        print(f"key         : {GROUP_KEY}, {len(entries)} entries")
    else:
        print(f"key         : none (v1: membership derived from member names, layout.md §8.2)")
    def _r1(key: str) -> str:
        return "present" if key in b.fields else "ABSENT"
    print(f"R1          : {FORMAT_KEY} {_r1(FORMAT_KEY)}, {PARENT_KEY} {_r1(PARENT_KEY)}, "
          f"{GROUPING_TOOL_KEY} {_r1(GROUPING_TOOL_KEY)}")
    if errors:
        return fail(errors)
    print(f"GATES OK ({grp})")
    return 0


def fail(errors: list[str]) -> int:
    for e in errors[:20]:
        print(f"FAIL {e}")
    if len(errors) > 20:
        print(f"FAIL ... and {len(errors) - 20} more")
    return 1


def show(path: Path) -> int:
    """Print the taillis.* keys and the format verdict of any GGUF (read-only)."""
    real = path.resolve(strict=True)
    r = GGUFReader(real)
    tensors = list(r.tensors)
    members = [t.name for t in tensors if MEMBER_FILE_RE.match(t.name)]
    print(f"file        : {real}  {real.stat().st_size} bytes ({human(real.stat().st_size)})")
    print(f"tensors     : {len(tensors)} ({sum(1 for t in tensors if EXPERT_RE.match(t.name))} "
          f"flat expert, {len(members)} member-shaped)")
    for k, f in r.fields.items():
        if k in RESERVED_KV or not k.startswith("taillis"):
            continue
        c = f.contents()
        if isinstance(c, list):
            c = (f"[{len(c)} entries] first={c[0]!r} last={c[-1]!r}" if c else "[]")
        elif len(str(c)) > 72:
            c = str(c)[:69] + "..."
        print(f"  {k} = {c} ({f.types[0].name})")
    fver = r.fields.get(FORMAT_KEY)
    version = int(fver.contents()) if fver is not None and fver.types[0] == GGUFValueType.UINT32 else None
    has_key = GROUP_KEY in r.fields
    if version == 1 and not has_key:
        print(f"format      : v1 (derived from member names, layout.md §8.2) — "
              f"READER NOT SHIPPED YET (docs/R2-reader-request.md)")
    elif version == 1:
        print(f"format      : MALFORMED under layout.md §8.1 (version=1 AND {GROUP_KEY} present)")
    elif has_key:
        print(f"format      : v0 (group key is the declaration; version "
              f"{'absent' if fver is None else f'= {version}'})")
    else:
        print(f"format      : flat/ungrouped (no group key, version "
              f"{'absent' if fver is None else f'= {version}'})")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--groups", type=int, default=1,
                   help="even-ish split of ne[2] into G contiguous members (default 1 = trivial)")
    p.add_argument("--spec", type=str, default=None,
                   help="explicit member sizes along ne[2], comma-separated, must sum to ne[2]")
    p.add_argument("--format", choices=("v0", "v1"), default="v0",
                   help="v0 (default): taillis.expert_groups IS the declaration, version=0; "
                        "v1 (OPT-IN, layout.md §8): grouping derived from member names, "
                        "version=1, no group key — aubier cannot read v1 yet")
    p.add_argument("--provenance", action="append", default=[], metavar="K=V",
                   help="stamp taillis.provenance.K=V (repeatable; layout.md §7)")
    p.add_argument("--verify", action="store_true",
                   help="re-parse both files and run the byte-level gates")
    p.add_argument("--show", action="store_true",
                   help="print the taillis.* keys and format verdict of --out (read-only)")
    args = p.parse_args()
    provenance = parse_provenance(args.provenance)
    if args.show:
        sys.exit(show(args.out if args.out is not None else args.src))
    if args.verify:
        if args.out is None:
            raise SystemExit("FATAL: --verify needs --out")
        sys.exit(verify(args.src, args.out, provenance, args.format))
    if args.out is None:
        raise SystemExit("FATAL: give --out (or --verify)")
    if args.spec is not None and args.groups != 1:
        raise SystemExit("FATAL: give --groups or --spec, not both")
    build(args.src, args.out, args.groups, args.spec, provenance, args.format)


if __name__ == "__main__":
    main()
