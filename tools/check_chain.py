#!/usr/bin/env python3
"""Verify the R3 comparison chain (taillis/layout.md §9) — FORMAT OBLIGATION checker.

The chain is P (both engine arms) -> uniform shadow (oracle arm) -> artifact
(store arm); a shuffled control shares the artifact's declared comparator and
shadow identity (§9.3.6).  This tool enforces the file-format side of §9: the
keys exist with the right values, the shadow identity is what the artifact's
own table + parent say it must be, and sibling members really do carry the
parent's bytes.  It is NOT the quality instrument (that is §K, §9.3.4).

Usage
    tools/check_chain.py --artifact A.gguf [--parent P.gguf] [--shadow S.gguf]
                         [--control C.gguf] [--rebuild-shadow DIR]

Gates (layout.md §9.4), each failure NAMED with its mechanism; exit code is the gate (G7):
  G1  the artifact parses and its grouping is declared (v0 key or v1
      derivation, §8.1); per-logical invariants of §8.2.3 hold
      (sum(ne[2])==expert_count, ne[2]>0, uniform ne[0]/ne[1], k unique);
  G2  mixed-ness (§9.2): any logical with non-uniform member types =>
      taillis.uniform_shadow_sha256 REQUIRED (absent => FAIL "R3.2"); a
      uniform artifact carrying it => FAIL (forbidden);
  G3  parent: sha256(--parent file) == taillis.parent_sha256, else FAIL
      naming both values;
  G4  shadow identity: derive S from the artifact's own table (uniformity +
      canonical k checked first — "shadow not rebuildable: non-uniform
      partition" / "k not canonical"); SYNTHESIZE the canonical stream from
      (parent, spec) — no shadow file needed — and compare to the key
      (mismatch => FAIL "shadow identity mismatch").  --shadow FILE is, in
      addition, hashed in the file direction and its member slabs compared
      against the parent's slabs (the writer's gate 1, re-run);
  G5  sibling byte gate: every member whose type == the parent's type for its
      logical must carry the parent's slab bytes.  HARD when the artifact
      records no provenance.imatrix_sha256 (the E2 proof applies; a mismatch
      means wrong parent or wrong weights); WARN-only when an imatrix is
      recorded (outside the E2 proof — the warning names both possibilities,
      it never silently passes);
  G6  control gates (--control): the control re-runs G1-G5 on its own bytes
      (validity is derived, not asserted — §9.3.6) and must share the
      artifact's parent key, shadow key and per-logical member sizes, keep
      the global type multiset, stay within ±384 B whole-file (the D5 class),
      differ in assignment, and carry arm=shuffled-control + shuffle_seed +
      type_file_sha256;
  G7  exit code = gate; output prints COUNTS, SIZES, PATHS and each failure
      with its mechanism.

The canonical content stream (§9.2) — sha256 over, for each tensor of the
shadow's table in TABLE ORDER: u32le(len(name)) || UTF-8(name) ||
u32le(ggml_type) || u32le(n_dims) || u32le(ne[0..n_dims-1]) ||
u64le(payload_nbytes) || payload bytes — lives in exactly ONE function here
(canonical_stream_sha) serving BOTH directions, so the definition cannot
drift: the file direction hashes a real shadow file's table; the synthesize
direction builds the same table from (parent table, spec).  The KV block is
excluded BY CONSTRUCTION (E2-M4: content-identical shadows, different
whole-file shas), so the key is NOT verifiable with sha256sum — only here.

--rebuild-shadow DIR shells out to tools/group_experts.py --src P --spec S
(subprocess: §9.3.3's "rebuildable via group_experts.py" is tested against
the REAL tool, any --format), re-verifies the rebuild with the writer's own
--verify, then hashes it in the file direction.  Output paths go through the
writer's check_write_scope allowlist (/var/lib/taillis/models, /tmp).

No engine change is involved or checked: the reader knows one taillis.* key
(aubier gguf_model.h:183) and ignores the rest (§7.2 rule 2).  Needs gguf-py
(venvs/convert/bin/python3); re-execs under that venv when not importable.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent


def ensure_venv() -> None:
    """gguf-py lives in venvs/convert; re-exec under its python (eval-test pattern)."""
    try:
        import gguf  # noqa: F401
    except ImportError:
        venv_py = REPO / "venvs" / "convert" / "bin" / "python3"
        if not venv_py.exists():
            sys.exit("FATAL: gguf-py not importable and venvs/convert missing")
        os.execv(str(venv_py), [str(venv_py), __file__, *sys.argv[1:]])


ensure_venv()

from gguf import GGUFReader, GGUFValueType  # noqa: E402

# The writer's own definitions, imported (not copied) so the two tools cannot
# drift apart on key names, the expert surface, or the write-scope allowlist.
from group_experts import (  # noqa: E402
    EXPERT_RE,
    FORMAT_KEY,
    GROUP_KEY,
    MEMBER_FILE_RE,
    MEMBER_RE,
    PARENT_KEY,
    PROVENANCE_PREFIX,
    check_write_scope,
    human,
    sha256_file,
    sha256_payload,
)

SHADOW_KEY = "taillis.uniform_shadow_sha256"
TYPE_FILE_KEY = PROVENANCE_PREFIX + "type_file_sha256"
ARM_KEY = PROVENANCE_PREFIX + "arm"
SEED_KEY = PROVENANCE_PREFIX + "shuffle_seed"
IMATRIX_KEY = PROVENANCE_PREFIX + "imatrix_sha256"
SIZE_TOL = 384  # §9.3.6: the D5 whole-file class bound for a control


class Entry(NamedTuple):
    """One tensor of the shadow's table, its payload located in an open fd."""
    name: str
    ttype: int
    ne: tuple[int, ...]
    fd: int
    offset: int
    nbytes: int


def canonical_stream_sha(entries: list[Entry]) -> tuple[str, list[str]]:
    """THE §9.2 definition — the only place the encoding lives in this repo.

    For each tensor of the shadow's tensor table in TABLE ORDER, concatenate
        u32le(len(name)) || UTF-8(name) || u32le(ggml_type) || u32le(n_dims)
        || u32le(ne[0..n_dims-1]) || u64le(payload_nbytes) || payload bytes
    and hash the concatenation.  Payloads are streamed chunk-wise (never the
    whole file in memory).  Returns (stream digest, per-entry PAYLOAD digests)
    from a single read pass, so callers get slab shas for free.
    """
    total = hashlib.sha256()
    slabs: list[str] = []
    for e in entries:
        nb = e.name.encode("utf-8")
        if len(nb) > 0xFFFFFFFF or any(d > 0xFFFFFFFF for d in e.ne):
            raise SystemExit(f"FATAL: {e.name} exceeds the canonical stream's u32 fields")
        total.update(struct.pack("<I", len(nb)))
        total.update(nb)
        total.update(struct.pack("<I", e.ttype))
        total.update(struct.pack("<I", len(e.ne)))
        for d in e.ne:
            total.update(struct.pack("<I", d))
        total.update(struct.pack("<Q", e.nbytes))
        h = hashlib.sha256()
        off = 0
        while off < e.nbytes:
            n = min(8 * 1024 * 1024, e.nbytes - off)
            buf = os.pread(e.fd, n, e.offset + off)
            if len(buf) != n:
                raise SystemExit(f"FATAL: short read at {e.offset + off} ({e.name})")
            total.update(buf)
            h.update(buf)
            off += n
        slabs.append(h.hexdigest())
    return total.hexdigest(), slabs


def file_entries(fd: int, reader: GGUFReader) -> list[Entry]:
    """The FILE direction: one Entry per tensor of a real file, TABLE ORDER."""
    return [Entry(t.name, int(t.tensor_type), tuple(int(x) for x in t.shape),
                  fd, t.data_offset, t.data.nbytes) for t in reader.tensors]


class SpecMismatch(Exception):
    """The parent's table and the spec disagree (ne[2]/divisibility) — a named
    checker outcome routed through G7, never a traceback."""


def shadow_entries(parent: GGUFReader, fd: int, spec: dict[str, list[int]]) -> list[Entry]:
    """The SYNTHESIZE direction (§9.2): the shadow's table as a pure function
    of the parent's table + spec — non-expert tensors verbatim; each expert
    logical replaced by G members <logical>.g<k>.weight at the PARENT's type,
    ne[2]=size_k, payload = the parent's byte slab.  Table order = parent
    order with the members in place of the logical."""
    out: list[Entry] = []
    for t in parent.tensors:
        ne = tuple(int(x) for x in t.shape)
        if not EXPERT_RE.match(t.name):
            out.append(Entry(t.name, int(t.tensor_type), ne, fd, t.data_offset, t.data.nbytes))
            continue
        logical = t.name[:-len(".weight")]
        ne2 = ne[2]
        if ne2 == 0 or t.data.nbytes % ne2:
            raise SpecMismatch(
                f"{t.name}: nbytes {t.data.nbytes} not divisible by ne[2]={ne2} — "
                f"parent table and spec disagree, shadow synthesis impossible")
        stride = t.data.nbytes // ne2
        lo = 0
        for k, s in enumerate(spec[logical]):
            out.append(Entry(f"{logical}.g{k}.weight", int(t.tensor_type),
                             (ne[0], ne[1], s), fd, t.data_offset + lo * stride, stride * s))
            lo += s
        if lo != ne2:
            raise SpecMismatch(
                f"{t.name}: spec sums to {lo}, parent ne[2] is {ne2} — parent table "
                f"and spec disagree, shadow synthesis impossible")
    return out


def parse_grouping(reader: GGUFReader, errors: list[str]) -> dict[str, list[tuple[int, str, object]]]:
    """G1: §8.1 dispatch (v0 key / v1 derivation) + §8.2.3 invariants, on the
    artifact's own table.  Returns logical -> [(k, member name, tensor)],
    members sorted by ascending k (= the expert order, §8.2.3)."""
    by_name = {t.name: t for t in reader.tensors}
    version: int | None = None
    fver = reader.fields.get(FORMAT_KEY)
    if fver is not None:
        if fver.types[0] != GGUFValueType.UINT32:
            errors.append(f"{FORMAT_KEY} is {fver.types[0].name}, not UINT32 (layout.md §8.1)")
        else:
            version = int(fver.contents())

    logicals: dict[str, list[tuple[int, str, object]]] = {}
    if GROUP_KEY in reader.fields:
        if version == 1:
            errors.append(f"{GROUP_KEY} present with {FORMAT_KEY}=1 — malformed (layout.md §8.1)")
        field = reader.fields[GROUP_KEY]
        if field.types[0] != GGUFValueType.ARRAY or field.types[-1] != GGUFValueType.STRING:
            errors.append(f"{GROUP_KEY} is not a string array ({field.types})")
        else:
            claimed: set[str] = set()
            for i, e in enumerate(field.contents()):
                mm = MEMBER_RE.match(str(e))
                if not mm:
                    errors.append(f"{GROUP_KEY}[{i}]='{e}' does not end in .g<digits>")
                    continue
                tname = f"{e}.weight"
                claimed.add(tname)
                if not EXPERT_RE.match(mm.group(1) + ".weight"):
                    errors.append(f"{GROUP_KEY}[{i}]: logical '{mm.group(1)}.weight' is "
                                  f"outside the expert surface (layout.md §8.2.6)")
                if tname not in by_name:
                    errors.append(f"{GROUP_KEY}[{i}]: member '{tname}' not in file")
                    continue
                logicals.setdefault(mm.group(1), []).append((int(mm.group(2)), tname, by_name[tname]))
            # a member-shaped tensor the key does NOT claim is undeclared
            # grouping — in v0 the key is the ONLY declaration (layout.md §2)
            unclaimed = sorted(n for n in by_name
                               if MEMBER_FILE_RE.match(n) and n not in claimed)
            if unclaimed:
                errors.append(f"G1: member-shaped tensor(s) not named in {GROUP_KEY}: "
                              f"{unclaimed[:4]}" + (f" (+{len(unclaimed) - 4} more)"
                                                    if len(unclaimed) > 4 else ""))
    elif version == 1:
        stray: list[str] = []
        for tname in by_name:
            mm = MEMBER_FILE_RE.match(tname)
            if not mm:
                continue
            if EXPERT_RE.match(mm.group(1) + ".weight"):
                logicals.setdefault(mm.group(1), []).append((int(mm.group(2)), tname, by_name[tname]))
            else:
                stray.append(tname)
        if stray:
            errors.append(f"member-shaped tensors outside the expert surface: {sorted(stray)[:4]}")
    else:
        errors.append(f"G1: no grouping declared ({GROUP_KEY} absent, {FORMAT_KEY} "
                      f"{'absent' if version is None else f'= {version}'}) — not a grouped artifact")
        return logicals
    for logical, _members in list(logicals.items()):
        if logical + ".weight" in by_name:
            errors.append(f"G1: '{logical}.weight' is grouped but ALSO present flat (layout.md §8.2.5)")
            del logicals[logical]

    f = reader.fields.get("general.architecture")
    arch = str(f.contents()) if f is not None and f.types[0] == GGUFValueType.STRING else None
    expert_count: int | None = None
    if arch is None:
        errors.append("G1: general.architecture absent — cannot read <arch>.expert_count")
    else:
        ec = reader.fields.get(f"{arch}.expert_count")
        if ec is None or ec.types[0] != GGUFValueType.UINT32:
            errors.append(f"G1: {arch}.expert_count absent/not u32 — cannot check sum(ne[2])")
        else:
            expert_count = int(ec.contents())

    for logical, members in logicals.items():
        members.sort(key=lambda x: x[0])
        ks = [k for k, _, _ in members]
        dups = sorted({k for k in ks if ks.count(k) > 1})
        if dups:
            errors.append(f"G1: {logical}: duplicate member index k={dups} (layout.md §8.2.3)")
        shapes = {tuple(int(x) for x in t.shape[:2]) for _, _, t in members}
        if len(shapes) > 1:
            errors.append(f"G1: {logical}: non-uniform member ne[0]/ne[1]: {sorted(shapes)}")
        if any(int(t.shape[2]) <= 0 for _, _, t in members):
            errors.append(f"G1: {logical}: non-positive member ne[2]")
        if expert_count is not None:
            total = sum(int(t.shape[2]) for _, _, t in members)
            if total != expert_count:
                errors.append(f"G1: {logical}: sum(ne[2])={total} != {arch}.expert_count={expert_count}")
    return logicals


def derive_spec(logicals: dict[str, list[tuple[int, str, object]]],
                errors: list[str]) -> dict[str, list[int]]:
    """§9.3.3's shadow-rebuildability preconditions, each a NAMED outcome:
    uniform member-size partition across logicals; k exactly 0..G-1 per
    logical (the parent is flat is checked by the caller)."""
    if not logicals:
        return {}
    per: dict[str, list[int]] = {}
    for logical, members in logicals.items():
        ks = [k for k, _, _ in members]
        if ks != list(range(len(members))):
            errors.append(f"G4: {logical}: k not canonical {ks} — shadow not rebuildable: "
                          f"§9.3.3 wants exactly 0..G-1")
        per[logical] = [int(t.shape[2]) for _, _, t in members]
    partitions = {tuple(v) for v in per.values()}
    if len(partitions) != 1:
        errors.append("G4: shadow not rebuildable: non-uniform partition across logicals ("
                      + ", ".join(f"{l}={per[l]}" for l in sorted(per)) + ")")
        return {}
    sizes = list(next(iter(partitions)))
    return {l: list(sizes) for l in per}


def check_artifact(tag: str, path: Path, parent_path: Path | None, shadow_path: Path | None,
                   synth_cache: dict) -> dict:
    """G1-G5 on one artifact; prints its section; returns the G6 summary."""
    errors: list[str] = []
    warns: list[str] = []
    real = path.resolve(strict=True)
    print(f"\n== {tag}: {real}  {real.stat().st_size} bytes ({human(real.stat().st_size)})")
    try:
        reader = GGUFReader(real)
    except Exception as e:
        print(f"FAIL G1: {real} does not parse ({type(e).__name__}: {e})")
        return {"failed": True, "errors": [f"G1: does not parse ({e})"], "warns": []}

    logicals = parse_grouping(reader, errors)
    n_members = sum(len(v) for v in logicals.values())
    mixed_l = [l for l, ms in logicals.items()
               if len({int(t.tensor_type) for _, _, t in ms}) > 1]
    is_mixed = bool(mixed_l)
    print(f"G1: {len(logicals)} logical(s), {n_members} member(s), "
          f"declaration {'v1-derived' if GROUP_KEY not in reader.fields else 'v0 key'}; "
          f"{'MIXED' if is_mixed else 'uniform'}"
          + (f" ({', '.join(sorted(mixed_l))})" if is_mixed else ""))

    # ---- G2: mixed-ness vs the shadow key (layout.md §9.2) ----------------
    key_val: str | None = None
    kf = reader.fields.get(SHADOW_KEY)
    if kf is not None:
        if kf.types[0] != GGUFValueType.STRING:
            errors.append(f"G2: {SHADOW_KEY} is {kf.types[0].name}, not STRING (layout.md §9.2)")
        else:
            key_val = str(kf.contents())
    if is_mixed and key_val is None:
        errors.append(f"G2: R3.2 — MIXED artifact ({len(mixed_l)} logical(s) with non-uniform "
                      f"member types) but {SHADOW_KEY} absent: REQUIRED (layout.md §9.2)")
    if not is_mixed and key_val is not None:
        errors.append(f"G2: R3.2 — uniform artifact carries {SHADOW_KEY}: FORBIDDEN "
                      f"(layout.md §9.2: required on mixed, forbidden on uniform)")
    if key_val is not None:
        print(f"G2: {SHADOW_KEY} present "
              f"({'required: MIXED' if is_mixed else 'FORBIDDEN on uniform — failure recorded'})")
    else:
        print(f"G2: {SHADOW_KEY} absent "
              f"({'REQUIRED on MIXED — failure recorded' if is_mixed else 'not required: uniform'})")

    # ---- G3: the declared comparator --------------------------------------
    parent_key = None
    pkf = reader.fields.get(PARENT_KEY)
    if pkf is None:
        errors.append(f"G3: {PARENT_KEY} absent (layout.md §7.1/§9.3.1)")
    elif pkf.types[0] != GGUFValueType.STRING:
        errors.append(f"G3: {PARENT_KEY} is {pkf.types[0].name}, not STRING")
    else:
        parent_key = str(pkf.contents())
    parent_reader: GGUFReader | None = None
    parent_fd = -1
    preal: Path | None = None
    if parent_path is not None:
        preal = parent_path.resolve(strict=True)
        want = sha256_file(preal)
        if parent_key is None:
            errors.append(f"G3: cannot compare — {PARENT_KEY} absent; sha256({preal}) = {want}")
        elif parent_key != want:
            errors.append(f"G3: {PARENT_KEY} {parent_key} != sha256({preal}) {want}")
        else:
            print(f"G3: parent_sha256 == sha256({preal}) = {want}")
        try:
            parent_reader = GGUFReader(preal)
            # §9.3.3: the parent must be FLAT (same refusal shape as the writer's)
            if GROUP_KEY in parent_reader.fields:
                errors.append(f"G4: parent {preal} carries {GROUP_KEY} — parent is not flat")
            member_srcs = [t.name for t in parent_reader.tensors if MEMBER_FILE_RE.match(t.name)]
            if member_srcs:
                errors.append(f"G4: parent {preal} carries {len(member_srcs)} member-shaped "
                              f"tensors (e.g. {member_srcs[0]}) — parent is not flat")
        except Exception as e:
            errors.append(f"G4: parent {preal} does not parse ({type(e).__name__}: {e})")
            parent_reader = None
        if parent_reader is not None:
            parent_fd = os.open(preal, os.O_RDONLY)
    else:
        print("G3: --parent not given: comparator value taken as declared, NOT verified")

    # ---- G4: shadow identity, both directions -----------------------------
    spec = derive_spec(logicals, errors)
    synth_sha: str | None = None
    synth_slabs: list[str] | None = None
    spec_sizes = list(next(iter(spec.values()))) if spec else None
    if spec and parent_reader is not None:
        # The synth direction is a PARENT-anchored claim: if the given parent
        # lacks the flat expert tensors, "synthesis" would silently degrade to
        # hashing the parent's own table (a region file content-identical to
        # the shadow would then "verify" trivially) — void, not a pass.
        pbn = {t.name: t for t in parent_reader.tensors if EXPERT_RE.match(t.name)}
        missing = sorted(l for l in spec if f"{l}.weight" not in pbn)
        if missing:
            errors.append(f"G4: parent {preal} lacks flat expert tensor(s) "
                          f"{[l + '.weight' for l in missing]} — shadow synthesis impossible")
        else:
            bad = []
            for logical in sorted(spec):
                p = pbn[f"{logical}.weight"]
                ne2 = int(p.shape[2])
                if sum(spec[logical]) != ne2 or ne2 == 0 or p.data.nbytes % ne2:
                    bad.append(f"{logical}: spec sums to {sum(spec[logical])}, "
                               f"parent ne[2]={ne2}")
            if bad:
                errors.append("G4: spec/parent mismatch — shadow synthesis impossible: "
                              + "; ".join(bad))
            else:
                cache_key = (str(preal), tuple(spec_sizes))
                if cache_key not in synth_cache:
                    try:
                        synth_cache[cache_key] = canonical_stream_sha(
                            shadow_entries(parent_reader, parent_fd, spec))
                    except SpecMismatch as e:  # defensive; preflight above is exact
                        errors.append(f"G4: {e}")
                    else:
                        synth_sha, synth_slabs = synth_cache[cache_key]
    if key_val is not None:
        if synth_sha is None:
            if parent_path is None and shadow_path is None:
                errors.append("G4: cannot verify the shadow key — no --parent and no "
                              "--shadow given")
            elif parent_path is None:
                # §9.2's OR: the file direction at the --shadow block below
                # verifies the key without a parent; the synth arm is skipped,
                # visibly, not failed.
                print("G4: synth direction skipped — no --parent (key verified in "
                      "the file direction, layout.md §9.2)")
            # else: synthesis impossible — already named where it failed
        elif synth_sha != key_val:
            errors.append(f"G4: shadow identity mismatch — {SHADOW_KEY} {key_val} != "
                          f"synthesized-from(parent,spec) {synth_sha}")
        else:
            print(f"G4: synth  {synth_sha}  == {SHADOW_KEY}")
    elif is_mixed:
        print("G4: no shadow key on a mixed artifact (already a G2 failure)")

    shadow_fd = -1
    sreader = None
    sreal = None
    if shadow_path is not None:
        try:
            sreal = shadow_path.resolve(strict=True)
            sreader = GGUFReader(sreal)
            shadow_fd = os.open(sreal, os.O_RDONLY)
        except Exception as e:  # unparsable/unreadable --shadow: a named outcome
            errors.append(f"G4: --shadow {shadow_path} does not parse "
                          f"({type(e).__name__}: {e}) — not a GGUF this checker can hash")
            sreader = None
    if sreader is not None:
        file_sha, file_slabs = canonical_stream_sha(file_entries(shadow_fd, sreader))
        if key_val is not None and file_sha != key_val:
            errors.append(f"G4: shadow identity mismatch — {SHADOW_KEY} {key_val} != canonical "
                          f"stream of {sreal} {file_sha}")
        if synth_sha is not None and file_sha != synth_sha:
            errors.append(f"G4: shadow identity mismatch — canonical stream of {sreal} "
                          f"{file_sha} != synthesized-from(parent,spec) {synth_sha}")
        if key_val is None and synth_sha is None:
            print(f"G4: shadow file stream {file_sha} (no key/synth to compare against)")
        else:
            print(f"G4: file   {file_sha}  ({sreal})")
        # the writer's gate 1, re-run: the shadow's member slabs == parent slabs
        if synth_slabs is not None:
            synth_by_name = {e.name: s for e, s in
                             zip(shadow_entries(parent_reader, parent_fd, spec), synth_slabs)}
            n_slab = n_slab_ok = 0
            for t in sreader.tensors:
                if not MEMBER_FILE_RE.match(t.name) or t.name not in synth_by_name:
                    continue
                n_slab += 1
                if sha256_payload(shadow_fd, t.data_offset, t.data.nbytes) == synth_by_name[t.name]:
                    n_slab_ok += 1
                else:
                    errors.append(f"G4: shadow slab mismatch — {t.name} payload != parent slab "
                                  f"(wrong --shadow or wrong --parent)")
            print(f"G4: shadow slabs {n_slab_ok}/{n_slab} byte-identical to the parent slabs")

    # ---- G5: sibling byte gate --------------------------------------------
    hard = IMATRIX_KEY not in reader.fields
    n_sib = n_sib_ok = 0
    if parent_reader is not None:
        parent_by_name = {t.name: t for t in parent_reader.tensors}
        art_fd = os.open(real, os.O_RDONLY)
        try:
            for logical, members in sorted(logicals.items()):
                p = parent_by_name.get(logical + ".weight")
                if p is None:
                    errors.append(f"G5: parent has no flat tensor '{logical}.weight' — "
                                  f"wrong --parent?")
                    continue
                ne2 = int(p.shape[2])
                stride: int | None = None
                if ne2 and p.data.nbytes % ne2 == 0:
                    stride = p.data.nbytes // ne2
                else:
                    errors.append(f"G5: {logical}: parent nbytes {p.data.nbytes} not "
                                  f"divisible by ne[2]={ne2}")
                lo = 0
                for _k, tname, t in members:  # ascending k = the expert order (§8.2.3)
                    hi = lo + int(t.shape[2])
                    if stride is not None and 0 <= lo < hi <= ne2:
                        if int(t.tensor_type) == int(p.tensor_type):
                            n_sib += 1
                            want = sha256_payload(parent_fd, p.data_offset + lo * stride,
                                                  t.data.nbytes)
                            got = sha256_payload(art_fd, t.data_offset, t.data.nbytes)
                            if want == got:
                                n_sib_ok += 1
                            else:
                                msg = (f"G5: {tname} payload sha {got} != parent slab "
                                       f"{want} — wrong parent or wrong weights")
                                if hard:
                                    errors.append(msg + " [HARD: no "
                                                  f"{IMATRIX_KEY} recorded — the E2 proof applies]")
                                else:
                                    warns.append(msg + f" [WARN: {IMATRIX_KEY} recorded — "
                                                  "outside the E2 proof; named, never silent]")
                    lo = hi
        finally:
            os.close(art_fd)
        mode = "HARD (no imatrix recorded — E2 proof applies)" if hard else \
               f"WARN-only ({IMATRIX_KEY} recorded — outside the E2 proof)"
        print(f"G5: siblings {n_sib_ok}/{n_sib} byte-identical to the parent slabs — {mode}")
    else:
        print("G5: --parent not given: sibling gate NOT run")

    if parent_fd >= 0:
        os.close(parent_fd)
    if shadow_fd >= 0:
        os.close(shadow_fd)
    for w in warns:
        print(f"WARN {w}")
    for e in errors:
        print(f"FAIL {e}")
    return {
        "tag": tag, "path": str(real), "size": real.stat().st_size,
        "failed": bool(errors), "errors": errors, "warns": warns,
        "mixed": is_mixed, "parent_key": parent_key, "shadow_key": key_val,
        "synth_sha": synth_sha, "spec": spec_sizes, "n_members": n_members,
        "types": {l: [int(t.tensor_type) for _, _, t in ms] for l, ms in logicals.items()},
        "arm": (str(reader.fields[ARM_KEY].contents()) if ARM_KEY in reader.fields else None),
        "has_seed": SEED_KEY in reader.fields,
        "has_type_file": TYPE_FILE_KEY in reader.fields,
    }


def gate_control(art: dict, ctl: dict) -> list[str]:
    """G6 (layout.md §9.3.6): the control shares identity, conserves bytes,
    differs in assignment, and carries its own keys.  The control's own
    G1-G5 failures were already printed by its check_artifact section."""
    errors: list[str] = []
    if art.get("failed") or ctl.get("failed") or "path" not in art or "path" not in ctl:
        errors.append("G6: control gates not comparable — one side failed to parse")
        return errors
    if not art.get("mixed"):
        errors.append("G6: --control requires a MIXED allocated artifact (a uniform "
                      "artifact has no shadow identity to share, layout.md §9.2)")
    print(f"\n== G6: control {ctl['path']}  {ctl['size']} bytes "
          f"vs allocated {art['path']}  {art['size']} bytes")
    if art.get("parent_key") != ctl.get("parent_key"):
        errors.append(f"G6: control {PARENT_KEY} {ctl.get('parent_key')} != allocated "
                      f"{art.get('parent_key')}")
    if art.get("shadow_key") != ctl.get("shadow_key"):
        errors.append(f"G6: control {SHADOW_KEY} {ctl.get('shadow_key')} != allocated "
                      f"{art.get('shadow_key')}")
    if art.get("spec") is None or ctl.get("spec") is None or art["spec"] != ctl["spec"]:
        errors.append(f"G6: control per-logical member sizes {ctl.get('spec')} != "
                      f"allocated {art.get('spec')}")
    ta = sorted(t for seq in art["types"].values() for t in seq)
    tc = sorted(t for seq in ctl["types"].values() for t in seq)
    if ta != tc:
        errors.append(f"G6: control global type multiset {tc} != allocated {ta}")
    else:
        print(f"G6: global type multiset conserved ({len(ta)} members)")
    delta = ctl["size"] - art["size"]
    if abs(delta) > SIZE_TOL:
        errors.append(f"G6: control whole-file size differs by {delta:+d} B "
                      f"(limit ±{SIZE_TOL} B, layout.md §9.3.6) — a type moved onto a "
                      f"different-SIZE member is the prime suspect")
    else:
        print(f"G6: whole-file delta {delta:+d} B (limit ±{SIZE_TOL} B)")
    if art["types"] == ctl["types"]:
        errors.append("G6: control assignment IDENTICAL to the allocated artifact — "
                      "not a control (layout.md §9.3.6)")
    else:
        diff = [l for l in art["types"] if art["types"][l] != ctl["types"].get(l)]
        print(f"G6: assignment differs in {len(diff)} logical(s): {', '.join(sorted(diff))}")
    if ctl.get("arm") != "shuffled-control":
        errors.append(f"G6: control {ARM_KEY} = {ctl.get('arm')!r}, expected "
                      f"'shuffled-control' (layout.md §9.3.6)")
    if not ctl.get("has_seed"):
        errors.append(f"G6: control {SEED_KEY} absent (layout.md §9.3.6)")
    if not ctl.get("has_type_file"):
        errors.append(f"G6: control {TYPE_FILE_KEY} absent (layout.md §9.3.6)")
    if not any(e.startswith("G6:") for e in errors):
        print("G6: control gates OK")
    for e in errors:
        print(f"FAIL {e}")
    return errors


def rebuild_shadow(artifact: dict, parent_path: Path, out_dir: Path,
                   key_val: str | None, synth_sha: str | None, spec_sizes) -> list[str]:
    """§9.3.3 against the REAL tool: group_experts.py --src P --spec S, then the
    writer's own --verify, then the file-direction hash of the rebuild."""
    errors: list[str] = []
    if spec_sizes is None:
        return ["rebuild-shadow: no uniform spec derivable from the artifact (see named "
                "errors above) — shadow not rebuildable"]
    preal = parent_path.resolve(strict=True)
    out = out_dir.resolve() / f"{Path(artifact['path']).stem}.shadow-rebuilt.gguf"
    check_write_scope(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        errors.append(f"rebuild-shadow: {out} already exists; remove it or pick another --rebuild-shadow")
        for e in errors:
            print(f"FAIL {e}")
        return errors
    spec_arg = ",".join(str(s) for s in spec_sizes)
    tool = REPO / "tools" / "group_experts.py"
    build = subprocess.run(
        [sys.executable, str(tool), "--src", str(preal), "--out", str(out),
         "--spec", spec_arg, "--format", "v0"],
        capture_output=True, text=True)
    if build.returncode != 0:
        errors.append(f"rebuild-shadow: group_experts.py --src {preal} --spec {spec_arg} "
                      f"exited {build.returncode}:\n{build.stdout}\n{build.stderr}")
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)  # a failed build leaves a partial; never keep it
        for e in errors:
            print(f"FAIL {e}")
        return errors
    verify = subprocess.run(
        [sys.executable, str(tool), "--verify", "--src", str(preal), "--out", str(out),
         "--format", "v0"],
        capture_output=True, text=True)
    if verify.returncode != 0:
        errors.append(f"rebuild-shadow: the writer's own --verify REJECTED the rebuild "
                      f"(exit {verify.returncode}):\n{verify.stdout}\n{verify.stderr}")
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)  # a rejected rebuild is garbage; never keep it
        for e in errors:
            print(f"FAIL {e}")
        return errors
    try:
        rreader = GGUFReader(out)
    except Exception as e:
        errors.append(f"rebuild-shadow: the rebuilt {out} does not parse "
                      f"({type(e).__name__}: {e})")
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
        for e in errors:
            print(f"FAIL {e}")
        return errors
    fd = os.open(out, os.O_RDONLY)
    try:
        sha, _slabs = canonical_stream_sha(file_entries(fd, rreader))
    except SpecMismatch as e:  # unreachable: the writer verified table/spec consistency
        errors.append(f"rebuild-shadow: {e}")
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
        for err in errors:
            print(f"FAIL {err}")
        return errors
    finally:
        os.close(fd)
    print(f"\n== rebuild-shadow: group_experts.py --src {preal} --spec {spec_arg} "
          f"--format v0 -> {out}  {out.stat().st_size} bytes ({human(out.stat().st_size)})")
    print(f"rebuild-shadow: writer --verify: PASS; canonical stream {sha}")
    if key_val is not None and sha != key_val:
        errors.append(f"rebuild-shadow: shadow identity mismatch — {SHADOW_KEY} {key_val} "
                      f"!= rebuilt-file stream {sha}")
    if synth_sha is not None and sha != synth_sha:
        errors.append(f"rebuild-shadow: shadow identity mismatch — rebuilt-file stream {sha} "
                      f"!= synthesized-from(parent,spec) {synth_sha}")
    if not errors:
        print(f"rebuild-shadow: == {SHADOW_KEY}"
              + (f" == synthesized {synth_sha}" if synth_sha else ""))
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifact", type=Path, required=True, help="the grouped artifact to check")
    p.add_argument("--parent", type=Path, default=None,
                   help="the flat parent P; without it G3/G4-synth/G5 are NOT verified")
    p.add_argument("--shadow", type=Path, default=None,
                   help="a real uniform-shadow file; hashed in the file direction and "
                        "slab-checked against the parent (G4)")
    p.add_argument("--control", type=Path, default=None,
                   help="the shuffled-control artifact; runs G6 (§9.3.6)")
    p.add_argument("--rebuild-shadow", type=Path, default=None, metavar="DIR",
                   help="rebuild the shadow with tools/group_experts.py into DIR and "
                        "re-verify (§9.3.3 against the real tool)")
    args = p.parse_args()

    if (args.control is not None or args.rebuild_shadow is not None) and args.parent is None:
        raise SystemExit("FATAL: --control/--rebuild-shadow need --parent")

    synth_cache: dict = {}
    art = check_artifact("allocated artifact" if args.control else "artifact",
                         args.artifact, args.parent, args.shadow, synth_cache)
    errors: list[str] = list(art.get("errors", []))

    if args.control is not None:
        ctl = check_artifact("control", args.control, args.parent, None, synth_cache)
        errors += ctl.get("errors", [])
        errors += gate_control(art, ctl)

    if args.rebuild_shadow is not None:
        errors += rebuild_shadow(art, args.parent, args.rebuild_shadow,
                                 art.get("shadow_key"), art.get("synth_sha"),
                                 art.get("spec"))

    total = len(errors)
    print(f"\n== {total} failure(s)"
          + ("" if total else " — ALL GATES OK"))
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
