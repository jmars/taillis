#!/usr/bin/env python3
"""Build the MALFORMED grouped GGUFs — two v0 (key-based) and four v1 (derived).

The aubier reader (engine/gguf_model.h) must REJECT every one; a test that cannot
fail is not a test.  Files are tiny, synthetic, and carry no model data.

v0 fixtures — taillis.expert_groups IS the declaration (layout.md §2):

  taillis-t21-bad-sum.gguf
        two members with ne[2] = 60 + 60 = 120 != 128 named in
        taillis.expert_groups -> the reader's sum(member ne[2]) == N_EXPERT
        CHECK must fire when logical() is called (gguf_model.h:318).
  taillis-t21-bad-flat-and-grouped.gguf
        'blk.0.ffn_gate_up_exps.weight' present FLAT and a '.g0' member naming
        it in the key -> the reader must refuse the file at load time
        (gguf_model.h:253).

v1 fixtures — format_version = 1, NO group key: the grouping is DERIVED from
the member names, so these attack the derivation (taillis/layout.md §8.2).
They are fixtures for AUBIER'S R2 READER REQUEST (docs/R2-reader-request.md);
taillis-side, the writer refuses each of them as a SOURCE (member-shaped names
=> not a flat parent), which eval/test_group_experts_v1.py checks:

  taillis-r2-bad-dup-k.gguf
        two tensor-table entries BOTH named
        'blk.0.ffn_gate_up_exps.g0.weight' -> the new duplicate-k CHECK must
        fire (gguf-py refuses duplicate add_tensor names, MEASURED, so the
        second entry's name is patched .g1 -> .g0 in place: same byte length).
  taillis-r2-bad-flat-and-grouped.gguf
        the logical present FLAT alongside its .g0 member -> the surviving
        key-free consistency check (§8.2.5) must fire.
  taillis-r2-bad-sum.gguf
        members summing to 120 -> the SURVIVING sum(ne[2]) == N_EXPERT CHECK
        (gguf_model.h:318) must fire exactly as in v0.
  taillis-r2-non-expert-member.gguf
        'blk.0.attn_q_a.g0.weight' — a member of a NON-expert logical -> the
        derivation must ignore it and the reader's existing UNCLASSIFIED pass
        must flag it (gguf_model.h:533-538, §8.2.6).

Usage: tools/make_malformed_grouped.py [--out-dir /tmp]
Prints COUNTS, SIZES and PATHS only.  Exit code is the gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGUFValueType
except ImportError as e:  # pragma: no cover - environment error, not logic
    sys.exit(f"FATAL: gguf-py not importable ({e}); use venvs/convert/bin/python3")

GROUP_KEY = "taillis.expert_groups"
FORMAT_KEY = "taillis.format_version"


def add_key(writer: GGUFWriter, entries: list[str]) -> None:
    writer.add_key_value(GROUP_KEY, entries, GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)


def member_tensor(ne2: int) -> np.ndarray:
    # numpy shape (ne2, 4, 8) -> ggml ne = (8, 4, ne2): ne[2] = ne2 experts
    return np.zeros((ne2, 4, 8), dtype=np.float32)


def write(path: Path, tensors: list[tuple[str, np.ndarray]], entries: list[str]) -> None:
    if path.exists():
        path.unlink()
    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    for name, arr in tensors:
        w.add_tensor(name, arr)
    add_key(w, entries)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}  {path.stat().st_size} bytes  tensors={len(tensors)} key_entries={len(entries)}")


def write_v1(path: Path, tensors: list[tuple[str, np.ndarray]]) -> None:
    """A v1 file: format_version = 1, NO group key (layout.md §8.1)."""
    if path.exists():
        path.unlink()
    w = GGUFWriter(str(path), arch="", use_temp_file=False)
    w.add_key_value(FORMAT_KEY, 1, GGUFValueType.UINT32)
    for name, arr in tensors:
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}  {path.stat().st_size} bytes  tensors={len(tensors)} format_version=1 (no group key)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1: members' ne[2] sum to 120, not 128
    bad_sum = args.out_dir / "taillis-t21-bad-sum.gguf"
    write(bad_sum,
          [("blk.0.ffn_gate_up_exps.g0.weight", member_tensor(60)),
           ("blk.0.ffn_gate_up_exps.g1.weight", member_tensor(60))],
          ["blk.0.ffn_gate_up_exps.g0", "blk.0.ffn_gate_up_exps.g1"])

    # 2: the logical tensor is grouped by the key AND present flat
    bad_both = args.out_dir / "taillis-t21-bad-flat-and-grouped.gguf"
    write(bad_both,
          [("blk.0.ffn_gate_up_exps.weight", member_tensor(128)),
           ("blk.0.ffn_gate_up_exps.g0.weight", member_tensor(128))],
          ["blk.0.ffn_gate_up_exps.g0"])

    # ------------------------------------------------------------------ v1
    # 3: duplicate k — two table entries with the SAME member name.  gguf-py
    # refuses duplicate add_tensor names (MEASURED 2026-10-15: ValueError), so
    # write .g0 + .g1 legally and patch the second name to .g0 in place — the
    # rename is byte-length-preserving, so no offset moves.
    dup_k = args.out_dir / "taillis-r2-bad-dup-k.gguf"
    write_v1(dup_k,
             [("blk.0.ffn_gate_up_exps.g0.weight", member_tensor(64)),
              ("blk.0.ffn_gate_up_exps.g1.weight", member_tensor(64))])
    g1 = b"blk.0.ffn_gate_up_exps.g1.weight"
    g0 = b"blk.0.ffn_gate_up_exps.g0.weight"
    assert len(g1) == len(g0)
    raw = bytearray(dup_k.read_bytes())
    first = raw.find(g1)
    assert first >= 0 and raw.find(g1, first + 1) < 0, "g1 name not unique in file"
    raw[first:first + len(g1)] = g0
    dup_k.write_bytes(bytes(raw))
    assert raw.count(g0) == 2 and raw.count(g1) == 0, "patch did not produce duplicate k"
    print(f"patched {dup_k}  -> two entries named {g0.decode()} (duplicate k)")

    # 4: the logical is grouped by its member AND present flat (surviving check)
    bad_both_v1 = args.out_dir / "taillis-r2-bad-flat-and-grouped.gguf"
    write_v1(bad_both_v1,
             [("blk.0.ffn_gate_up_exps.weight", member_tensor(128)),
              ("blk.0.ffn_gate_up_exps.g0.weight", member_tensor(128))])

    # 5: members' ne[2] sum to 120, not 128 (the surviving sum check, v1-shaped)
    bad_sum_v1 = args.out_dir / "taillis-r2-bad-sum.gguf"
    write_v1(bad_sum_v1,
             [("blk.0.ffn_gate_up_exps.g0.weight", member_tensor(60)),
              ("blk.0.ffn_gate_up_exps.g1.weight", member_tensor(60))])

    # 6: a member of a NON-expert logical — the derivation must ignore it and
    # the unclassified pass must flag it
    bad_nonexpert = args.out_dir / "taillis-r2-non-expert-member.gguf"
    write_v1(bad_nonexpert,
             [("blk.0.attn_q_a.weight", member_tensor(2)),
              ("blk.0.attn_q_a.g0.weight", member_tensor(2))])


if __name__ == "__main__":
    main()
