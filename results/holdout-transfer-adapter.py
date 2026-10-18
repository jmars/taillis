#!/usr/bin/env python3
"""ADAPTER (part B of taillis-v4): aubier's per-position route dump -> the
format tools/t22b_cross_request.py reads.

AUBIER (read-only, ~/futaie/aubier/logs/engine/m1-u6/trace-*.txt), one row per
(layer, position):
    R <layer> <pos> <sel0..3> <w0..w3>          <- the 4 ids are 8 fields in
                                                   (2 ids), not T tokens
OURS (tools/t22_signal_study.parse_trace / t22b):
    R <ubatch> L <layer> T <n_tokens> <4*n_tokens ids>

ADAPTER DECISION (stated, because it is the one judgement here): aubier's rows
carry EXACTLY ONE position each, so there is no T>1 row and therefore no
row-shape signal separating prefill from decode.  Inventing T>1 by grouping
consecutive positions would assert ubatch boundaries aubier never measured.
Instead the phase is carried by WHICH FILE SET is written: `--mode prompt`
emits only positions < n_prompt, `--mode decode` only positions >= n_prompt,
`--mode whole` everything.  Every emitted row is T=1 and exactly 4 ids.
CONSEQUENCE, and it must be read with every number: t22b derives its regime
label from T, so in a phase-filtered set the populated "regime decode" block
IS the filtered phase (prompt or decode as named by --mode), and the
"regime prefill" block is structurally empty.  The phase is named by the run,
never by t22b's regime label.

n_prompt is not hardcoded and not taken from a brief: it is READ from aubier's
own run record (*.stdout.txt, "n_prompt=N + n_decode=128 = M positions").
Weights are used only as an instrument check (a router's 4 selected weights sum
to 1); they are discarded from the output, as our format has no weight column.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUBIER = Path.home() / "futaie" / "aubier" / "logs" / "engine" / "m1-u6"
N_EXPERTS = 128
N_LAYERS = 36
NAMES = ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4")


def trace_path(name: str) -> Path:
    return AUBIER / (f"trace-{name}.txt" if name.startswith("A") else f"trace-ctrl_{name}.txt")


def read_profile(name: str) -> tuple[int, int]:
    """(n_prompt, n_decode) from aubier's own stdout record — measured, not cited."""
    txt = trace_path(name).with_suffix(".stdout.txt").read_text()
    m = re.search(r"n_prompt=(\d+) \+ n_decode=(\d+) = (\d+) positions", txt)
    if not m:
        sys.exit(f"FATAL: {name}: no 'n_prompt=... positions' line in its stdout record")
    return int(m.group(1)), int(m.group(2))


def load(name: str):
    """-> (per[(layer)] = {pos: (ids[4])}, stats).  Validates the columns."""
    path = trace_path(name)
    per: dict[int, dict[int, tuple[int, ...]]] = {}
    rows = 0
    for line in path.open():
        if not line.startswith("R "):
            continue
        p = line.split()
        if len(p) != 11:
            sys.exit(f"FATAL: {path}: row with {len(p)} fields, expected 11: {line!r}")
        layer, pos = int(p[1]), int(p[2])
        ids = tuple(int(x) for x in p[3:7])
        w = [float(x) for x in p[7:11]]
        if len(set(ids)) != 4:
            sys.exit(f"FATAL: {path}: L{layer} pos{pos}: repeated expert id {ids}")
        if any(not (0 <= i < N_EXPERTS) for i in ids):
            sys.exit(f"FATAL: {path}: L{layer} pos{pos}: ids {ids} outside [0,{N_EXPERTS})")
        if abs(sum(w) - 1.0) > 1e-3:
            sys.exit(f"FATAL: {path}: L{layer} pos{pos}: weights {w} sum {sum(w):.6f} != 1")
        if pos in per.setdefault(layer, {}):
            sys.exit(f"FATAL: {path}: duplicate row L{layer} pos{pos}")
        per[layer][pos] = ids
        rows += 1
    if sorted(per) != list(range(N_LAYERS)):
        sys.exit(f"FATAL: {path}: layers {sorted(per)} != 0..{N_LAYERS - 1}")
    nprompt, ndecode = read_profile(name)
    npos = {len(v) for v in per.values()}
    if npos != {nprompt + ndecode}:
        sys.exit(f"FATAL: {path}: positions/layer {npos} != n_prompt+n_decode "
                 f"= {nprompt + ndecode}")
    if {p for v in per.values() for p in v} != set(range(nprompt + ndecode)):
        sys.exit(f"FATAL: {path}: positions are not exactly 0..{nprompt + ndecode - 1}")
    return per, dict(rows=rows, n_prompt=nprompt, n_decode=ndecode,
                     positions_per_layer=nprompt + ndecode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("prompt", "decode", "whole"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    print(f"ADAPTER mode={a.mode} -> {a.out}")
    print(f"{'trace':5s} {'prompt':>6s} {'decode':>6s} {'kept/L':>6s} {'rows':>6s} {'ids':>9s} {'w-sum':>7s}")
    for name in NAMES:
        per, st = load(name)
        lo, hi = (0, st["n_prompt"]) if a.mode == "prompt" else (
            (st["n_prompt"], st["positions_per_layer"]) if a.mode == "decode" else
            (0, st["positions_per_layer"]))
        kept = 0
        out = a.out / f"dump-{name}.txt"
        with out.open("w") as fh:
            fh.write(f"# ADAPTER taillis-v4: aubier {trace_path(name).name} mode={a.mode} "
                     f"n_prompt={st['n_prompt']} n_decode={st['n_decode']} "
                     f"kept positions [{lo},{hi}) of {st['positions_per_layer']}\n")
            for layer in range(N_LAYERS):
                for pos in range(lo, hi):
                    ids = per[layer][pos]
                    fh.write(f"R {pos} L {layer} T 1 {' '.join(map(str, ids))}\n")
                kept = len(per[layer])
        # read the written file back: it is the artifact the rule consumes
        back = 0
        for line in out.open():
            if line.startswith("R "):
                back += 1
        expect = N_LAYERS * (hi - lo)
        if back != expect:
            sys.exit(f"FATAL: {out}: wrote {back} rows, expected {expect}")
        summary[name] = dict(**{k: v for k, v in st.items() if k != "rows"},
                             mode=a.mode, range=[lo, hi],
                             tokens_per_layer=hi - lo, rows_written=back,
                             source_rows=st["rows"])
        print(f"{name:5s} {st['n_prompt']:6d} {st['n_decode']:6d} {hi-lo:6d} {back:6d} "
              f"[0,128)  1.000")
    (a.out / "adapter-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"rows/trace/layer = {sorted({v['tokens_per_layer'] for v in summary.values()})}"
          f"  (floor per EVAL trace is {2 * 100} tokens/layer: 100 in EACH half)")
    print(f"summary: {a.out / 'adapter-summary.json'}")


if __name__ == "__main__":
    main()
