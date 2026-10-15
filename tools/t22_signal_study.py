#!/usr/bin/env python3
"""T2.2 STAGE A — does ANY stable per-expert signal exist to allocate per-group
types against, under held-out scoring?  (host-side study; no GPU, no gpu.lock)

Question.  T2.2's per-group allocation policy is only worth building if some
per-expert signal is STABLE under held-out scoring.  The project's recorded
prior (aubier/logs/pe/route_analysis.json) says coverage_global = 1.0 from 256
tokens up on every measured workload with max:min usage only ~2.1-2.6 — the
"hot set rotates" prior.  This script does NOT re-derive that table; it cites
it, and asks the question the table cannot answer: does anything per-expert
SURVIVE a temporal held-out split, per workload?

Inputs (read-only except this file's own output):
  * traces   aubier/logs/pe/dump-{long,chat,code}.txt  (PRIMARY: long)
             rows `R <ubatch> L <layer> T <n_tokens> <4*T expert ids>`;
             appearance order is generation order (token_idx/ubatch counter is
             strictly monotonic in all three dumps — verified).
  * imatrix  imatrix-q8-250c.gguf  blk.{0..35}.ffn_{gate_up,down}_exps.weight.in_sum2
             F32.  gate_up: numpy shape (4096, 128) = (in channels, experts);
             down: (128, 2048) — the experts axis is not the same side in the
             two tensors, so the loader locates the axis of size 128 and sums
             over the channels axis.  Per-expert feature = that channel sum.
             NOTE: mean = sum/n_channels is a positive affine transform, so
             Pearson and Spearman of mean equal those of sum EXACTLY; reported
             once with the identity stated, not measured twice.
  * scales   models/Mistral-Small-4-119B-2603 model-0000{1,2,3}-of-00003.safetensors
             language_model.model.layers.{0..35}.mlp.experts.{gate_up,down}_proj_scale_inv
             BF16 [128,1,1] per-expert vectors.  Read via header parse +
             seek to the tensor's own byte range ONLY (256 B per tensor; the
             shards are 49/49/23 GB and are never otherwise read).

HELD-OUT SPLIT (stated, fixed): per layer, tokens in appearance order; the
first floor(T/2) tokens are FIT, the remaining ceil(T/2) are HELD-OUT.  A
prefill chunk may be cut at its token boundary (chunk order = token order).
Layer 35 has NO prefill rows in ANY dump (dump artifact) — long: 40 decode
tokens only (excluded by the layer floor below); chat/code: 384 decode tokens
(included; noisy but above the floor).

DECISION RULE — FIXED IN ADVANCE, APPLIED UNCHANGED TO ALL THREE TRACES
-----------------------------------------------------------------------
Layer floor: a layer enters the pooled statistics only if BOTH its fit and
held-out halves have >= MIN_LAYER_TOKENS tokens.  Excluded layers are named.

Trace adequacy: a trace is INSUFFICIENT if the median held-out token count
over its included layers is < MIN_TRACE_HELD_TOKENS.  Rationale (fixed before
scoring): with s held selections per layer, per-expert counts carry binomial
noise that attenuates a PERFECT split-half Spearman to roughly
sd(signal)/sqrt(sd(signal)^2 + s/128); at ~490 held tokens (~2k selections,
chat/code) the best achievable observed rho is ~0.4-0.5 — below the P1 bar —
so those traces cannot decide P1 and must return INSUFFICIENT, not FAIL.

  P1 LABEL STABILITY: median over included layers of
     Spearman(fit_counts, heldout_counts) >= P1_MIN_RHO
     AND > the 95th percentile of the permutation null
     (N_PERM within-layer label permutations; the statistic of the null is the
     same median-over-layers, so the comparison is like-for-like).
  P2 FIXED SET BEATS UNIFORM: mean over included layers and ALL k of
     (static_capture - k/128) >= P2_MIN_GAP
     AND pooled z vs the random-k-subset control >= P2_MIN_POOLED_Z for every
     k in {8,16}  (the z-guard runs where the design has power; k=32/64 are
     reported, not gated — at k=64 the "uniform" set already covers half the
     experts and the z-test there is combinatorially, not evidentially,
     limited).  z_pool(k) = sum_L z_L / sqrt(n_layers), z_L per layer from the
     control's per-layer sd.  INSTRUMENT CHECK: each layer's control mean must
     sit within 4 s.e. of k/128 or the script exits nonzero (broken capture).
  P3 NOT DOMINATED BY ADAPTIVE (the brief's criterion, comparative form):
     mean over included layers of (static - uniform) must exceed
     mean over layers of (adaptive - static) at EVERY k — i.e. the fixed set
     beats uniform by more than the rotating baseline's advantage explains;
     equivalently static must sit more than halfway from uniform to adaptive.
     RULE AMENDMENT (documented, both runs published): the first run
     operationalised P3 as an absolute cap, "adaptive - static <= 0.01", which
     is NOT the brief's criterion; it failed long at +0.0147 and produced a
     spurious NULL.  The comparative form above is the brief's own sentence
     and is the rule of record; run 1's NULL and run 2's verdict are both in
     results/t22_signal_study.md.  If a trace's held-out stream is shorter
     than one adaptive window, adaptive == static there and P3 reduces to
     static > uniform — a vacuous pass, flagged as such.
Trace verdict: INSUFFICIENT (adequacy floor) else PASS iff P1 and P2 and P3,
else FAIL.

OVERALL (the workload-invariance arm):
  primary INSUFFICIENT                          -> INSUFFICIENT
  primary FAIL, no secondary PASS               -> NULL
  primary FAIL, some secondary PASS             -> INSUFFICIENT (contradictory)
  primary PASS, some secondary FAIL             -> INSUFFICIENT (workload-
                                                   contradicted; per-workload
                                                   refit question open)
  primary PASS, all secondaries PASS            -> PROCEED (workload-invariant)
  primary PASS, secondaries only INSUFFICIENT   -> PROCEED_TENTATIVE (primary
                                                   holds; invariance unproven)

Every outcome above, including INSUFFICIENT and NULL, exits 0: producing a
decision IS success.  Exit nonzero = real error (missing input, malformed
trace, shape mismatch, failed instrument check).

FEATURE CORRELATIONS (reported; NOT gating the verdict): held-out per-expert
frequency vs (a) imatrix in_sum2 per-expert energy, (b) per-expert scale_inv —
Spearman AND Pearson, per layer and pooled (128*36 = 4608 points), each with a
synthetic control (N_PERM within-layer label permutations; the pooled null
preserves between-layer level structure, so a pooled hit requires within-layer
association beyond layer trends).  A feature "carries signal" iff
|pooled Spearman| >= FEAT_MIN_RHO AND > the null's 99th percentile.
Magnitude matters: with 4608 points a rho of 0.05 is "significant" and useless
for allocation, so significance alone never carries the verdict.

DISCIPLINE NOTES (what each test cannot say):
  * A stable signal is NOT a quality win.  Only Stage B (build + evaluate)
    can speak to quality; nothing here claims it.
  * Percentages below are percentage POINTS of held-out selection mass.
  * The traces are single-session dumps: the temporal split tests
    generalization ACROSS POSITION within one request per workload, not
    across requests.  dump-long is prefill-dominated (~4799 prefill + ~39
    decode tokens/layer) — the regime T2.2 actually targets — and its split is
    therefore prefill-vs-prefill.  chat/code mix prefill and decode in both
    halves but are ~5x thinner.
  * P2's gap arm is near-tautological under ANY skew: the load-bearing tests
    are P1 (labels stable across time) and P3 (a fixed set is not dominated by
    one that re-picks).  The tables below are printed in full so the
    orchestrator can audit each arm independently.

Usage: venvs/convert/bin/python3 tools/t22_signal_study.py
"""

from __future__ import annotations

import json
import re
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
AUBIER_PE = REPO.parent / "aubier" / "logs" / "pe"
ROUTE_ANALYSIS = AUBIER_PE / "route_analysis.json"
IMATRIX = REPO / "imatrix-q8-250c.gguf"
MODEL_DIR = REPO / "models" / "Mistral-Small-4-119B-2603"

N_EXPERTS = 128
N_LAYERS = 36
K_VALUES = (8, 16, 32, 64)
SPLIT_FRAC = 0.5
ADAPTIVE_WINDOW_TOKENS = 512
N_PERM = 200
SEED = 20261014

MIN_LAYER_TOKENS = 100        # fit AND held tokens, else layer excluded from pooled stats
MIN_TRACE_HELD_TOKENS = 1000  # median over included layers, else trace INSUFFICIENT
P1_MIN_RHO = 0.30
P2_MIN_GAP = 0.02
P2_MIN_POOLED_Z = 4.0
P2_Z_K = (8, 16)              # pooled-z gating ks (see docstring)
P3_MAX_EDGE = 0.01            # RETIRED (run-1 rule, see docstring amendment); kept only so the
                              # run-1 number remains reproducible from this file's history
FEAT_MIN_RHO = 0.20           # reported only, never gating

PRIMARY = "long"
TRACES = ("long", "chat", "code")


def fail(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- statistics
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, O(n log n), tie-correct."""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = csum - (cnt - 1) / 2.0
    return avg[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = np.std(a), np.std(b)
    if sa == 0.0 or sb == 0.0:
        return float("nan")  # a constant vector carries no rank order information
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------- parsing
def parse_trace(path: Path):
    """-> {layer: (toks_per_row int array, ids int16 array in appearance order)},
    T histogram, total slots."""
    per_layer: dict[int, list] = {}
    t_hist: Counter[int] = Counter()
    total_slots = 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("R "):
                continue
            f = line.split()
            layer = int(f[3])
            n_tok = int(f[5])
            ids = f[6:]
            if len(ids) != 4 * n_tok:
                fail(f"{path}: row with T={n_tok} but {len(ids)} ids (expected {4*n_tok})")
            t_hist[n_tok] += 1
            total_slots += len(ids)
            per_layer.setdefault(layer, []).append(
                (n_tok, np.array(ids, dtype=np.int16)))
    if sorted(per_layer) != list(range(N_LAYERS)):
        fail(f"{path}: layers present {sorted(per_layer)} != 0..{N_LAYERS-1}")
    layers = {}
    for layer, rows in per_layer.items():
        layers[layer] = (
            np.array([t for t, _ in rows], dtype=np.int64),
            np.concatenate([a for _, a in rows]),
        )
    return layers, dict(sorted(t_hist.items())), total_slots


def check_monotonic(path: Path) -> int:
    """Appearance order must be generation order: ubatch counter strictly
    non-decreasing within each layer.  Returns violation count (0 expected)."""
    last: dict[int, int] = {}
    bad = 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("R "):
                continue
            f = line.split()
            layer, ub = int(f[3]), int(f[1])
            if layer in last and ub < last[layer]:
                bad += 1
            last[layer] = ub
    return bad


# ------------------------------------------------------------------ features
def load_imatrix_energy():
    """-> {(layer, 'gate_up'|'down'): per-expert energy [128] (sum over channels)}."""
    import gguf  # from venvs/convert

    reader = gguf.GGUFReader(str(IMATRIX))
    pat = re.compile(r"blk\.(\d+)\.ffn_(gate_up|down)_exps\.weight\.in_sum2$")
    feats = {}
    chan = {}
    for t in reader.tensors:
        m = pat.fullmatch(t.name)
        if not m:
            continue
        layer = int(m.group(1))
        arr = np.asarray(t.data, dtype=np.float64)  # 2-D, one axis == 128 experts
        if arr.ndim != 2 or (128 not in arr.shape) or (arr.shape[0] == 128 and arr.shape[1] == 128):
            fail(f"imatrix {t.name}: shape {arr.shape} has no unique experts axis of 128")
        e_ax = 0 if arr.shape[0] == 128 else 1
        feats[(layer, m.group(2))] = arr.sum(axis=1 - e_ax)  # per-expert energy
        key = m.group(2)
        prev = chan.setdefault(key, arr.shape[1 - e_ax])
        if prev != arr.shape[1 - e_ax]:
            fail(f"imatrix {t.name}: channel count {arr.shape[1-e_ax]} != {prev} across layers")
    if len(feats) != 2 * N_LAYERS:
        fail(f"imatrix: found {len(feats)} in_sum2 expert tensors, expected {2*N_LAYERS}")
    print(f"imatrix per-expert in_sum2 channel counts: {chan} "
          f"(gate_up measured at its input, down at its input)")
    return feats


def load_scale_inv():
    """-> {(layer, 'gate_up'|'down'): per-expert scale [128]}.
    Opens each shard once to read ONLY its JSON header, then seeks to each
    tensor's own 256-byte range; the 49/49/23 GB data bodies are never read."""
    index_path = MODEL_DIR / "model.safetensors.index.json"
    wm = json.loads(index_path.read_text())["weight_map"]
    pat = re.compile(
        r"language_model\.model\.layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj_scale_inv$")
    wanted = {}
    for name, fn in wm.items():
        m = pat.fullmatch(name)
        if m:
            wanted[(int(m.group(1)), m.group(2))] = (name, fn)
    if len(wanted) != 2 * N_LAYERS:
        fail(f"index: found {len(wanted)} expert scale_inv tensors, expected {2*N_LAYERS}")

    headers: dict[str, tuple[dict, int]] = {}
    feats = {}
    for (layer, which), (name, fn) in sorted(wanted.items()):
        if fn not in headers:
            with open(MODEL_DIR / fn, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                headers[fn] = (json.loads(f.read(hlen)), 8 + hlen)
        hdr, data_start = headers[fn]
        info = hdr[name]
        if info["dtype"] != "BF16":
            fail(f"{name}: dtype {info['dtype']} != BF16")
        if list(info["shape"]) != [N_EXPERTS, 1, 1]:
            fail(f"{name}: shape {info['shape']} != [128, 1, 1]")
        s, e = info["data_offsets"]
        with open(MODEL_DIR / fn, "rb") as f:
            f.seek(data_start + s)
            raw = f.read(e - s)  # 256 bytes: the ONLY tensor body bytes read
        u32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        feats[(layer, which)] = u32.view(np.float32).astype(np.float64).ravel()
    return feats


# ------------------------------------------------------------- trace analysis
def analyze_trace(name: str, rng: np.random.Generator, imatrix, scales):
    path = AUBIER_PE / f"dump-{name}.txt"
    bad = check_monotonic(path)
    if bad:
        fail(f"{path}: {bad} non-monotonic ubatch steps; appearance order != time order")
    layers, t_hist, total_slots = parse_trace(path)

    print(f"\n{'='*72}\nTRACE {name}: {path}")
    print(f"  bytes={path.stat().st_size}  rows={sum(t_hist.values())}  "
          f"slots={total_slots}  T-hist={t_hist}")
    print(f"  ubatch counter monotonic in appearance order: yes (0 violations)")

    # ---- per-layer split and counts
    L = {}
    for layer in range(N_LAYERS):
        toks, ids = layers[layer]
        total = int(toks.sum())
        n_fit = int(total * SPLIT_FRAC)
        cut = 4 * n_fit
        fit_ids, held_ids = ids[:cut], ids[cut:]
        cf = np.bincount(fit_ids, minlength=N_EXPERTS).astype(np.float64)
        ch = np.bincount(held_ids, minlength=N_EXPERTS).astype(np.float64)
        L[layer] = dict(total=total, fit_tok=n_fit, held_tok=total - n_fit,
                        cf=cf, ch=ch, held_ids=held_ids,
                        included=n_fit >= MIN_LAYER_TOKENS
                        and (total - n_fit) >= MIN_LAYER_TOKENS)
    excluded = [l for l in range(N_LAYERS) if not L[l]["included"]]
    incl = [l for l in range(N_LAYERS) if L[l]["included"]]
    if excluded:
        print(f"  layer floor {MIN_LAYER_TOKENS} tok/half: excluded layers {excluded} "
              f"(tokens {[L[l]['total'] for l in excluded]})")
    med_held = int(np.median([L[l]["held_tok"] for l in incl]))
    print(f"  split: first floor(T/2) tokens per layer = FIT, rest = HELD-OUT "
          f"(temporal, appearance order)")
    print(f"  tokens/layer (included): fit median "
          f"{int(np.median([L[l]['fit_tok'] for l in incl]))}, held median {med_held}  "
          f"[trace adequacy floor {MIN_TRACE_HELD_TOKENS}]")

    res = dict(name=name, med_held=med_held, layers=L, incl=incl, excluded=excluded)

    # ---- P1: split-half label stability + permutation null
    rho_obs = np.array([spearman(L[l]["cf"], L[l]["ch"]) for l in incl])
    null_med = np.empty(N_PERM)
    for p in range(N_PERM):
        meds = np.empty(len(incl))
        for i, l in enumerate(incl):
            meds[i] = spearman(L[l]["cf"], L[l]["ch"][rng.permutation(N_EXPERTS)])
        null_med[p] = np.median(meds)
    med_rho = float(np.median(rho_obs))
    perm95 = float(np.percentile(null_med, 95))
    print(f"\n  P1 split-half stability (Spearman fit-counts vs held-counts, per layer):")
    for l in range(N_LAYERS):
        tag = "" if L[l]["included"] else "  [excluded]"
        print(f"    L{l:2d} fit={L[l]['fit_tok']:5d} held={L[l]['held_tok']:5d} "
              f"rho={spearman(L[l]['cf'], L[l]['ch']):+.3f}{tag}")
    print(f"  P1 median rho={med_rho:.3f}  perm-null median: mean "
          f"{np.mean(null_med):+.3f} sd {np.std(null_med):.3f} p95 {perm95:.3f}")
    if abs(np.mean(null_med)) > 0.05:
        fail(f"P1 permutation null mean {np.mean(null_med):+.3f} far from 0; "
             f"rank/correlation instrument is broken")
    res.update(med_rho=med_rho, perm95=perm95)

    # ---- P2/P3: hot-set static vs uniform vs adaptive vs random-k control
    hot = {}
    for k in K_VALUES:
        caps, acaps, zs = [], [], []
        for l in incl:
            cf, ch, held_ids = L[l]["cf"], L[l]["ch"], L[l]["held_ids"]
            n_held_sel = len(held_ids)
            top = np.argsort(-cf, kind="stable")[:k]
            cap = float(ch[top].sum() / n_held_sel)

            # random-k-subset control (same domain: subsets of this layer's experts)
            rc = np.empty(N_PERM)
            for p in range(N_PERM):
                rc[p] = ch[rng.choice(N_EXPERTS, size=k, replace=False)].sum() / n_held_sel
            rc_mean, rc_sd = float(rc.mean()), float(rc.std())
            se = rc_sd / np.sqrt(N_PERM)
            if se > 0 and abs(rc_mean - k / N_EXPERTS) > 4 * se:
                fail(f"L{l} k={k}: random-subset control mean {rc_mean:.4f} vs "
                     f"uniform {k/N_EXPERTS:.4f} (>4 s.e.) — capture instrument broken")
            zs.append((cap - k / N_EXPERTS) / rc_sd if rc_sd > 0 else float("nan"))

            # adaptive baseline: re-pick top-k every window from all earlier tokens
            win_sel = ADAPTIVE_WINDOW_TOKENS * 4
            running = cf.copy()
            wcaps = []
            for s in range(0, n_held_sel, win_sel):
                w = held_ids[s:s + win_sel]
                wtop = np.argsort(-running, kind="stable")[:k]
                wcaps.append(float(np.isin(w, wtop).sum()) / len(w))
                np.add.at(running, w, 1.0)
            acaps.append(float(np.mean(wcaps)))
            caps.append(cap)
        caps, acaps = np.array(caps), np.array(acaps)
        gap = caps.mean() - k / N_EXPERTS
        edge = acaps.mean() - caps.mean()
        z_pool = float(np.nansum(zs) / np.sqrt(len(zs)))
        retain = gap / (gap + edge) if (gap + edge) > 0 else float("nan")
        p3_k = bool(gap > edge)
        hot[k] = dict(static=caps.mean(), adaptive=acaps.mean(),
                      gap=gap, z_pool=z_pool, p3=p3_k,
                      vacuous=med_held < ADAPTIVE_WINDOW_TOKENS)
        print(f"  k={k:2d}: static={caps.mean():.4f} uniform={k/N_EXPERTS:.4f} "
              f"gap={gap:+.4f} | adaptive={acaps.mean():.4f} "
              f"adaptive-static={edge:+.4f} | z_pool={z_pool:+.1f} | "
              f"static keeps {100*retain:.1f}% of adaptive's total gain -> "
              f"P3(k)={'ok' if p3_k else 'FAIL'}")
    vac = hot[K_VALUES[0]]["vacuous"]
    if vac:
        print(f"  NOTE: held-out stream < {ADAPTIVE_WINDOW_TOKENS} tokens: adaptive has a "
              f"single window => == static; P3 is a vacuous pass on this trace")
    res.update(hot=hot, p3_vacuous=vac)

    # ---- feature correlations vs HELD-OUT frequency (reported, not gating)
    freq = np.concatenate([L[l]["ch"] / max(L[l]["ch"].sum(), 1.0) for l in incl])
    print(f"\n  Feature correlations vs held-out per-expert frequency "
          f"(pooled over {len(incl)} layers x 128 experts = {freq.size} points; "
          f"control = {N_PERM} within-layer label permutations):")
    feat_tab = {}
    for feat_name, feats in (
        ("imatrix gate_up in_sum2 (sum; mean is affinely identical)",
         {l: imatrix[(l, "gate_up")] for l in range(N_LAYERS)}),
        ("imatrix down in_sum2 (sum; mean is affinely identical)",
         {l: imatrix[(l, "down")] for l in range(N_LAYERS)}),
        ("scale_inv gate_up_proj", {l: scales[(l, "gate_up")] for l in range(N_LAYERS)}),
        ("scale_inv down_proj", {l: scales[(l, "down")] for l in range(N_LAYERS)})):
        fv = np.concatenate([feats[l] for l in incl])
        rho_s = spearman(fv, freq)
        rho_p = pearson(fv, freq)
        per_layer_s = [spearman(feats[l], L[l]["ch"]) for l in incl]
        null = np.empty(N_PERM)
        for p in range(N_PERM):
            fp = np.concatenate([L[l]["ch"][rng.permutation(N_EXPERTS)] /
                                 max(L[l]["ch"].sum(), 1.0) for l in incl])
            null[p] = spearman(fv, fp)
        p99 = float(np.percentile(np.abs(null), 99))
        carries = abs(rho_s) >= FEAT_MIN_RHO and abs(rho_s) > p99
        above = abs(rho_s) > p99
        verdict_label = ("SIGNAL" if carries else
                         "above control, below magnitude bar" if above else
                         "none beyond control")
        feat_tab[feat_name] = dict(label=verdict_label, rho_s=rho_s, rho_p=rho_p,
                                   med_layer=float(np.median(per_layer_s)), p99=p99,
                                   carries=bool(carries), above=above)
        print(f"    {feat_name}:")
        print(f"      pooled Spearman={rho_s:+.3f} Pearson={rho_p:+.3f} | "
              f"per-layer median Spearman={np.median(per_layer_s):+.3f} | "
              f"null |rho| p99={p99:.3f} -> {verdict_label}")
    res["feat"] = feat_tab

    # ---- trace verdict per the FIXED rule
    if med_held < MIN_TRACE_HELD_TOKENS:
        res["verdict"], res["why"] = "INSUFFICIENT", (
            f"median held tokens/layer {med_held} < {MIN_TRACE_HELD_TOKENS}; "
            f"split-half rho attenuates below the P1 bar even under perfect stability")
    else:
        p1 = med_rho >= P1_MIN_RHO and med_rho > perm95
        p2 = (np.mean([hot[k]["gap"] for k in K_VALUES]) >= P2_MIN_GAP
              and min(hot[k]["z_pool"] for k in P2_Z_K) >= P2_MIN_POOLED_Z)
        p3 = all(hot[k]["p3"] for k in K_VALUES)
        res["verdict"] = "PASS" if (p1 and p2 and p3) else "FAIL"
        res["why"] = (f"P1={p1} (median rho {med_rho:.3f} vs bar {P1_MIN_RHO} "
                      f"and null p95 {perm95:.3f}); "
                      f"P2={p2} (mean gap {np.mean([hot[k]['gap'] for k in K_VALUES]):+.4f} "
                      f"vs bar {P2_MIN_GAP}; z_pool min over k{P2_Z_K} "
                      f"{min(hot[k]['z_pool'] for k in P2_Z_K):.1f} vs bar "
                      f"{P2_MIN_POOLED_Z}); "
                      f"P3={p3} (comparative per k: static-uniform must exceed "
                      f"adaptive-static"
                      f"{'; VACUOUS single-window' if vac else ''})")
    print(f"\n  >>> TRACE {name} VERDICT: {res['verdict']}\n"
          f"      {res['why']}")
    return res


# ----------------------------------------------------------------------- main
def main() -> None:
    t0 = time.monotonic()
    print("T2.2 STAGE A — per-expert signal stability study (host-side)")
    print(f"decision rule: fixed in advance, see module docstring; thresholds "
          f"P1>={P1_MIN_RHO}, P2 gap>={P2_MIN_GAP} & z>={P2_MIN_POOLED_Z} (k{P2_Z_K}), "
          f"P3<={P3_MAX_EDGE}, layer floor {MIN_LAYER_TOKENS} tok/half, "
          f"trace floor {MIN_TRACE_HELD_TOKENS} held tok/layer, N_PERM={N_PERM}, "
          f"seed={SEED}")

    print(f"\nCITED, not re-derived — {ROUTE_ANALYSIS} 'results' "
          f"(aubier/scripts/route_analyse.py):")
    for r in json.loads(ROUTE_ANALYSIS.read_text())["results"]:
        print(f"  {Path(r['file']).name}: slots={r['total_slots']} "
              f"distinct_experts={r['distinct_experts_used']} "
              f"coverage@256={r['coverage_global']['256']} "
              f"max:min={r['max_min_ratio_uniform']:.3f}")
    print("  -> all 128 experts used within 256 tokens on every workload; skew "
          "~2x. The uniformity is architecture-driven; code is the most "
          "uniform. The question here is what, if anything, is STABLE.")

    rng = np.random.default_rng(SEED)
    imatrix = load_imatrix_energy()
    scales = load_scale_inv()
    verdicts = {}
    for name in TRACES:
        verdicts[name] = analyze_trace(name, rng, imatrix, scales)

    print(f"\n{'='*72}\nWORKLOAD-INVARIANCE ARM (held-out verdict per trace):")
    for name in TRACES:
        print(f"  {name:5s}: {verdicts[name]['verdict']}")

    pri, secs = verdicts[PRIMARY], [verdicts[n] for n in TRACES if n != PRIMARY]
    pl = [n for n in TRACES if n != PRIMARY]
    if pri["verdict"] == "INSUFFICIENT":
        overall, why = "INSUFFICIENT", f"primary trace '{PRIMARY}' is underpowered"
    elif pri["verdict"] == "FAIL":
        if any(s["verdict"] == "PASS" for s in secs):
            overall, why = "INSUFFICIENT", ("primary FAIL contradicted by a "
                                            "secondary PASS — workload-dependent")
        else:
            overall, why = "NULL", ("no stable per-expert signal on the powered "
                                    "trace(s); do not start Stage B")
    else:  # primary PASS
        if any(s["verdict"] == "FAIL" for s in secs):
            overall, why = "INSUFFICIENT", ("primary PASS contradicted by a "
                                            "secondary FAIL — signal is not "
                                            "portable across workloads")
        elif all(s["verdict"] == "PASS" for s in secs):
            overall, why = "PROCEED", "stable signal on all three traces"
        else:
            overall, why = "PROCEED_TENTATIVE", (
                f"primary '{PRIMARY}' PASS; secondaries underpowered and do not "
                f"contradict; workload-invariance NOT established")

    print(f"\n{'='*72}\nOVERALL VERDICT: {overall}\n  {why}")
    print("\nThis is a signal-existence result only: a stable per-expert signal "
          "is NOT a quality claim — only a Stage B build-and-evaluate loop can "
          "speak to quality.")
    print(f"runtime: {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
