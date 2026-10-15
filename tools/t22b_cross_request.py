#!/usr/bin/env python3
"""T2.2 STAGE A+ — THE CROSS-REQUEST TEST.  (host-side; no GPU, no gpu.lock)

Question.  taillis-t22a established that WITHIN one request the per-expert
selection profile is strongly stable (per-layer split-half Spearman 0.72-0.84
on dump-long; a fixed top-k keeps 93.4-96.1% of what per-window adaptive
reselection captures).  But a shipped allocation policy is fixed per MODEL,
not per request.  The deciding question for T2.2 Stage B is therefore: do
DIFFERENT requests agree on WHICH experts are hot?  If they do not, a fixed
per-group type allocation cannot help no matter how stable any single request
is, and Stage B must not be built.

Inputs — the FOUR INDEPENDENT traces under aubier/logs/pe/ (read-only),
rows `R <ubatch> L <layer> T <n_tokens> <4*T expert ids>`:

  EXCLUDED, with measured provenance (this run re-measured, did not just cite):
    dump-all.txt       35,085 lines — EXACT union of chat+code+long (all
                       routing rows of those three are contained in it; it is
                       NOT a union of all four — gate's rows are absent).
                       Using it as further "requests" would double-count three
                       of the four traces and MANUFACTURE cross-trace agreement.
    dump-all.nodec.txt 34,571 lines — separate earlier no-decode collection
                       variant; near-zero routing-row overlap with the four
                       (5,007/34,354 shared).  Not an admissible same-config
                       session, and not one of the four independent traces.
    dump-all.crashed.txt 108 lines — fragment of a crashed run.
  The four used here are split/collected as: chat, code, long are the three
  workload splits of one server run (their union == dump-all.txt); gate is a
  separate run.  The brief named these four as the independent sessions;
  gate turns out CORRUPT (below), so the usable n is THREE.

PROVENANCE CAVEAT (measured this run, applies to the whole T2.2 evidence
  chain incl. t22a, which used dump-long unqualified):
  * chat, code, long were collected from ONE server process
    (aubier scripts/route_session.py session6: builds/hip-route with
    GGML_ROUTE_DUMP=ON, port 8753, temperature 0, seed 42) — the ONLY
    difference between the three traces is the REQUEST: w1_chat = a ~80-token
    essay prompt + its 384 generated tokens; w2_code = a Python-function
    prompt + its ~384 generated tokens; w3_longctx = ~4.8k tokens of C++
    source (llama-graph.cpp lines 1999-2330) + a question.  Each trace's
    "prefill" regime rows are the batched evaluations of that request's
    context (e.g. chat: one T=80 prompt ubatch + one T=512 chunk); its
    "decode" rows are the generated tokens.  Cross-trace comparisons are
    therefore internally consistent, and this is the right design for the
    cross-REQUEST question — but see the next caveat.
  * session6.log:24 records GATE2 stock-vs-patched-ON: FAIL — the dump-ON
    server's OUTPUT TEXT differed from stock on the gate prompts (dump OFF
    was output-identical, GATE1 PASS).  So the captured routing may differ
    from stock routing.  This caveat applies EQUALLY to all three traces
    (structure-of-agreement findings survive it), but absolute hot sets
    should not be quoted as stock routing, and any downstream use (incl.
    t22a's absolute numbers) carries it.

PROBE ARTIFACT (measured this run; forced an input-cleaning step, NOT a rule
  change — thresholds untouched): every one of the three usable traces
  contains EXACTLY ONE byte-identical 512-token ubatch per layer (same T,
  same ids).  It is 512/592 = 86.5% of chat's and code's prefill tokens/layer
  and 512/4799 = 10.7% of long's.  Counting it twice would MANUFACTURE
  cross-request agreement (a "transfer" that is the same tokens capturing
  themselves).  This script therefore removes, from every trace, any ubatch
  row whose (T, ids) digest also appears in another trace — uniformly, with
  per-trace drop counts printed, BEFORE any statistic is computed.  Decode
  rows are verified to contain zero cross-trace duplicates.  Consequence:
  chat/code prefill drop to 80/86 unique tokens/layer — BELOW
  MIN_LAYER_TOKENS=100 — so the prefill regime has no evaluable pairs left
  and returns INSUFFICIENT(structural) under the unchanged rule; the decode
  regime (genuinely distinct generated content, 383 tokens/layer) is the
  only adequate cross-request comparison this trace set can support.
  t22a INHERITANCE: t22a's numbers were computed on the SAME undeduplicated
  streams (its long prefill carried the probe at 10.7% of tokens; its
  chat/code whole-stream numbers were probe-dominated) — its headline
  conclusions on long are directionally unlikely to flip at that dilution,
  but the inheritance is real and is recorded in the results write-up.

REGIME ASYMMETRY IS CONTROLLED, NOT POOLED: chat/code are decode-dominated
(T==1 rows), long is prefill-dominated (T>1 rows, ~97% of tokens).  A pooled
number would confound REQUEST differences with REGIME differences.  Every
comparison below is therefore WITHIN regime:
  decode  = rows with T==1  -> eligible traces {chat, code}
            (gate: CORRUPT, excluded whole — see below; long excluded:
             ~39 decode tokens/layer < MIN_LAYER_TOKENS)
  prefill = rows with T>1   -> eligible traces {chat, code, long}
            (gate: zero prefill rows even before corruption)
No pooled-across-regime statistic is produced anywhere in this script.

DATA FINDING (measured this run, supersedes the brief's assumption):
  dump-gate.txt is CORRUPT and is excluded WHOLE: 4,963/4,975 routing rows
  contain out-of-range "expert ids" (-1072730037, 1029760064, ...) whose u32
  bit patterns decode as float32 values — the gate collection run's dumper
  wrote the wrong tensor.  Only the first ~12 rows are sane.  Salvaging the
  12 good rows (~0.3 tokens/layer) would fabricate a trace from cherry-picked
  rows; the gate workload therefore contributes NOTHING here, the usable n is
  THREE traces (briefed n=4), and the decode regime rests on ONE workload
  pair (chat<->code, two ordered directions) — annotated THIN below.  The
  decode-regime verdict cannot by itself establish cross-REQUEST (as opposed
  to cross-PAIR) agreement.  A re-collect of the gate workload on the GPU
  would be needed to restore the briefed n=4.

DEFINITIONS (per layer, per regime; "selections" = the 4 routed expert ids of
a token; captured mass is a fraction of selection SLOTS):
  U      = k/128, the uniform expectation.
  topA   = per-expert counts of stream A -> top-k expert set (argsort, stable).
  C_X->Y = captured mass on Y of the top-k set FIT on X's FULL stream
           (cross-request transfer).  NOTE the deliberate asymmetry: X fits on
           its full stream while the within-trace diagonal D fits on half —
           this biases FOR transfer, so a collapse under it is conclusive.
  D_Y    = within-trace ceiling: top-k fit on Y's FIRST half (appearance
           order), captured mass on Y's SECOND half.  Same split as t22a.
  z_pool_cross    = sum_L z_L / sqrt(n_L), z_L = (C_L - U)/sd of the
           random-k-subset control ON Y'S OWN STREAM (the control carries the
           stream's discreteness/overdispersion; a binomial would understate
           sd because a token's 4 experts are distinct and correlated).
  z_pool_within   = same pooling for D_L vs a control on Y's held half.
  overlap  |A ∩ B|/k and Jaccard per layer, pooled as the mean over layers;
  null     = per layer, 200 random k-subsets vs Y's actual top-k, pooled the
           same way (like-for-like).  Under H0 "X's picks are unrelated to
           Y's", X's top-k behaves w.r.t. Y like a random k-subset — so the
           null randomises ONE side only, keeping Y's actual top-k.

DECISION RULE — FIXED IN ADVANCE, APPLIED UNCHANGED (thresholds are inherited
project bars, not invented here; provenance stated):
  MIN_LAYER_TOKENS = 100 (t22a's layer floor).  A layer enters pair (X->Y)
    iff X has >= 100 full-stream tokens there AND Y has >= 100 in BOTH halves
    (both C and D must be defined on the same layer set).
  Z_BAR = 4.0 (t22a's P2_MIN_POOLED_Z).  RETAIN_MIN = 0.5 (t22a's P3
    "more than halfway" principle, applied to the within-trace refit instead
    of the adaptive baseline).  N_PERM = 200 (brief).
  Per ordered pair, per k in {8,16,32,64}, per regime:
    INSUFFICIENT   no layer passes the floors, or z_pool_within < Z_BAR
                   (Y's own within-trace advantage is not established, so
                   transfer onto it cannot be judged);
    COLLAPSE       z_pool_cross < Z_BAR  — cross capture is statistically
                   indistinguishable from the uniform expectation on Y;
    WEAK_TRANSFER  z_pool_cross >= Z_BAR but retain < RETAIN_MIN, where
                   retain = (C - U)/(D - U) — transfer is real but keeps less
                   than half of what within-request calibration achieves;
    TRANSFERS      z_pool_cross >= Z_BAR and retain >= RETAIN_MIN.
  Per regime, per k (gated at k in {8,16}; 32/64 reported, not gated — same
    rationale as t22a): all evaluable TRANSFERS -> GENERALISES; all evaluable
    TRANSFERS/WEAK_TRANSFER with some WEAK -> GENERALISES_WEAK; all evaluable
    COLLAPSE -> DO_NOT_BUILD; transfer mixed with COLLAPSE -> INSUFFICIENT
    (pair-contradicted); no evaluable pair -> INSUFFICIENT (structural).
    k=8 vs k=16 disagree -> INSUFFICIENT (k-contradicted).
  OVERALL over the two regimes (mapping AMENDED after run 1 — see the
    RULE AMENDMENT note at the end of this section):
    no evaluable regime -> INSUFFICIENT;
    all evaluable DO_NOT_BUILD, supported by >= 1 non-THIN regime
      -> DO_NOT_BUILD (a fixed per-group allocation does not generalise;
      Stage B must not be built on this evidence);
    all evaluable DO_NOT_BUILD, THIN-supported only -> INSUFFICIENT (a kill
      may not rest on a single workload pair);
    any DO_NOT_BUILD alongside a generalising regime -> INSUFFICIENT
      (regime-contradicted); a merely-INSUFFICIENT regime elsewhere does
      NOT block the kill (t22a precedent: a FAIL with only-INSUFFICIENT
      secondaries still decides);
    all GENERALISES -> PROCEED_TENTATIVE; all GENERALISES or
    GENERALISES_WEAK -> PROCEED_TENTATIVE_WEAK; PROCEED_* likewise require
    >= 1 non-THIN supporting regime, else INSUFFICIENT; else INSUFFICIENT.
    Every PROCEED_* is explicitly TENTATIVE: the usable n (3 requests from
    ONE server run; gate corrupt) cannot establish per-MODEL generality —
    see below.

  RULE AMENDMENT (documented, both runs published): run 1 left the overall
    mapping incomplete — its code dropped INSUFFICIENT regimes from the
    overall set and had no THIN-support guard, so a THIN single-pair decode
    regime alone produced a global DO_NOT_BUILD while the richer prefill
    regime was INSUFFICIENT(mixed).  That resolution was never part of the
    stated rule and claims more than the design can support; the guards in
    this mapping are the rule of record.  Run 1's output is preserved
    verbatim as results/t22b_cross_request_run1_literal_rule.txt; run 2
    (mapping fixed, inputs still undeduplicated) as
    results/t22b_cross_request_run2_amended_mapping.txt.  RUN 3 = rule of
    record: same mapping plus the probe-artefact dedup above (an input
    correction, not a threshold change), preserved as
    results/t22b_cross_request_run3_rule_of_record.txt.

ARM B (overlap) is reported per layer, pooled, with its null, but does NOT
gate the verdict: the headline is the operational ARM C, per the brief.
WHAT WOULD MAKE THIS TEST FAIL: COLLAPSE across the pairs of a regime — that
is the outcome the rule is built to detect, and it is a real possibility the
within-request stability of t22a does NOT rule out (shared hot sets could be
a per-request phenomenon).

WHAT THE USABLE n CANNOT SUPPORT (stated here because the rule cannot):
  * per-MODEL generality — 3 requests, 1 server run, 1 model (and a
    dump-patch build whose output differs from stock, GATE2 FAIL);
  * any QUALITY claim — stability/transfer of selection mass says nothing
    about perplexity or output quality; only a Stage B build-and-evaluate
    loop can say that, and it is gated on this result;
  * a tight estimate of retain — each retain divides two noisy pooled gaps.
  What would settle it properly: a MULTI-SESSION trace collected over the
  real corpus (a GPU collection run, dozens of independent sessions), split
  per session, with this same rule applied unchanged.  That is the honest
  next step rather than a bigger claim from 3 usable sessions.

Usage: venvs/convert/bin/python3 tools/t22b_cross_request.py
       (every outcome incl. INSUFFICIENT/DO_NOT_BUILD exits 0: producing a
        decision IS success.  Exit nonzero = real error: missing/malformed
        trace, layer set mismatch, failed instrument check.)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_signal_study import N_EXPERTS, N_LAYERS, check_monotonic, fail, parse_trace

REPO = Path(__file__).resolve().parent.parent
# PATH-ONLY parameterisation (taillis-trace Stage A++): the RULE and its
# thresholds are untouched — only where traces are read from, which trace
# names are used, and where the detail JSON is written. Unset env = the
# original behaviour, byte-for-byte.
AUBIER_PE = Path(os.environ.get("T22B_TRACE_DIR",
                                str(REPO.parent / "aubier" / "logs" / "pe")))
OUT_JSON = Path(os.environ.get(
    "T22B_OUT_JSON", str(REPO / "results" / "t22b_cross_request_detail.json")))

TRACES = (tuple(os.environ["T22B_TRACES"].split(","))
          if os.environ.get("T22B_TRACES")
          else ("chat", "code", "gate", "long"))
REGIMES = ("decode", "prefill")
K_ALL = (8, 16, 32, 64)
K_GATED = (8, 16)
N_PERM = 200
SEED = 20261014
MIN_LAYER_TOKENS = 100  # t22a's floor
Z_BAR = 4.0             # t22a's P2_MIN_POOLED_Z
RETAIN_MIN = 0.5        # t22a's P3 halfway principle


# ------------------------------------------------------------------- loading
def counts_of(ids: np.ndarray) -> np.ndarray:
    return np.bincount(ids, minlength=N_EXPERTS).astype(np.float64)


def _row_digest(T: int, ids: np.ndarray) -> bytes:
    return T.to_bytes(4, "little") + hashlib.md5(ids.tobytes()).digest()


def parse_rows(name: str):
    """Validate + parse.  Returns dict(rows=, t_hist=, total_slots=,
    total_rows=) with rows[layer] = list of (T, ids_array) in appearance
    order, or dict(corrupt=True, ...) if the file has out-of-range ids —
    corrupt traces are excluded WHOLE, never row-skipped."""
    path = AUBIER_PE / f"dump-{name}.txt"
    bad = check_monotonic(path)
    if bad:
        fail(f"{path}: {bad} non-monotonic ubatch steps; appearance != time order")
    bad_rows = 0
    total_rows = 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("R "):
                continue
            total_rows += 1
            f = line.split()
            n_tok = int(f[5])
            if len(f) - 6 != 4 * n_tok:
                fail(f"{path}: row with T={n_tok} but {len(f)-6} ids")
            if any(not (0 <= int(v) < N_EXPERTS) for v in f[6:]):
                bad_rows += 1
    if bad_rows:
        return dict(corrupt=True, bad_rows=bad_rows, total_rows=total_rows)
    layers, t_hist, total_slots = parse_trace(path)
    rows = {}
    for layer, (toks, ids) in layers.items():
        ends = np.cumsum(4 * toks)
        beg = ends - 4 * toks
        rows[layer] = [(int(t), ids[b:e])
                       for t, b, e in zip(toks, beg, ends)]
    return dict(corrupt=False, rows=rows, t_hist=t_hist,
                total_slots=total_slots, total_rows=total_rows)


def build_dup_index(parsed: dict) -> dict[int, set]:
    """layer -> set of MULTI-TOKEN row digests (T > 1) that appear in >= 2
    traces (shared ubatches such as the route-dump probe).  T==1 rows are
    exempt: a 4-expert combination recurs across independent requests by
    chance (birthday over C(128,4) with ~13.8k draws/trace/layer — the
    observed 13/8/7 collisions in ~35 layers are that noise), and dropping
    them would remove genuine independent data.  A T>1 row with identical
    ids cannot collide by chance; it can only be the same content."""
    idx: dict[int, dict] = {}
    for l in range(N_LAYERS):
        seen: dict[bytes, set] = {}
        for name, p in parsed.items():
            for T, ids in p["rows"][l]:
                if T > 1:
                    seen.setdefault(_row_digest(T, ids), set()).add(name)
        idx[l] = {d for d, s in seen.items() if len(s) >= 2}
    return idx


def assemble(name: str, p, dup: dict[int, set]):
    """Drop cross-trace duplicate ubatches (probe artefact), split regimes."""
    rows = p["rows"]
    reg: dict[str, dict[int, dict]] = {r: {} for r in REGIMES}
    dropped = {r: [0, 0] for r in REGIMES}  # rows, tokens
    n_decode = n_prefill = 0
    for layer in range(N_LAYERS):
        acc = {r: [] for r in REGIMES}
        for T, ids in rows[layer]:
            r = "decode" if T == 1 else "prefill"
            if T > 1 and _row_digest(T, ids) in dup[layer]:
                dropped[r][0] += 1
                dropped[r][1] += T
                continue
            acc[r].append((T, ids))
            if r == "decode":
                n_decode += 1
            else:
                n_prefill += 1
        for r in REGIMES:
            if acc[r]:
                reg[r][layer] = dict(
                    toks=np.array([t for t, _ in acc[r]], dtype=np.int64),
                    ids=np.concatenate([i for _, i in acc[r]]))
    return dict(name=name, reg=reg, total_slots=p["total_slots"],
                n_decode=n_decode, n_prefill=n_prefill, dropped=dropped)


def prep_trace(tr):
    """Per (regime, layer): full/fit/held counts + top-k orderings."""
    out = {}
    for r in REGIMES:
        for l, d in tr["reg"][r].items():
            ids, toks = d["ids"], d["toks"]
            total = int(toks.sum())
            n_fit = total // 2
            cut = 4 * n_fit  # same token-boundary cut as t22a
            fit, held = ids[:cut], ids[cut:]
            c, cf, ch = counts_of(ids), counts_of(fit), counts_of(held)
            out[(r, l)] = dict(tok_total=total, n_fit=n_fit, n_held=total - n_fit,
                               c=c, cf=cf, ch=ch,
                               order=np.argsort(-c, kind="stable"),
                               orderf=np.argsort(-cf, kind="stable"))
    return out


def draw_masks(rng: np.random.Generator):
    """Per (layer, k): N_PERM random k-subsets, shared across pairs/uses."""
    masks = {}
    for l in range(N_LAYERS):
        masks[l] = {}
        for k in K_ALL:
            p = rng.random((N_PERM, N_EXPERTS))
            masks[l][k] = np.argsort(p, axis=1)[:, :k]
    return masks


# ------------------------------------------------------------------- verdict
def pair_verdict(L, z_within, z_cross, retain):
    if not L:
        return "INSUFFICIENT", "no layer passes the floors"
    if not (z_within >= Z_BAR):
        return "INSUFFICIENT", (f"ceiling not established on eval trace "
                                f"(z_pool_within={z_within:.2f} < {Z_BAR})")
    if z_cross >= Z_BAR:
        if retain >= RETAIN_MIN:
            return "TRANSFERS", (f"z_pool_cross={z_cross:.2f} >= {Z_BAR}, "
                                 f"retain={retain:.2f} >= {RETAIN_MIN}")
        return "WEAK_TRANSFER", (f"z_pool_cross={z_cross:.2f} >= {Z_BAR} but "
                                 f"retain={retain:.2f} < {RETAIN_MIN}")
    return "COLLAPSE", (f"z_pool_cross={z_cross:.2f} < {Z_BAR}: capture "
                        f"indistinguishable from uniform")


def k_verdict(vs: list[str]) -> str:
    ev = [v for v in vs if v != "INSUFFICIENT"]
    if not ev:
        return "INSUFFICIENT"
    if all(v == "TRANSFERS" for v in ev):
        return "GENERALISES"
    if all(v in ("TRANSFERS", "WEAK_TRANSFER") for v in ev):
        return "GENERALISES_WEAK"
    if all(v == "COLLAPSE" for v in ev):
        return "DO_NOT_BUILD"
    return "INSUFFICIENT"  # transfer mixed with collapse


def combine_ks(cls_by_k: dict[int, str]) -> str:
    a, b = cls_by_k[K_GATED[0]], cls_by_k[K_GATED[1]]
    if a == b:
        return a
    if {a, b} == {"GENERALISES", "GENERALISES_WEAK"}:
        return "GENERALISES_WEAK"
    return "INSUFFICIENT"  # k-contradicted


def main() -> None:
    t0 = time.monotonic()
    print("T2.2 STAGE A+ — CROSS-REQUEST test (host-side)")
    print(f"rule (FIXED IN ADVANCE, full statement in module docstring): "
          f"floors {MIN_LAYER_TOKENS} tok/layer (t22a), Z_BAR={Z_BAR} "
          f"(t22a P2_MIN_POOLED_Z), RETAIN_MIN={RETAIN_MIN} (t22a P3 halfway), "
          f"N_PERM={N_PERM}, seed={SEED}; gated k{K_GATED}; everything WITHIN "
          f"regime (decode=T==1, prefill=T>1); no pooled-across-regime number.")

    rng = np.random.default_rng(SEED)
    masks = draw_masks(rng)

    parsed = {}
    for n in TRACES:
        p = parse_rows(n)
        if p["corrupt"]:
            print(f"\nDATA FINDING — {AUBIER_PE / f'dump-{n}.txt'} is CORRUPT and "
                  f"is EXCLUDED WHOLE: {p['bad_rows']}/{p['total_rows']} routing rows "
                  f"contain out-of-range expert ids (u32 bit patterns of float32 "
                  f"values — broken dumper in that collection run).  Rows are "
                  f"never skipped selectively: a trace built from cherry-picked "
                  f"rows would be fabricated.  See module docstring.")
            continue
        parsed[n] = p
    if len(parsed) < 2:
        fail(f"only {len(parsed)} usable traces; the cross-request test needs >= 2")
    used = tuple(parsed)
    dup = build_dup_index(parsed)
    n_dup_layers = sum(1 for l in range(N_LAYERS) if dup[l])
    raw = {}
    for n in used:
        raw[n] = assemble(n, parsed[n], dup)
    dd = {n: raw[n]["dropped"] for n in used}
    dec_dups = {n: dd[n]["decode"][0] for n in used}
    if any(v > 0 for v in dec_dups.values()):
        print(f"NOTE: decode regime had cross-trace duplicate ubatches: "
              f"{dec_dups} (dropped; they were shared content, not independent "
              f"requests)")
    print(f"\nUSABLE TRACES: {used}  (briefed four minus corrupt/derived; "
          f"usable n = {len(used)})")
    print(f"DEDUP (probe artefact): cross-trace duplicate ubatches present at "
          f"{n_dup_layers}/{N_LAYERS} layers; dropped (rows, tokens) per trace: "
          + "; ".join(f"{n} decode={tuple(dd[n]['decode'])} "
                      f"prefill={tuple(dd[n]['prefill'])}" for n in used)
          + "  — prefill tokens drop because every trace carried the SAME "
          "512-token ubatch; counting it in two traces would manufacture "
          "agreement (see module docstring PROBE ARTIFACT).")
    prepped = {n: prep_trace(raw[n]) for n in used}

    # ---- exclusion statement (re-measured this run, not cited)
    print("\nEXCLUDED as further 'requests' (not independent sessions; would "
          "double-count and MANUFACTURE agreement):")
    print("  dump-all.txt / dump-all.nodec.txt / dump-all.crashed.txt — "
          "see module docstring for measured provenance (dump-all.txt is an "
          "exact union of chat+code+long; gate is independent of it; nodec is "
          "a separate earlier no-decode collection; crashed is a 108-line "
          "fragment).")

    # ---- sizes per trace/regime
    print(f"\n{'='*72}\nTRACE SIZES after dedup (rows; tokens per layer over "
          f"layers present, min..median; floor {MIN_LAYER_TOKENS}):")
    for n in used:
        tr = raw[n]
        line = f"  {n:5s}: rows={tr['n_decode'] + tr['n_prefill']:6d} " \
               f"(decode {tr['n_decode']}, prefill {tr['n_prefill']})  slots={tr['total_slots']}"
        for r in REGIMES:
            ls = prepped[n]
            toks = [ls[(r, l)]["tok_total"] for l in range(N_LAYERS) if (r, l) in ls]
            if toks:
                line += f" | {r}: {min(toks):5d}..{int(np.median(toks)):5d} tok/L " \
                        f"({sum(t >= MIN_LAYER_TOKENS for t in toks)}/{len(toks)} L >= floor)"
            else:
                line += f" | {r}: none"
        print(line)

    # ---- eligible sets and ordered pairs per regime
    reg_pairs: dict[str, list[tuple[str, str]]] = {}
    for r in REGIMES:
        names = [n for n in used
                 if any(prepped[n][(r, l)]["tok_total"] >= MIN_LAYER_TOKENS
                        for l in range(N_LAYERS) if (r, l) in prepped[n])]
        reg_pairs[r] = [(x, y) for x in names for y in names if x != y]
        print(f"  regime {r}: eligible traces {names} -> "
              f"{len(reg_pairs[r])} ordered pairs")

    detail: dict = {"dedup": dict(layers_with_dup=n_dup_layers, dropped=dd),
                    "meta": dict(seed=SEED, n_perm=N_PERM,
                                 min_layer_tokens=MIN_LAYER_TOKENS,
                                 z_bar=Z_BAR, retain_min=RETAIN_MIN,
                                 k_gated=list(K_GATED),
                                 excluded=["dump-all.txt", "dump-all.nodec.txt",
                                           "dump-all.crashed.txt"])}

    # ---- ARM B/C per regime
    regime_cls: dict[str, str] = {}
    for r in REGIMES:
        print(f"\n{'='*72}\nREGIME {r.upper()} (within-regime only)")
        pairs = reg_pairs[r]
        if not pairs:
            print("  no ordered pairs — regime INSUFFICIENT (structural)")
            regime_cls[r] = "INSUFFICIENT"
            detail[r] = dict(pairs={}, capture={}, overlap={},
                             verdict="INSUFFICIENT", thin=True,
                             note="no ordered pairs: eligible traces below the "
                                  "floor on every layer (after dedup: chat 80, "
                                  "code 86, long 4287 prefill tokens/layer)")
            continue

        # layer sets: X full >= floor AND Y both halves >= floor
        layer_sets = {}
        for (x, y) in pairs:
            ls = [l for l in range(N_LAYERS)
                  if (r, l) in prepped[x] and (r, l) in prepped[y]
                  and prepped[x][(r, l)]["tok_total"] >= MIN_LAYER_TOKENS
                  and prepped[y][(r, l)]["n_fit"] >= MIN_LAYER_TOKENS
                  and prepped[y][(r, l)]["n_held"] >= MIN_LAYER_TOKENS]
            layer_sets[(x, y)] = ls

        # ---- ARM B: overlap across requests (unordered pairs; reported, not gating)
        print(f"\n  ARM B — top-k OVERLAP across requests (unordered pairs, "
              f"pooled mean over included layers; null = {N_PERM} random "
              f"k-subsets vs the eval trace's actual top-k, pooled like-for-like):")
        print(f"    {'k':>2} {'pair':>11} {'nL':>3} {'|∩|/k':>7} {'medL':>6} "
              f"{'Jac':>6} {'nullμ':>7} {'p95':>6} {'p99':>6} {'>p95':>5}")
        btab = {}
        for k in K_ALL:
            for x, y in pairs:
                if x > y:  # unordered
                    continue
                L = layer_sets[(x, y)]
                if not L:
                    print(f"    {k:2d} {x+'/'+y:>11} {0:3d}  -- no layers --")
                    continue
                ov = np.empty(len(L)); ja = np.empty(len(L))
                null_ov = np.empty((len(L), N_PERM))
                for i, l in enumerate(L):
                    a, b = prepped[x][(r, l)], prepped[y][(r, l)]
                    A = a["order"][:k]; B = b["order"][:k]
                    inter = np.intersect1d(A, B).size
                    ov[i] = inter / k
                    ja[i] = inter / (2 * k - inter) if inter else 0.0
                    Bm = np.zeros(N_EXPERTS, dtype=bool); Bm[B] = True
                    null_ov[i] = Bm[masks[l][k]].sum(axis=1) / k
                null_pooled = null_ov.mean(axis=0)
                nmean, nsd = float(null_pooled.mean()), float(null_pooled.std())
                nse = nsd / np.sqrt(N_PERM)
                if nse > 0 and abs(nmean - k / N_EXPERTS) > 4 * nse:
                    fail(f"overlap null mean {nmean:.4f} vs uniform "
                         f"{k/N_EXPERTS:.4f} (>4 s.e.) — null instrument broken")
                obs = float(ov.mean())
                p95, p99 = (float(np.percentile(null_pooled, 95)),
                            float(np.percentile(null_pooled, 99)))
                btab[(k, x, y)] = dict(nL=len(L), obs=obs, med=float(np.median(ov)),
                                       jac=float(ja.mean()), null_mean=nmean,
                                       p95=p95, p99=p99,
                                       above=bool(obs > p95))
                print(f"    {k:2d} {x+'/'+y:>11} {len(L):3d} {obs:7.3f} "
                      f"{np.median(ov):6.3f} {ja.mean():6.3f} {nmean:7.3f} "
                      f"{p95:6.3f} {p99:6.3f} {('YES' if obs > p95 else 'no'):>5}")

        # per-layer overlap grid, k=8 (the brief asks for per-layer reporting)
        k0 = 8
        prs = [p for p in pairs if p[0] < p[1] and layer_sets[p]]
        if prs:
            print(f"    per-layer |∩|/k at k={k0} (columns = pairs "
                  f"{' '.join(f'{x}/{y}' for x, y in prs)}):")
            for l in range(N_LAYERS):
                cells = []
                for (x, y) in prs:
                    if l in layer_sets[(x, y)]:
                        a, b = prepped[x][(r, l)], prepped[y][(r, l)]
                        cells.append(f"{np.intersect1d(a['order'][:k0], b['order'][:k0]).size/k0:.2f}")
                    else:
                        cells.append(" --")
                print(f"      L{l:02d} {' '.join(f'{c:>5}' for c in cells)}"
                      + ("  [excluded everywhere]" if all(c == " --" for c in cells) else ""))

        # ---- within-trace ceiling D per trace (t22a cross-check)
        print(f"\n  WITHIN-TRACE CEILING D (split-half within regime; t22a's "
              f"split; the diagonal of the matrix below).  t22a reference for "
              f"long/prefill/k=8 was static=0.2382 vs uniform=0.0625 — measured "
              f"on a WHOLE-STREAM split (mixed regimes), so expect ~=, not ==:")
        for k in (8, 16):
            U = k / N_EXPERTS
            cells = []
            for n in used:
                Ls = [l for l in range(N_LAYERS)
                      if (r, l) in prepped[n]
                      and prepped[n][(r, l)]["n_fit"] >= MIN_LAYER_TOKENS
                      and prepped[n][(r, l)]["n_held"] >= MIN_LAYER_TOKENS]
                if not Ls:
                    cells.append(f"{n}: no layers")
                    continue
                Ds, zw = [], []
                for l in Ls:
                    b = prepped[n][(r, l)]
                    nhy = 4 * b["n_held"]
                    D = float(b["ch"][b["orderf"][:k]].sum()) / nhy
                    M = masks[l][k]
                    rw = b["ch"][M].sum(axis=1) / nhy
                    se = rw.std() / np.sqrt(N_PERM)
                    if rw.std() == 0 or (se > 0 and abs(rw.mean() - U) > 4 * se):
                        fail(f"L{l} k={k} {r} {n} (diag): random control broken")
                    Ds.append(D)
                    zw.append((D - U) / rw.std())
                Dp, z_pw = float(np.mean(Ds)), float(np.nansum(zw) / np.sqrt(len(zw)))
                cells.append(f"{n}: D={Dp:.4f} z={z_pw:+.1f} (nL={len(Ls)})")
                detail.setdefault("ceiling", {}).setdefault(r, {})[f"k{k}|{n}"] = \
                    dict(D=Dp, z_within=z_pw, nL=len(Ls))
            print(f"    k={k:2d}  U={U:.4f}  " + "  ".join(cells))

        # ---- ARM C: captured mass X-by-Y (the headline)
        print(f"\n  ARM C — CAPTURED SELECTION MASS (HEADLINE): fit top-k on X's "
              f"full stream, eval on Y's full stream; diagonal = within-trace "
              f"ceiling D (fit on Y's first half, eval on second; t22a's split, "
              f"within regime).  Uniform expectation U = k/128.")
        ctab: dict = {}
        pv: dict[str, dict[int, str]] = {f"{x}>{y}": {} for x, y in pairs}
        for k in K_ALL:
            U = k / N_EXPERTS
            print(f"\n    k={k}  U={U:.4f}   [cell = C_X->Y; '*' = pair INSUFFICIENT]")
            hdr = "        fit->eval " + " ".join(f"{y:>8}" for y in sorted({y for _, y in pairs}))
            print(hdr)
            for x in sorted({x for x, _ in pairs}):
                cells = []
                for y in sorted({y for _, y in pairs}):
                    if x == y:
                        cells.append("     diag")
                        continue
                    L = layer_sets[(x, y)]
                    if not L:
                        pv[f"{x}>{y}"][k] = "INSUFFICIENT"
                        cells.append("       --")
                        continue
                    Cs, Ds, zc, zw = [], [], [], []
                    for l in L:
                        a, b = prepped[x][(r, l)], prepped[y][(r, l)]
                        nsy = 4 * b["tok_total"]
                        nhy = 4 * b["n_held"]
                        C = float(b["c"][a["order"][:k]].sum()) / nsy
                        D = float(b["ch"][b["orderf"][:k]].sum()) / nhy
                        M = masks[l][k]
                        rc = b["c"][M].sum(axis=1) / nsy
                        rw = b["ch"][M].sum(axis=1) / nhy
                        for ctrl, side in ((rc, "full"), (rw, "held")):
                            se = ctrl.std() / np.sqrt(N_PERM)
                            if ctrl.std() == 0 or (se > 0 and
                                                   abs(ctrl.mean() - U) > 4 * se):
                                fail(f"L{l} k={k} {r} {x}->{y} ({side}): random "
                                     f"control mean {ctrl.mean():.4f} vs uniform "
                                     f"{U:.4f} — capture instrument broken")
                        Cs.append(C); Ds.append(D)
                        zc.append((C - U) / rc.std())
                        zw.append((D - U) / rw.std())
                    Cp, Dp = float(np.mean(Cs)), float(np.mean(Ds))
                    z_pc = float(np.nansum(zc) / np.sqrt(len(zc)))
                    z_pw = float(np.nansum(zw) / np.sqrt(len(zw)))
                    retain = (Cp - U) / (Dp - U) if Dp > U else float("nan")
                    v, why = pair_verdict(L, z_pw, z_pc, retain)
                    pv[f"{x}>{y}"][k] = v
                    ctab[(r, k, x, y)] = dict(nL=len(L), C=Cp, D=Dp, U=U,
                                              retain=float(retain),
                                              z_cross=z_pc, z_within=z_pw,
                                              verdict=v, why=why)
                    cells.append(f"{Cp:8.4f}" + ("" if v != "INSUFFICIENT" else "*"))
                print(f"        {x:9s}" + " ".join(f"{c:>8}" for c in cells))
            # per-pair verdict lines for this k
            for x, y in pairs:
                e = ctab.get((r, k, x, y))
                if e:
                    print(f"      {x}>{y}: C={e['C']:.4f} U={e['U']:.4f} "
                          f"D={e['D']:.4f} retain={e['retain']:.2f} "
                          f"z_cross={e['z_cross']:+.1f} z_within={e['z_within']:+.1f} "
                          f"-> {e['verdict']}  ({e['why']})")
                else:
                    print(f"      {x}>{y}: INSUFFICIENT (no layers pass floors)")

        # ---- regime verdict per gated k, then combined
        print(f"\n  REGIME {r} VERDICT (gated k{K_GATED}):")
        cls_by_k = {}
        for k in K_GATED:
            vs = [pv[f"{x}>{y}"][k] for x, y in pairs]
            cls_by_k[k] = k_verdict(vs)
            print(f"    k={k:2d}: {cls_by_k[k]:14s} "
                  f"(pairs: {' '.join(f'{x}>{y}=' + pv[f'{x}>{y}'][k] for x, y in pairs)})")
        regime_cls[r] = combine_ks(cls_by_k)
        thin = len({x for x, _ in pairs}) < 3
        detail[r] = dict(
            pairs={f"{x}>{y}": {k: dict(verdict=pv[f'{x}>{y}'][k]) for k in K_ALL}
                   for x, y in pairs},
            capture={f"{k}|{x}>{y}": v for (rr, k, x, y), v in ctab.items() if rr == r},
            overlap={f"{k}|{x}/{y}": v for (k, x, y), v in btab.items()},
            verdict=regime_cls[r], thin=thin)
        print(f"    => regime {r}: {regime_cls[r]}"
              + ("  [THIN: fewer than 3 eligible traces — a single workload "
                 "pair cannot by itself establish cross-REQUEST (as opposed "
                 "to cross-PAIR) agreement]" if thin else ""))

    # ---- OVERALL (mapping of record; see docstring RULE AMENDMENT)
    print(f"\n{'='*72}\nOVERALL VERDICT (rule as stated, applied unchanged):")
    evaluable = {r: v for r, v in regime_cls.items() if v != "INSUFFICIENT"}
    thin_sup = {r for r, v in evaluable.items() if detail[r]["thin"]}
    if not evaluable:
        overall, why = "INSUFFICIENT", "no regime yielded an evaluable pair"
    elif all(v == "DO_NOT_BUILD" for v in evaluable.values()):
        if thin_sup and len(thin_sup) == len(evaluable):
            overall, why = "INSUFFICIENT", (
                "every evaluable regime says DO_NOT_BUILD, but the support is "
                "THIN-only (single workload pair per regime); the richer "
                "regime is INSUFFICIENT — a kill may not rest on one pair")
        else:
            overall = "DO_NOT_BUILD"
            why = ("cross-trace capture collapses to the uniform expectation "
                   "in every evaluable regime: hot sets are a per-request "
                   "phenomenon; a fixed per-group allocation cannot help; "
                   "Stage B must not be built on this evidence")
    elif any(v == "DO_NOT_BUILD" for v in evaluable.values()):
        overall, why = "INSUFFICIENT", (
            f"regime-contradicted: {dict(regime_cls)}")
    elif all(v == "GENERALISES" for v in evaluable.values()) and \
            any(not detail[r]["thin"] for r in evaluable):
        overall = "PROCEED_TENTATIVE"
        why = (f"hot experts transfer across requests in every evaluable "
               f"regime {dict(regime_cls)}, keeping >= {RETAIN_MIN:.0%} of the "
               f"within-request refit gain — but n={len(raw)} requests from "
               f"one server run cannot establish per-MODEL generality: "
               f"TENTATIVE")
    elif all(v in ("GENERALISES", "GENERALISES_WEAK")
             for v in evaluable.values()) and \
            any(not detail[r]["thin"] for r in evaluable):
        overall = "PROCEED_TENTATIVE_WEAK"
        why = (f"transfer real in every evaluable regime {dict(regime_cls)} "
               f"but some of it retains < {RETAIN_MIN:.0%} of the within-request "
               f"refit gain; n={len(raw)} requests from one server run cannot "
               f"establish per-MODEL generality")
    else:
        overall, why = "INSUFFICIENT", f"mixed regime verdicts {dict(regime_cls)}"
    print(f"  >>> {overall}\n      {why}")
    # factual structure summary (what the pair verdicts were, per regime, k=8)
    print("  structure (k=8 pair verdicts):")
    for r in REGIMES:
        vs = [f"{x}>{y}:{detail[r]['capture'].get(f'8|{x}>{y}', {}).get('verdict', 'n/a')}"
              for (x, y) in reg_pairs[r]]
        if vs:
            print(f"    {r}: {' '.join(vs)}")

    print(f"\nWHAT n={len(used)} USABLE SESSIONS CANNOT SUPPORT (fixed in the docstring, "
          "unchanged by the outcome): per-MODEL generality (3 requests, one "
          "server run, one model, dump-patch build with GATE2 FAIL), any "
          "QUALITY claim (only a Stage B "
          "build-and-evaluate loop can speak to quality), and a tight "
          "retain estimate.  What would settle it properly: a MULTI-SESSION "
          "trace over the real corpus (GPU collection run, dozens of "
          "independent sessions), this same rule applied unchanged.")
    print("This is a transfer-of-selection-mass result only: it is NOT a "
          "quality claim.")

    detail["overall"] = dict(verdict=overall, why=why, regimes=regime_cls)
    OUT_JSON.write_text(json.dumps(detail, indent=1, sort_keys=True))
    print(f"per-layer detail: {OUT_JSON}")
    print(f"runtime: {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
