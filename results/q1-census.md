# Q1 — THE POPULATION CENSUS (format-proposal §5 Q1; gates R4)

Date: 2026-10-15. Unit: taillis-q1. Rule of record: `tools/t22b_cross_request.py`
(sha256 `d586482fc2f6fcadef8deb3398681df54d4ba73e3dfa95ee3a7886b45f0c72ef`,
byte-unchanged throughout). Evidence files: `results/q1-census-evidence/`;
per-conversation traces + `split-validation.txt`:
`/var/lib/taillis/trace-a++/conversations/`.

## The question

G3 measured cross-request transfer of per-expert hot sets on K=12 MIXED
8-conversation streams (1056/1056 pair verdicts TRANSFERS); G4 measured the
opposite on single prompts. Q1 asks whether the discrepancy is a GRANULARITY
artifact: does the same rule, same thresholds, same traces, still hold when the
unit is ONE CONVERSATION (96 units instead of 12)? If transfer holds between
independent single conversations, R4's objection (a static per-group
allocation fitted on one unit may not serve another) is answered at the unit
the product serves; if it collapses there, the 8-conversation mixture was
doing the work and R4's premise dies before any artifact is built.

## Method

* SPLIT: the 12 slice traces were partitioned into 96 per-conversation trace
  files using the measured `dump_bytes [start,end)` ranges in
  `logs/prov-all.json` (2 requests' rows never overlap; ranges are contiguous
  and the last ends at EOF). Each segment file carries the parent's 1-line
  `# route-dump v1 ...` header (the rule never parses the header, but this
  makes every segment a standalone well-formed trace). Written to
  `/var/lib/taillis/trace-a++/conversations/` (parents read-only).
* RULE: unchanged, PATH-ONLY env (`T22B_TRACE_DIR`/`T22B_TRACES`/
  `T22B_OUT_JSON`), exactly as `run_t22b.sh` does it. One delta, applied at
  import time by a 6-line runner (verbatim in
  `q1-census-evidence/run_k32.py`): `K_ALL` restricted to (8,16,32) —
  dropping k=64, which the rule's own docstring classes "reported, not
  gated" and which Stage A++ measured as a deterministic Monte-Carlo abort at
  132 pairs (it aborts a fortiori at 9120). No threshold, floor, or verdict
  code was touched; the gated set stays k={8,16}.
* PREFLIGHT (new, host-side, no rule change): every `fail()` abort condition
  of the rule is a deterministic function of (masks, count vectors) — the
  masks are drawn once from the fixed seed. `q1-census-evidence/preflight.py`
  reuses the rule's own code to compute every control z the k<=32 runs can
  hit, before running them. Predictions: slices abort-free at k<=32 (max
  |z|=3.710); conversations abort-free at k=8/16 but abort at the FIRST k=32
  cell, (eval=conv-1134, L7, full, z=4.117; 88 bad cells at k=32, 0 at
  k=8/16). Both predictions were then MEASURED (below).

## Split validation (all MEASURED; `split-validation.txt` has the per-segment table)

* 96/96 segments parse; every segment contains all 36 layers; 0 rows with
  out-of-range ids (G5); 0 rows with T/4 id-shape mismatches.
* Row accounting: parent rows − 72 warmup rows = segment rows, exact on all
  12 slices (total 187,631 − 864 = 186,767). The 72 rows outside every
  `dump_bytes` are the server warmup (2 ubatches x 36 rows: 35x T=2 + 1x
  T=1 each), present per slice, not part of any request.
* Token accounting vs prov, exact on 96/96 segments under one measured
  dumper pattern: prefill tokens summed over layers 0..34 ==
  `tokens_evaluated` (layer 35 carries 0 prefill — the known dumper
  artifact); decode rows/layer == `tokens_predicted` − 1 on layers 0..34 and
  == `tokens_predicted` on layer 35 (the missing L35 prefill row reappears
  as one extra L35 decode row). Totals per conversation are exact.
* R counter: per process, monotonic ACROSS the 8 requests of a slice
  (segments start at R=3; ranges contiguous within each slice). R is used by
  the rule ONLY in `check_monotonic` (non-decreasing within each layer —
  satisfied by every segment); it enters no statistic (`parse_trace` reads
  only layer, T, ids). Segments were NOT renumbered; the data is as
  captured.

## Reproduction control (deliverable 3)

Re-run of the UNCHANGED rule over the 12 slice traces (run A): **stdout
byte-identical to `results/t22b_run.log`** — including decode 132/132
TRANSFERS at k=8/16/32 and the same deterministic FATAL at the k=64 decode
control (L0, slice-00->slice-11, mean 0.4929 vs 0.5000), exit 1. The census
record reproduces exactly; nothing drifted in the pipeline.

Slice-level k<=32 completion (run B; the "drop k=64 subset" the brief
blesses): exit 0, **792/792 pair verdicts TRANSFERS** (2 regimes x 3 k x 132
pairs; zero WEAK/COLLAPSE/INSUFFICIENT), regime verdicts decode
GENERALISES + prefill GENERALISES, OVERALL **PROCEED_TENTATIVE** — the
census verdict stands without k=64. Rule-native JSON:
`q1-census-evidence/t22b_slice_k81632.json`.

## Slice level vs conversation level (deliverables 2+4)

Decode regime, conversation level: **INSUFFICIENT (structural), 0/96
traces eligible.** `tokens_predicted <= 64` per request (n_predict 64), and
the rule's floor needs >=100 full-stream tokens for the fit side and >=100
in BOTH halves of the eval side per layer — 64 < 100, so no single
conversation can EVER enter a decode-regime pair in this collection. The
finest unit supporting the decode regime is an 8-conversation aggregate.
This is a property of the collection (n_predict=64), not of the rule.

Prefill regime (89/96 conversations eligible; 7 excluded whole —
`tokens_evaluated` 41..77 < the 100-token layer floor; 2 more,
conv-0000/conv-0728 at 168/106 tokens, are eligible as fit sides but fail
the eval-side halves floor, making 88+88 ordered pairs INSUFFICIENT per k;
7,832 ordered pairs per k = 89x88):

| level / regime | k | pairs evaluated | C_med | U | retain_med | z_cross_med | tally |
|---|---|---|---|---|---|---|---|
| slice decode (8-conv) | 8 | 132 | 0.1882 | 0.0625 | 1.016 | +34.2 | 132 TRANSFERS |
| slice decode (8-conv) | 16 | 132 | 0.2989 | 0.1250 | 1.012 | +35.1 | 132 TRANSFERS |
| slice decode (8-conv) | 32 | 132 | 0.4685 | 0.2500 | 1.019 | +33.6 | 132 TRANSFERS |
| slice prefill (8-conv) | 8 | 132 | 0.2409 | 0.0625 | 1.009 | +47.0 | 132 TRANSFERS |
| slice prefill (8-conv) | 16 | 132 | 0.3640 | 0.1250 | 1.009 | +47.4 | 132 TRANSFERS |
| slice prefill (8-conv) | 32 | 132 | 0.5385 | 0.2500 | 1.007 | +44.0 | 132 TRANSFERS |
| **conv prefill (1-conv)** | 8 | 7,656 | **0.2222** | 0.0625 | **0.910** | +37.9 | **7,580 TRANSFERS / 76 WEAK / 176 INSUF** |
| **conv prefill (1-conv)** | 16 | 7,656 | **0.3401** | 0.1250 | **0.920** | +38.2 | **7,635 TRANSFERS / 21 WEAK / 176 INSUF** |
| conv prefill k=32 | 32 | 0 | — | 0.2500 | — | — | run aborted at first cell (predicted, see below) |

*Metric note (review correction): the `_med` columns are computed as
`sorted()[n//2]` — the UPPER-MIDDLE order statistic, not the arithmetic median
(they will not reproduce under `statistics.median()`; e.g. slice decode k=8
prints retain 1.016 where the true median is 1.010). Immaterial to every
verdict, which rest on `z_cross` and `retain >= RETAIN_MIN`, but stated so the
table is not read as medians.*

Conversation-level run (run C): completed ALL of k=8 and k=16 (15,312
evaluable pair verdicts), then aborted at the k=32 control in the exact
cell the preflight predicted (FATAL `L7 k=32 prefill conv-0000->conv-1134
(full)`, mean 0.2412 vs 0.2500 = z 4.117; rule exit 1, JSON not written).
Measured confirmation of the preflight: 0 bad cells at k=8/16, first
k=32 cell exactly as computed.

## Q1 verdict (the plain statement)

**The agentic premise does NOT depend on the 8-conversation mixture — but it
is no longer unanimous, and only the prefill regime is testable at the
single-conversation unit.**

* PREFILL: transfer between INDEPENDENT single conversations holds with
  **zero COLLAPSE in 15,312 evaluable pair verdicts**. A top-k fitted on one
  conversation captures on an unseen conversation 3.6x the uniform
  expectation at k=8 (C_med 0.2222 vs 0.0625; slice level 3.9x), at
  z_cross_med +38. Per the rule's own verdict functions applied to its own
  complete k=8/16 outputs (run aborted before printing regime/overall
  verdicts; reconstruction used the rule's imported `k_verdict`/
  `combine_ks` unmodified): k=8 GENERALISES_WEAK, k=16 GENERALISES_WEAK ->
  prefill regime **GENERALISES_WEAK** (non-THIN, 89 eligible traces) ->
  overall **PROCEED_TENTATIVE_WEAK** (decode INSUFFICIENT does not block;
  t22a precedent).
* The shrinkage against the mixture is real but small: TRANSFERS drops from
  100% to 97.0% (k=8) / 97.5% (k=16); retain_med drops 1.01 -> 0.91 (a
  cross-conversation fit keeps 91% of what a within-conversation refit
  keeps, still far above RETAIN_MIN=0.5); the WEAK pairs (76+21) keep
  0.31..0.50 and concentrate on short conversations at BOTH ends — as the
  EVAL side conv-0658 takes 12 of 76 (k=8) and 5 of 21 (k=16), and as the FIT
  side conv-0658 (643 evaluated tokens) produces 48 of 76 (k=8) and 10 of 21
  (k=16): a top-k fitted on that ONE short conversation transfers weakly onto
  48/88 (~55%) of its targets, and the next fit-side contributors are also
  short (conv-0630 382 tok, conv-0000 168 tok). The fit-side concentration is
  the operative bound on R4's premise (a single conversation is the fitting
  unit), not the eval-side one.
* DECODE: untestable at the unit the product serves (0/96 eligible). The
  396/396 slice-level decode TRANSFERS cannot be decomposed into
  single-conversation evidence with n_predict=64 traces. R4's decode-regime
  objection is therefore UNANSWERED, not refuted, at conversation
  granularity.

For R4 specifically: a static per-group allocation fitted on ONE conversation
transfers to OTHER single conversations in prefill at the unit the product
serves; the mixture was not manufacturing the G3 PREFILL result — G3's decode
half remains mixture-only evidence. **The G4 single-prompt collapse was itself
measured in the DECODE regime (its prefill was structurally INSUFFICIENT), so
this census and G4 share NO regime**: Q1 is therefore not a clean
granularity-vs-population split, and population is not the only live difference
between G3 and G4. The G4 collapse remains consistent with this data.

## Trap checks (deliverable 5)

* **G7 (shared rendered prefix / dedup)**: MEASURED, and the summary's
  attribution is corrected: the slice-level dedup drops (70 rows / 140
  tokens per slice, all T=2, 35/36 layers) are EXACTLY the 72-row server
  warmup's T=2 rows (2 x 35) — NOT the corpus's rendered prefix. Request
  rows contain ZERO byte-identical ubatches across traces at either level
  (the shared prefix is fused with unique content in every first prefill
  ubatch, so no T>1 digest is ever shared). At conversation level the
  warmup rows are outside all `dump_bytes`, and the rule's DEDUP line
  confirms: **0/36 layers, 0 rows dropped on all 96 traces**. So the
  dedup never fired on request data at either level; any common-prefix
  routing mass is present-and-undeduplicated EQUALLY at both levels and
  cannot manufacture a level DIFFERENCE (it biases both toward transfer;
  INTERPRETATION — now MEASURED as well: the longest common prefix across all
  96 prompts is **40 bytes (~10 tokens, median 0.7% of prompt bytes**;
  pairwise p50 0.8%, p90 17%, p99 30%, max 81%, the max occurring only among
  conversations sharing a first rendered block).
* **G8 (capture vs coverage)**: this write-up reports only the rule's
  capture metrics (C, D, retain, z). The validator's coverage@top-N numbers
  are a different metric and are not cited or compared anywhere above.
* **G5 (id range)**: 0 out-of-range ids in all 96 segments (and the rule
  re-checked on parse).
* **G6 (derived traces as requests)**: the census read ONLY
  `conversations/` with exactly the 96 named traces; `dump-gate-inertness.txt`
  (corrupt-by-design instrument check file) was never in any T22B_TRACES
  list; no dump-all-equivalent exists in this collection.

## Caveats (carried from STAGE_A++_SUMMARY + added)

1. Inertness strict gate FAILED upstream (one deterministic near-tie token
   flip per prompt; dump-ON perturbs compute): traces are routing of
   patched-ON compute, exactly reproducible. Applies equally to both levels.
2. ONE model, ONE build, ONE machine; temperature 0, seed 42. Per-model
   generality untested. TENTATIVE qualifiers stand.
3. k=64 untested here (excluded by brief; unchanged rule provably aborts
   there — reproduced in run A). k=32 at conversation level is MEASURED
   ABORTING at the first pair under the rule's own 200-mask control (MC
   multiplicity, z invariant to N_PERM): k=32 conversation-level pair
   verdicts are NOT obtainable from the unchanged rule and are not reported
   as if they were. Gated verdicts rest on k=8/16 only.
4. Decode-regime starvation is a collection property (n_predict=64): any
   future single-conversation decode claim needs a re-collection with
   longer generations (GPU run; out of scope here — no GPU was used).
5. retain is a ratio of two noisy pooled gaps (rule docstring); the
   1.01 -> 0.91 drop is a median over 7,656 ratios and is directionally
   meaningful, not a tight estimate.
6. The 7 excluded + 2 eval-INSUFFICIENT conversations are the SHORT end
   (evaluated 41..168 tokens); the census population is conversations with
   >=~200 evaluated tokens. The product population includes short prompts
   too short for ANY within-rule judgement — a boundary the rule states,
   not a defect introduced here.

## Artifacts

* `results/q1-census-evidence/` — run A diff base is
  `/var/lib/taillis/trace-a++/results/t22b_run.log` (byte-identical stdout
  reproduced as `t22b_repro.log`, /tmp); run B log + rule-native JSON; run C
  log (harvest source); preflight + abort-prediction + runner scripts; rule
  sha256; `split-validation.txt` (per-segment table).
* `/var/lib/taillis/trace-a++/conversations/dump-conv-*.txt` — the 96
  per-conversation traces (integer expert-id sequences only; no corpus
  text, no session paths anywhere in any artifact — verified by scan).
