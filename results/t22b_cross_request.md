# T2.2 Stage A+ — the CROSS-REQUEST test (t22b)

Instrument: `tools/t22b_cross_request.py` (added alongside `tools/t22_signal_study.py`; the
t22a instrument and its gate are untouched). Runs (all `venvs/convert/bin/python3`, exit 0,
byte-identical across re-runs modulo the runtime line):

| run | file | what it was |
|---|---|---|
| 1 | `results/t22b_cross_request_run1_literal_rule.txt` | rule as first written; inputs undeduplicated; OVERALL mapping had an unstated branch (gap) |
| 2 | `results/t22b_cross_request_run2_amended_mapping.txt` | OVERALL mapping completed + THIN guard; inputs still undeduplicated |
| 3 | `results/t22b_cross_request_run3_rule_of_record.txt` | **rule of record**: same rule + probe-artefact dedup (input correction, thresholds untouched) |
| — | `results/t22b_cross_request_detail.json` | per-layer detail (overlap, capture, z, verdicts, dedup record) |

**Verdict (rule of record): INSUFFICIENT.** The only regime with an admissible cross-request
comparison (decode: chat↔code, both directions, k=8/16/32/64) shows **no transfer** — cross
capture sits at the uniform expectation (z_pool −0.9…−2.8 vs bar +4.0) while each trace's own
within-request hot set is strongly real (z_within +29…+41) — but it is a single workload pair
(THIN), and the prefill regime has **no admissible pairs at all** after the probe-artefact
dedup (chat 80 / code 86 tokens/layer < floor 100; long alone remains). A kill of Stage B on
this evidence would over-claim; a build is equally unjustified. Stage B stays gated OFF.

## 1. Question

t22a showed the per-expert selection profile is strongly stable WITHIN a request (split-half
rho 0.72–0.84; fixed top-k keeps 93.4–96.1% of adaptive reselection). But a shipped allocation
is fixed per MODEL, not per request: **do different requests agree on WHICH experts are hot?**
If not, a fixed per-group allocation cannot help and Stage B must not be built.

## 2. Inputs, exclusions — all re-measured this run, not cited

Four independent traces under `aubier/logs/pe/`, rows `R <ubatch> L <layer> T <n_tokens> <4T ids>`.

**Excluded as further "requests"** (would double-count and manufacture agreement):
- `dump-all.txt` (35,085 lines): exact union of chat+code+long — all routing rows of those
  three are contained in it. MEASURED refinement of the brief's description: it is **not** a
  union of all four (gate's rows are absent from it).
- `dump-all.nodec.txt` (34,571): separate earlier no-decode collection variant; near-zero
  routing-row overlap with the four (5,007/34,354 shared). Not an admissible same-config session.
- `dump-all.crashed.txt` (108): fragment of a crashed run.

**Data findings that changed the design** (each measured, both the finding and its consequence
recorded; the decision rule's thresholds were never touched):

1. **`dump-gate.txt` is corrupt — excluded WHOLE.** 4,963/4,975 routing rows contain
   out-of-range "expert ids" (e.g. −1072730037, 1029760064) whose u32 bit patterns decode as
   float32 values: that collection run's dumper wrote the wrong tensor. Salvaging the ~12 sane
   rows (~0.3 tokens/layer) would fabricate a trace from cherry-picked rows. The briefed n=4 is
   therefore n=3. Restoring the gate workload needs a GPU re-collect.
2. **Provenance caveat — GATE2 FAIL.** chat/code/long were collected from ONE server process
   (`aubier/scripts/route_session.py` session6: `builds/hip-route` with `GGML_ROUTE_DUMP=ON`,
   port 8753, temperature 0, seed 42) — the ONLY difference between the three traces is the
   request (essay / Python-code / 4.8k-token C++ long context + question). Internally
   consistent for cross-request comparison. BUT `session6.log:24` records **GATE2
   stock-vs-patched-ON: FAIL** (dump OFF was output-identical, GATE1 PASS): the dump-ON server's
   outputs differed from stock, so the captured routing may not be stock routing. This caveat
   applies EQUALLY to all three traces and **inherits to t22a**, which used `dump-long`
   unqualified.
3. **Probe artefact — forced an input-cleaning step.** Every usable trace contains EXACTLY ONE
   byte-identical 512-token ubatch per layer (35/36 layers; same T, same ids) — 512/592 = 86.5%
   of chat's and code's prefill tokens/layer, 512/4799 = 10.7% of long's. Counting it in two
   traces would manufacture "agreement" (the same tokens capturing themselves). The instrument
   drops every multi-token ubatch whose (T, ids) digest appears in another trace, uniformly,
   before any statistic (T==1 rows are exempt: 4-expert combinations collide by chance across
   independent requests — the observed 0/0/0 decode drops and the birthday count over
   C(128,4)≈10.7M with ~13.8k draws/trace/layer confirm the decode regime carries none).

**Regime control (binding brief constraint):** decode = rows T==1, prefill = rows T>1. All
statistics are within-regime; no pooled-across-regime number is produced anywhere.

## 3. Sizes after dedup (tokens/layer, min..median; floor 100)

| trace | decode | prefill | eligible in |
|---|---|---|---|
| chat | 383..383 (36/36 L) | 80..80 (0/35 L) | decode |
| code | 383..383 (36/36 L) | 86..86 (0/35 L) | decode |
| long | 39..39 (0/36 L) | 4287..4287 (35/35 L) | prefill (ceiling only) |

Regime pairs: decode {chat, code} → 2 ordered pairs; prefill {long} → 0 pairs.

## 4. Within-trace ceiling D (split-half, within regime; the diagonal)

| k | U | chat (decode) | code (decode) | long (prefill) |
|----|----|----|----|----|
| 8 | 0.0625 | **0.3138** z=+41.1 (36 L) | **0.2061** z=+33.8 | no layers (excl. decode) |
| 16 | 0.1250 | **0.4438** z=+40.7 | **0.3278** z=+35.2 | no layers |
| 8 | 0.0625 | — prefill chat/code: no layers (80/86 tok, halves < floor) | | **0.2389** z=+40.7 (35 L) |
| 16 | 0.1250 | | | 0.3728 z=+43.2 |

Cross-check against t22a: its headline long/prefill/k=8 static was **0.2382** (measured on the
whole-stream split, mixed regimes); this instrument's within-regime diagonal gives **0.2389** —
consistent, as expected (≈, not ==: different split composition). The instrument reproduces the
t22a number it was supposed to reproduce.

## 5. ARM B — top-k overlap across requests, with permutation nulls

Only one unordered pair is admissible (decode chat/code; nL=36). Pooled mean over layers of
|∩|/k; null = 200 random k-subsets vs the eval trace's actual top-k, pooled like-for-like
(null mean checked within 4 s.e. of k/128 — instrument check passed):

| k | \|∩\|/k | median L | Jaccard | null μ | p95 | p99 | obs > p95? |
|---|---|---|---|---|---|---|---|
| 8 | 0.059 | 0.000 | 0.034 | 0.063 | 0.094 | 0.101 | no |
| 16 | 0.104 | 0.062 | 0.057 | 0.123 | 0.144 | 0.156 | no |
| 32 | 0.236 | 0.250 | 0.136 | 0.251 | 0.269 | 0.278 | no |
| 64 | 0.522 | 0.531 | 0.355 | 0.500 | 0.511 | 0.515 | yes (trivially: half the space) |

Decode hot-expert sets of the two requests are **uncorrelated** (k=8 overlap at null level).
Per-layer k=8 values: 0.00 in 19/36 layers, max 0.50 (L32), pooled 0.059. Prefill had no
admissible pairs (below), so no prefill overlap is reported by the rule of record.

## 6. ARM C — captured selection mass (HEADLINE)

Fit top-k on X's full stream, eval on Y's full stream; U = k/128; D = within-trace ceiling of Y.

**decode, k=8** (U=0.0625; cells C_X→Y):

| fit→eval | chat | code |
|---|---|---|
| chat | D=0.3138 | **0.0577** |
| code | **0.0622** | D=0.2061 |

- chat→code: C=0.0577, z_cross=−1.3, z_within=+33.8, retain≈0 → **COLLAPSE**
- code→chat: C=0.0622, z_cross=−1.0, z_within=+41.1, retain≈0 → **COLLAPSE**

k=16: C=0.1138/0.1172 vs U=0.1250, z_cross −1.9/−1.5 → COLLAPSE. k=32/64 (reported, not
gated): same picture — cross capture never exceeds uniform by more than noise (k=64: z −2.8/+0.3).

**prefill: no ordered pairs** — after dedup chat/code carry 80/86 prefill tokens/layer, below
the 100-token floor on every layer; long alone remains, and a regime of one trace cannot answer
a cross-request question. Regime verdict: INSUFFICIENT (structural).

**The asymmetry that proves the artefact mattered** (same traces, same rule, before/after
dedup): run 1 — before the shared probe ubatch was removed — reported prefill chat↔code overlap
0.907 and cross capture z=+47 with retain 1.35 ("TRANSFERS"). After removing the ONE shared
ubatch, the entire prefill comparison falls below the rule's floors and the only clean regime
shows no transfer at all. The pre-"cleaning" transfer was the probe capturing itself, not
request agreement. Both runs are preserved (run 1 / run 3 files above).

## 7. Decision rule (fixed in advance; where it came from; what was amended)

Thresholds are inherited project bars, none invented here: floor 100 tok/layer (t22a),
Z_BAR=4.0 (t22a `P2_MIN_POOLED_Z`), RETAIN_MIN=0.5 (t22a's P3 "more than halfway" principle),
N_PERM=200 (brief), gated k∈{8,16} (t22a's rationale). Pair verdicts: INSUFFICIENT (floors, or
ceiling not established: z_within < Z_BAR) / COLLAPSE (z_cross < Z_BAR) / WEAK_TRANSFER
(z_cross ≥ Z_BAR, retain < 0.5) / TRANSFERS. Regime: all-TRANSFERS → GENERALISES; all
COLLAPSE → DO_NOT_BUILD; mixed → INSUFFICIENT. OVERALL (amended after run 1, both runs
published): a DO_NOT_BUILD or PROCEED needs ≥1 non-THIN supporting regime — a kill may not rest
on a single workload pair; a merely-INSUFFICIENT regime elsewhere does not block a kill (t22a
precedent). Every PROCEED_* is TENTATIVE by construction.

**Verdicts (rule of record):**
- decode: DO_NOT_BUILD, **THIN** (single workload pair, two directions)
- prefill: INSUFFICIENT (structural — no admissible pairs)
- **OVERALL: INSUFFICIENT** — "every evaluable regime says DO_NOT_BUILD, but the support is
  THIN-only; a kill may not rest on one pair."

## 8. What this evidence does and does not say

- MEASURED: on the only admissible cross-request pair (decode, chat↔code — genuinely distinct
  generated content, zero shared rows), cross-trace capture sits at the uniform expectation at
  every k while within-request decode hot sets are strongly real (D z +29…+41). Within-request
  stability (t22a) therefore does NOT imply between-request sharing in this pair.
- MEASURED: the only pre-request-agreement signal in these traces (run 1's prefill
  chat↔code overlap 0.91) was an artefact of one shared ubatch and disappears under dedup.
- INTERPRETATION (not gated, n=1 pair): generated-text (decode) hot sets look per-request.
  If that generalised, a fixed per-model allocation would capture ≈ uniform mass on decode.
- NOT CLAIMED: any quality/perplexity statement — only a Stage B build-and-evaluate loop can
  speak to quality, and it stays gated OFF on this result. Percentages are fractions of
  expert-selection slots.

## 9. What n=3 cannot support; what would settle it

Three requests from one server run of one model (dump-patch build, GATE2 FAIL) cannot
establish per-MODEL generality, cannot disentangle workload from request-mix, and give no
admissible prefill comparison at all. What would settle it properly: **a MULTI-SESSION trace
collected over the real corpus (a GPU collection run, dozens of independent requests/sessions),
collected with a dump path that is output-identical to stock (fix the GATE2 FAIL first), split
per session, with this same rule applied unchanged.** That is the honest next step rather than
a bigger claim from 3 sessions. (A gate-workload re-collect would also be needed to restore the
briefed n=4.)

## 10. Cannot-verify list

- Whether the captured routing equals stock routing (GATE2 FAIL; no GPU here to re-verify).
- Cross-request agreement in the prefill regime (no admissible pairs after dedup).
- Whether the probe ubatch's identical routing across traces reflects determinism or logging
  duplication (mechanism inside the patched build; not verifiable host-side).
- Any quality claim (Stage B territory).

## 11. Inheritance caveat for t22a (record; t22a files not modified by this unit)

t22a's numbers were computed on the SAME undeduplicated streams. Its PRIMARY (long) carried the
probe at 10.7% of prefill tokens; its chat/code whole-stream numbers were probe-dominated
(53% of chat's whole stream). t22a's headline conclusions are unlikely to flip at that dilution
(its long diagonal reproduces here: 0.2389 vs 0.2382), but the t22a write-up carries no
GATE2/probe caveat — the orchestrator may want to append one.
