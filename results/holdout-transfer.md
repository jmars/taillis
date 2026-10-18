# TRANSFER OF PER-EXPERT HOT SETS ACROSS AN INDEPENDENT HELD-OUT POPULATION

Date: 2026-10-18. Unit: taillis-v4, part B. No GPU, no model run, no `gpu.lock`:
this is pure trace analysis. Rule of record: `tools/t22b_cross_request.py`,
**byte-unchanged** — `sha256sum` = `d586482fc2f6fcadef8deb3398681df54d4ba73e3dfa95ee3a7886b45f0c72ef`
(MEASURED this session, and identical to the hash recorded in `results/q1-census.md`).
It was driven PATH-ONLY (`T22B_TRACE_DIR` / `T22B_TRACES` / `T22B_OUT_JSON`).
Every threshold below (100 tokens/layer, Z_BAR=4.0, RETAIN_MIN=0.5, N_PERM=200,
seed 20261014, gated k∈{8,16}) is the rule's own, unedited.

Claim markers: **MEASURED** = read off a run in this session; **INTERPRETATION**
= my reading of those numbers; **INFERRED** = derived from another measured
number; **PROJECTION** = a consequence I claim follows but did not run.

Inputs (aubier, READ-ONLY): the m1-u6 route-trace set, `trace-A1..A6`
(derivation) and `trace-ctrl_B1..B4` (HELD OUT), rows
`R <layer> <pos> <sel0..3> <w0..w3>`, one row per (layer, position), 36 layers.
No corpus text and no session paths are reproduced here; everything below is an
integer statistic.

---

## 0. The headline, in the rule's own vocabulary

| configuration | phase content | ordered pairs | pairs passing the floors | regime verdict |
|---|---|---|---|---|
| **DECODE-only** | 128 decode positions/layer | 90 | **0** | **INSUFFICIENT** (structural) |
| **PROMPT-only** | 27–116 prompt positions/layer | 2 | **0** | **INSUFFICIENT** (structural) |
| **MIXED whole-stream** (diagnostic, *not* a regime) | prompt+decode, 155–244 pos/layer | 90 | **18** | **INSUFFICIENT** (pair-contradicted: 2 TRANSFERS, 16 COLLAPSE) |

**OVERALL (rule applied unchanged): INSUFFICIENT.** The mapping's own reason string
reads "no regime yielded an evaluable pair", which is imprecise here: 18 pairs
*were* evaluated in the mixed configuration, whose regime k-verdict is
`INSUFFICIENT` (pair-contradicted) — see §3c. MEASURED, exit code 0 on all three
runs (in this rule, INSUFFICIENT is a decision, not an error; only a malformed
trace exits nonzero).

**The DECODE regime is NOT testable on this data, and the reason is arithmetic,
not a judgement.** The rule's floor is `MIN_LAYER_TOKENS=100` tokens per layer
on the fit side **and in each half of the eval trace** — so an eval trace must
carry **≥ 200 tokens per layer**. A held-out prompt has **128** decode
positions per layer (MEASURED, §1), i.e. a 64/64 split: 64 < 100 in both
halves, 0/36 layers qualify, on every pair. The fit side is satisfied
(128 ≥ 100); it is the within-trace ceiling `D` on the eval side that has no
room. The brief's premise — "128 decode positions makes the decode regime
testable" — holds only for the fit side; **MEASURED refutation** of that
premise. No floor was relaxed and no pair was manufactured to get around it.

---

## 1. Adapter: aubier's rows → the format the rule reads

The rule expects `R <ubatch> L <layer> T <n_tokens> <4*n_tokens ids>` and
derives its regime from `T` (`decode` = T==1, `prefill` = T>1). Aubier's rows
are **one position each with exactly 4 ids**: there is no T at all, hence no
row-shape signal separating prefill from decode.

**Design chosen (the brief's clean option):** the phase is carried by WHICH
FILE SET is written, never by an invented `T`. `--mode prompt` writes only
positions `< n_prompt`, `--mode decode` only positions `>= n_prompt`,
`--mode whole` everything. Every emitted row is one position, `T 1`, 4 ids;
weights are dropped (our format has no weight column) after being used as an
instrument check.

**Read with every number:** because `T==1` everywhere, the rule's populated
regime block is labelled `decode` in *all three* runs, and its `prefill` block
is structurally empty. The phase is named by the run (`-decode` / `-prompt` /
`-whole`), **never** by the rule's regime label. The `prefill` block of any run
below is empty for this reason alone, so nothing in it is a prefill measurement.

`n_prompt` is taken from aubier's **own run record**, not from the brief and not
from a hardcoded table: the adapter parses `n_prompt=N + n_decode=128 = M
positions` out of each trace's `.stdout.txt` and cross-checks it against
`M == positions present per layer`. The trace data therefore validates the
prompt lengths rather than being fitted to them, and the two independent
sources agree for all 10 traces (MEASURED):

| trace | n_prompt | n_decode | positions/layer (counted) | kept in `prompt` | kept in `decode` |
|---|---|---|---|---|---|
| A1 | 66 | 128 | 194 | 66 | 128 |
| A2 | 65 | 128 | 193 | 65 | 128 |
| A3 | 116 | 128 | 244 | 116 | 128 |
| A4 | 53 | 128 | 181 | 53 | 128 |
| A5 | 48 | 128 | 176 | 48 | 128 |
| A6 | 67 | 128 | 195 | 67 | 128 |
| B1 | 27 | 128 | 155 | 27 | 128 |
| B2 | 71 | 128 | 199 | 71 | 128 |
| B3 | 36 | 128 | 164 | 36 | 128 |
| B4 | 111 | 128 | 239 | 111 | 128 |

Validation performed on every source row before a byte was written (all
MEASURED, 0 failures): 11 whitespace fields; **all 4 ids inside `[0,128)`**
(0 out-of-range ids in any of the 10 traces — the corruption mode that excluded
`dump-gate.txt` in our own collection is absent here); the 4 ids distinct within
a row; the 4 router weights summing to 1 within 1e-3 (all 69,840 rows: max
|Σw−1| = 1.3e-07, MEASURED); no duplicate `(layer,pos)`; layers exactly `0..35`; positions exactly
`0..M-1` per layer, i.e. a contiguous position axis; and after writing, each
emitted file is read back and its row count equals `36 × kept positions`
(MEASURED: 972–8784 rows/file).

**Independent corroboration of the phase boundary (MEASURED, and it is the
phase control).** The treat arm is a different decode trajectory, so the ctrl
and treat captures must agree on prompt positions and diverge in decode. At
layer 0, per trace, prompt positions bit-exact / decode positions equal:

| trace | prompt positions identical | decode positions identical |
|---|---|---|
| B1 | **27/27** | 12/128 |
| B2 | **71/71** | 3/128 |
| B3 | **36/36** | 2/128 |
| B4 | **111/111** | 45/128 |

The 100% agreement stops exactly at `n_prompt` in all four held-out traces, and
the decode agreement (12/3/2/45 of 128) reproduces aubier's recorded numbers
exactly. The phase boundary the adapter uses is therefore measured twice, from
two independent artifacts, and agrees.

## 2. Exclusions (re-measured, not cited)

| input | disposition | evidence this session |
|---|---|---|
| `trace-ctrl_B1-rep2.txt` | **excluded** | `sha256sum` = `d3cfb390…8ef94ee`, **identical** to `trace-ctrl_B1.txt` — the byte-identical repeat behind aubier's 0.5521→0.5704 correction. Counting both would double one request. |
| `trace-A5-rebuild.txt` | **excluded** | `sha256sum` = `ed0523a4…808e6c39`, **identical** to `trace-A5.txt`. |
| `trace-treat-B1..B4` | **excluded whole** | NOT extra power: a different DECODE trajectory (§1 table: layer-0 decode agrees on only 12/3/2/45 of 128). Treating it as a fifth held-out request would treat a divergent generation as independent evidence of the same prompt. |
| `trace-ctrl32_B1` | not used | 32-layer variant, not a member of the m1-u6 36-layer set. |

Our own 96 per-conversation census traces are **not** inputs here (the brief's
scope; they are also under a read-only tree).

## 3. Results per configuration

Every command below ran from the taillis repo root; `EXIT=0` is the gate.
Adapter: `venvs/convert/bin/python3 results/holdout-transfer-adapter.py
--mode {prompt,decode,whole} --out /tmp/t22b_v4/{prompt,decode,whole}` →
**EXIT=0, EXIT=0, EXIT=0**.

Rule (one invocation per phase set, path-only, unedited):

```
T22B_TRACE_DIR=/tmp/t22b_v4/<mode> T22B_TRACES=A1,A2,A3,A4,A5,A6,B1,B2,B3,B4 \
T22B_OUT_JSON=$PWD/results/holdout-transfer-<mode>.json \
venvs/convert/bin/python3 tools/t22b_cross_request.py > results/holdout-transfer-<mode>.txt
```

| run | output | exit |
|---|---|---|
| decode-only | `results/holdout-transfer-decode.txt` | **0** |
| prompt-only | `results/holdout-transfer-prompt.txt` | **0** |
| mixed whole-stream | `results/holdout-transfer-whole.txt` | **0** |
| descriptive recompute (§5) | `results/holdout-transfer-descriptive.txt` | **0** |

The rule's own dedup reports `cross-trace duplicate ubatches present at 0/36
layers` in every run — correct and uninformative here, because the probe-artefact
test is byte-identity of a **multi-token** ubatch and every adapter row is T=1.
See §6 for why that matters on this data.

### 3a. DECODE-only — INSUFFICIENT (structural)

MEASURED: all 10 traces carry 128 decode tokens/layer, 36/36 layers ≥ the
100-token *fit* floor, so all 10 are eligible and 90 ordered pairs are formed —
and **0 layers** pass on the eval side, because `n_fit = n_held = 64 < 100`.
Every pair prints `INSUFFICIENT (no layers pass floors)`; both gated k give
`INSUFFICIENT`; regime `INSUFFICIENT`; `prefill` empty; OVERALL `INSUFFICIENT`.

This is exactly the regime our own census could not test either (0/96 eligible
at 64 generated tokens). **The independent population does not close the hole,
because the hole is the floor, not the population:** 128 decode positions is
still 72 short of the 200 an eval trace needs. Q1's "R4's decode-regime
objection is UNANSWERED, not refuted" therefore stands — now with a second,
independent measurement of *why*.

### 3b. PROMPT-only — INSUFFICIENT (structural)

MEASURED: 27–116 prompt positions/layer. Only A3 (116) and B4 (111) reach the
100-token *fit* floor; 2 ordered pairs (`A3>B4`, `B4>A3`), **0 layers** pass
(halves 58/58 and 55/56 < 100), so both are `INSUFFICIENT`, the regime is
`INSUFFICIENT` and flagged `THIN` (fewer than 3 eligible traces). OVERALL
`INSUFFICIENT`.

### 3c. MIXED whole-stream — the only configuration the floor admits, and it is NOT a regime

Prompt and decode positions pooled per trace; totals 155–244 per layer, so
**A3 (244) and B4 (239)** are the only eval traces with ≥200 tokens and
therefore the only ones that can carry the within-trace ceiling. 18 of the 90
ordered pairs are evaluable, all at `nL=36` layers, with the ceiling established
on both eval traces (`z_within` = +47.5 / +50.1 at k=8).

**This number conflates the two regimes and its rule-label `decode` is an
artifact of T==1** (~47% of A3's and ~46% of B4's pooled tokens are prompt
positions). It is reported as a *floor-admissible diagnostic*, not as a regime
verdict, and it gates nothing.

Per-pair verdicts, k=8 and k=16 (gated ks; MEASURED, `results/holdout-transfer-whole.txt`):

| pair | k=8 C | retain | z_cross | verdict | k=16 C | retain | z_cross | verdict |
|---|---|---|---|---|---|---|---|---|
| **A3→B4** | **0.4713** | **1.05** | **+49.46** | **TRANSFERS** | **0.6435** | **1.03** | **+47.35** | **TRANSFERS** |
| **B4→A3** | **0.5339** | **0.98** | **+51.07** | **TRANSFERS** | **0.6979** | **0.95** | **+46.51** | **TRANSFERS** |
| A1→A3 | 0.0851 | 0.05 | +2.09 | COLLAPSE | 0.1443 | 0.03 | +1.36 | COLLAPSE |
| A1→B4 | 0.0896 | 0.07 | +2.80 | COLLAPSE | 0.1581 | 0.07 | +2.84 | COLLAPSE |
| A2→A3 | 0.0597 | −0.01 | −0.11 | COLLAPSE | 0.1154 | −0.02 | −0.76 | COLLAPSE |
| A2→B4 | 0.0634 | 0.00 | −0.11 | COLLAPSE | 0.1198 | −0.01 | −0.57 | COLLAPSE |
| A4→A3 | 0.0764 | 0.03 | +1.32 | COLLAPSE | 0.1386 | 0.02 | +0.94 | COLLAPSE |
| A4→B4 | 0.0769 | 0.04 | +1.45 | COLLAPSE | 0.1344 | 0.02 | +0.61 | COLLAPSE |
| A5→A3 | 0.0453 | −0.04 | −2.25 | COLLAPSE | 0.1042 | −0.03 | −1.80 | COLLAPSE |
| A5→B4 | 0.0480 | −0.04 | −2.31 | COLLAPSE | 0.1069 | −0.04 | −1.96 | COLLAPSE |
| A6→A3 | 0.0379 | −0.05 | −2.85 | COLLAPSE | 0.0744 | −0.08 | −4.38 | COLLAPSE |
| A6→B4 | 0.0378 | −0.06 | −3.47 | COLLAPSE | 0.0698 | −0.11 | −5.43 | COLLAPSE |
| B1→A3 | 0.0520 | −0.02 | −1.09 | COLLAPSE | 0.0907 | −0.06 | −2.82 | COLLAPSE |
| B1→B4 | 0.0531 | −0.02 | −1.37 | COLLAPSE | 0.0919 | −0.07 | −3.20 | COLLAPSE |
| B2→A3 | 0.0486 | −0.03 | −1.38 | COLLAPSE | 0.0984 | −0.04 | −2.27 | COLLAPSE |
| B2→B4 | 0.0548 | −0.02 | −1.17 | COLLAPSE | 0.1079 | −0.03 | −1.81 | COLLAPSE |
| B3→A3 | 0.0572 | −0.01 | −0.52 | COLLAPSE | 0.1034 | −0.04 | −1.48 | COLLAPSE |
| B3→B4 | 0.0530 | −0.02 | −1.25 | COLLAPSE | 0.0953 | −0.06 | −2.65 | COLLAPSE |

Uniform expectation U = 0.0625 (k=8) / 0.1250 (k=16). Capture medians: the 16
COLLAPSE pairs sit at 0.0378–0.0896 (k=8), i.e. 0.6–1.4× U, and 11 of 16 are
*below* U; the 2 TRANSFERS pairs sit at 0.4713 / 0.5339 = 7.5× / 8.5× U.

Regime verdict under the rule: `k=8: INSUFFICIENT`, `k=16: INSUFFICIENT`
(transfer mixed with collapse — pair-contradicted), regime `INSUFFICIENT`,
OVERALL `INSUFFICIENT`.

Rule-wording observation (reported, not fixed — the file is off limits): the
OVERALL reason string says "no regime yielded an evaluable pair" although 18
pairs *were* evaluated; that message is what the mapping prints whenever a
regime's k-verdict is `INSUFFICIENT`, and it does not distinguish
"nothing evaluable" from "pair-contradicted". The per-pair verdicts and the
capture JSON carry the real detail.

## 4. What the one live structure is — and why it is not cross-request generality

MEASURED, independent of the verdict: the per-layer top-8 routing overlap.

| pair | mean overlap over 36 layers | rule's null p95 (k=8) |
|---|---|---|
| A3/B4 | **0.736** | 0.087 |
| every other pair with layers | 0.024–0.090 | 0.087 |

MEASURED, from the prompts' own token streams
(`tokens/tokens-*.txt`, prompt positions only): the two prompts that transfer
have an **identical distinct-token SET** — Jaccard 1.00 on a vocabulary of 12
distinct tokens — while sharing no prefix (common-prefix length 0) and having a
sequence similarity of 0.489. Their pairwise token-set Jaccard against every
other trace is 0.00–0.04.

**INTERPRETATION.** A3 and B4 are *different* prompts of the *same content
class*: number lists drawn from the same small vocabulary in a different order.
Every cross-class pair on the same population collapses to uniform. So the
cross-population transfer measured here is a **within-content-class** effect
(a hot set fitted on one prompt captures mass on another prompt built from the
same token population), not evidence that hot sets generalise across request
*content*. For a shipped per-model allocation this is the decisive distinction:
such a policy is worth its cost only if the deployment population is narrow
enough that "another request" means "another example of the same kind".

**The rule cannot see this, and that is a real limitation of applying it to
per-position dumps** (INTERPRETATION, stated as an objection, not a defect I
patched): its independence check is byte-identity of multi-token ubatches. Here
every row is T=1, so the check is structurally inert, and two prompts that share
a token population but not a byte pass as independent. Recorded as an input
caveat for this result, not applied to our own t22b/q1 runs (their rows are real
ubatches, where the check does operate).

## 5. Descriptive only — the direction the rule cannot gate

`results/holdout-transfer-descriptive.txt` recomputes the rule's own operator
(captured mass `C` of X's top-k fit on X's full stream, evaluated on Y's full
stream) with **no floor, no z, no retain and no verdict**, on the phase-filtered
sets, to answer "is there a direction at all". It must not be quoted as a
TRANSFERS/COLLAPSE result, and it adds nothing to the rule's eligible set.

At k=8, U = 0.0625 (×U in parentheses):

| pair (content class) | decode-only C | prompt-only C |
|---|---|---|
| A3→B4 / B4→A3 (number list) | 0.4428 (7.1×) / 0.5227 (8.4×) | 0.5071 (8.1×) / 0.5349 (8.6×) |
| A2→B2 / B2→A2 (code) | 0.2240 (3.6×) / 0.2321 (3.7×) | 0.3311 (5.3×) / 0.3111 (5.0×) |
| A1→A4 / A4→A1 (prose) | 0.1272 (2.0×) / 0.1433 (2.3×) | 0.1334 (2.1×) / 0.1275 (2.0×) |
| A2→A3 (code→number) | 0.0466 (0.7×) | 0.0854 (1.4×) |
| A6→B4 (code→number) | 0.0400 (0.6×) | 0.0459 (0.7×) |
| A5→B4 (prose→number) | 0.0515 (0.8×) | 0.0719 (1.2×) |

The pattern is the same in both phases: same-class pairs are 2–9× uniform,
cross-class pairs are 0.6–1.5× uniform and frequently *below* it. INTERPRETATION:
the DECODE regime, could it be gated, would most likely return the same
content-class structure as the mixed run — not a collapse and not generality.

## 6. What this does and does not establish

* **DOES (MEASURED):** on an independent held-out population, cross-request
  hot-set transfer is strong between prompts of the same content class and
  vanishes (at or below uniform) between prompts of different classes, in the
  one configuration where the rule's floors admit an evaluation.
* **DOES (MEASURED):** the DECODE regime remains INSUFFICIENT on 128-position
  decode sets under the unchanged floors — the ceiling needs 200 tokens/layer.
  That is a finding about the data's size, not a failure of the rule, and it
  means **this population does not close our decode hole**.
* **DOES NOT:** say anything about quality, perplexity or speed. It is a
  transfer-of-selection-mass result only.
* **DOES NOT:** establish per-model generality. n=10 traces, one model, one
  collection session, one prompt-design template per class (2 prompts of the
  number-list class among 10 traces), and the two TRANSFERS pairs rest on a
  *single* prompt pair each. Any "generalises" reading of §4 is a reading of
  n=1 pair per class.
* **What would settle it:** a held-out set whose eval traces carry ≥200 tokens
  **per regime per request** (e.g. ≥200 generated tokens per prompt), and ≥3
  independent prompts *per content class*. Then the rule's decode floor is met
  without pooling and the same thresholds can decide the regime.
* Our own census verdict is unchanged and not contradicted: PREFILL transfer
  across single conversations stands (PROCEED_TENTATIVE_WEAK), decode remains
  UNANSWERED, and this run adds a second independent measurement of the reason.

## 7. Files produced by this unit

| file | content |
|---|---|
| `results/holdout-transfer.md` | this write-up |
| `results/holdout-transfer-adapter.py` | the adapter (copy of the script that ran; also at `/tmp/t22b_v4/adapter.py`) |
| `results/holdout-transfer-decode.txt` / `.json` | rule output, decode-only set, EXIT 0 |
| `results/holdout-transfer-prompt.txt` / `.json` | rule output, prompt-only set, EXIT 0 |
| `results/holdout-transfer-whole.txt` / `.json` | rule output, mixed whole-stream set, EXIT 0 |
| `results/holdout-transfer-descriptive.txt` | §5 descriptive recompute, EXIT 0 |

Nothing under the aubier tree and nothing in the rule was written, moved or
edited; aubier's trees were opened read-only.
