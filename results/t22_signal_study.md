# T2.2 Stage A — does any STABLE per-expert signal exist to allocate per-group types against?

Unit taillis-t22a · 2026-10-14 · instrument: `tools/t22_signal_study.py` · host-side only, no GPU.

## Verdicts

| run | rule | long (PRIMARY) | chat | code | OVERALL |
|---|---|---|---|---|---|
| 1 | P3 mis-implemented as absolute cap `adaptive−static ≤ 0.01` | **FAIL** (P3 +0.0147 vs cap 0.01) | INSUFFICIENT | INSUFFICIENT | **NULL** |
| 2 (**rule of record**) | P3 as the brief's comparative criterion: `static−uniform > adaptive−static` at every k | **PASS** | INSUFFICIENT | INSUFFICIENT | **PROCEED_TENTATIVE** |

**Rule amendment (explicit, both runs published).** Run 1's P3 was my own absolute cap, not the
brief's criterion. The brief says: "A policy is only interesting if the fixed set beats uniform on
held-out by more than the rotating baseline's advantage explains" — comparative, not absolute. Run 1
failed that arm by 0.005 and produced a spurious NULL. Run 2 implements the brief's sentence; the
substantive change: run 1's raw numbers were already consistent with a strong stable signal
(P1 median ρ=0.774, P2 z≈42, static keeps 93–96% of the adaptive policy's total gain). Both runs
exit 0; both full outputs are in this directory (`*_run1_misruled.txt`, `*_run2_rule_of_record.txt`).

**Answer to the unit question:** on the only trace with adequate power (dump-long), YES — per-expert
routing labels are strongly stable under a temporal held-out split within the request, a static
per-layer top-k set beats uniform by 18–31 points of held-out selection mass (z vs random-subset
control ≈ 35–42), and an adaptive re-picking baseline recovers only ~4–7% more than the static set.
BUT: (a) this is stability WITHIN ONE REQUEST — cross-request/cross-workload stability is UNRESOLVED
because chat/code are ~5× underpowered, hence PROCEED_TENTATIVE, not PROCEED; (b) a stable signal is
not a quality claim — only a Stage B build-and-evaluate loop can speak to quality.

## Held-out split (stated)

Per layer, tokens in appearance order (ubatch counter verified strictly monotonic, 0 violations, all
three dumps): first ⌊T/2⌋ tokens = FIT, rest = HELD-OUT. dump-long: 2419/2419 tokens per layer
(35 layers; ~4799 prefill + 39 decode tokens/layer — the split is therefore prefill-vs-prefill, the
regime T2.2 targets). dump-chat: 487/488. dump-code: 490/491. Layer 35 has NO prefill rows in ANY
dump (dump artifact); long L35 (40 tokens) is excluded by the 100-token/half layer floor; chat/code
L35 (384 decode tokens) is included.

## Decision rule (fixed in advance; full text in the script docstring)

- Layer floor 100 tok/half; trace floor: median held tokens/layer ≥ 1000 else INSUFFICIENT.
  Rationale: at ~488 held tokens (~1.95k selections over 128 experts) binomial noise attenuates a
  PERFECT split-half Spearman to ~0.4–0.5 — below reachability of a decisive verdict at the P1 bar;
  chat/code therefore cannot decide P1 and must return INSUFFICIENT, not FAIL.
- **P1 label stability:** median over layers of Spearman(fit counts, held counts) ≥ 0.30 and above
  the permutation null's p95 (200 within-layer label permutations; null statistic = same
  median-over-layers). Instrument self-check: null mean must be ≈ 0 (observed −0.001…+0.001) else
  exit nonzero.
- **P2 fixed set beats uniform:** mean over layers and k ∈ {8,16,32,64} of (static − k/128) ≥ 0.02,
  AND pooled z vs a 200-draw random-k-subset control ≥ 4.0 at k ∈ {8,16} (the guard runs where the
  design has power; k=32/64 reported, not gated). Instrument self-check per layer×k: control mean
  must sit within 4 s.e. of k/128 (held everywhere, all traces) — else exit nonzero.
- **P3 not dominated by adaptive** (comparative, run 2): (static−uniform) > (adaptive−static) at
  every k; adaptive re-picks top-k every 512 held-out tokens from all strictly-earlier tokens.
  On chat/code the held-out stream is shorter than one window → adaptive ≡ static → P3 vacuous
  (flagged).
- Overall map: primary INSUFFICIENT→INSUFFICIENT; primary FAIL w/o secondary PASS→NULL; primary
  FAIL w/ secondary PASS→INSUFFICIENT; primary PASS w/ secondary FAIL→INSUFFICIENT
  (workload-contradicted); primary PASS all PASS→PROCEED; primary PASS secondaries only
  INSUFFICIENT→**PROCEED_TENTATIVE** (invariance not established).

## Results — dump-long (PRIMARY; 1755 rows, 677,480 slots, T-hist {1:1405, 191:35, 512:315})

- **P1: median split-half ρ = 0.774** (per-layer range 0.72–0.84 over L0–L34); permutation null
  p95 = 0.029. Every included layer individually clears the 0.30 bar. The "hot set rotates" prior is
  refuted *within this request*: the first and second halves of one ~4.8k-token prefill rank experts
  almost identically.
- **Hot-set test** (captured fraction of held-out selection mass, mean over 35 included layers):

| k | static | uniform | gap | adaptive | adaptive−static | z_pool | static keeps |
|---|---|---|---|---|---|---|---|
| 8 | 0.2382 | 0.0625 | +0.1757 | 0.2506 | +0.0125 | +41.6 | 93.4% |
| 16 | 0.3717 | 0.1250 | +0.2467 | 0.3872 | +0.0156 | +42.4 | 94.1% |
| 32 | 0.5569 | 0.2500 | +0.3069 | 0.5762 | +0.0193 | +42.1 | 94.1% |
| 64 | 0.7886 | 0.5000 | +0.2886 | 0.8002 | +0.0116 | +34.9 | 96.1% |

  Static beats uniform 3–4× over at k=8; the online re-picking policy adds only ~1.2–1.9 points on
  top of a ~25-point advantage. P2 and P3 pass with wide margin.
- **Feature correlations vs held-out frequency** (4480 pooled points; 200-perm within-layer control;
  verdict needs |ρ| ≥ 0.20 AND above the control's p99):

| feature | pooled Spearman | pooled Pearson | per-layer median ρ | null p99 | verdict |
|---|---|---|---|---|---|
| imatrix gate_up in_sum2 (sum per expert; mean affinely identical) | **+0.215** | +0.212 | **+0.475** | 0.043 | **SIGNAL** |
| imatrix down in_sum2 (sum per expert) | +0.156 | −0.006 | +0.341 | 0.048 | above control, below magnitude bar |
| scale_inv gate_up_proj | +0.034 | +0.051 | +0.086 | 0.043 | none beyond control |
| scale_inv down_proj | −0.032 | +0.047 | −0.056 | 0.063 | none beyond control |

  The imatrix per-expert activation energy (gate_up input) is a real, offline-computable covariate
  of routing frequency at the per-layer median (+0.475) — the strongest Stage-B-relevant positive
  finding, because a policy built on it needs no trace-fitting at all. The static FP8 weight scales
  carry NO routing signal in any trace. Pearson vs Spearman disagree on `down` (−0.006 vs +0.156):
  the rank association is real but the level relationship is not linear — use ranks, not raw energy.

## Results — chat & code (workload-invariance arm)

- chat: split 487/488; median split-half ρ = 0.302; gaps +0.033…+0.058 (z_pool 6.6–7.9); verdict
  **INSUFFICIENT** (488 < 1000 floor). code: split 490/491; median ρ = 0.264; gaps
  +0.032…+0.065 (z_pool 8.6–10.1); verdict **INSUFFICIENT** (491 < 1000).
- Reading: both traces point the SAME direction as long (positive stability, static > uniform) at
  magnitudes consistent with a strong signal attenuated by a ~0.4–0.5 reliability ceiling — but they
  cannot decide it, and no direction is claimed from them. **Whether the signal is portable across
  workloads is UNRESOLVED**, hence PROCEED_TENTATIVE rather than PROCEED. The same feature
  correlations are consistent-directionally (gate_up energy: per-layer median +0.114 chat, +0.366
  code) but weaker/noisier; only long has the power to call the feature.
- Features on chat/code: scale_inv gate_up (chat) and gate_up+down (code) land "above control, below
  magnitude bar" (pooled |ρ| 0.13–0.18 vs p99 ≈ 0.02) — suggestive, not called.

## What these traces do NOT support (discipline)

1. **Within-request ≠ cross-request.** Each dump is ONE session (one prefill + its decode). The
   split tests generalization across position within a request. It says nothing about a set fitted
   on request A transferring to request B, except weakly via the imatrix-feature correlation.
2. **Chat/code underpowered** — the invariance arm is one powered trace plus two consistency
   checks, not a proof of invariance. If Stage B needs a per-workload claim, collect ≥2k-token
   held-out halves per workload (≈ a 4× larger dump) or accept per-request online refitting.
3. **Stability ≠ quality.** Nothing here measures perplexity, latency, or accuracy. A stable signal
   is necessary-not-sufficient for T2.2; only build-and-evaluate can speak to quality.
4. All percentages are percentage POINTS of held-out selection mass (4 selections/token/layer).
5. P2's gap arm is near-tautological under any skew (route_analysis.json already shows max:min
   ≈ 2.1–2.6 with coverage 1.0 at 256 tokens on all three workloads — cited, not re-derived). The
   load-bearing tests are P1 and P3.
6. dump-long's held-out half is ~98% prefill; decode-time routing (chat's dominant regime) is
   unresolved at this sample size. Layer 35's missing prefill rows are a dump artifact (all three
   dumps); its imatrix/scale features exist and were used only where its counts passed the floor.

## Gate / cannot-verify

- `venvs/convert/bin/python3 tools/t22_signal_study.py` → exit 0, runtime 6.1 s (run 1: 7.8 s).
  Exit 0 = a decision was produced (any verdict, incl. INSUFFICIENT/NULL); nonzero = real error
  (malformed trace, shape mismatch, failed instrument self-checks).
- Cannot verify host-side: anything about GPU dispatch behaviour, whether the imatrix corpus overlaps
  the trace prompts (it is not recorded in the imatrix metadata), and the semantics of the trace's
  `token_idx` beyond "monotonic ubatch counter" (aubier is read-only for this unit).

## Artifacts

- `tools/t22_signal_study.py` — instrument + fixed rule (docstring = authoritative rule text).
- `results/t22_signal_study_run1_misruled.txt` — run 1 full output (NULL; retired P3).
- `results/t22_signal_study_run2_rule_of_record.txt` — run 2 full output (PROCEED_TENTATIVE).
