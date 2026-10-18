# COLD-BAND — is there a stable contiguous cold band in the expert axis?

**Deciding measurement for `layout.md` §10.7** (should `taillis.variant_band` exist at all, or should
the variant use the manifest row-range design of `aubier/docs/expert-swap.md` §4?). Unit
`taillis-coldband`, 2026-10-18. Pure trace analysis: **no GPU, no model run, no artifact touched**;
both input trees read-only. Claims are marked **MEASURED** / **INTERPRETATION** / **PROJECTION**;
**RELAYED** marks a fact taken from the brief or from a prior record and used as an input without
re-deriving it.

## Verdict (short)

> **NO STABLE BAND.** Under the pre-declared rule, in **both** regimes, on **both** populations, for
> every width tested. The best contiguous width-4 window carries **0.78–0.89× the uniform share**
> (criterion (i) asks for ≤ 0.50×), it is coldest-or-near in **0–6 of the 35–36 defined layers**
> (criterion (ii) asks ≥ 30), it is **not** reproduced on the held-out halves of both populations
> (criterion (iii)),
> and it does not clear the permutation null (criterion (iv)).
>
> The decisive fact is stronger than "the band is not cold enough": **the coldest contiguous window
> is exactly as cold as random relabelling of the same count vectors makes it** — observed per-layer
> minimum **0.973–1.022×** the min-of-125 null mean (MEASURED). There is no contiguous cold
> *structure* to stabilise, per layer or file-wide.
>
> §10.7's own consequence therefore applies: **use the manifest row-range design and leave
> `variant_band` unimplemented.** An absent key costs nothing (§7.2 rules 1–2), which was the point of
> putting it in the `taillis.*` namespace.

---

## 1. Data, populations, exclusions

Two populations, deliberately pooled across the lane seam.

* **G — general sample (RELAYED + MEASURED).** aubier's `m1-u6` routing traces: 10 sessions,
  `A1…A6` (derivation set) and `ctrl_B1…B4` (held-out set). Rows are
  `R <layer> <pos> <sel0..3> <w0..w3>`. Prompt length per session is **MEASURED from each session's
  own run log** (`n_prompt=P + n_decode=D`), not assumed: 66/65/116/53/48/67 for A1–A6 and
  27/71/36/111 for B1–B4. The four B values **reproduce the brief's stated lengths exactly**
  (RELAYED values 27/71/36/111 vs MEASURED 27/71/36/111), which is the cross-check that the
  extraction is right. All sessions have 128 decode positions.
* **T — target corpus (MEASURED).** 96 per-conversation routing dumps of the agent population the
  product serves: `R <ubatch> L <layer> T <n_tokens> <4*T expert ids>`, 186,767 data rows. Regime is
  `T > 1 ⇒ prefill`, `T == 1 ⇒ decode` (the split q1-census used; it also absorbs the L35 dumper
  artifact — see §2).

**Exclusions, and why** (all RELAYED as pre-verified in the brief; the two duplicate claims are
re-verified here by sha256):

| excluded | reason |
|---|---|
| `trace-ctrl_B1-rep2` | **MEASURED** byte-identical to `ctrl_B1` (sha256 `d3cfb390ba976044…`) — a repeat, not a second session |
| `trace-A5-rebuild` | **MEASURED** byte-identical to `A5` (sha256 `ed0523a483932a94…`) |
| `trace-treat-B1…B4` | the **treatment** arm: canonical (pre-permutation) router ids but a *different decode trajectory* — layer-0 prompt phase is bit-exact ctrl-vs-treat while layer-0 decode agrees on only 12/128, 3/128, 2/128, 45/128 (RELAYED). One prompt must contribute one trajectory |
| `trace-ctrl32_B1` | same prompt as `ctrl_B1`, re-run at a different activation precision, and not in the named set — excluded to avoid double-counting a prompt |

## 2. Validation, run before any statistic

**MEASURED, 0 problems.** Every expert id in `[0,128)`; 36 layers present; row counts consistent with
the file; prompt-phase row counts equal to the stated prompt lengths.

| check | result |
|---|---|
| G: sessions | 10; rows per session 5,580–8,784 |
| G: positions vs `n_prompt`/`n_decode` | exact for all 10 sessions (e.g. 66/128, 27/128) |
| G: layers per regime | 36/36 in prefill, 36/36 in decode, all sessions |
| G: max expert id | 127 |
| T: conversations | 96; 186,767 data rows |
| T: ids per row | `== 4 × T` on every row; all ids in `[0,128)` |
| T: ubatch order | monotone non-decreasing in all 96 files |
| T: layer coverage | 349 prefill row-sets, **all 349 with 35 of 36 layer rows** (no L35 row); 4,942 decode row-sets, of which 4,846 carry all 36 layers and **96 are the lone L35 row** emitted inside a prefill ubatch's group — the known dumper artifact (RELAYED from q1-census; **MEASURED** here as 349/96) |
| T: L35 prefill | **zero** prefill rows in every conversation ⇒ T/prefill has **35 defined layers** |

**A zero vector is absent, not cold.** T/prefill's L35 has no rows in any conversation, so its share
vector is all zeros and would score as the coldest possible band. Undefined layers are **excluded
from every statistic** and named (only `T/prefill: [35]`). This is the one place where a careless
reading of the same data would have manufactured a "STABLE BAND" for free.

**Adequacy** (pre-declared floor: median ≥ 100 tokens per layer): G/prefill **660**, G/decode **1,280**,
T/prefill **146,007**, T/decode **4,846** tokens per layer. All four cells are adequate, so no cell
returns `INSUFFICIENT`. **MEASURED** caveat: T/decode's 4,846 tokens are pooled over 96 conversations
(`n_predict=64` per conversation — the G10 collection limit), so decode is adequate *in aggregate*,
not per session.

## 3. Metric

**SELECTION COUNT per (layer, expert)**, each of the 4 ids in a row counting once, normalised per
layer (every layer sums to 1). Uniform share of one expert is 1/128; **a width-w window at uniform is
`w/128`** (0.03125 for w=4). This is the only metric computable on both populations — G carries
weights, T does not — and it is never mixed with weight mass. Weight mass appears **only** as a
labelled secondary check in §8.

**Prefill and decode are reported separately throughout** (never pooled), because our own census found
the regimes behave differently.

## 4. Deliverable 1 — per-layer coldest contiguous window

Coldest `[lo, lo+w)` in each layer (non-wrapping, ties → lowest `lo`), and how many **distinct**
windows the 36 layers choose:

| cell | w | distinct windows chosen | available | mean of the per-layer minimum | uniform w/128 |
|---|---|---|---|---|---|
| G/prefill | 3 | 31 | 126 | 0.00503 (**0.21×**) | 0.02344 |
| G/prefill | **4** | **33** | 125 | 0.00859 (**0.27×**) | 0.03125 |
| G/prefill | 5 | 30 | 124 | 0.01236 (0.32×) | 0.03906 |
| G/prefill | 8 | 30 | 121 | 0.02719 (0.44×) | 0.06250 |
| G/decode | 4 | 33 | 125 | 0.01203 (0.39×) | 0.03125 |
| T/prefill | 3 | 30 | 126 | 0.00665 (0.28×) | 0.02344 |
| T/prefill | **4** | **31** of 35 | 125 | 0.01074 (**0.34×**) | 0.03125 |
| T/prefill | 8 | 30 | 121 | 0.02971 (0.48×) | 0.06250 |
| T/decode | 3 | 32 | 126 | 0.00691 (0.29×) | 0.02344 |
| T/decode | **4** | **30** | 125 | 0.01132 (**0.36×**) | 0.03125 |
| T/decode | 8 | 29 | 121 | 0.03197 (0.51×) | 0.06250 |

**MEASURED: the layers almost never agree.** 29–34 distinct windows across the 35–36 layers (out of
121–126 available), in every cell and at every width. §10.2 declares **one** band for the whole file, so this
is already the crux: a single `[lo,hi)` that must be cold in *all* 36 layers has almost no support.

**But the per-layer minimum is also not evidence of structure** — see §6. The per-layer minimum is
the min over 125 windows, so it is *expected* to be well below uniform even in pure relabelled noise
(null mean 0.27×–0.39× uniform, i.e. the same as observed). Read deliverable 1 only against that
null; against uniform alone it overstates coldness by ~4×.

## 5. Deliverable 2 — the best global band

The single window minimising the summed normalised share over all layers:

| cell | w | band | mean per-layer mass | ratio to uniform | exact-coldest layers | layers warmer than uniform | layers above ½×uniform | rank of S |
|---|---|---|---|---|---|---|---|---|
| G/prefill | 3 | 74–76 | 0.01799 | 0.768 | 1/36 | 7 | 26 | 1/126 |
| G/prefill | **4** | **74–77** | **0.02687** | **0.860** | 0/36 | **9** | **29** | 1/125 |
| G/prefill | 5 | 35–39 | 0.03476 | 0.890 | 0/36 | 10 | 31 | 1/124 |
| G/prefill | 8 | 112–119 | 0.05607 | 0.897 | 0/36 | 11 | 35 | 1/121 |
| G/decode | 4 | 67–70 | 0.02771 | 0.887 | 0/36 | 11 | 33 | 1/125 |
| T/prefill | 3 | 32–34 | 0.01839 | 0.785 | 0/35 | 4 | 29 | 1/126 |
| T/prefill | **4** | **31–34** | **0.02433** | **0.778** | 0/35 | **4** | **31** | 1/125 |
| T/prefill | 5 | 30–34 | 0.03133 | 0.802 | 0/35 | 4 | 31 | 1/124 |
| T/prefill | 8 | 30–37 | 0.05127 | 0.820 | 0/35 | 8 | 33 | 1/121 |
| T/decode | 3 | 32–34 | 0.01899 | 0.810 | 1/36 | 6 | 31 | 1/126 |
| T/decode | **4** | **31–34** | **0.02643** | **0.846** | 2/36 | **8** | **33** | 1/125 |
| T/decode | 8 | 31–38 | 0.05570 | 0.891 | 0/36 | 10 | 36 | 1/121 |

("rank of S = 1" is by construction — the band *is* the argmin. The informative number is the null
comparison in §6.)

**Where the global band is NOT cold (MEASURED, width 4).** Mass **warmer than uniform** in
G/prefill layers **3,4,7,16,17,20,26,27,28**; G/decode **1,4,8,10,18,20,24,25,26,31,32**;
T/prefill **6,8,23,33**; T/decode **6,8,12,13,14,23,25,33**. Above **½×uniform** (i.e. failing
criterion (i) *at layer level*) in 29–33 of the 35–36 layers of every cell. The per-layer mass under
the G/prefill band ranges 0.0091–0.0644 — a 7× spread around a 2.7% mean.

**The axis has almost no contiguous dynamic range (MEASURED — the finding that matters most).**
Across all 125 candidate windows: 47–62% of them are *already* colder than uniform, and the hottest
window of all is only **1.11–1.35× uniform** (T/prefill, the flattest cell and the largest population:
hottest at lo 95–99 worth 1.20–1.35×, coldest at lo 30–34 worth 0.78–0.82×). So the whole contiguous axis spans roughly **0.8×–1.35× uniform**: there is no cold region,
only mild, broad, position-irrelevant variation. Under §10.2 the band would be a permanent structural
commitment to one of these nearly-equivalent windows.

## 6. Deliverable 3 — the permutation null

Null: per-layer **independent random relabelling of the 128 experts** (shares preserved per layer, so
the null is exact for the window statistic and destroys only expert identity and cross-layer
alignment). N = 2,000, seed 20261018. Two statistics, because they answer different questions:

* `p_min` — the **selection-adjusted** test: is the best of 125 windows colder than the best of 125
  windows under relabelling?
* `p_cand` — the test of the **specific** window: is *this* window colder than the same position
  under relabelling? (Used as rule criterion (iv), as pre-declared; it is the optimistic of the two,
  because the position was itself chosen from the data.)

| cell | observed min S | null min S (mean ± sd) | **p_min** | z_min | p_cand | z_cand | layers with per-layer p<0.05 | Bonferroni |
|---|---|---|---|---|---|---|---|---|
| G/prefill | 0.9674 | 0.9027 ± 0.0337 | **0.9815** | +1.92 | 0.0525 | −1.49 | 2/36 | 0 |
| G/decode | 0.9975 | 0.9602 ± 0.0263 | **0.9395** | +1.42 | 0.0370 | −1.69 | 0/36 | 0 |
| T/prefill | 0.8514 | 0.8863 ± 0.0303 | **0.1319** | −1.15 | 0.0015 | −2.54 | 2/35 | 0 |
| T/decode | 0.9513 | 0.9408 ± 0.0284 | **0.6112** | +0.37 | 0.0110 | −2.16 | 3/36 | 0 |

**MEASURED, and it is the metric trap in reverse:** the observed best-of-125 window is **not** below
the null's best-of-125 in any cell (`p_min` 0.13–0.98), and in **three of the four cells it is *warmer* than the null
mean** (z_min +0.37 to +1.92). `p_cand` looks small for T (0.0015) only
because it tests a window that was selected as the minimum — the same statistic, selection-adjusted,
gives 0.13. **A "cold band" here is what min-over-125-windows looks like on data with no contiguous
structure.** Any argument from `p_cand` alone would be reading the instrument.

**Why the observed minimum can be warmer than the null's (MEASURED diagnostic, INTERPRETATION of it).**
A cold band common to the layers would make the window surface *lumpier* than independence predicts
(cross-layer agreement inflates the variance of `S(lo)`). Measured spread of the surface across the
125 windows against the relabelled null's spread: **G/prefill 0.0913 vs 0.1029 (0.888×)**,
**G/decode 0.0581 vs 0.0739 (0.786×)**, **T/prefill 0.0952 vs 0.0948 (1.004×)**,
**T/decode 0.0736 vs 0.0818 (0.899×)**. The real surface is as flat as — or flatter than — the
no-structure null: there is **no excess cross-layer agreement anywhere in the expert axis**. So the
positive z is not "the data is actively warm"; it is "the surface is flat, and the null's lumpiness
is absent".


**Per-layer null (MEASURED):**

| cell | observed per-layer minimum (mean) | null min-of-125 mean | ratio obs/null | layers below null mean | layers below null 5th pct |
|---|---|---|---|---|---|
| G/prefill | 0.00859 (0.275×u) | 0.00858 | **1.000** | 19/36 | 2/36 |
| G/decode | 0.01203 (0.385×u) | 0.01176 | **1.022** | 15/36 | 0/36 |
| T/prefill | 0.01074 (0.344×u) | 0.01072 | **1.002** | 14/35 | 2/35 |
| T/decode | 0.01132 (0.362×u) | 0.01163 | **0.973** | 17/36 | 3/36 |

The per-layer "very cold window" (0.27×–0.39× uniform, §4) is **statistically indistinguishable from
noise** once its own selection is accounted for. Even a *per-layer* band — which §10.2 cannot express
anyway — would be built on nothing.

## 7. Deliverable 4 — stability

**Within-population split-half (MEASURED).** G halves = the pre-existing A (derivation) / B (held-out)
split; T halves = conversations by parity of their file index (pre-declared). Spearman is over the 128
per-expert shares, per layer, median over layers; null = per-layer relabelling of the second half
(1,000 permutations).

| cell | n layers | median ρ | pooled ρ | null mean | p | per-layer window agreement | file-wide window | p |
|---|---|---|---|---|---|---|---|---|
| G/prefill | 36 | 0.5739 | 0.5663 | 0.0000 | 0.001 | **6/36** | **111 vs 45** | 0.001 |
| G/decode | 36 | 0.5212 | 0.5034 | −0.0002 | 0.001 | **3/36** | **34 vs 5** | 0.005 |
| T/prefill | 35 | 0.9756 | 0.9781 | +0.0001 | 0.001 | 20/35 | 31 vs 31 | 0.001 |
| T/decode | 36 | 0.8726 | 0.8866 | −0.0005 | 0.001 | 15/36 | 31 vs 32 | 0.001 |

Per-expert *profiles* are stable (ρ 0.52–0.98, far above the null). **The chosen window is not**: within
the general sample's own halves it comes out at entirely different places (111 vs 45; 34 vs 5), and even
inside the target corpus only 15–20 of 35–36 layers pick the same window. Note the tension: a stable
per-expert profile with an unstable window is exactly what "no contiguous structure" looks like.

**Cross-population (MEASURED).** Aligned on the 35 (prefill) / 36 (decode) layers both populations
define; the band **application** uses each population's own whole cell.

| regime | median ρ | pooled ρ | null mean | p | same file-wide window? | layers picking the same window | p |
|---|---|---|---|---|---|---|---|
| prefill | 0.1652 | 0.1782 | −0.0002 | 0.001 | **no** (74 vs 31) | **1/35** | 0.266 |
| decode | 0.2694 | 0.2829 | +0.0006 | 0.001 | **no** (67 vs 31) | **3/36** | 0.003 |

| band applied | window | mean per-layer mass | ratio to uniform | rank among the target's 125 windows | layers warmer than uniform |
|---|---|---|---|---|---|
| G's prefill band → T | 74–77 | 0.03248 | **1.039** | 84/125 | **18/35** |
| T's prefill band → G | 31–34 | 0.02852 | 0.913 | 19/125 | 11/36 |
| G's decode band → T | 67–70 | 0.02805 | 0.898 | 7/125 | 12/36 |
| T's decode band → G | 31–34 | 0.02890 | 0.925 | 6/125 | 13/36 |

**MEASURED: the general sample's prefill band is *warmer than uniform* on the target corpus**
(1.039×, rank 84 of 125, 18 of 35 layers warmer than uniform). Not cold, and not transported.
**INTERPRETATION:** the weak but real cross-population ρ (0.17 / 0.27, ~8σ / ~15σ above its null) says a
*scattered, per-(layer, expert)* ranking transfers a little; the *contiguous window* does not transfer
at all. Note ρ's caveat: this null controls for identity, not for sampling attenuation, so 0.17 is a
lower bound on the true profile correlation — it does not change the band conclusion, which rests on
the window statistics and their own nulls.

## 8. Secondary check — aubier's WEIGHT MASS (never mixed with selection counts)

| regime | band by weight mass | band by selection count | same window? | weight ratio at its own band | weight ratio at the selection band | median per-layer ρ(selection, weight) |
|---|---|---|---|---|---|---|
| prefill | 0–3 | 74–77 | **no** | 0.808 | 0.817 | 0.959 |
| decode | 65–68 | 67–70 | **no** | 0.893 | 0.894 | 0.972 |

The two metrics agree almost perfectly *per layer* (ρ 0.96–0.97) and still land on **different
windows**, because an argmin amplifies the last 1%. By weight mass no window reaches ½× uniform
either (0.81 / 0.89). This is the G8 lesson in miniature: two metrics that look alike and are not —
and a warning against ever gating a file-format decision on a ~1% argmin gap.

## 9. Deliverable 5 — the pre-declared rule, applied mechanically

The rule, **fixed in the analysis script before any data was read** and not moved afterwards (verbatim
from `results/cold-band-evidence/cold_band.py`, including the definitions that had to be fixed:
non-wrapping windows, ties → lowest `lo`, "coldest-or-near" = mass ≤ **1.25×** that layer's own
minimum, ADEQUATE = median ≥ 100 tokens/layer, candidate = the width-4 window chosen **on the general
sample**, null = per-layer relabelling, N=2,000, seed 20261018):

> **STABLE BAND** iff the same width-4 window is
> **(i)** materially cold — mean per-layer mass ≤ 0.5 × the uniform share (0.015625);
> **(ii)** coldest-or-near in ≥ 30 of 36 layers;
> **(iii)** reproduced on held-out halves of **both** populations — (i) and (ii) hold for that window
> in each half of each population;
> **(iv)** outside the permutation null — p < 0.01 in both populations.
> **NO STABLE BAND** otherwise; **INSUFFICIENT** if a population cannot supply the samples.
> Overall: STABLE BAND iff an adequate regime PASSES and no adequate regime returns NO STABLE BAND.

| | prefill | decode |
|---|---|---|
| candidate window (chosen on G) | **74–77** | **67–70** |
| (i) G mean mass | 0.02687 = **0.860×u** ✗ | 0.02771 = **0.887×u** ✗ |
| (i) T mean mass | 0.03248 = **1.039×u** ✗ | 0.02805 = **0.898×u** ✗ |
| (ii) G near-coldest | **1/36** ✗ | **6/36** ✗ |
| (ii) T near-coldest | **0/35** ✗ | **4/36** ✗ |
| (iii) G halves | A 0.864×, 0/36 · B 0.854×, 2/36 ✗ | A 0.873×, 1/36 · B 0.908×, 3/36 ✗ |
| (iii) T halves | H1 1.023×, 1/35 · H2 1.056×, 1/35 ✗ | H1 0.908×, 2/36 · H2 0.888×, 4/36 ✗ |
| (iv) p_candidate | G 0.0450, T 0.6982 ✗ | G 0.0275, T 0.0715 ✗ |
| **verdict** | **NO STABLE BAND** | **NO STABLE BAND** |

**OVERALL = NO STABLE BAND.** Every criterion fails in both regimes; no criterion is borderline in a
way a threshold could flip — (i) misses by 1.7×, (ii) by ≥ 24 layers, (iii) in all four halves of both
regimes, (iv) by ≥ 2.5× in the p value. **Threshold robustness (MEASURED, no threshold was changed):**
criterion (i) must hold in *both* populations, so a cut-off would have to rise above **1.039× uniform**
(prefill) / 0.898× (decode) — i.e. above uniform itself — to pass; criterion (ii) would need relaxing
from 30 layers to **6 or fewer**; (iv) would need p < 0.05 and would still fail in the prefill regime.
No plausible single-threshold change flips the verdict. The mirror direction (band chosen on the target corpus, 31–34)
fails the same criteria symmetrically: G ratio 0.913 (prefill) / 0.925 (decode), near-cold in 0/36 and
1/36 layers, p 0.159 / 0.102.

## 10. Deliverable 6 — the interpretive caveat, stated plainly

**A cold band is a strange home for a variant (INTERPRETATION, and it is the reason the §10.7 question
has a cheap answer).** Cold means little traffic: even the *theoretical* best window here carries
0.78–0.89× of uniform traffic for 4 of 128 experts. Adapting experts that carry ~2.4–2.8% of traffic
instead of 3.1% changes very little about the model's behaviour — the variant would be nearly
unobservable, which is the opposite of a reason to build a structural key for it. And §10.2 property 4
records that **the band's member type is a quality CEILING**: the band forces the variant into the
cheapest type the band can afford, while an adapted expert wants *more* bits, not fewer. Those pull in
opposite directions, and "cold enough to be free" and "worth adapting" are the same inequality read
from both ends. **So even if the data had said STABLE BAND, "STABLE" must not be read as "build it":**
the correct reading would have been "a stable cold band exists and is a poor adapt home", which is a
reason to prefer the manifest design anyway (it can name a *hot* scattered set, or a different set per
layer, or a different set per variant).

Two further consequences of the measurements, both MEASURED:

1. **§10.2's one-band-for-the-file constraint is the binding one.** The layers choose 29–34 distinct
   windows of 121–126 available. Whatever cold structure exists is layer-specific, so a single
   `u32[2]` cannot express it — and a per-layer band would need 36 values, which is the manifest list
   by another name (and still could not express a scattered set, §10.1).
2. **The cold set, if any, is scattered, not contiguous (INTERPRETATION, from the two measurements in
   §7 and §6).** With a stable per-expert profile (ρ 0.52–0.98 within population) and *no* stable
   window at all, the experts that are relatively cold are not adjacent. §10.1 already says a
   scattered set cannot be a group and cannot be a band; this measurement supplies the "is scattered"
   half.

## 11. What this measurement does and does not license

**Licensed (PROJECTION, §10.7's own consequence):** implement the variant as the **manifest row-range
design** (`aubier/docs/expert-swap.md` §4) and **leave `taillis.variant_band` unimplemented**. §10.7
says exactly this for a null result; the null result holds in both regimes, both populations, all four
widths, and against a selection-adjusted null.

**Not licensed / not measured:**

* This is a **routing-mass** measurement. It says nothing about quality capacity or interference — a
  variant could have a *format* reason to want a contiguous home (whole-group replacement,
  GGUF-native bytes, §10.2 property 3) that is independent of traffic. That argument stands on its own
  and does not need this measurement; what this measurement removes is the *traffic* argument for it.
* Prefill and decode are **both** adequate here (unlike Q1, which was prefill-only), but T/decode is
  4,846 tokens per layer pooled over 96 conversations — ~50 tokens per conversation per layer. Decode
  claims are aggregate, not per-session.
* Prompt and corpus content never left the integer statistics in the evidence: no corpus text, no
  session paths, nothing but counts, shares and ratios.

## 12. Reproduction

Everything is one script, one seed, one run (~26 s, no GPU):

```
venvs/convert/bin/python3 results/cold-band-evidence/cold_band.py > cold-band-run.log
```

| evidence file | contents |
|---|---|
| `results/cold-band-evidence/cold_band.py` | the whole measurement; pre-declared rule and every fixed definition in the module docstring |
| `results/cold-band-evidence/cold-band-run.log` | full console output: validation, adequacy, per-layer windows, global bands, nulls, stability, rule |
| `results/cold-band-evidence/cold-band-results.json` | every number above, including per-layer masses and per-layer p-values |

Inputs are read-only: aubier's `m1-u6` routing traces (10 sessions) and the 96 per-conversation dumps
of the agent corpus. The script names them; this write-up carries no session paths and no corpus text —
only counts, shares and ratios. Nothing outside `results/` was written; nothing committed.
