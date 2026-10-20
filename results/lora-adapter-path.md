# LORA — the ADAPTER → F16 DELTA → OVERLAY path

The variant programme is named after LoRA swap, and until this unit the half that turns an
**adapter** into overlay bytes did not exist: `tools/repack_overlay.py` took a delta that was
already materialised (`--delta`) or a synthetic one (`--delta-synth`) and had no adapter input at
all (FINDINGS U3). This unit builds that half, and cross-checks its arithmetic against the two
independent references FINDINGS §U bought.

**No GPU. No base written. Nothing committed. Every adapter below is SYNTHETIC and the files say
so** — no trained adapter exists in this project yet, and a synthetic adapter is labelled by its
sha256 in `merge_delta`, never by a path that could be read as a trained one.

> **EVIDENCE REBUILT AFTER A REBOOT.** The machine rebooted mid-unit and `/tmp` is tmpfs, so every
> fixture, probe and log was lost. The w1/w3 order measurement was **re-taken** and returned the
> same numbers bit for bit; all six gates were **re-run** on one binary (the final
> `tools/repack_overlay.py`) and every number below is from that run. The gate scripts are in
> `results/lora-gates/` and are byte-identical to the ones that ran (they build their fixtures
> beside themselves, so copy them into `/tmp/lora-gate/` to reproduce).

---

## 0 · The answer, first

| question | answer |
|---|---|
| **can a PEFT adapter be turned into overlay bytes without forking the quantise path?** | **YES.** `--lora` produces exactly the array `--delta` used to supply — a per-expert `(n_ov, ne1, ne0)` delta — and hands it to the *existing* cut → `merged_f16` → `llama-quantize` implementation. There is still one place where an override's bytes are made, and `--verify --f16 --lora` re-derives the payload through it. |
| **is the merge arithmetic right?** | **YES, to the byte.** Bit-identical to `llama-export-lora` on the dense case (3/3 scales, 0/4096 elements differing), and bit-identical to an independent numpy oracle on the expert case (0 differing of 16 777 216 and 8 388 608 elements). |
| **the w1+w3 fusion — which half of `ffn_gate_up_exps` is which?** | **MEASURED, not assumed:** rows `[0,2048)` are **w1 (gate)**, rows `[2048,4096)` are **w3 (up)**, elementwise identical to the checkpoint's own FP8 weights. |
| **the SHARED-pair hazard** | **REFUSED by name** (`lora:shared-pair-on-expert-tensor`). It is the shape `llama-export-lora` ABORTS on, and the shape that would silently broadcast one delta to 128 experts. |
| **does anything silently get dropped?** | **No, and the difference is DECLARED.** Dense-module pairs cannot be expressed by an expert-slice override at all; they are counted in `merge_delta` (`dense_skipped=N`) and warned. Expert pairs no override covers are counted (`unapplied=N`). |
| **what a GGUF LoRA cannot express** | §6. Short version: it *can* carry a per-expert 3-D pair (`lora_a` ne=(in,r,n_exp)), and it can NOT carry a shared one — and our overlay path consumes the PEFT safetensors a trainer writes, not the GGUF LoRA a converter would derive. |

---

## 1 · The adapter format accepted

PEFT safetensors, read with `safetensors.safe_open` (lazy, one tensor at a time — a full-expert
adapter is ~2.7 GiB at r=16 and is never materialised whole):

```
<module>.lora_A.weight      numpy (r, in)      PEFT's own layout
<module>.lora_B.weight      numpy (out, r)
```

`<module>` is the **per-expert** module path, and the accepted spellings are finite and explicit
(anything else is a named refusal, never a guessed strip):

| module | accepted spellings |
|---|---|
| the checkpoint's own (layout.md §6) | `layers.<l>.experts.<e>.w1` / `.w3` / `.w2` |
| HF-format aliases | `layers.<l>.mlp.experts.<e>.gate_proj` / `.up_proj` / `.down_proj` |
| PEFT prefix | optional `base_model.model.` in front of either |
| model prefix | one of `language_model.model.`, `model.`, or none |

`w1` ≡ `gate_proj`, `w3` ≡ `up_proj`, `w2` ≡ `down_proj`. `adapter_config.json` is **required to
exist** (beside the adapter, or `--lora-config`): it is where `lora_alpha` comes from, and alpha
*is* the merge scale, so a missing config is refused rather than guessed.

**Refused, each named** (gate (f) runs all thirteen):

| code | what it refuses, and why it is not "supported anyway" |
|---|---|
| `lora:shared-pair-on-expert-tensor` | a pair with NO expert index against a 3-D expert tensor. Artifact §6: this is the shape `llama-export-lora` aborts on and the runtime would broadcast to all 128 experts. |
| `lora:unmappable-module` | a module path that matches nothing accepted — or a tensor that is not a `lora_A`/`lora_B` at all (a DoRA magnitude vector, an embedding adapter, a bias delta). |
| `lora:shape-mismatch` | `lora_A` ≠ `(r, ne0)`, or `lora_B` ≠ `(block rows, r)` — i.e. the adapter was not trained on this base tensor. |
| `lora:rank-mismatch` | the tensors disagree on rank, or `adapter_config.r` disagrees with them. One adapter, one scale (`alpha/rank`). |
| `lora:uncovered-expert` | an override id the adapter has no pair for: the file would DECLARE that expert and leave it at its base value. |
| `lora:incomplete-pair` | `lora_A` without `lora_B`. |
| `lora:duplicate-pair` | two names for one `(expert, projection)` (e.g. `w1` and `gate_proj` both present). |
| `lora:unsupported-config` | `use_rslora` (`alpha/sqrt(rank)`), `use_dora` (not `B@A` at all), non-empty `rank_pattern`/`alpha_pattern` (per-module scales), `lora_alpha: 0` (llama.cpp reads 0 as ABSENT; PEFT would read it as a no-op — refusing beats picking), `peft_type` ≠ LORA. |
| `lora:config-missing` | no `adapter_config.json` — alpha cannot be guessed. |
| `lora:delta` | `--lora` together with `--delta`/`--delta-synth` (one delta source, so none is silently ignored). |
| *(plain refusal)* | `--lora` with `--identity`: a delta merged into a quantised payload has no meaning on the verbatim-copy path. |

---

## 2 · The mapping, and the measurement that pins the fusion

`W = W + (alpha/rank) · B @ A`, per expert, into the override tensor's own numpy `(ne1=out,
ne0=in)` space — which is the space `--delta` already consumed, so no new payload path exists.
`lora_a`'s GGML `ne=(in, r)` is numpy `(r, in)`, i.e. **PEFT's `lora_A.weight` exactly**: the two
conventions meet with no transpose in our code (`tools/repack_overlay.py:883`).

| adapter module | GGUF tensor | rows of the expert's numpy `(out, in)` slab |
|---|---|---|
| `w1` / `gate_proj` | `blk.<l>.ffn_gate_up_exps` | `[0, ne1/2)` |
| `w3` / `up_proj` | `blk.<l>.ffn_gate_up_exps` | `[ne1/2, ne1)` |
| `w2` / `down_proj` | `blk.<l>.ffn_down_exps` | the whole out axis |

**The row order is the load-bearing fact and it is MEASURED** (`results/lora-gates/order_probe.py`).
The source carries every expert as its own tensor (`layers.<l>.experts.<e>.w1/w2/w3`, layout.md §6)
and the GGUF fuses w1+w3; a wrong half is invisible once merged, because nothing in the format or
in any checker can see it. So the F16 GGUF's fused slab is compared against the checkpoint's own
FP8 weights, dequantised with the source's `qscale_weight`:

```
GGUF tensor : blk.0.ffn_gate_up_exps.weight ne=(4096, 4096, 128) type=F16
expert 0 slab: numpy (4096, 4096) (out, in), rms=0.012613
source w1 (2048, 4096) rms=0.012982; w3 (2048, 4096) rms=0.012233

GGUF rows [0:2048)    ~ w1 corr=1.00000000   ~ w3 corr=0.00487185
GGUF rows [2048:4096) ~ w1 corr=0.00487185   ~ w3 corr=1.00000000
max|top-w1| = 2.98023e-08   max|bot-w3| = 2.98023e-08
max|top-w3| = 0.333023      max|bot-w1| = 0.333023
CONTROL top ~ w1(expert 1) corr=0.00267285

GGUF blk.0.ffn_down_exps.weight ne=(2048, 4096, 128)
source w2 (4096, 2048) (out, in); GGUF slab (4096, 2048) (ne1, ne0)
corr(slab, w2) = 0.99999998   corr(slab, w2.T) = 0.00037324
max|slab-w2| = 3.05176e-05
```

MEASURED. `max|top-w1| = 3.0e-8` is the F16 store's own rounding, i.e. the two are the same
matrix; the crossed correlations are noise, and the **wrong-expert control** (expert 1's w1 against
expert 0's rows) is noise too, so the match is attributable to the right expert and not to the
weights being similar. The `ffn_down_exps` case is the orientation control: `corr(slab, w2)` at
1.0 against `corr(slab, w2.T)` at 0.0004 rules out the transposed arrangement on a **non-square**
tensor, where both arrangements would be shape-legal.

The placement is also **stamped into the artifact**: `merge_delta` carries
`map=ffn_gate_up_exps[w1:0:2048,w3:2048:4096];ffn_down_exps[w2:0:4096]`, so a later reader can see
which half the tool believed it was writing without re-running anything.

### The scale (FINDINGS U1/U2, re-verified here)

`scale = alpha/rank` when the config declares `lora_alpha`, else **1.0** — the user-scale-only
branch of `llama-graph.cpp:1625`. PEFT's own default (alpha defaults to `r`) gives the same 1.0, so
both readings agree. The competing conventions are refused or ruled out by measurement, not by
argument: rsLoRA is refused at load; `alpha/sqrt(rank)` fails gate (c)'s comparison by ~40 000 ULP.

---

## 3 · What was built

One file, `tools/repack_overlay.py` (+510/−10, uncommitted):

| piece | where |
|---|---|
| the accepted-module grammar, the projection table, the dense allow-list | `:761-795` |
| `lora_scale(alpha, rank)` — the one scale, shared with gate (c) | `:797` |
| `LoraAdapter` — pairs, `bounds()` (fused blocks), `pair()` (shape/coverage), `check_for()`, `delta_for()` (F32 `B@A`), `map_note()` | `:819-911` |
| `load_lora()` — reads the adapter + config, every load-time refusal | `:913` |
| `lora_plan_report()` — pre-write cross-check against the whole override plan, the `merge_delta` note, the declared-not-applied counts | `:1082` |
| wiring: `build()` (`:1440`, before the base is hashed), `requantised_payload()` (`:1207`), `verify()` (`:1980`) | — |

Three decisions worth naming:

1. **The delta is built in FLOAT32, not F16.** `--delta` arrives F16 because a GGUF holds F16; an
   adapter is a rank-r *factorisation*, and rounding `B@A` down before the merge throws away
   precision that costs nothing to keep. `merged_f16` still does the single F16 store, so §5.3's
   "merge at F16" contract is untouched and the only rounding is the one it always had.
2. **`merge_delta` carries the adapter's sha256, the config's sha256, alpha, rank, scale, the
   applied/unapplied/dense counts and the mapping** — not a path. A path can be replaced under the
   same name; a hash cannot. `--verify --lora` checks the hash it is given against the declaration
   and refuses a different adapter rather than using it.
3. **The adapter is read and cross-checked BEFORE the base is hashed**, so every refusal in gate
   (f) leaves nothing behind and costs a second, not a minute.

---

## 4 · The gates — exit codes and numbers

Command per gate: `venvs/convert/bin/python3 /tmp/lora-gate/gate_<x>.py` (scripts published in
`results/lora-gates/`). Base for (a)(b)(d)(e)(f): `t14-armA-imatrix.gguf` (the only base whose
imatrix, type file and F16 source are all on disk — the same constraint S1b documented), F16
`Mistral-Small-4-119B-2603-f16.gguf`, `recipes/example-t1.entries.txt`, layer 20.

### (a) A ZERO DELTA IS A NO-OP — `exit 0`, ALL OK

An adapter with `lora_B` all zeros, against the same build with no adapter at all:

| | no adapter | zero-B |
|---|---|---|
| `blk.20.ffn_gate_up_exps.ov0.weight` payload (8 650 752 B) | — | **byte-identical** |
| `blk.20.ffn_down_exps.ov1.weight` payload (5 373 952 B) | — | **byte-identical** |
| `taillis.overlay_digest.ov0` | `ecbc62dd…3777d2` | `ecbc62dd…3777d2` |
| `taillis.overlay_digest.ov1` | `55174723…17baca` | `55174723…17baca` |
| `taillis.overlay_identity` | `true` (derived) | `true` (derived) |
| KV keys | 16 | 17 (one extra: `merge_delta`) |

**STATED DEVIATION.** The brief asked for a BYTE-IDENTICAL FILE; what holds is a byte-identical
**payload**. The files cannot be identical and must not be: the zero-B build really did read an
adapter, so its `taillis.provenance.merge_delta` records that (a different input was applied), and
demanding the key's absence would force the provenance to lie. What the gate therefore compares is
everything a LOADER uses — every tensor's payload bytes, every per-tensor digest — **plus every
other KV key**, and it asserts that the difference is exactly `["taillis.provenance.merge_delta"]`
(the `GGUF.kv_count` the reader synthesises moves from 13 to 14 and is excluded as the reader's own
count, not file metadata). If anything else moved, the gate fails.

Corroboration from outside the gate: `smoke-nolora.gguf`, built hours earlier by an earlier
revision of the file, has the **same sha256** (`0b05ac04…d962`) as `gate-a-noadapter.gguf` built by
the final one — the no-adapter payload path is deterministic across the adapter work.

### (b) KNOWN ARITHMETIC, independent numpy oracle — `exit 0`, ALL OK

Adapter: layer 20, experts 0,1,2,3, r=4, alpha=2.0 (scale 0.5). Overrides declare experts **0,1
only**, so the undeclared experts' pairs are a leak canary. Oracle reads the safetensors and the
F16 slabs itself; it never imports the tool.

| tensor | expert | exact vs F32 oracle | F64 oracle arm |
|---|---|---|---|
| `ffn_gate_up_exps` | 0 | **0** differing of 16 777 216 (0 ulp) | 1230 differ, max **1 ulp** |
| `ffn_gate_up_exps` | 1 | **0** of 16 777 216 | 1227 differ, max 1 ulp |
| `ffn_down_exps` | 0 | **0** of 8 388 608 | 619 differ, max 1 ulp |
| `ffn_down_exps` | 1 | **0** of 8 388 608 | 589 differ, max 1 ulp |

The F64 arm is the honest limit of "exactly": 0.007 % of elements land within an F16 rounding tie
of the boundary, where a float64 sum and our float32 sum round differently. Both are correct; the
F32 arm is the one that must be bit-identical, and it is.

**THE GATE CAN FAIL** — three wrong conventions, same comparator:

| wrong arm | result |
|---|---|
| scale = `1/rank` (alpha dropped) | 16 652 392 … 16 637 672 elements differ, max 38 434 ulp |
| the two fused halves SWAPPED (w1 into w3's rows) | 16 732 742 / 16 728 627 differ, max 40 085 ulp |
| expert 0's slab holding expert 1's delta | 16 732 366 / 16 728 661 differ, max 40 141 ulp |

"Unchanged on every other expert and every other tensor": the base file's sha256 is identical
before and after the build (`5a11cfc5…653d2c`); the overlay contains exactly two tensors with
exactly two expert slabs each (`ne[2]=2`) and `taillis.overlay_experts.ov0 = ov1 = "0,1"`; the
undeclared experts 2,3 appear nowhere, and their 6 pairs are declared as
`WARNING: 6 adapter pair(s) are NOT applied (2 ffn_down_exps, 4 ffn_gate_up_exps)` with
`applied=6,unapplied=6` in `merge_delta`.

### (c) INDEPENDENT EXECUTABLE CROSS-CHECK, DENSE CASE — `exit 0`, ALL OK

The same (A, B, alpha, r) merged twice on the same F16 tensor (`blk.0.attn_q.weight`, ne=(64,64), a
hand-built llama-arch base since the vendored converter cannot run): by `llama-export-lora` from a
hand-built GGUF LoRA, and by **our** `lora_scale()` + `merged_f16()` on `scale·B@A`.

| case | export-lora's own line | our merged F16 | wrong-convention arm |
|---|---|---|---|
| r=8, alpha=2 | `calculated_scale=0.250000 rank=8` | **BIT-IDENTICAL** (0/4096, 0 ulp) | alpha/sqrt(r)=0.707107 → 4095/4096 differ, max 40 482 ulp |
| r=8, alpha **absent** | `calculated_scale=1.000000 rank=8` | **BIT-IDENTICAL** (0/4096) | 0.353553 → 4094/4096 differ, max 39 372 ulp |
| r=2, alpha=8 | `calculated_scale=4.000000 rank=2` | **BIT-IDENTICAL** (0/4096) | 5.65685 → 4094/4096 differ, max 41 259 ulp |

**SCOPE, stated so it is not over-read: this covers the DENSE case only**, because a GGUF LoRA
cannot express a per-expert delta in the form `llama-export-lora` accepts without a 3-D pair — and
even with one, the reference is the runtime's own convention, not a dense-merge comparison. The
**expert** case is gate (b), an independent numpy oracle rather than an executable reference. What
(c) pins is what the two paths share: the scale, the product order, the A orientation, and the
single F16 store.

### (d) ALPHA ABSENT ⇒ SCALE 1.0 — `exit 0`, ALL OK

An adapter whose config carries no `lora_alpha` (r=8, one expert of `ffn_down_exps`):

| | |
|---|---|
| the tool's own declaration | `alpha=absent (scale 1.0) … scale=1` and `merge_delta=…,alpha=absent,rank=8,scale=1,…` |
| `W + 1.0·B@A` | **0 of 8 388 608** elements differ (0 ulp) — the alpha-absent branch |
| `W + (1/rank)·B@A` | **8 377 881 of 8 388 608** differ, max 40 935 ulp |
| delta is not a no-op | delta rms 0.00233939 (a zero delta would make both arms match) |

### (e) PROVENANCE — `exit 0`, ALL OK

Artifact `gate-e-lora.gguf` (5 375 424 B, sha256 `af9dd24f…60ee`): one override tensor,
`blk.20.ffn_down_exps.ov0.weight` ne=(2048,4096,2) IQ2_S.

| check | result |
|---|---|
| `taillis.provenance.imatrix_sha256` | `dd02a7e6…e1d59` (present) |
| `taillis.provenance.type_file_sha256` | `929d0935…2360eb` (present) |
| `taillis.provenance.quantize_flags` / `quantize_tool` | present / `llama-quantize@50e3e3e48` |
| `taillis.overlay_digest.ov0` | `c77df7ea…45fa` (present) |
| `taillis.overlay_identity` | ABSENT — derived: the payload really differs from the base |
| `taillis.provenance.merge_delta` | `lora:sha256=45541482…e211,config=a90e6c5b…29a2,alpha=2,rank=4,scale=0.5,applied=2,unapplied=4,dense_skipped=0,map=…` |
| `--verify` (base only) | **exit 0**, `GATES OK`; `payload dig : 1/1 tensor(s) hash to their declared digest` |
| `--verify` on a copy with **one payload byte flipped** | **exit 1**, `PAYLOAD DIGEST MISMATCH — the file declares taillis.overlay_digest.ov0 = c77df7ea…` |
| `--verify --f16 --lora <the adapter>` | **exit 0**, `reproduce : … BYTE-IDENTICAL to the re-quantised F16 slice (merge: lora:sha256=45541482…)`, `reproduced : 1/1 … (2 expert slab(s) compared)` |
| `--verify --f16 --lora <a DIFFERENT adapter>` | **exit 1**, `the adapter's sha256 4fb906c8… is not the declared taillis.provenance.merge_delta` |

The last row is the one worth keeping: the reproduction path does not merely re-run the quantiser,
it refuses to run it on an adapter that is not the one the file says built it.

### (f) REFUSALS, each NAMED — `exit 0`, 13/13

Each case asserts a nonzero exit, the named code in the output, **and** that no `--out` file was
created. All are refused before the base is hashed (≤ ~12 s each, dominated by opening the 35 GB
base's tensor table):

```
ok   unmappable module name                                        lora:unmappable-module
ok   SHARED 2-D pair on a 3-D expert tensor                        lora:shared-pair-on-expert-tensor
ok   B's out axis wrong for the fused block                        lora:shape-mismatch
ok   rank disagrees with the config                                lora:rank-mismatch
ok   an override expert the adapter does not cover                 lora:uncovered-expert
ok   lora_A without lora_B                                         lora:incomplete-pair
ok   two names for one (expert, projection)                        lora:duplicate-pair
ok   rsLoRA (alpha/sqrt(rank) training)                            lora:unsupported-config
ok   DoRA                                                          lora:unsupported-config
ok   per-module alpha_pattern                                      lora:unsupported-config
ok   --lora with --identity                                        (delta on a verbatim cut)
ok   --lora together with --delta-synth                            lora:delta
ok   no adapter_config.json beside the adapter                     lora:config-missing
```

The shared-pair refusal, verbatim, because it is the unit's design constraint:

```
FATAL: [lora:shared-pair-on-expert-tensor] adapter_model.safetensors:
  '…layers.20.mlp.experts.gate_up_proj.lora_A.weight' targets
  'layers.20.mlp.experts.gate_up_proj', which has NO expert index. The base tensor
  'ffn_gate_up_exps'/'ffn_down_exps' is 3-D (one weight per expert), so a 2-D pair here is ONE
  SHARED update: llama-export-lora ABORTS on exactly this shape (GGML_ASSERT(ggml_can_repeat),
  exit 134, MEASURED) and the runtime kernel would broadcast it to all 128 experts. Emit one pair
  per expert (e.g. '...experts.<e>.w1' / '.w3' / '.w2') instead — this tool will not apply one
  delta to 128 experts
```

---

## 5 · Artefacts

All in `/var/lib/taillis/models/lora-gate/` (write scope: that tree or `/tmp`; bases untouched;
nothing here is committed).

| file | size | sha256 | what |
|---|---|---|---|
| `gate-b-known.gguf` | 14 026 496 B | `643be79b…b0565` | 2 tensors × experts `0,1`, r=4 alpha=2 (scale 0.5); digests `ee7a3adc…` (gate_up) and `c77df7ea…` (down) |
| `gate-e-lora.gguf` | 5 375 424 B | `af9dd24f…60ee` | 1 tensor × experts `0,1`, the provenance/reproduction artifact |
| `gate-d-noalpha.gguf` | 2 688 448 B | `50ac8a4f…c419` | 1 tensor × expert `0`, alpha ABSENT (scale 1.0), digest `4362dc8d…` |
| `gate-a-noadapter.gguf` / `gate-a-zerob.gguf` | 14 026 176 / 14 026 528 B | `0b05ac04…d962` / `e2c2fa98…b146` | the zero-delta pair; payload-identical, one KV key apart |

Note that `gate-b-known.gguf`'s down digest (`c77df7ea…`) equals `gate-e-lora.gguf`'s
(`c77df7ea…`): same base, same ids, same adapter → same payload, even though `k` is 1 in one file
and 0 in the other. The payload is a function of (base, ids, delta), not of the `k` token — a small
independent corroboration of the per-tensor key design.

---

## 6 · What a GGUF LoRA cannot express, and what our path cannot

**GGUF LoRA (the runtime format) — `CODE` + MEASURED by the xcheck unit, re-stated here because it
is the reason the refusals exist:**

- It CAN express a per-expert delta: a 3-D pair (`lora_a` ne=(in,r,n_exp), `lora_b`
  ne=(r,out,n_exp)) merged per expert at α/r — MEASURED on all 128 experts of
  `blk.0.ffn_gate_up_exps` (S = 0.4999999, and the per-expert deltas genuinely differ).
- It CANNOT express a **shared** one: `llama-export-lora` aborts
  (`GGML_ASSERT(ggml_can_repeat)`, exit 134, a partial output file left behind) on a 2-D pair
  against a 3-D tensor. Nothing in the runtime refuses it cleanly, and `as->ne[2]` is the expert
  count the kernel indexes by, so `ne[2]==1` would be "one update broadcast to every expert" —
  silently wrong, not an error (xcheck caveat (b): the loader checks only `ne[0]`/`ne[1]`).

**Our path** consumes the PEFT safetensors a *trainer* writes, per-expert, and produces the
quantised overlay directly. It does **not** read a GGUF LoRA, and that is a deliberate interface
choice, not a gap: the GGUF LoRA is a derived artifact of a converter we cannot run
(`conversion.py`/`gguf-py` are missing from the vendored tree — do not try to fix or fetch it), and
its per-expert 3-D form is exactly the shape whose `ne[2]` nothing checks. The arithmetic is the
same either way; only the reader would differ.

**What our path cannot express, declared rather than dropped:** a LoRA on a **dense** module
(attention, router `ffn_gate_inp`, `mlp.shared_experts.*`). The `--requantise` payload path carries
expert-slice overrides only — a whole-tensor override has no expert axis to crop an imatrix to and
is an identity cut — so a dense delta has nowhere to land. Recognised dense modules are counted
(`dense_skipped=N`), warned about in the build log, and stamped in `merge_delta`; an *unrecognised*
module name is a hard refusal, so a new architecture fails loudly instead of being skipped.

---

## 7 · Boundaries of the claims, and what is not established

1. **No trained adapter exists.** Every adapter here is synthetic (fixed seed). The gates prove the
   *mechanism*, not that a trained variant improves anything — that is aubier's S3(b), and the
   artifact to measure it on is `gate-b-known.gguf` or a real one built the same way.
2. **One layer, two to four experts, two tensors.** Nothing here says a 36-layer × 128-expert
   adapter builds correctly end to end; what it says is that the per-expert arithmetic and the
   placement are right, and that the coverage checks catch what is not declared. The scale of a real
   build is a cost question (an F16 delta at 4096×4096×128 is 4.3 GiB per tensor-layer).
3. **Gate (a) compares payloads, not whole files** (§4a) — a file-level claim would require the
   provenance to omit a fact it knows.
4. **Gate (c) is the dense case only** (§4c). The expert case stands on the numpy oracle, not on an
   executable reference, because a GGUF LoRA cannot express the shared form and a 3-D per-expert
   GGUF LoRA would only re-measure the convention gate (c) already pins.
5. **`ne1` even is assumed to mean "w1|w3 fused 50/50".** MEASURED on this base (4096 = 2×2048).
   A future arch with an unequal fusion would pass the evenness check and be placed wrongly; the
   guard is `map=` in the artifact plus this paragraph, not a check the format can make.
6. **The `--quant-type IQ2_XXS` positional and the base's type file** are S1b's recipe, unchanged;
   nothing here re-litigates the imatrix crop.
7. **Cost:** the six gates take ~12 min wall clock, dominated by hashing the 35 GB base once per
   build (the F16's 238 GB is skipped with `--input-sha256`, as S1b does). No GPU, no gpu.lock.
