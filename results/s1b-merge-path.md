# S1b — the merge path: an overlay whose bytes are QUANTISED, not copied

aubier's `docs/expert-swap.md` §5.3 fixes the rule:

> **merge at F16, then re-quantize with the SAME imatrix and the SAME quant spec.** Never merge a
> float delta into a low-bit base and re-quantize.

This unit implements it in `tools/repack_overlay.py` (`--requantise`, `--delta`, `--delta-synth`),
answers the gate that decides whether the rule can be followed, and hands over the two artefacts
aubier asked for. **No GPU was taken; no base was written; nothing is committed.**

Everything below is MEASURED on this host unless marked otherwise. **A gate is an exit code**, and
each one is quoted with its exit code and its command.

---

## 0 · The answer, first

| question | answer |
|---|---|
| **(a) does re-quantising the BASE's own F16 slice at the base's own type, with the base's own imatrix, give the base's bytes back EXACTLY?** | **YES — for all three (logical, type) cases measured**, 0 differing bytes over 24 expert slabs and 84 944 656 payload bytes. The quantisation path is faithful, so every later difference is the delta. |
| **can §5.3 be followed EXACTLY?** | **Yes, with the crop now WRITTEN INTO THE SPEC** (`layout.md` V4.4, settled 2026-10-19): llama-quantize **cannot accept the base's full imatrix for a slice** — it is a hard error, not a warning. The matrix is per-**(expert, column)** and must be cropped to the same experts, and the crop is recorded as `taillis.provenance.expert_crop` (§1 is the mechanism). |
| non-identity artefact | `/var/lib/taillis/models/MS4-overlay-L20-e8-merged.gguf`, 56 100 480 B, sha256 `082dc9f9…2c12`, 8 experts × 2 expert tensors in **one layer**, provenance complete, **per-tensor payload digests**. |
| corrupted twin | `…-merged-corrupt.gguf` (one byte flipped): `--verify` **exit 1 with NO `--f16`**, from the digest alone, naming `blk.20.ffn_gate_up_exps.ov0.weight` and both hashes. That is the asymmetry the key exists for — §5. |

---

## 1 · THE MECHANISM FINDING: "the same imatrix" cannot mean the same *file*

MEASURED, twice, from both ends:

1. **But the source.** `src/llama-quant.cpp:1284` accepts an imatrix entry for a tensor only if
   `it->second.size() == tensor->ne[0] * tensor->ne[2]`.
2. **And the use.** `:784-786`:

   ```cpp
   auto imatrix_for_row = [=](int64_t row_global) {
       return imatrix ? imatrix + (row_global / nrows_per_expert) * n_per_row : nullptr;
   };
   ```

   → the matrix is indexed by **expert**, one `ne[0]`-length weight vector per expert, shared by
   all `nrows_per_expert` rows of that expert. (Consistent with the imatrix's own tensor shape,
   MEASURED: `blk.20.ffn_gate_up_exps.weight.in_sum2` is `(4096, 128)` ggml, i.e. `(128, 4096)`
   numpy = expert × column.)

**Consequence, MEASURED by trying it.** A 2-expert slice needs `4096 × 2 = 8192` weights; the base's
own matrix has `4096 × 128 = 524288`, and llama-quantize **refuses** rather than misapplying:

```
====== llama_model_quantize_impl: imatrix size 524288 is different from tensor size 8192 for blk.20.ffn_gate_up_exps.weight
llama_model_quantize: failed to quantize: imatrix size 524288 is different from tensor size 8192 …
```

That refusal is the *good* outcome (the alternative is "most of the model quantised without an
imatrix", the code's own words at `:1293-1296`). So the pipeline crops:

```
expert e owns imatrix bytes [e*ne0, (e+1)*ne0), and no row of expert e reads any other block
⇒ crop(ids) = concat(arr[e*ne0:(e+1)*ne0] for e in ids)      EXACT, LOSSLESS
```

Implemented as `write_cropped_imatrix()`; stamped into `quantize_flags` as `--expert-crop <ids>`
(OUR clause — llama-quantize has no such flag and cannot have one, because the crop is what makes
its length check pass). `.stats` is deliberately **not** copied: it describes the full 128-expert
tensor and would be a false declaration about a slice (the loader does not read it).

**The decision this needs from aubier/the spec, not from me:** §5.3's "the SAME imatrix" has to be
read as *the same matrix, cropped to the same experts* — the literal reading is unexecutable. The
rule's purpose (the requantisation must not draw on a different importance estimate) is untouched
by the crop, since every expert's weights are the base's own.

---

## 2 · The base, and why this base

MEASURED by reading tensor tables:

| base | §5.3 followable? |
|---|---|
| **`t14-armA-imatrix.gguf`** (34 924 187 712 B, sha256 `5a11cfc5…653d2c`, flat, 579 tensors, `gate_up` **IQ2_XXS** ×36, `down` **IQ2_S** ×32 / **IQ3_S** on layers 4,28,32,35) | **YES, and it is the only kind of file where the answer is yes.** Its own KV names the matrix (`quantize.imatrix.file`), the type file is in `recipes/example-t1.entries.txt`, and the F16 it was quantised from is on disk. |
| the shipped `…UD-IQ2_XXS.gguf` | **NO.** Its imatrix is Unsloth's and is not on disk; we know it moves 14.4 % of the weight bytes (FINDINGS D4) but we cannot re-run it. |
| `t4a-uniform-minus23p5.gguf` | yes in principle (it stamps `imatrix_sha256` + `quantize_flags`), but its expert types are the −23.5 % control's, so it is not the product base. |

**This is a real constraint on the variant programme, not a local inconvenience:** §5.3 can only be
satisfied against a base **this lane built**, because only then are all three inputs (imatrix, quant
spec, ≥16-bit source) on disk. A variant against the shipped product file can only be produced by
borrowing a matrix we do not have.

The tool CHECKS the one fact the base can prove: `quantize.imatrix.entries_count` must equal the
matrix handed in (432 = 1728/4). A different count is a named failure; a different recorded *path*
is only a warning (paths move, and the stamped sha256 is this file's own).

---

## 3 · GATE (a) — re-quantise the base's own F16 slice and compare bytes

The F16 source is `Mistral-Small-4-119B-2603-f16.gguf`
(237 992 005 440 B, sha256 `83b58fbcc27456bb5cb334ade30ad92aa6a01cd922f00bcc471dc60501a9af3b`,
independently re-derived with `sha256sum` this session and equal to the value `t4a` recorded).

Method per case: cut the experts' rows out of the F16 into a **throwaway GGUF holding only those
rows** (277 MB was the whole probe; the real runs are 33 MiB / 21 MiB), quantise it with the base's
own type file + cropped imatrix + positional `IQ2_XXS` and `nthread 8`, then compare the produced
bytes with **the base file's own slab bytes**.

| case | base type | experts | payload compared | slabs differing | exit |
|---|---|---|---|---|---|
| `blk.20.ffn_gate_up_exps` | IQ2_XXS | 0,1,13,73,76,121,122,127 | 34 603 008 B | **0 / 8** | 0 |
| `blk.20.ffn_down_exps` | IQ2_S | same 8 | 21 495 808 B | **0 / 8** | 0 |
| `blk.4.ffn_down_exps` | **IQ3_S** | same 8 | 28 835 840 B | **0 / 8** | 0 |
| `blk.20.ffn_gate_up_exps` | IQ2_XXS | 0,1 (the first probe) | 8 650 752 B | **0 / 2** | 0 |

Total: **24 expert slabs, 84 944 656 bytes, 0 differing bytes.** MEASURED (`--verify` prints
`diffs=0 bytes in 0/8 expert(s)`; the same comparison is made inside `build()` and reported as
`base cmp : … 0/8 expert slab(s) differ from the base`).

**Why this had to be measured and not assumed.** Three ways it could have failed, all invisible
without the run: the type file's patterns are anchored on `blk.<l>.<logical>.weight`, so renaming
the slice tensor would silently send it to the fall-through (caught instead by the tool's read-back
type CHECK); the imatrix crop could have been indexed wrong (row vs expert-major); and the F16
"source" could be a *different* F16 (the tool refuses on an ne[0..1] mismatch). It returned exact
equality, so the pipeline is faithful and **every difference in §4 is the delta**, which is exactly
what S3(b) needs to be interpretable.

---

## 4 · The artefacts handed over

All three are in `/var/lib/taillis/models/` (write scope: models dir or /tmp; bases are read-only).

Layer **20**, experts **`0,1,13,73,76,121,122,127`** — the **8 hottest experts of that layer on the
measured routing traces** (`aubier/logs/pe/dump-{chat,code,long}.txt`, selections with id < 128:
1840, 1670, 925, 868, 527, 523, 520, 518). Chosen hot so that aubier's S3(b) — *"logits measurably
DIFFERENT at the first token that routes to that expert"* — has the best chance of being hit early;
the artefact is only useful if a token actually reaches it. Both expert tensors of that one layer
are overridden (`ffn_gate_up_exps` IQ2_XXS + `ffn_down_exps` IQ2_S), so one file covers both.

| artefact | size | sha256 | what |
|---|---|---|---|
| `MS4-overlay-L20-e8-requant.gguf` | 56 100 416 B (53.50 MiB) | `b97b0807cc240231b654cf5face83f378cc021c0d56386c13fd46fad95860766` | **the identity-by-construction overlay**: requantised from the F16 with the base's own spec, **no delta** ⇒ byte-identical to the base ⇒ `overlay_identity = true`, carrying the full quantisation provenance (PERMITTED on an identity overlay — §6). The strongest possible S3(a) artefact: its bytes are NOT a copy, they are what the quantiser produces. |
| `MS4-overlay-L20-e8-merged.gguf` | 56 100 480 B (53.50 MiB) | `082dc9f937791eb2c085f19c80927457fdf94ddc12acbc6600de98d152dd2c12` | **THE NON-IDENTITY OVERLAY**: the same cut, merged at F16 with a **synthetic** delta (`merge_delta=synthetic:seed=20261019,relrms=0.02`, i.e. N(0,(0.02·rms(slice))²), PCG64, deterministic), then re-quantised with the base's own spec. `overlay_identity` absent. **The artefact to gate S3(b) on.** |
| `MS4-overlay-L20-e8-merged-corrupt.gguf` | 56 100 480 B (same) | `d7ac9f3db9c538bf0d520f8409b29e5152ef5b0723503d439cd77d47ca95724c` | **the corrupted twin**: byte `0xa7→0xa6` at file offset 15 140 480 = `data_offset 1664 + j3·stride 4 325 376 + 2 162 688`, i.e. inside expert **73**'s slab of `blk.20.ffn_gate_up_exps.ov0`. `--verify` rejects it from the DIGEST, no `--f16` (§5). |

**Their own declared keys** (each file states what built it; the digests are what an engine or a
checker can test without the 222 GiB F16):

| key | `…-requant.gguf` | `…-merged.gguf` |
|---|---|---|
| `taillis.overlay_digest.ov0` (gate_up, 34 603 008 B payload) | `d76cc156f943bccc4e73eb2b6b36d645585e7ee6340576be93edcd9d700c9b16` | `6b6cff78aeec542cfa42bb37edd10d6fae463cd2ab7b19e26625e40a8c2c3d64` |
| `taillis.overlay_digest.ov1` (down, 21 495 808 B payload) | `6388ce2a6bb82c3f7ef61ef394cf01df9a2411a4ca6e9632212622d97eb52342` | `1136e46f58c648647ef2c1109626e7f1dc66a7290d9bda010389f6d1cf9f5971` |
| `taillis.overlay_identity` | `true` (derived) | absent |
| `taillis.provenance.expert_crop` | `ov0=0,…,127;ov1=0,…,127` | same |
| `taillis.parent_sha256` | `5a11cfc5…653d2c` (armA) | same |
| `taillis.provenance.{imatrix_sha256, type_file_sha256, quantize_flags, quantize_tool, merge_delta}` | present, `merge_delta` absent | present, `merge_delta=synthetic:…` |

**The delta is SYNTHETIC and the file says so.** No trained adapter exists in this project yet; the
stamped value is `synthetic:seed=…,relrms=…,slice_rms=…`, never a path that could be mistaken for a
real one. `--delta FILE` (a converted adapter's slice, tensors named exactly as the override
tensors) is the trained path and shares the code; only `build_delta()` differs.

**Merge convention, stated because it is a choice:** accumulate in F32, store F16
(round-to-nearest-even). Merging two F16 arrays elementwise in F16 rounds twice; an F32 reference of
221 GiB does not exist here. Measured effect of the synthetic delta at `relrms=0.02`:
`slice_rms` 0.014329629 (gate_up), 0.014162895 (down) → `delta_rms` 2.8659e-4 / 2.8326e-4,
`elems_changed` 132 714 295 / 66 363 451 of 134 217 728 / 67 108 864 elements.

**What the delta does to the QUANTISED bytes** (the number aubier needs for S3(b)): **all 16 slabs
change**, by 10.2 % of bytes (gate_up, 3 524 761 / 34 603 008) and 11.6 % (down, 2 497 995 /
21 495 808), spread evenly across experts — 300 964 to 446 417 bytes each (per-expert table in
`logs/s1b/verify-merged.log`). A uniform synthetic delta perturbs uniformly; a trained adapter would
not, which is a reason to re-run this table when a real adapter lands.

---

## 5 · The verification path: the payload DIGEST, and a limit it CLOSES

`--verify` previously treated a payload that differs from its base as an ERROR. That is right for an
identity cut and wrong for a variant: on a variant the bytes are *supposed* to differ, so the base
comparison sees nothing. Measured consequence, and the reason the format grew a key:

```
file                                --verify (base only), BEFORE the digest
…-merged.gguf (clean)               exit 0
…-merged-corrupt.gguf (1 byte)      exit 0        <-- a corrupted overlay passing its own gate
```

**`taillis.overlay_digest.<ovk>` closes it (V4.4, settled 2026-10-19, REQUIRED on every overlay).**
One scalar sha256 per override tensor, keyed by the same `k` token as its id key, hashed over the
payload as stored — the tensor's `nbytes` bytes at its `data_offset`, and for an expert band the
per-expert slabs in ascending id order. (Alignment PADDING after a tensor is not payload and is not
covered.) `--verify` recomputes it from the file's own bytes: **no `--f16`, no quantiser, no
reproduction needed.**

The asymmetry, all four runs without `--f16`:

| file | digest check | `--verify` |
|---|---|---|
| `…-merged.gguf` (clean, non-identity) | `2/2` tensors hash to their declared keys | **0** |
| `…-merged-corrupt.gguf` (1 byte flipped inside expert 73) | `1/2`; names the tensor + both hashes | **1** |
| `…-requant.gguf` (identity by construction) | `2/2` | **0** |
| `…-merged.gguf` with its digest keys stripped | `0/2`; names each tensor missing its key | **1** |

The twin's failure line (verbatim):

```
FAIL blk.20.ffn_gate_up_exps.ov0.weight: PAYLOAD DIGEST MISMATCH — the file declares
taillis.overlay_digest.ov0 = 6b6cff78…3d64 but the payload hashes to ab04516e…e5c1
(34603008 byte(s) of IQ2_XXS). The payload is not the one this overlay was built with …
```

Note the second tensor still verifies in the same run: the check is per-tensor, so a single flipped
byte does not simply fail the file wholesale — it names the ONE tensor that moved.

**What the digest does and does not prove.** It proves the payload is *the payload this file was
built with* — integrity and self-consistency. It does **not** prove those bytes are what §5.3's
recipe produces, because a builder could stamp a digest over wrong bytes. That second, stronger
property is what `--verify --f16 …` still does: it re-runs the quantiser from the declared
`imatrix_sha256` / `type_file_sha256` / `quantize_flags` (refusing the inputs if they do not hash to
the declarations) and compares byte for byte. Both were run on every artefact here:

| artefact | digest alone | + reproduction |
|---|---|---|
| `…-requant.gguf` | 0 | **0**, `reproduced: 2/2` |
| `…-merged.gguf` | 0 | **0**, `reproduced: 2/2` (16 expert slabs) |
| `…-merged-corrupt.gguf` | **1** | **1** |

**`taillis.provenance.expert_crop` is stamped and CROSS-CHECKED (V4.4).** Value:
`ov0=<ids>;ov1=<ids>`, the crop each override tensor's imatrix was taken with. It is the *build* fact
and the id key is the *format* fact; `--verify` requires them to be EQUAL, because a payload quantised
with a different crop than the ids it declares would pass every other check with the wrong weights.
MEASURED negative: rewriting the declaration to `…,126` against an id key of `…,127` gives exit 1
("the imatrix was cropped to DIFFERENT experts than this tensor declares"). A verbatim identity cut
quantises nothing and therefore declares no crop.

---

## 6 · Identity and provenance are INDEPENDENT (V4.4, settled)

The first draft of V4.4 made the quantisation provenance **forbidden** on an identity overlay, on the
reasoning *"identity ⇒ a verbatim byte copy ⇒ nothing was quantised"*.

**`MS4-overlay-L20-e8-requant.gguf` falsifies that inference, and V4.4 now says so in as many words.**
Its bytes are byte-identical to the base's, so `overlay_identity` is (correctly) `true` — and a
quantiser absolutely did run, with an imatrix, a type file and a flag string, and it produced those
bytes. "Identity" is a statement about **bytes**; "was quantised" is a statement about **provenance**,
and this payload path separates them.

The settled rule, implemented: the pair is **required on a non-identity overlay, permitted on an
identity one** — never demanded there, never refused. MEASURED both ways: `…-merged.gguf` (non-identity)
carries it and must; `…-requant.gguf` (identity) carries it and verifies **exit 0**, and a verbatim
identity cut carries none and also verifies **exit 0**.

`overlay_identity` stays DERIVED on the quantised path — the tool compares every payload with its base
slab and stamps the flag iff they are all equal — and G8.3's asymmetry is unchanged: declaring `true`
while a payload differs FAILS, declaring false/absent while every payload is equal FAILS. Deriving what
you stamp is not cheating; the checker still tests the claim against the bytes.

Also noted for §7.1: `taillis.provenance.merge_delta` is **not yet a defined name**. It is stamped

## 7 · What this does NOT establish

- **Not that a variant is better, or even plausible.** The delta is synthetic. This unit makes a
  variant *exactly representable*; quality is the training record's problem (§7 of aubier's plan).
- **Not that the crop is the only possible mechanism.** It is the only one that runs against stock
  llama-quantize; a tool that accepted a per-expert imatrix plus a row range would remove the crop.
  The spec settles the crop as the build obligation (§1), so this is now a format fact, not a choice
  this unit made.
- **Not that a payload digest proves CORRECTNESS.** It proves the payload is the one the file was
  built with. Whether those bytes are what §5.3's recipe produces is the reproduction's job (§5).
- **Not a speed claim.** Nothing here changes what the engine executes; the swap happens at load.
- **Not a grouped-base claim.** All of §3–§5 is on a FLAT base. The grouped SET path is still
  unmeasured (LEDGER L43 FINDING-3), so a grouped-base variant remains *serverable and unaudited*.
- **Not a full-file reproduction gate for a whole model.** `--verify --f16` re-quantises only the
  overridden tensors (seconds to minutes), by construction.

---

## 8 · Cost, and the gates that were re-run

- F16 sha256 (222 GiB): **14m36s**, once, then supplied via `--input-sha256` for the builds — the
  tool prints that it *recorded* a supplied value instead of measuring one.
- Build: armA hash + 2 tensors requantised + F16 cut ≈ 1–2 min per artefact after the sha was known
  (GGUFReader on the 222 GiB source is 11 s: that is the KV parse, not the data).
- Re-quantisation itself: ~0.2 s per tensor slice (2-expert probe: `quantize time = 178.59 ms`).
- **No GPU taken, no `gpu.lock` touched, no base file written.**
- **Existing suites re-run, all green, exit codes:** `eval/test_check_chain.py` **0** (40 checks),
  `eval/test_stamp_gguf.py` **0** (25 checks), `eval/test_group_experts_v1.py` **0** (42 checks),
  `eval/test_corpus.py` **0** (ALL PASS).
- **S1's own paths re-run after the refactor** (the identity writer and the group arithmetic are
  shared code, so they are regression surfaces): an identity cut on the FLAT armA base → build **0**,
  `--verify` **0** (payloads all equal, provenance absent — OPTIONAL and truly absent there); an
  identity cut on the GROUPED base `MS4-119B-UD-IQ2_XXS-g3.gguf` (3-way tiling
  `[0,43) [43,86) [86,128)`, override `41,42` from segment `g0`) → build **0**, `--verify` **0**; and
  an override straddling `42,43` still REFUSED by name ("a group boundary is not an expert
  boundary"). Both S1 artefacts now carry payload digests and verify `1/1` on them.

### The fix round (digest · provenance-on-identity · expert_crop), every gate re-run

| gate | command | exit |
|---|---|---|
| gate (a) idempotence, UNCHANGED | `build` of `…-requant.gguf` → `base cmp 0/8` and `0/8`, `--verify` `diffs=0 bytes in 0/8 expert(s)` ×2 | **0** |
| identity + provenance ACCEPTED | `--verify …-requant.gguf` (identity declared, imatrix/type/flags/tool present) | **0** |
| identity + none, no refusal | `--verify` on the S1 flat identity cut (no provenance at all) | **0** |
| digest, clean non-identity | `--verify …-merged.gguf` — `2/2` tensors hash to their keys | **0** |
| **digest, corrupt twin, NO `--f16`** | `--verify …-merged-corrupt.gguf` — `1/2`, names the tensor and both hashes | **1** |
| digest, MISSING keys | overloaded `…-merged.gguf` with the two digest keys stripped — `0/2`, names both tensors | **1** |
| `expert_crop` disagreement | `…-merged.gguf` with `ov0=…,126` declared against an id key of `…,127` | **1** |
| reproduction (stronger check) | `--verify --f16 …` on `…-requant.gguf` and `…-merged.gguf` — `reproduced: 2/2` each (16 slabs) | **0**, **0** |
| reproduction on the twin | same command on `…-merged-corrupt.gguf` | **1** |
| S1 flat identity | build, verify | **0**, **0** |
| S1 grouped identity | build, verify | **0**, **0** |
| `eval/test_check_chain.py` | — | **0** (40 checks) |
| `eval/test_stamp_gguf.py` | — | **0** (25 checks) |
| `eval/test_group_experts_v1.py` | — | **0** (42 checks) |
| `eval/test_corpus.py` | — | **0** (ALL PASS) |

No new test file is added, deliberately: the gates that matter here **are** byte/hash comparisons
that can fail, and each is re-runnable from §9. A synthetic mini-model unit test would exercise the
same code against easier bytes and prove less.

## 9 · Re-runnable commands (verbatim; `LD_LIBRARY_PATH` comes from the binary's own directory)

```bash
BASE=/var/lib/taillis/models/t14-armA-imatrix.gguf
F16=models/Mistral-Small-4-119B-2603-f16.gguf
F16SHA=83b58fbcc27456bb5cb334ade30ad92aa6a01cd922f00bcc471dc60501a9af3b
Q=~/futaie/aubier/builds/cpu/bin/llama-quantize
PY=venvs/convert/bin/python3
EXP=0,1,13,73,76,121,122,127

# GATE (a): requantise with NO delta -> must reproduce the base byte for byte
$PY tools/repack_overlay.py --base $BASE --f16 $F16 --input-sha256 $F16SHA \
  --out /tmp/gateA.gguf --requantise \
  --override "20:ffn_gate_up_exps:$EXP" --override "20:ffn_down_exps:$EXP" \
  --imatrix imatrix-q8-250c.gguf --type-file recipes/example-t1.entries.txt \
  --llama-quantize $Q --workdir /tmp/gateA
$PY tools/repack_overlay.py --verify --base $BASE --overlay /tmp/gateA.gguf     # exit 0, diffs=0

# THE DIGEST GATE — no --f16, no quantiser: what an engine or a checker can run
$PY tools/repack_overlay.py --verify --base $BASE \
  --overlay /var/lib/taillis/models/MS4-overlay-L20-e8-merged.gguf            # exit 0, 2/2 digests OK
$PY tools/repack_overlay.py --verify --base $BASE \
  --overlay /var/lib/taillis/models/MS4-overlay-L20-e8-merged-corrupt.gguf    # exit 1, names ov0

# the merged artefact, and the STRONGER check: reproduce the payload from the declared spec
$PY tools/repack_overlay.py --base $BASE --f16 $F16 --input-sha256 $F16SHA \
  --out /tmp/merged.gguf --requantise --delta-synth 20261019:0.02 \
  --override "20:ffn_gate_up_exps:$EXP" --override "20:ffn_down_exps:$EXP" \
  --imatrix imatrix-q8-250c.gguf --type-file recipes/example-t1.entries.txt \
  --llama-quantize $Q --workdir /tmp/merged
$PY tools/repack_overlay.py --verify --base $BASE --overlay /tmp/merged.gguf \
  --f16 $F16 --delta-synth 20261019:0.02 --imatrix imatrix-q8-250c.gguf \
  --type-file recipes/example-t1.entries.txt --llama-quantize $Q --workdir /tmp/v   # exit 0, 2/2
```

To corrupt a twin: flip one byte at `tensor.data_offset + j*stride + k` (see §4 for the arithmetic
actually used). To check `--verify` is not vacuous, run the same verify on the flipped file and
require **exit 1** naming the tensor — which is what the digest delivers, with no F16 in sight, and
the reproduction delivers independently of the file's own declarations.
