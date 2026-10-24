# S7ART — the four artefacts aubier's S7 needs, over the SHIPPED base

aubier is building **S7** (`aubier/docs/expert-swap.md`): an overlay whose experts are
overridden at a **different type** than the base's, so the store must give those experts
their own segment. This unit emits the four files that unblock it, over the **product
file** — `~/futaie/aubier/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf` — plus the two
prerequisite changes to `tools/repack_overlay.py` that make a different-type build possible
at all.

**No GPU was taken, no base was written, `~/futaie/aubier` was read only, nothing was
committed.** Every claim below is marked **MEASURED** (I ran it on this host), **CODE** (a
property of the tool's own text) or **INTERPRETATION**. **A gate is an exit code**, quoted
with its command.

---

## 0 · The answer, first

Layer **20**, experts **0, 1, 13, 73, 76, 121, 122, 127** (the same layer and the same eight
as the existing `MS4-overlay-L20-e8-*` handover, so the two sets are comparable), **both**
expert tensors — `ffn_gate_up_exps` and `ffn_down_exps`. Directory
`/var/lib/taillis/models/s7art-L20-e8/` (new, nothing else lives there).

| artefact | file | size | sha256 | the ONE property it demonstrates |
|---|---|---|---|---|
| **identity** | `MS4-overlay-L20-e8-identity.gguf` | 56 099 648 B (53.50 MiB) | `16a65cbd278aa5a2ef433d8a7f0b5517795eb243ef2c703d91ce85f08e8adbd1` | **the read path is transparent**: every overridden expert's bytes EQUAL the base's slab, 16/16 units, 53.50 MiB compared byte-for-byte (**MEASURED**, `--verify` exit 0). Base types preserved (IQ2_XXS / IQ2_S), `overlay_identity=true`. |
| **type-only control** | `MS4-overlay-L20-e8-q5k-typeonly.gguf` | 138 413 056 B (132.00 MiB) | `7a06af7258e3d526e71509ece3cc8579bd4d1a7c5d1e4b606058e2c9a995e836` | **isolates the TYPE change**: both tensors read back as **Q5_K** over an IQ2_XXS / IQ2_S base, NO delta merged, and the bytes necessarily DIFFER from the base (per-expert 11 534 336 B vs 4 325 376 B, and 5 767 168 B vs 2 686 976 B). |
| **variant** | `MS4-overlay-L20-e8-q5k-variant.gguf` | 138 413 184 B (132.00 MiB) | `9c591fbfd0e4a53162a69f7106bb060c19399a735ff9a48dcb64dfca878295ed` | **isolates the DELTA**: the same Q5_K cut with a delta merged (`merge_delta=synthetic:seed=20261019,relrms=0.02`), reproduced **byte-identically** from its declared spec (**MEASURED**, `--verify --f16` exit 0, 2/2 tensors, 16 slabs). |
| **corrupted twin** | `MS4-overlay-L20-e8-q5k-variant-corrupt.gguf` | 138 413 184 B | `f68035efd7a5515a07abc917d72fc1fe5201568868081b7cd18b5e22888730cd` | **the digest is the only check that can see a corrupted non-identity payload**: ONE payload byte flipped (file byte 39 604 160, `0xf0→0xf1`, `ov0` expert index 3 = expert 73, expert-local byte 5 000 000) and `--verify` **exits 1 naming the tensor and both hashes, with no `--f16` and no reproduction** (**MEASURED**). |

Per-tensor payload digests as stamped and independently re-derived (both agree —
**MEASURED** twice, once by the writer, once by re-reading the files):

| artefact | `overlay_digest.ov0` (`ffn_gate_up_exps`) | `overlay_digest.ov1` (`ffn_down_exps`) |
|---|---|---|
| identity | `9f8463e8977d46a8f6f07108cd0b133e37cfc6ff3aa12ec9653b1b28d25fa152` | `51adff2cf4aac67f0c17db10d0b2474dc48d2fe64de438faf8827ef60aa3b028` |
| type-only | `3ddf320629a13239695b45599d78d9cc074e7507433071e54c297078a878a070` | `05e49ca9a02831f4d4d74af88721318bb276eb1e44a6e89b776b8302e9a909c1` |
| variant | `ebff425ea9f4d47f9999c1c5be5b76d403105e2df7356f32a33755a8b532899d` | `8f2ff76a6e6c770485ac895cae47f16bc6ab1f039c7837f070fe845fd6952b92` |
| corrupted twin | declares `ebff425e…` but hashes to `13f30539fde427d1ea19508ce31d260d4fa7249b95c981667e3b3301890da9a2` | declares `8f2ff76a…` and hashes to it (**OK**) |

All four carry `taillis.parent_sha256 = 82a676844e27065abe36cf9292603b6e2484a23ca5665a6d8a0bc62ac478beb3`
(= sha256 of the shipped base, **MEASURED** by `sha256sum`), per-tensor
`taillis.overlay_experts.<ovk>` and `taillis.overlay_digest.<ovk>`, and on the variant the
full `merge_delta` string. **No model KV, no `taillis.format_version`** on any of them
(V4.3 clause 5 / §8.1) — read back from each file, not asserted (`--show`, §6).

**Both tensors were Q5_K-expressible, so the brief's STOP condition did not fire.**
`ne[0]` is 4096 (`ffn_gate_up_exps`) and 2048 (`ffn_down_exps`); Q5_K's block is 256
(**MEASURED**, `gguf.quants.GGML_QUANT_SIZES`), so both divide. The tool now checks this
per override and **stops** rather than substituting a type (§2).

---

## 1 · aubier's bar for the type-only control is wrong, and it is wrong in a way worth writing down

aubier's S7 brief asserts that the **type-only control should be byte-identical to the
base**. It cannot be, and the reason is the whole content of the regime they are asking us
to build:

* **MEASURED, from the files.** The base's `blk.20.ffn_gate_up_exps` is `IQ2_XXS`,
  4 325 376 bytes per expert; the control's same eight experts at `Q5_K` are 11 534 336
  bytes per expert. The down tensor: 2 686 976 vs 5 767 168. **A payload cannot equal a
  slab of a different size**, so "byte-identical" is not a strong bar the control misses —
  it is a bar that does not typecheck.
* **The bar they want belongs to the OTHER artefact.** "Byte-identical to the base" is the
  identity overlay's property (gate (a): 16/16 slabs, 53.50 MiB, **MEASURED**), and it is
  what proves the read path is transparent. The type-only control's job is the opposite:
  it must differ, so that `variant − type-only` can attribute the remaining difference to
  the delta alone.
* **CODE, and it is the same sentence in the spec.** layout.md V4.3 clause 5 splits the
  overlay class into exactly two regimes: **SAME-TYPE** (aubier S1–S5), where the override's
  type EQUALS the base group's and the overlay is a source-pointer swap, and
  **DIFFERENT-TYPE** (S7), where the override's type deliberately DIFFERS and *"the store
  must then give those experts their own SEGMENT, because a slice carries one type"*. A
  control that came back byte-identical would be evidence the build **failed** to change
  the type.

**INTERPRETATION, stated as such:** if aubier's S7 harness asserts identity on the control,
it will fail against a correct artefact and pass only against a broken one — i.e. the
assertion inverts the gate. The control's gate should be: types read back as the new type,
and payload ≠ base.

**And our own checker was making the same mistake, which is how I found it.** On a
different-type tensor `--verify` compared nothing and then counted the tensor as
`byte-identical`, so the summary read `2/2 override tensors byte-identical` for a file where
nothing was compared — a line aubier could have read as evidence *for* their bar. Fixed
(**CODE**): a tensor that was not compared is never counted equal, and the summary now says
`0/2 … (0/0 compared unit(s)) — 2 tensor(s) NOT COMPARED: different-type regime, no
cross-type byte comparison exists (V4.3 clause 5)` (**MEASURED**, all four `--verify` runs
re-run after the fix).

---

## 2 · The two prerequisite changes to `tools/repack_overlay.py` (ours, and they gate everything)

### 2.1 The imatrix requirement now mirrors llama-quantize's own line

**MEASURED** at `aubier/vendor/llama.cpp/src/llama-quant.cpp:856-875`
(`tensor_requires_imatrix`): an importance matrix is required for
`IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, IQ1_S`, and for `Q2_K` **only** when the file type
is `MOSTLY_Q2_K_S`. Nothing else.

**CODE.** `IMATRIX_TYPES` + `type_requires_imatrix(target, ftype)` encode exactly that, and
the requirement is now applied in **both directions**:

| direction | behaviour | evidence |
|---|---|---|
| a type that REQUIRES one, none given | **refuse, by name** | **MEASURED** exit 1: `FATAL: target type(s) IQ2_XXS REQUIRE an importance matrix (llama-quant.cpp:856 tensor_requires_imatrix) and no --imatrix was given…` |
| a type that CANNOT use one, one offered | **refuse, by name** | **MEASURED** exit 1: `FATAL: --imatrix was given but the target type(s) Q5_K cannot use one — llama-quantize requires an importance matrix only for IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS and for Q2_K inside a Q2_K_S file (llama-quant.cpp:856). Stamping an imatrix sha256 (and a crop) on a build that never read it is a FALSE PROVENANCE claim — the same lie as a missing one, pointed the other way — so it is refused, not warned about` |
| the same, on `--verify` (an imatrix in hand for a q5_k overlay) | **refuse, by name** | **MEASURED** exit 1 (`…Passing one here would be reproducing with an input the build never read`) |
| a q5_k build, no imatrix | **succeeds** | **MEASURED** exit 0 — this is the type-only and the variant build |

**MEASURED, no partial file is left behind** by either refusal (`/tmp/s7art/*.gguf` empty
afterwards), because the decision is taken before the writer is opened.

### 2.2 No false provenance on a q5_k overlay

**CODE.** `taillis.provenance.imatrix_sha256` and `taillis.provenance.expert_crop` are
stamped **only** when a matrix was actually read and cropped; `quantize_flags` records the
invocation that really ran (`Q5_K 8` here — no `--imatrix`, no `--expert-crop`, no
`--tensor-type-file`). `--verify` now gates the imatrix half **by the type, not by the
flag**, in both directions: absent where a type requires one is a FAIL, and present where
no type can use one is a FAIL.

**The gate can fail — proved by asymmetry.** Stamping an imatrix sha onto a copy of the
q5_k type-only file (`tools/stamp_gguf.py`) makes `--verify` **exit 1** with:
`FAIL taillis.provenance.imatrix_sha256 present but this overlay's type(s) Q5_K cannot use
an importance matrix (llama-quant.cpp:856) — a FALSE PROVENANCE claim: nothing was
imatrix-weighted, so the stamp names a matrix no quantiser read` (**MEASURED**).

The honest file therefore says: *this payload was made by llama-quantize at Q5_K, from the
F16 slice, with no matrix*. Which is true.

### 2.3 The different-type regime had to become expressible

**CODE.** `--target-type TYPE` sets the type every override must be **produced** at; absent,
it is the base segment's own type (the same-type regime, unchanged). The produced type is
read back from the quantiser's output and checked against the target; the tensor is written
at that type, and `overlay_identity` is **derived false** when any target differs from its
base segment (no byte comparison is even defined across types).

**Regression, MEASURED — the same-type path still works end to end.** The previous unit's
artefact `/var/lib/taillis/models/MS4-overlay-L20-e8-merged.gguf` still verifies and still
**reproduces 2/2 byte-identically** from its declared spec (exit 0), and a fresh same-type
build over `t14-armA-imatrix.gguf` with `--imatrix imatrix-q8-250c.gguf --type-file
recipes/example-t1.entries.txt` still stamps `imatrix_sha256`, `expert_crop=ov0=0,1` and
`type_file_sha256`, still returns the base's bytes exactly (`0/2 expert slab(s) differ`),
and verifies exit 0. And the same overlay verified **without** `--imatrix` still refuses
(exit 1), because IQ2_XXS/IQ2_S require one.

**INTERPRETATION — a fact aubier needs before planning a same-type overlay:** the SHIPPED
base was quantised with **unsloth's** matrix (`quantize.imatrix.file =
Mistral-Small-4-119B-2603-GGUF/imatrix_unsloth.gguf`, `chunks_count=102`,
`entries_count=432`, **MEASURED** from the base's own KV), which is **not on this host**.
So a same-type overlay over the shipped base is not reproducible here at all; a
different-type q5_k overlay needs no matrix, which is precisely why S7 is the buildable
path today.

---

## 3 · The gates, each with its exit code

Commands are verbatim in §6. `$D = /var/lib/taillis/models/s7art-L20-e8`.

| gate | what it must show | command | result |
|---|---|---|---|
| **(a)** | identity overlay == the base's slabs, byte for byte | `--verify --base BASE --overlay $D/…-identity.gguf` | **exit 0** — `2/2 override tensors byte-identical (16/16 compared unit(s))`, 53.50 MiB compared, `diffs=0 bytes in 0/8 expert(s)` per tensor, `G3` passes |
| **(b)** | type-only control: q5_k on every overridden tensor, bytes DIFFER from the base, and differ from the variant | `--show` + `--verify` on the control; digest/size comparison | **exit 0** — both tensors read back `Q5_K`; **size differs** (11 534 336 vs 4 325 376 B/expert and 5 767 168 vs 2 686 976), and `overlay_digest.ov0 = 3ddf3206…` ≠ the base-slab digest `9f8463e8…` (`ov1`: `05e49ca9…` ≠ `51adff2c…`); vs the variant, **34 113 345 of 92 274 688 bytes differ** (`ov0`) and **17 068 248 of 46 137 344** (`ov1`). The control's own summary line reads `payload: 0/2 override tensors byte-identical (0/0 compared unit(s)) — 2 tensor(s) NOT COMPARED: different-type regime` |
| **(c)** | the variant merges the delta and reproduces from it; `variant − type-only` is the delta's effect only | `--verify --f16 F16 --delta-synth 20261019:0.02 --llama-quantize Q` | **exit 0** — `reproduced: 2/2 expert-slice tensor(s) BYTE-IDENTICAL to a fresh re-quantisation … (16 expert slab(s) compared)`; and the control reproduces 2/2 with **no** delta from the same command shape, so each file is byte-exactly its own declared inputs |
| **(d)** | the corrupted twin is refused BY NAME from the digest, no F16 | `--verify --base BASE --overlay $D/…-corrupt.gguf` (no `--f16`) | **exit 1** — `FAIL blk.20.ffn_gate_up_exps.ov0.weight: PAYLOAD DIGEST MISMATCH — the file declares taillis.overlay_digest.ov0 = ebff425e… but the payload hashes to 13f30539…` |
| **(e)** | the imatrix relaxation, all three directions | §2.1 table | q5_k without imatrix **exit 0**; IQ2_XXS without imatrix **exit 1**; imatrix offered to q5_k **exit 1**; and the false-provenance stamp **exit 1** (§2.2) |
| **(f)** | every artefact's KV read back and printed | `--show --overlay <each>` | §6 — the handover table above is **measured from the files**, not asserted |

### 3.1 What "differs only by the delta's effect" means here, precisely

**MEASURED + CODE, and the comparison is named rather than implied:**

1. Both files were produced by **one** code path (`requantised_payload`) from the **same**
   F16 rows (`Mistral-Small-4-119B-2603-f16.gguf`, sha256
   `83b58fbcc27456bb5cb334ade30ad92aa6a01cd922f00bcc471dc60501a9af3b`, re-measured here and
   identical to the value S1b recorded), at the **same** target type (`Q5_K`), through the
   **same** quantiser invocation (`Q5_K 8`, `llama-quantize@50e3e3e48`).
2. The **only** differing input is the merged delta: the variant's
   `merge_delta=synthetic:seed=20261019,relrms=0.02`, the control declares none.
3. Each payload is **byte-reproducible from its own declared inputs** (both `--verify
   --f16` runs above are byte-identical, which also measures that the quantiser is
   deterministic on this input: the same slice and type give the same bytes).
4. The measured difference is therefore attributable: **34 113 345 / 92 274 688** bytes
   (`ov0`) and **17 068 248 / 46 137 344** (`ov1`) — 37.0% of each payload — all of it from
   the delta, since nothing else differs.

**INTERPRETATION (and it is a limit worth naming):** the delta is **synthetic**
(`synthetic:…`, never `lora:` or `sha256:`), because no trained adapter exists for these
experts. It proves the *path*, not any quality claim. A real variant is the same command
with `--lora ADAPTER --lora-config CONFIG`.

---

## 4 · What is NOT promised to aubier

1. **These are not a served configuration.** They are one layer, eight experts, two
   tensors — 132 MiB of override, not a model.
2. **An engine that cannot yet build a set-slice will refuse them, loudly.** That is the
   intended outcome: the overlay's `ne[2]` is 8, the base's segments are `[0,128)` (the
   shipped base is **flat**, MEASURED: no `taillis.expert_groups`, no `.g<k>` members, 579
   tensors), so the ids sit inside one base group — but at a **different type**, so S7's
   job is to give those experts their own segment. A loader that silently installs Q5_K
   bytes into an IQ2_XXS slot is the failure mode; `overlay_digest` and the size check are
   what make it detectable.
3. **No quality claim.** The variant's delta is synthetic. The type-only control is
   strictly *more* bits than the base, so it is not a quality *experiment* either — it is a
   type-change control.
4. **The overlays carry no model KV and no `format_version`**, by design. A loader must take
   architecture, block count and expert count from the BASE.
5. **Nothing here was committed.** `tools/repack_overlay.py` is modified in the working
   tree; the artefacts record it as `repack_overlay.py@9c1efb05…+dirty` (the `+dirty` is
   honest — the tool changed after the identity build and before the others, and the
   recorded string says so).

---

## 5 · The tool changes, file by file

`tools/repack_overlay.py` only (425 insertions, 121 deletions):

* `IMATRIX_TYPES`, `type_requires_imatrix()`, `type_sizes()`, `expert_stride()` — the
  quantiser's own requirement line, and the expressibility check that stops a build rather
  than substituting a type.
* `build()` — `--target-type`; the imatrix decision per override against the positional
  ftype; `imatrix_sha256`/`expert_crop`/`type_file_sha256` stamped **only** when the
  corresponding input was really used; `overlay_identity` derived false for a different-type
  payload; the tensor written at the type the quantiser produced (read back, checked).
* `requantised_payload()` / `run_llama_quantize()` / `read_quantised()` — imatrix and type
  file optional and passed only when true; the positional ftype is the target unless
  `--quant-type` says otherwise; the produced type is checked against the target.
* `verify()` — the same-type/different-type regime read off the file (V4.3 clause 5); no
  cross-type byte comparison is invented; the reproduction's required inputs are decided by
  the file's own types; the imatrix provenance gated in both directions; `complete` (G8.3's
  "every payload equals its base") excludes different-type tensors by construction; and an
  uncompared tensor is no longer counted as `byte-identical` in the summary.

---

## 6 · Re-runnable commands (verbatim)

```bash
BASE=/home/jaye/futaie/aubier/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf   # READ-ONLY
F16=/var/lib/taillis/models/Mistral-Small-4-119B-2603-f16.gguf                   # READ-ONLY
F16SHA=83b58fbcc27456bb5cb334ade30ad92aa6a01cd922f00bcc471dc60501a9af3b
Q=/home/jaye/futaie/aubier/builds/cpu/bin/llama-quantize
PY=venvs/convert/bin/python3
EXP=0,1,13,73,76,121,122,127
D=/var/lib/taillis/models/s7art-L20-e8

# (a) the identity overlay: a VERBATIM cut, at the base's own types
$PY tools/repack_overlay.py --base $BASE --out $D/MS4-overlay-L20-e8-identity.gguf --identity \
  --override "20:ffn_gate_up_exps:$EXP" --override "20:ffn_down_exps:$EXP"

# (b)+(e) the TYPE-ONLY control: Q5_K, no delta, NO --imatrix (llama-quant.cpp:856)
$PY tools/repack_overlay.py --base $BASE --f16 $F16 --input-sha256 $F16SHA \
  --out $D/MS4-overlay-L20-e8-q5k-typeonly.gguf --requantise --target-type Q5_K \
  --override "20:ffn_gate_up_exps:$EXP" --override "20:ffn_down_exps:$EXP" \
  --llama-quantize $Q --workdir /tmp/s7art/typeonly

# (c) the VARIANT: the same cut with a deterministic synthetic delta merged
$PY tools/repack_overlay.py --base $BASE --f16 $F16 --input-sha256 $F16SHA \
  --out $D/MS4-overlay-L20-e8-q5k-variant.gguf --requantise --target-type Q5_K \
  --delta-synth 20261019:0.02 \
  --override "20:ffn_gate_up_exps:$EXP" --override "20:ffn_down_exps:$EXP" \
  --llama-quantize $Q --workdir /tmp/s7art/variant

# (d) the corrupted twin: ONE payload byte of ov0, expert index 3 (expert 73)
cp --no-clobber $D/MS4-overlay-L20-e8-q5k-variant.gguf \
                $D/MS4-overlay-L20-e8-q5k-variant-corrupt.gguf
printf '\xf1' | dd of=$D/MS4-overlay-L20-e8-q5k-variant-corrupt.gguf bs=1 seek=39604160 conv=notrunc

# the gates
$PY tools/repack_overlay.py --verify --base $BASE --overlay $D/MS4-overlay-L20-e8-identity.gguf        # exit 0
$PY tools/repack_overlay.py --verify --base $BASE --overlay $D/MS4-overlay-L20-e8-q5k-typeonly.gguf    # exit 0
$PY tools/repack_overlay.py --verify --base $BASE --overlay $D/MS4-overlay-L20-e8-q5k-variant.gguf     # exit 0
$PY tools/repack_overlay.py --verify --base $BASE --overlay $D/MS4-overlay-L20-e8-q5k-variant-corrupt.gguf  # exit 1, names ov0
# the reproduction gates (the only content check on a payload supposed to differ)
$PY tools/repack_overlay.py --verify --base $BASE --overlay $D/MS4-overlay-L20-e8-q5k-typeonly.gguf \
  --f16 $F16 --llama-quantize $Q --workdir /tmp/s7art/t-repro                              # exit 0, 2/2
$PY tools/repack_overlay.py --verify --base $BASE --overlay $D/MS4-overlay-L20-e8-q5k-variant.gguf \
  --f16 $F16 --delta-synth 20261019:0.02 --llama-quantize $Q --workdir /tmp/s7art/v-repro  # exit 0, 2/2
# the KV, read back
for f in identity q5k-typeonly q5k-variant q5k-variant-corrupt; do
  $PY tools/repack_overlay.py --show --overlay $D/MS4-overlay-L20-e8-$f.gguf; done
```

### 6.1 The KV as read back (gate (f), MEASURED)

`identity`: 2 tensors — `blk.20.ffn_gate_up_exps.ov0.weight` `ne=(4096,4096,8)` **IQ2_XXS**
34 603 008 B, `blk.20.ffn_down_exps.ov1.weight` `ne=(2048,4096,8)` **IQ2_S** 21 495 808 B;
`overlay_experts.ov0/ov1 = 0,1,13,73,76,121,122,127`; digests as tabled; `parent_sha256`;
`overlay_identity = True`; `provenance.overlay_tool`. **No quantisation provenance** —
nothing was requantised.

`q5k-typeonly`: the same two tensors at **Q5_K** (92 274 688 B and 46 137 344 B), the same
id keys and digests, `parent_sha256`, `provenance.overlay_tool`,
`provenance.input_sha256 = 83b58fbc…`, `provenance.quantize_flags = "Q5_K 8"`,
`provenance.quantize_tool = "llama-quantize@50e3e3e48"`. **No `imatrix_sha256`, no
`expert_crop`, no `type_file_sha256`** — by rule, not by omission.

`q5k-variant`: identical surface plus
`provenance.merge_delta = "synthetic:seed=20261019,relrms=0.02,slice_rms=0.014329629"`.

`q5k-variant-corrupt`: the variant's KV unchanged (that is the point — the file still
*claims* the right digests; the bytes do not match them).
