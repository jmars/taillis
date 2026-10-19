# taillis

**What the weights are.** Bit-level and layout-level decisions about Mistral Small 4's expert
layers, expressed in a GGUF that [aubier](https://github.com/jmars/aubier) — the engine — reads.
The interface between the two projects is one file format, specified in [`layout.md`](layout.md).

**Mistral Small 4 is the only target — this is not a general MoE quantizer.** The shapes are known
(36 all-MoE layers, 128 routed experts of which 4 are active, a fused `ffn_gate_up_exps` and an
`ffn_down_exps` per layer), so nothing here is built to generalise.

Claims are marked **MEASURED** (with the artifact), **INTERPRETATION**, or **PROJECTION**.
Anything unmarked is intent. A gate is an **exit code**, never the tail of stdout. A fix is proved
by **asymmetry** — it must fail before and pass after.

---

## What this is working on now: variant (LoRA) swap

A trained adapter is exported as an **OVERLAY** file and swapped in at load time: the overlay
replaces a chosen set of `(layer, expert)` weight slices, while the base model, the router tensors
it does not override, the shared expert, attention and the KV path stay untouched.

The format problem is real and narrow. A GGUF carries one type per tensor, and a variant's experts
need to be addressable as a unit *by the engine*, without a side file. `layout.md` **V4** specifies
the overlay class:

- `.ov<k>` override tensors, either an **expert slice** (`ne[2] == len(experts)`) or a **whole
  tensor** — the kind is derived from the base tensor's rank, never declared;
- **the manifest lives in the file**, not in a side JSON: the expert list is one scalar key per
  tensor, bound to the `k` already in the tensor name; the target group and the type are *derived*
  and then checked. Declaring them twice is how a format grows a second thing that can disagree
  with the first;
- `taillis.parent_sha256` names the base, so a comparator is a property of the artifact.

`tools/repack_overlay.py` is the exporter (S1 of aubier's plan): it cuts rows out of a base GGUF,
writes them back as an overlay, and proves the round trip byte-for-byte — on a flat base *and* on a
grouped one, with an uneven tiling so that "the segments come from the base's own declaration"
cannot be confused with "the segments were assumed to be even".

**S1b, the merge path, answers the question the format cannot:** a variant's bytes are *quantised*
from the ≥16-bit source, not copied — merge at F16, then re-quantise with the base's own imatrix and
type spec. It reproduces the base's bytes **exactly** when no delta is merged (24 expert slabs, three
types, 0 differing bytes), so every later difference is the delta. One mechanism fact decides how
that can be done at all: **llama-quantize refuses the base's imatrix for a slice** — the matrix is
per-(expert, column) and its length must be `ne[0]·ne[2]` — so "the same imatrix" has to mean the
same matrix *cropped to the same experts* (`results/s1b-merge-path.md` §1). And a variant overlay is
**unauditable against its base by construction** — its bytes are supposed to differ, so that finding
became the format's `taillis.overlay_digest.<ovk>` (a per-tensor payload sha256, required on every
overlay): `--verify` recomputes it and rejects a one-byte corruption **by name, with no F16 and no
quantiser**. `--verify --f16` still goes further and reproduces the payload from the declared spec.

## What was measured and then dropped

The project began as a byte-reduction campaign: fewer bytes per expert, because in the engine bytes
are a residency variable and the CPU miss path is at the memory ceiling. **That campaign is
concluded, negatively, and the number is the result:**

| what we measured | result |
|---|---|
| a **−23.5 %** expert-byte cut, uniform, on the product path | **+14.69 % relative perplexity** (2,391 paired positions; CI [+0.089, +0.181]). That is **9.20 %PPL per MiB/expert removed** against a 1.25 bar. It would have bought +2.2 tok/s |
| a **workload imatrix** vs a uniform one at equal bytes | **+0.252 % PPL** — a tight null at this quant level, and therefore not a lever that can recover 14.7 points |
| per-expert routing hot sets | **concentrated and request-general on agentic streams** (zero collapse in 15,312 pair verdicts at the single-conversation unit), but contiguous grouping averages the signal away — the hottest *contiguous* chunk is only 1.26× the coldest at G=3 — and every grouping costs the engine more in store graphs than the bytes save |

So: the quality price of a byte cut at this level is real and large, and the allocation machinery
that was meant to afford it is worth a quarter of a point. That is why the work moved to variants,
which change **which weights an expert uses**, not how many bytes it occupies.

## What is in here

| path | what |
|---|---|
| [`layout.md`](layout.md) | **the format.** Grouped experts (R2), provenance and the version scalar (R1), the comparison chain (R3), and the overlay/variant class (V4) |
| `tools/` | the writers and checkers: `group_experts.py`, `stamp_gguf.py`, `check_chain.py`, `repack_overlay.py`, plus the corpus/imatrix/slice tools |
| `eval/` | self-tests. Each is written to be able to fail; several exist because an earlier version could not |
| `results/` | the write-ups behind the numbers above, with their evidence |
| `recipes/` | a machine-readable per-tensor-type recipe |
| `corpus/` | the builder for the calibration corpus — **the corpus itself is private and is not committed** |

## Known facts carried here so nobody re-derives them

| fact | where it was measured |
|---|---|
| the source checkpoint is **natively FP8 (E4M3)**, not BF16, and every E4M3 value maps exactly into F16 | its `params.json` and shard headers |
| the F16 GGUF we run is **221.647 GiB**, arch `mistral4`, 579 tensors | `results/t1-f16.md` |
| our recipe is already heterogeneous per tensor: router `F32`×36, attention `Q8_0`/`Q5_K`/`F32`, shared expert `Q6_K`/`Q5_K`, **`ffn_gate_up_exps` `IQ2_XXS`×36**, **`ffn_down_exps` `IQ3_S`×32 / `IQ2_S`×4** | introspected from the shipped `UD-IQ2_XXS` |
| the quantization block axis is `ne[0]`, so slicing along `ne[2]` never disturbs a block — which is what makes any group, and any overlay, carry an arbitrary type | `layout.md` §1 |
| requantizing an existing 2-bit GGUF is invalid (`llama-quantize --allow-requantize` says so itself); every artifact here is built from the F16 | `llama-quantize --help` |

## Non-goals

- Not a general MoE quantizer, and not support for any model but Mistral Small 4.
- Not a re-quantization of the existing GGUFs.
- Not a speed claim. Grouping and type mixing change which kernel runs, so any speed effect is a
  measurement, not an assumption — and in both directions.
