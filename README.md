# taillis

**Per-expert quantization of Mistral Small 4's expert layers.** Bits go where the tokens go.

**Mistral Small 4 is the only target — this is not a general MoE quantizer.** The shapes are known
(36 all-MoE layers, 128 routed experts of which 4 are active, a fused `ffn_gate_up_exps` and a
`ffn_down_exps` per layer), so nothing here is built to generalise, and code that would only pay off
on a second model is not written.

Claims in this repo are marked **MEASURED** (with the artifact), **INTERPRETATION**, or
**PROJECTION**. Anything unmarked is a statement of intent.

---

## Why: one of the two reasons is speed, not fidelity

1. **Quality.** Today every one of a layer's 128 experts shares a single type, because the GGUF
   carries one type per tensor. Experts carry very different numbers of tokens, so they should not
   all be quantized as if they did.
2. **Bytes are a residency variable.** In the engine ([aubier](https://github.com/jmars/aubier)), each
   layer's experts are **split by where their weights live** and computed concurrently. That design
   has a balance point `f` (the resident fraction) that moves with every byte of expert weight:
   **MEASURED arithmetic** in aubier's `docs/TARGET-ARCHITECTURE.md` §11 puts `f ≈ 0.58` at the ideal
   for one 16 GB card. Shrinking the expert set, or keeping the same size but protecting it better,
   buys residency — which is throughput.

So this is the representation half of a running engine, not a size-reduction exercise for its own sake.

---

## Where this sits

`aubier` owns **where the weights live and who computes them**. `taillis` owns **what the weights
are**. The interface between them is a single GGUF file, specified in [`layout.md`](layout.md).

---

## Status: stage 0 — scaffold. Nothing here is measured yet.

Two stages, split because the first needs no engine change and the second does.

### T1 — workload imatrix + per-tensor-type recipe (works with unmodified `llama.cpp`)

Collect activation importance with `llama-imatrix` over **agent-shaped traffic** — and this project
has an unusually on-target calibration corpus already on disk: real agent session transcripts
(`~/.local/state/hax/sessions/**/*.jsonl`). Then one `llama-quantize --imatrix <f>
--tensor-type-file <recipe>` run from the BF16 source.

This stage is fork-independent, is a serial consumer of the same GPU the engine work uses, and
produces a shippable recipe even if T2 never lands.

### T2 — per-expert / per-group types (needs `layout.md` plus a loader extension in aubier)

Group a layer's expert axis into a few classes by measured routing hotness, give each class its own
type, and let the engine treat each group as a store it already knows how to compute and sum. The
hard part is *expressing* it: see `layout.md`.

---

## What is already known (all MEASURED elsewhere; carried here so nobody re-derives it)

| fact | where it was measured |
|---|---|
| Our current recipe is already heterogeneous per tensor: router `F32`×36, attention `Q8_0`×144/`Q5_K`×72/`F32`×108, shared expert `Q6_K`×36/`Q5_K`×72, **`ffn_gate_up_exps` `IQ2_XXS`×36**, **`ffn_down_exps` `IQ3_S`×32, `IQ2_S`×4** | introspected from `UD-IQ2_XXS` |
| Expert tensor shapes: `ffn_gate_up_exps` `(4096, 4096, 128)`, `ffn_down_exps` `(2048, 4096, 128)`, router `(4096, 128)` | introspected from `UD-IQ2_XXS` |
| **ds4 does NOT allocate bits per expert.** Its `add_experts()` assigns one type across all 288 experts; what varies per expert in ds4 is the *imatrix weighting* | `antirez/ds4` `gguf-tools/glm53_quantize.py`, `gguf-tools/imatrix/README.md` |
| ds4's measured imatrix payoff, Q4 routed experts, 100 continuations: avg NLL **0.177357819 → 0.173895148 (−1.95% rel)**, case wins 54/46. Vendor-reported, not reproduced here — **this is the honest prior, and it is small** | `antirez/ds4` README |
| A **static** hot-expert set is dead on this model; the hot set **rotates** | aubier routing trace |
| GGUF carries **one type per tensor**; `llama-quantize --tensor-type` addresses whole tensors | `llama-quantize --help` |

The consequence of the last two rows together is the whole design problem, and it is stated in
`layout.md`.

---

## Entry cost

**Corrected 2026-10-11 by reading the downloaded checkpoint's own headers — the earlier "225.3 GiB
BF16 source" was wrong on both counts.**

`mistralai/Mistral-Small-4-119B-2603` is a **natively FP8 (E4M3) checkpoint**, not BF16:
`quantization = {qformat_weight: "fp8_e4m3", qscheme_act: "TENSOR"}`, with the non-converted modules
(`model.vision_tower`, `model.multi_modal_projector`, `lm_head`) left in BF16 — MEASURED from
`params.json` and from the shard headers (shard 1: 4824 BF16 + 2395 F8_E4M3 tensors).

| what | measured |
|---|---|
| one full copy of the weights | **112.6 GiB** (`total_size` 120,922,7xx,xxx, identical in both index files) |
| representations shipped | **two**, describing the same weights: `consolidated-*.safetensors` (42,741 tensors) and `model-*.safetensors` (1,485 tensors) |
| what we downloaded | **242 GB — both copies**, byte-exact vs the HF manifest (34 min at ~64 MB/s) |

**The practical consequence is good news, and it replaces an assumption with a fact:** the earlier
claim that T1 needs a ~240 GB BF16 download to avoid re-quantizing a 2-bit file was unnecessary.
`convert_hf_to_gguf.py` handles this checkpoint (`--fp8-as-q8` exists for the FP8 path), and because
**every E4M3 value maps exactly into F16**, converting with `--outtype f16` reproduces the source
weights losslessly at ~225 GiB — i.e. the "≥16-bit source" the quantizer needs is obtainable **from
what is already on disk**, and requantizing the existing IQ2_XXS GGUF remains the invalid option it
was. (The conversion is a CPU/disk job; it must not run concurrently with a timing measurement.)

**And a structural fact that changes how T2 looks:** the `consolidated` representation carries
**every expert individually** — `layers.N.experts.N.w1/w2/w3.weight` plus per-expert
`qscale_weight`/`qscale_act`, 4,608 tensors per projection = 36 layers × 128 experts exactly — where
the GGUF we run fuses w1+w3 into one `ffn_gate_up_exps` tensor per layer. So the source does not have
the "one type per tensor" problem we are solving for the *runtime* file, and it already carries
per-expert weight scales. See `layout.md` §6.

## Non-goals

- Not a general MoE quantizer, and not support for any model other than Mistral Small 4.
- Not a re-quantization of the existing GGUFs.
- Not a size chase: the target is **equal quality at fewer bytes, or more quality at equal bytes**.
- Not a speed claim. Every recipe is evaluated on the engine's pinned configuration and its **speed
  is recorded alongside its quality**, because mixing types inside a layer changes which GPU kernel
  path is selected (`get_mmvq_mmid_max_batch` is batch- and type-dependent in aubier's backend).

## Discipline

Same as the rest of the system: a gate is an **exit code**, never the tail of stdout; a fix is proved
by **asymmetry** (fails before, passes after); the engine's **pinned configuration is the control**
for every comparison, with the harness noise floor measured in-session; and no claimed number is
reported without the artifact that produced it.
