# layout.md — how a per-expert quantized file is expressed

**Status: PROPOSAL. Not implemented, not yet prototyped.** This document is the interface between
`taillis` (what the weights are) and `aubier` (where they live and who computes them). It is written
before the code so that the file format is a decision, not an accident.

---

## 1. The problem, stated exactly

GGUF carries **one quantization type per tensor** (MEASURED: `llama-quantize --help`, and the format
itself). For Mistral Small 4 that means all 128 experts of a layer share a type, because the layer
carries one expert tensor:

| logical tensor | shape | today's type |
|---|---|---|
| `blk.N.ffn_gate_up_exps.weight` | `(4096, 4096, 128)` | `IQ2_XXS` |
| `blk.N.ffn_down_exps.weight` | `(2048, 4096, 128)` | `IQ3_S` (32 layers) / `IQ2_S` (4) |

**MEASURED, and load-bearing for everything below: the expert axis is `ne[2]`.** The quantization
block dimension is `ne[0]` (4096 for `gate_up`, 2048 for `down`), which grouping along `ne[2]` does
not touch. So **any group may independently carry any type whose block size divides `ne[0]`** — the
grouping cannot invalidate a type's block structure. (This is the one structural fact that makes the
proposal cheap; it is worth re-verifying on any other tensor.)

## 2. The proposal: grouped expert tensors + one metadata key

Replace one expert tensor with **G tensors**, each covering a contiguous range of the expert axis, each
with its own type:

```
blk.3.ffn_gate_up_exps.g0.weight   ne[2]=96   IQ2_XXS
blk.3.ffn_gate_up_exps.g1.weight   ne[2]=24   IQ3_S
blk.3.ffn_gate_up_exps.g2.weight   ne[2]=8    IQ4_XS
```

plus one metadata key naming the grouping:

```
taillis.expert_groups = [ "blk.3.ffn_gate_up_exps.g0", "blk.3.ffn_gate_up_exps.g1", ... ]
```

**Rules:**

1. **The group list is ordered and contiguous**, and the ranges are derived by the reader as the
   cumulative `ne[2]` of the members. The writer must guarantee `sum(ne[2]) == 128` for Mistral
   Small 4 (128 is not a parameter here — this repo targets one model).
2. **Fallback is exact.** Key absent ⇒ the file is today's file and loads byte-identically. Key
   present with **one** group per logical tensor ⇒ loading, output and timing must be
   **byte-identical to the ungrouped file** (single group = the trivial partition). This is the
   bitwise-inert property that makes the format safe to introduce: it is a superset with a
   no-op encoding, exactly as the engine's flags are gated.
3. **Tensor count is bounded by classes, not experts.** G is small (2–4). One tensor per expert would
   be 128×36 expert tensors where there are now 72 — a ~9,000-tensor GGUF, for no benefit, because
   the *type* is the quantum, not the expert.

## 3. Why the engine side is small (and what is actually new)

`aubier` already builds its execution as a **store list**: for each MoE layer, a set of
(expert range, storage location) parts, each computed where it lives, with the partial outputs summed.
A **group is a store whose type differs** — so the group list maps onto the store list directly, each
group's matmul running with one uniform type. No per-element dispatch, no new kernel family.

**What is genuinely new is on the loader side, and only there:** accepting several GGUF tensors for
one logical expert tensor, where today a layer's expert tensor is created once and then *sliced into
parts by byte range* (`create_tensor_part`, `src/llama-fork/src/llama-model-loader.h:255`).

**Precision, because it has already been mis-stated once:** those parts are slices of a parent's bytes
onto a chosen **buffer type**. They vary **where bytes live**, never **how they are encoded** — a slice
inherits the parent's type. So the existing parts machinery is the **residency** half of this feature
and **not** the type half. T2's loader change is a small extension next to it (the parts pass already
loads `[expert_lo, expert_hi)` ranges), but it does not exist yet and must not be briefed as existing.

## 4. What this proposal does not solve

- **Which experts go in which group.** That is a policy question, not a format question
  (`policy/`), and the only measured input says a *static* hot set is dead on this model — the hot
  set rotates — so the policy must be scored against a rotating baseline.
- **Type mixing changes kernel selection.** In aubier's backend the MoE GPU path is chosen by batch
  size *and* type (`get_mmvq_mmid_max_batch`). Grouping therefore changes which kernel runs, so a
  quality win can arrive with a speed change in **either** direction. Every recipe evaluation records
  speed on the pinned configuration, not just quality.
- **Whether the whole idea pays.** See §5.

## 5. Entry criteria, and the prior we are competing against

T2 is entered only when:

1. **T1 has landed and been evaluated** — a workload imatrix and a whole-tensor recipe, measured
   against the engine's pin. Without that baseline, a T2 gain cannot be attributed to grouping.
2. **The policy has a rotating-hotness baseline to beat**, held out, not fitted on the trace used to
   build the groups.
3. **The loader extension exists** — that work belongs to `aubier`, and the group list must survive
   its regression gate (flag off ⇒ byte-identical).

**The prior is small and it is stated here so it cannot be forgotten later:** the closest measured
result is ds4's imatrix on Q4 routed experts, **−1.95% relative NLL** (vendor-reported, 100
continuations). Per-expert *type* allocation is not something ds4 does at all, so it has **no**
measured prior — a null result is a legitimate and useful outcome of this repository.
