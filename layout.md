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

## 6. Correction (2026-10-11): the SOURCE already carries experts individually

Read from the downloaded checkpoint's index this session — the `consolidated` representation names
**every expert as its own tensor**, with its own weight and activation scales:

```
layers.N.experts.E.w1.weight / .qscale_weight / .qscale_act     (w1 = gate)
layers.N.experts.E.w2.weight / .qscale_weight / .qscale_act     (w2 = down)
layers.N.experts.E.w3.weight / .qscale_weight / .qscale_act     (w3 = up)
```

4,608 tensors per projection = **36 layers × 128 experts**, exactly. The checkpoint is FP8 (E4M3,
`qscheme_act: TENSOR`), so those scales are part of the source, not something we invent.

**What this changes:**
- T2's grouping is **not** a slicing exercise on a fused tensor — the source is already per-expert. The
  work is in what we *write* (a runtime file has one type per tensor, so groups must become separate
  tensors) and in the **policy** (which experts share a group), not in decomposing the input.
- The per-expert weight scales are a free, upstream signal about each expert's dynamic range — useful
  as a **feature** in the allocation policy, and a sanity check against our own routing-hotness trace.
  They are not the same thing as hotness: a scale says how large the weights are, not how often the
  expert is chosen.
- Note the naming mismatch to bridge: the source has **w1 / w2 / w3** per expert; the GGUF we run fuses
  **w1+w3 into one `ffn_gate_up_exps` tensor** per layer and keeps w2 as `ffn_down_exps`. A grouped
  writer has to decide whether its groups respect the source's natural per-expert layout (then the
  fusion into `gate_up` happens on the runtime side) or fuse first and group after.

## 7. R1 — the format version key and the provenance block (spec)

Two additions to the `taillis.*` KV namespace, written by **our tools at build time**. Both are
writer-side only: no engine change is needed for correctness, and no existing artifact is touched.

### 7.1 Keys

```
taillis.format_version        : u32
taillis.parent_sha256         : string
taillis.provenance.<name>     : string, one key per build fact
```

- `taillis.format_version` : u32. R1 introduced the *key*; **its values and their dispatch are
  normative in §8.1, not here** — R2 narrowed the scalar to the grouping encoding (`0` = the
  `taillis.expert_groups` key is the declaration; `1` = the grouping is derived from the member
  names). Absent means `0`. This is the anchor that lets any later evolution of the format
  (R3 and beyond) be *distinguished at read time* instead of guessed from tensor shapes.
- `taillis.parent_sha256` : string. The sha256, **lowercase hex, over the WHOLE FILE BYTES** (header,
  KV block and tensor data alike) of the **flat file a grouped file derives from**. This is the
  declared comparator the per-file verification protocol needs: a quality delta between a grouped
  artifact and its parent can be attributed only against the exact bytes it was built from. **R3
  normatively references this key — do not rename it — and §9.3.1 amends what it NAMES for the
  region-quantised path: the flat-quant SIBLING comparator, not the immediate input.**
- `taillis.provenance.<name>` : string, one key per build fact. Defined names, used verbatim:

| key | value |
|---|---|
| `source_checkpoint` | `<hf repo id>@<revision>` |
| `converter` | `<script or 'convert_hf_to_gguf.py'>@<git commit>` |
| `input_sha256` | sha256 of the input GGUF file this artifact was built from |
| `quantize_tool` | `llama-quantize@<llama.cpp commit>` |
| `quantize_flags` | the flags string |
| `imatrix_sha256` | sha256 of the imatrix file (IQ recipes) |
| `grouping_tool` | `group_experts.py@<taillis git commit>` |
| `type_file_sha256` | sha256 of the `--tensor-type-file` (region-quantised artifacts, §9) |
| `arm` | `allocated` or `shuffled-control` (allocation campaigns, §9) |
| `shuffle_seed` | the permutation seed (shuffled controls only, §9) |

For a grouped artifact `taillis.parent_sha256` and `taillis.provenance.input_sha256` name the same
file (the flat parent) and their values may coincide. For a REGION-QUANTISED grouped artifact
`taillis.parent_sha256` names the flat-quant SIBLING comparator, not the immediate input (§9.3.1);
`taillis.provenance.input_sha256` names the immediate input.

A tool version may carry the suffix **`+dirty`** (`group_experts.py@<commit> +dirty`): the named
commit does **not** contain the tool file as it ran — it is untracked or locally modified — so the
pointer identifies the commit *and* warns that the tree at that commit is not the writer. This is
the K3 discipline in miniature: a provenance string must never name a tree lacking the tool that
produced the artifact.

### 7.2 Rules

1. **Every provenance key is optional.** A writer stamps what it knows; absence is not an error and
   is never a reason to reject the file.
2. **Unknown `taillis.*` keys are ignored by the reader.** Arbitrary unknown KV being stock-inert is
   a MEASURED precedent, not an argument: the shipped Unsloth file carries
   `quantize.imatrix.file/.dataset/.entries_count/.chunks_count` — keys stock llama.cpp has no
   schema for — and it is the very file the whole stock-llama.cpp campaign runs (format-proposal §0).
   `taillis.*` keys are the same class of thing.
3. **Absence of `taillis.format_version` means 0** (pre-R1 semantics), exactly as the absence of
   `taillis.expert_groups` means "flat".
4. `taillis.parent_sha256` is the sha of the **DECLARED COMPARATOR** (§9.3.1). (a) For a
   grouping-only artifact that is the **immediate flat parent** — the file the artifact was derived
   from, exactly as R1 defined it. The writer refuses to group an already-grouped file
   (`group_experts.py` refuses any already-grouped SOURCE outright: a source carrying
   `taillis.expert_groups`, `taillis.format_version = 1`, or member-shaped tensor names is not a
   flat parent — layout.md §8.3), so a grouping-only artifact's parent is always flat. (b) For a
   REGION-QUANTISED artifact
   (llama-quantize over a grouped intermediate, §9) the comparator is the **flat quant** built from
   the same >=16-bit source with the same quantize tool and flags at the uniform type — a SIBLING
   of the immediate input, not an ancestor; the immediate input is then recorded as
   `taillis.provenance.input_sha256` (§9.3.1). The re-stamp is a REQUIRED build step:
   llama-quantize copies `taillis.*` KV verbatim, so an unstamped region output INHERITS the
   intermediate's parent hash and misdeclares its comparator (MEASURED, FINDINGS §O / E2-M3).

### 7.3 Invariant: existing artifacts are never rewritten

The F16, arms A/B/S, `-g1`, `-g3`, the shipped quant, **and the two 2-layer slice fixtures in
`models/` (`mistral-small-4-2L-f16.gguf`, `mistral-small-4-2L-q8_0.gguf` — the parent/probe pair
the T2 gates and this unit's own stamp gates measure)** are **documented** (FINDINGS §K, memory)
and are **never rewritten to add these keys**. Rewriting arms A/B/S would change the very bytes
whose quality was measured in §K — invalidating the measurement for zero gain. The keys are stamped
at build time on *future* artifacts only (`convert_f16.sh` STEP 3, `group_experts.py` at write time,
the quantize wrapper when it exists). Any re-derivation of an existing arm under this spec is a NEW
artifact with its own name and its own `parent_sha256` chain.

**Added 2026-10-18, on the same rule (§8.4):** the two full-size R2 v1 fixtures
`MS4-119B-UD-IQ2_XXS-g1-v1.gguf` and `MS4-119B-UD-IQ2_XXS-g3-v1.gguf` are on this list from the
moment they were measured — as are `2L-v1-g3.gguf` and the v0 originals `-g1`, `-g3`, `2L-g3`,
which stay BECAUSE they are the continuity reference: the v0 pair carries the key-declared
grouping the v1 pair must be identical to, so rewriting either side destroys the comparison.
Regrouping under a different default is a new artifact (as this unit's `-v1` names are), never a
rewrite of a measured one.

## 8. R2 — v1: the grouping is derived from the member tensor names (spec)

**Status: COMPLETE, 2026-10-18.** The writer emits v1 **by default**
(`tools/group_experts.py --format`, v0 selectable) and the reader half LANDED in
`aubier` (`engine/gguf_model.h` `read_expert_groups()`, version dispatch); the handshake
§8.4 made binding is MEASURED on this host, including on both full-size grouped
artifacts — §8.4.

### 8.1 Version dispatch

- **v0** — `taillis.format_version` absent, or present and `0`. Two sub-cases, and the
  second is the one that matters for artifacts built outside the grouping writer:
  - the `taillis.expert_groups` string array is present and **is** the declaration
    (§2 semantics, unchanged); or
  - **neither the array nor any member-shaped name is present, and the file is simply
    FLAT** — there is no grouping to declare. A flat artifact must NOT stamp `1`: `1`
    is a claim about a derived grouping, and a file with no members has none.
    (Settled 2026-10-18: the first non-grouped artifact we built under R1 stamped `1`
    because the stamp was specified before this case existed. It is inert either way —
    the reader derives no members and reads the file flat — but a provenance record
    that says "grouping derived from member names" about a file that has no grouping
    is a lie of the same kind this document exists to prevent.)
- **v1** — `taillis.format_version == 1` (u32). Membership and order are **derived**
  from the tensor table (§8.2), and **`taillis.expert_groups` MUST NOT be present** —
  its presence in a v1 file is itself malformed.

*History of the scalar, so nobody trusts it backwards.* §7.1 provisionally defined `1`
as "the R1 keys are present". R2 narrows `1` to v1 semantics and recasts the v0 stamp
to `0`. MEASURED 2026-10-15: no artifact on disk carries the key at all
(`MS4-…-g1`, `MS4-…-g3`, `2L-g3` — none has `taillis.format_version`), so the value is
reclaimed freely. The pre-R2 writer stamped `1` into files that also carried the group
key — exactly the shape §8.1 declares malformed — but no such file was ever built; the
v0 writer stamps `0` from now on.

### 8.2 The derivation (normative — this is what the reader implements)

1. For each expert logical tensor of the model — the reader's own fixed surface,
   `blk.N.ffn_gate_up_exps` and `blk.N.ffn_down_exps`, `N` in `[0, 36)` (72 logicals;
   Mistral Small 4 is the only target) — scan the tensor table for names matching
   `^<logical>\.g([0-9]+)\.weight$`.
2. No match ⇒ the logical is **flat**; look it up as today.
3. One or more matches ⇒ sort the members by **ascending `k`**; ranges are the
   cumulative `ne[2]` in that order. Require: `k` unique per logical (a duplicate `k`
   is malformed and rejected loudly), at least one member, `ne[2] > 0` per member,
   uniform `ne[0]/ne[1]` across members, and `sum(ne[2]) == 128`.
4. `k` **contiguity (0..G-1) is NOT required** — `k` is purely an ordering token.
5. `<logical>.weight` present flat AND members present ⇒ malformed, rejected (the ONE
   surviving key-free consistency check; format-proposal §1 #4).
6. A `.g<k>.weight` whose prefix is not one of the 72 expert logicals is simply
   **unclassified**, and the reader's existing unclassified pass rejects it loudly
   (`engine/gguf_model.h:533-538`) — §1's check #10 by construction for the known
   surface.

What this buys, stated plainly: §1's checks #2/#3/#5 become **impossible, not merely
detectable** — there is no entry to be malformed (#2), no name to be missing (#3: you
only ever look up names found in the table), no list to duplicate (#5); and §1's silent
**order seam closes by construction** — order comes from `k`, the names are the only
channel, so there is no second ordering to disagree with. Checks #6-#8 are data
invariants and stay (§8.2.3).

### 8.3 Writer behaviour

`tools/group_experts.py --format {v0,v1}` (default **v1** since the §8.4 handshake was
measured, 2026-10-18; `--format v0` stays selectable and is the form every artifact built
before that date carries). v1 stamps `taillis.format_version = 1` plus the R1 keys
(§7.1) and emits **no** group key; v0 is today's semantics with the stamp set to `0`
(§8.1). The default covers `--verify` too, so the format a build stamps is the format a
verify expects; §8.4's evidence passes `--format` explicitly at every call so no gate
changes meaning when a default moves. The already-grouped-source refusal
scans the source's tensor table for member-shaped names (`<anything>.g<k>.weight`)
**and** refuses a source carrying `taillis.expert_groups` (a key-carrying source is
grouped, member-shaped names or not — otherwise a v1 output could inherit the stale key
and be §8.1-malformed), so it holds for v0 sources (key + names) and v1 sources (names
only) alike, and also refuses a source carrying member-shaped names outside the expert
surface (such a file is not a flat parent).

### 8.4 Migration order (binding)

**The reader must support v1 before any writer emits v1 by default.** The only reader
is aubier's engine, and until R2 an absent key was read as "flat file" and then failed
`logical()` on the missing flat tensor (`gguf_model.h:237-238` → `:326`) — to the pre-R2
reader a v1 file was indistinguishable from a corrupt flat one. Hence R2 shipped the
writer as an OPT-IN and the default stayed v0.

**SATISFIED 2026-10-18 — R2 IS COMPLETE.** Both halves landed and the handshake is
MEASURED on this host, CPU only (no GPU, no `-ngl`; `~/futaie/aubier` read-only — the
probe is compiled FROM `tools/gguf_reader_probe.cpp` INTO `/tmp` against
`aubier/builds/cpu` and run with `LD_LIBRARY_PATH=$HOME/futaie/aubier/builds/cpu/bin`):

- **READER (aubier, CODE):** `read_expert_groups()` at `engine/gguf_model.h:270-341` —
  version dispatch (absent/0 = the v0 loop verbatim; 1 = the key MUST be absent plus the
  derivation over the reader's own 72-logical checklist, ascending `k` = expert order;
  an unknown version is refused). Report: `~/futaie/aubier/results/loader-r2.md`.
- **HANDSHAKE (MEASURED):** the key-less v1 artifact derives exactly the segments the v0
  key declares. `2L-v1-g3.gguf` (v1, no key) and `2L-g3.gguf` (v0, key declares the
  grouping) print **IDENTICAL** probe stdout — `diff` EMPTY, exit 0 both:
  `blk.0.ffn_gate_up_exps.weight ne=[4096,4096,128,1] segs=3 [0,43)f16 [43,86)f16
  [86,128)f16` (nbytes 1442840576/1442840576/1409286144). The ranges came from the names
  alone.
- **DEFAULT FLIPPED:** `tools/group_experts.py --format` defaults to **v1** (build AND
  verify — the format a build stamps is the format a verify expects); `--format v0` stays
  selectable. Every gate below passes `--format` EXPLICITLY, so no check changed meaning
  when the default moved; `eval/test_group_experts_v1.py` asserts the default itself
  (part 1b) and was proven to FAIL when the default is mutated back to v0.
- **FULL-SIZE artifacts (MEASURED; NEW names, §7.3 — the v0 pair is untouched):**
  `MS4-119B-UD-IQ2_XXS-g1-v1.gguf` (34,924,187,808 B) and `…-g3-v1.gguf`
  (34,924,199,200 B), built by the flipped default from the shipped flat IQ2_XXS quant
  (`parent_sha256 = 82a67684…c478beb3`). `--verify --format v1`: **exit 0 both** —
  gate 1 72/72 (G=1) and 216/216 (G=3) member payloads byte-identical to the parent slab
  plus 507/507 verbatim tensors, gate 5 72/72 logicals with sum(member nbytes) == parent
  nbytes, R1 keys present, group key ABSENT.
- **F1 INERTNESS at FULL SIZE (MEASURED):** probe stdout on the flat quant and on the
  **G=1 v1** file is byte-identical — `diff` EMPTY, md5 `957a895288d0f26cd2b2ab44ab115384`
  for both. G=1 is invisible through the reader, not merely similar.
- **G=3 (MEASURED):** the v1 G=3 file reads back `segs=3 [0,43)/[43,86)/[86,128)` with the
  same types (`iq2_xxs` for `ffn_gate_up_exps`, `iq2_s` for the last member of
  `ffn_down_exps`) and the same nbytes as the v0 G=3 file — full stdout `diff` EMPTY; and
  all 72 logicals equal an **independent** derivation of §8.1/§8.2 — re-derived from the
  file's own declaration with gguf-py (cumulative `ne[2]` in declared/ascending-`k` order,
  gguf-py types and nbytes), NOT with aubier's reader — diff empty.
- **v0 CONTINUITY (MEASURED — this section's own control):** `-g1`, `-g3` and `2L-g3`
  still load (exit 0; 72/72/4 grouped logicals) and their segment lists still equal what
  the v0 key declares (independent derivation, diff empty); the v0 G=1 file is still
  byte-inert against the flat quant (stdout diff EMPTY). Nothing about v0 changed.

What remains binding: the ORDER still runs one way. A future format version may not
flip a default before the reader for it has landed and this same handshake has been
MEASURED on a full-size artifact — and no already-measured artifact is rewritten to get
there (§7.3).

## 9. R3 — the comparison chain as a format obligation (spec)

**Status: spec + taillis-side checker (`tools/check_chain.py`).** No engine change: the reader
knows one `taillis.*` key (`gguf_model.h:183`) and ignores the rest (§7.2 rule 2, inert by the
M2/M3 precedent). No existing artifact is rewritten (§7.3).

### 9.1 The uniform shadow, defined

The **uniform shadow** of a grouped artifact A is the file `tools/group_experts.py --src P --spec S`
produces, where P is A's flat parent (§7.2 rule 4) and S is A's member-size partition. Members
carry P's per-logical types; member payloads are P's byte slabs; all else verbatim. This is the H5
g3-uniform role: a pure re-partition of P that runs on BOTH engine arms (uniform member types can
go through the oracle, H1) and is byte-conserving (writer gate 1). MEASURED (FINDINGS §O): the
shadow's content is byte-identical to the uniform REGION-quantisation of the same source for
imatrix-free k-quants (43/43 on the 2L slice), so for every member of A whose type equals P's type
for its logical, A's bytes are IDENTICAL to the shadow's — the A-vs-shadow delta isolates the TYPE
POLICY with no requantisation confound. Scope: proven for imatrix-free k-quants only; IQ/imatrix
region passes are outside the proof and §9.4 gate 5 is scoped accordingly.

### 9.2 Keys

```
taillis.uniform_shadow_sha256        : string — REQUIRED on mixed-type grouped artifacts,
                                       FORBIDDEN on uniform-grouped ones. A MIXED artifact is
                                       one with any logical whose members do not share one
                                       type (H1's own oracle criterion).
taillis.provenance.type_file_sha256  : string — sha256 of the --tensor-type-file
                                       (region-quantised artifacts).
taillis.provenance.arm               : string — "allocated" or "shuffled-control"
                                       (allocation campaigns).
taillis.provenance.shuffle_seed      : string — the permutation seed (shuffled controls only).
```

`taillis.uniform_shadow_sha256` = sha256 over the shadow's **canonical content stream**: for each
tensor of the shadow's tensor table in TABLE ORDER, concatenate
`u32le(len(name)) || UTF-8(name) || u32le(ggml_type) || u32le(n_dims) || u32le(ne[0..n_dims-1]) ||
u64le(payload_nbytes) || payload bytes`; hash the concatenation. The domain EXCLUDES the KV block
deliberately: the stream is invariant to grouping_tool version, +dirty state, v0/v1 (§8) and
provenance keys — none of which are weights. MEASURED justification (FINDINGS §O): two shadows of
one (parent, spec) that are content-identical but whole-file-different. Therefore: this key is NOT
verifiable with `sha256sum`; it is verifiable with `tools/check_chain.py`, which computes the
stream from a shadow file OR directly from (P, S) — the shadow need not stay on disk to verify its
identity.

### 9.3 Rules (builder obligations)

1. Every grouped artifact carries `taillis.parent_sha256` = the DECLARED COMPARATOR (§7.2 rule 4,
   as amended by this section). (a) grouping-only artifacts: the immediate input. (b)
   region-quantised artifacts: the FLAT QUANT from the same >=16-bit source, same quantize tool
   and flags, at the uniform type — a SIBLING of the immediate input; the immediate input is
   recorded as `taillis.provenance.input_sha256`. llama-quantize copies `taillis.*` verbatim, so
   the region path MUST re-stamp `parent_sha256` after quantising (an unstamped output inherits
   the intermediate's parent and misdeclares the comparator — MEASURED, FINDINGS §O).
2. A mixed artifact MUST carry `uniform_shadow_sha256`; a uniform-grouped artifact MUST NOT.
3. Shadow rebuild rule: `group_experts.py --src <parent> --spec <S>`, ANY `--format`. Requirements
   for the rule to hold, each a NAMED checker outcome when violated: the artifact's partition is
   uniform across logicals (else "shadow not rebuildable: non-uniform partition"); each logical's
   k values are exactly 0..G-1 (else "k not canonical"); the parent is flat. A shadow that cannot
   be rebuilt is reported, never silently skipped.
4. EVALUATION PROTOCOL, normative for any T2/R4 quality claim: the chain is P (both arms) ->
   shadow (oracle arm) -> artifact (store arm); token lines are re-verified PER FILE and recorded
   as DIVERGENCE DETECTORS (H4: never the verdict); the shadow's oracle token line is EXPECTED
   byte-identical to P's oracle line (H2) — a divergence there is a gate failure (the file does
   not carry P's weights), not a measurement; the verdict is a paired average metric through the
   §K instrument (same windows, per-chunk PPL, pre-declared 0.5% rule, INSUFFICIENT possible); a
   null is publishable (§5); adverse direction stops the campaign (the K1 pattern).
5. Any ALLOCATION claim additionally requires the SHUFFLED CONTROL (§9.3.6 + §9.4 gates) measured
   through the same instrument with the same pre-declared rule; the verdict is read against the
   pre-declared rule, never post-hoc.
6. THE SHUFFLED CONTROL: a second region-quantised artifact from the same grouped intermediate,
   whose type assignment is a SEEDED permutation of the allocated assignment among members of
   **IDENTICAL SHAPE — the same `(ne[0], ne[1], ne[2])`, i.e. members that any type swap leaves
   the same size.** Same shape does **not** require the same logical: two `ffn_gate_up_exps`
   members of equal size in different layers are interchangeable and swapping between them is a
   valid, conserving permutation. Unchanged: boundaries, member sizes, parent, shadow. Exactly
   conserved: the global type multiset and total tensor bytes (whole-file within 384 B, the D5
   class).
   **MEASURED, and the reason this is not merely equal `ne[2]`:** `ne[2]` alone does not make two
   members interchangeable. A 43-expert `ffn_gate_up_exps` member and a 43-expert `ffn_down_exps`
   member hold a different number of neighbours per expert (4096-wide vs 2048-wide), so moving a
   type between them changes total bytes — the obvious cross-logical swap was built and measured at
   **−87,359,424 B** on the 2L chain (FINDINGS §O). Equal `ne[2]` at different widths is not a
   conserving permutation; equal `(ne[0],ne[1],ne[2])` is. G6 measures the conservation directly
   (§9.4), so a control built the wrong way is rejected before it is ever measured — but the rule
   is stated here so that no builder has to discover it. It carries the
   SAME parent_sha256 and uniform_shadow_sha256 VALUES as the allocated artifact, plus
   arm=shuffled-control, shuffle_seed, type_file_sha256. A control counts ONLY if the checker's
   control gates pass — validity is derived, not asserted. If the allocation's boundaries make
   equal-size classes degenerate, the builder keeps uniform boundaries or documents the weaker
   bytes-matched re-draw control in the campaign record; §9 does not recognise an undocumented
   control.
7. (Q2, folded) Writer rule-of-thumb: every group boundary inside [0, n_res) becomes a VRAM store
   boundary with its own (c,n) graph set and drain() pulls — keep boundaries >= the default
   n_res, or accept the extra slices knowingly.

### 9.4 The checker (`tools/check_chain.py`) — gates, each a named failure

- **G1** the artifact parses and its grouping is declared (v0 key or v1 derivation); per-logical
  invariants of §8.2.3 hold (sum(ne[2])==expert_count, ne[2]>0, uniform ne[0]/ne[1], k unique).
- **G2** mixed-ness: any logical with non-uniform member types => uniform_shadow_sha256 REQUIRED
  (absent => FAIL "R3.2"); a uniform artifact carrying it => FAIL (forbidden).
- **G3** parent: sha256(--parent file) == taillis.parent_sha256, else FAIL naming both values.
- **G4** shadow identity: derive S from the artifact's own table (uniformity + canonical k checked
  first); synthesize the canonical stream from (parent, spec) — no shadow file needed — and
  compare to the key (mismatch => FAIL "shadow identity mismatch"). If --shadow FILE is given,
  additionally hash ITS stream: it must equal both the synthesized stream and the key, and its
  member slabs must equal the parent's slabs (the writer's own gates re-run).
- **G5** sibling byte gate: for every member whose type == the parent's type for its logical,
  payload sha == the parent's slab sha. HARD when the artifact records no imatrix_sha256 (the E2
  proof applies; a mismatch means wrong parent or wrong weights); WARN-only when an imatrix is
  recorded (outside the E2 proof — the warning names both possibilities, it never silently
  passes).
- **G6** control gates (--control): same parent key, same shadow key, same per-logical member
  sizes, same global type multiset, |whole-file size difference| <= 384 B, assignment differs,
  arm/seed/type-file keys present.
- **G7** exit code = gate; output prints COUNTS, SIZES, PATHS and each failure with its mechanism.

## V4 — the variant band and the OVERLAY file class (spec)

**Provenance: written by aubier 2026-10-18 (a different aubier session from the engine-swap
work), and renamed `V4` by agreement on the taillis<->aubier bus the same day to free `R4` for the
format proposal's fixed-bytes hotness allocation (`docs/format-proposal.md` R4). The measurement
V4.7 asks for has since been RUN and returned **NO STABLE BAND** (`results/cold-band.md`);
V4.7's own consequence therefore applies.**

**Status: PROPOSAL, 2026-10-18. Nothing here is implemented.** Motivating record:
`../aubier/docs/expert-swap.md` (S1–S7), the engine-side plan for serving a *variant* — a set of
overridden `(layer, expert)` weights whose bytes come from a second file. This section amends the
FORMAT only; it decides no policy, and it creates no obligation on any artifact already on disk
(§7.3 stands unchanged).

**V4 is additive and inherits §7.2 rules 1–2 verbatim**: every new key is optional, absent keys mean
the file is exactly today's file, and unknown `taillis.*` keys are ignored by the reader. **V4 adds
no value to `taillis.format_version`** and does not touch §8.1 dispatch — a file may be v0 or v1 and
still carry a band.

### V4.1 Why: the one thing the format cannot currently express

A variant needs its target experts to be addressable as a *unit* by the engine, and it needs the
engine to know, without a side file, which experts are the adapt/swap home. Today that is
inexpressible: `taillis.expert_groups` says how the expert axis is *split for quantisation*, which
is a different question from *which experts a variant will replace*.

There is also a hard structural constraint that decides the design: **groups are contiguous ranges
that tile `[0,128)`** (§2 rule 1). A measured cold set like `{17, 42, 97}` therefore **cannot be a
group**, and cannot be a band. Any "variant = group replacement" design is available only if the
variant's experts are chosen *contiguously*, up front. V4 makes that option expressible; it does not
assert it is the right one.

### V4.2 `taillis.variant_band` — SUPERSEDED, DO NOT IMPLEMENT

**Status: DEAD. `taillis.variant_band` stays unimplemented (settled 2026-10-18).** V4.7's deciding
measurement was run and returned **NO STABLE BAND** in both regimes and both populations
(`results/cold-band.md`, gate re-run independently, exit 0): 29–34 distinct coldest windows across
35–36 layers out of 121–126 candidates, the best global width-4 band carrying 0.78–0.89× uniform, and
the per-layer coldest window exactly as cold as relabelling the same count vectors (observed/null
0.973–1.022, Bonferroni 0). A file-wide contiguous band is therefore not a place that is cold, and the
variant design is the **manifest row-range** of V4.3 below. The rest of this subsection is kept as the
record of what was considered, not as a specification.

```
taillis.variant_band : u32[2]   # [lo, hi), the reserved adapt/swap home in the EXPERT axis
```

Normative properties:

1. **Absent ⇒ no band** and behaviour is exactly today's. Present ⇒ `0 <= lo < hi <= 128`.
2. **Named as an expert RANGE, never as a group.** Grouping is a quantisation decision that changes
   between recipes; which experts are the adapt home is a model decision. Naming `g3` would make the
   band disappear when a later recipe re-groups.
3. **BAND EDGES MUST FALL ON GROUP BOUNDARIES** in every grouped expert logical tensor. This is the
   invariant that makes a variant a *whole-group replacement* (GGUF-native, no row arithmetic, and a
   member type may then differ from its neighbours without inventing per-expert tensors — §2 rule 3
   still holds, because the band is a group like any other). A grouped file whose boundaries cut the
   band MUST be refused at load with a named failure, not silently served.
4. **The band's member type is the variant's type** (engine-side same-type rule, `expert-swap.md`
   §5.3). Consequence, stated so nobody is surprised: **the band's type is a quality CEILING for
   every variant placed in it.** A cold band wants few bits; an adapted expert wants more. Those pull
   opposite ways and the choice belongs to the recipe, not the format.

### V4.3 The OVERLAY file class — `.ov<k>` tensors

An OVERLAY is a GGUF that is **not a model**: it carries only override tensors plus KV, and it is
read by an overlay loader, never by `GGUFModel::load()`.

```
blk.<l>.ffn_gate_up_exps.ov<K>.weight   ne = (4096, 4096, n_ov)
blk.<l>.<logical>.ov<K>.weight          ne = (…, …, n_ov)      one band of experts
```

1. **`.ov<k>` MUST NOT match `<logical>.g<digits>`.** It is chosen so §8.2's name-derived v1
   grouping cannot mistake an override for a grouping declaration. `group_member_logical()`
   (`aubier/engine/gguf_model.h`) already requires `.g` followed by digits only — V4 fixes the
   overlay naming so that requirement keeps holding *by construction*, and any future overlay naming
   scheme must preserve it.
2. **An OVERLAY MUST NOT contain `.g<digits>` members** and **a BASE MUST NOT contain `.ov<k>`
   tensors.** Symmetric refusal, both named failures.
3. **Two kinds of override tensor, and the kind is DERIVED from the base tensor's rank** — never from a
   second declaration:
   - **EXPERT-SLICE**: the base tensor has an expert axis (`ne[2]`), e.g. `ffn_gate_up_exps`. Then
     `ne[2] == n_ov == len(experts)`, every id lies inside **one** base group's range, and the tensor
     carries a matching `taillis.overlay_experts.<ovk>` key. (The band of V4.2 is dead; there is no band
     to also lie inside.)
   - **WHOLE-TENSOR**: the base tensor has **no** expert axis (rank ≤ 2) and the override replaces the
     whole tensor — `blk.<l>.ffn_gate_inp.ov<k>.weight`, the router, `ne = (4096,128)` per layer. It needs
     **no** id key, and its shape/type must equal the base tensor's exactly. This case was missing from
     the first draft and it blocks S4, which `aubier/docs/expert-swap.md` §4 calls *"the one to build
     first"*: the router-only variant is 72 MiB, exercises the entire overlay path, and touches no expert
     tensor. A format that cannot express the cheapest end-to-end test of itself is not finished.
4. **The overlay's `taillis.parent_sha256` IS the base file's sha256** (§7.1's "declared
   comparator", §7.2 rule 4). No new key is invented for this: `tools/check_chain.py` G3 already
   verifies `sha256(--parent) == parent_sha256`, so **G3 applies to overlays unchanged** — this is the
   whole reason overlays reuse the namespace rather than starting a parallel one.
5. **A type mismatch against the target group is a named failure at load**, not at first use. For a
   WHOLE-TENSOR override the check is against the base tensor's own type, and the shape must match too.
   **An overlay carries NO model KV** — no `general.architecture`, no `<arch>.block_count`, no expert
   count. It is not a model; a loader takes all of that from the BASE and must not read it from the
   overlay (and an overlay that carries it is malformed, because it is a second copy of a fact the base
   owns).
6. **THE MANIFEST IS IN THE FILE, and it is derived wherever derivation is possible.** This is the one
   place V4 departs from `aubier/docs/expert-swap.md` §4's draft, which put the override list in a side
   `manifest.json` — and it departs deliberately, for the reason R1 exists at all (FINDINGS §K/K3: a
   5.6% quality delta stayed unattributable because the files did not carry what built them). A side
   file is lab notes wearing a filename. So:

   | what | how |
   |---|---|
   | layer, logical, `k` | **DERIVED** from the tensor name `blk.<l>.<logical>.ov<k>.weight` — no declaration (§8.2's principle). |
   | the base it overrides | **DECLARED**, already: `taillis.parent_sha256` = the base file's sha (clause 4). |
   | expert ids per override tensor | **DECLARED**: ONE SCALAR STRING KEY PER OVERRIDE TENSOR, keyed by the `k` token ALREADY IN THE TENSOR NAME — `taillis.overlay_experts.ov3 = "17,42,97"`. `k` is **globally unique within an overlay file** (the exporter assigns it), so the key addresses exactly one tensor. A `.ov<k>` tensor with no matching key is a named failure, and so is a key with no tensor. |
   | the target group | **DERIVED**: the group of the base whose range contains the expert ids. All ids of one entry must fall in exactly ONE base group; a straddling entry is malformed. |
   | the type | **DERIVED** from that base group, and CHECKED equal to the tensor's type (clause 5). |
   | whether it is an identity overlay | **DECLARED**: `taillis.overlay_identity` (V4.4). |

   Alignment is by the `k` TOKEN, never by position. An earlier draft of this section aligned a single
   `taillis.overlay_experts` ARRAY to the tensor table by order; that was withdrawn (aubier raised it, and
   they are right) because a length check cannot catch a mispair: two entries with the same count swap
   cleanly, and if both fall inside the SAME base group the group and type checks pass too — so the loader
   would install expert 17's bytes at expert 42's slot, silently. Binding by `k` removes the alignment
   entirely, because `k` is in the tensor name. A side `manifest.json` remains legal as an **export-side
   convenience** (a human reading a shelf), never as the authority: a loader that needs it to place an
   override is refusing to work, and must say so rather than guess.

### V4.4 §7.1 provenance names added by V4

Extending the defined-name table (all `string`, all optional, `tools/stamp_gguf.py` writes them):

| key | value |
|---|---|
| `overlay_experts.<ovk>` | ONE SCALAR STRING PER OVERRIDE TENSOR, keyed by the exact `k` token in that tensor's name (`taillis.overlay_experts.ov3` for `blk.<l>.<logical>.ov3.weight`), value a comma-separated ASCENDING list of distinct ids in `[0,128)`. **`k` must be globally unique within the file** (V4.3 clause 6). REQUIRED on every overlay: a `.ov<k>` tensor with no matching key is a named failure, and so is a key with no tensor. |
| `overlay_identity` | `true` for a deliberate IDENTITY overlay (every override tensor's bytes equal its base slab) — the engine's paired gate uses one to prove the read path is transparent; `false` or absent otherwise. A FILE-LEVEL flag: it describes the whole overlay, not one tensor. |

| `overlay_tool` | the exporter that wrote the overlay, `tools/repack_overlay.py@<taillis git commit>[+dirty]` — same convention as `grouping_tool` (§7.1). Optional but stamped, for the K3 reason. |
| `overlay_digest.<ovk>` | sha256 of that override tensor's PAYLOAD, one per tensor, keyed by `k` like the ids. **REQUIRED on every overlay.** Without it a non-identity overlay is unauditable against its base *by construction*: its payload is supposed to differ from the base's, so nothing in the file can distinguish a correct override from a corrupted one. Stamped at build; `--verify` recomputes. |

**TWO KEYS ARE MANDATORY ON A NON-IDENTITY OVERLAY, and both are §7.1 names that already exist:** the
imatrix sha256 the override bytes were quantised with, and the type/quant spec used. Without them a
variant's measured quality is confounded with a requantisation, and any comparison will blame the variant
for the quantiser — the exact attribution failure R1 was built to stop (§5.3 of aubier's plan makes the
rule checkable; these keys are what make it checkable *from the file*).

**On an IDENTITY overlay they are OPTIONAL — and that is a correction to this spec.** The first draft said
they are FORBIDDEN there, on the reasoning "identity ⇒ a byte copy ⇒ nothing was quantised". **MEASURED
falsification:** re-quantising the base's own F16 slice at the base's own type with the base's own imatrix
returns the base's bytes EXACTLY (24 slabs, 84,944,656 bytes, 0 differing, across `IQ2_XXS`, `IQ2_S` and
`IQ3_S`), so an overlay can be identity-compatible *in bytes* **and** have been produced by quantisation.
The two properties are independent, and a rule keyed on one must not forbid the other's provenance.
Required on non-identity, optional on identity, and the checker accepts both.

**AND "the same imatrix" needs a mechanism, because the literal reading does not work.** `llama-quantize`
**refuses** a full imatrix for a slice: the matrix is per-`(expert, column)` and its length must equal
`ne[0] * ne[2]` (`llama-quant.cpp:1284`, used at `:784-786`). So the build rule is: **the SAME imatrix,
CROPPED to the same experts** — exact and lossless — recorded as `taillis.provenance.expert_crop`.

**`overlay_target` and `overlay_group` are NOT keys and must not be written.** An earlier draft of this
table declared them; both are derivable — the target logical is in the tensor name and the group follows
from the expert ids against the base's segments (V4.3 clause 6). Declaring them would be the same
redundancy disease §8/R2 removed: a second declaration of a fact the artifact already carries, and a
second thing that can disagree with it.

### V4.5 Checker gates — `tools/check_chain.py`

**G5 does NOT apply to an overlay and must be SKIPPED for one, with a named reason**: G5 requires a
member of the parent's type to carry the *parent's slab bytes*, which is precisely what an overlay
exists to change. The overlay's complement property is G8.3 instead.

- **G8** OVERLAY validity (V4.3): parses; no `.g<digits>` member; **the kind of each override is derived
  from the BASE tensor's rank** (V4.3 clause 3) — for an EXPERT-SLICE tensor, a matching
  `taillis.overlay_experts.<ovk>` key must exist and **every such key must have a matching tensor** (both
  directions, a named failure each — there is no positional alignment to get wrong, so a missing or
  orphaned key is the only pairing error possible); for a WHOLE-TENSOR override no key is expected and a
  stray key is FAIL; `k` is unique across the file; every entry parses as an ascending list of distinct ids
  in `[0,128)`; every expert-slice tensor's `ne[2] == len(entry)` and **every id of one entry falls in
  exactly ONE base group's range** (a straddling entry is FAIL); a whole-tensor override's shape and type
  EQUAL the base tensor's; every expert-slice tensor's type EQUALS its derived group's type;
  `overlay_target` / `overlay_group` are ABSENT (they are derived facts and their presence is FAIL — V4.4);
  the quantisation provenance (imatrix sha, type spec) is present on a NON-identity overlay and permitted on
  an identity one (V4.4); **every override tensor's payload hashes to its `taillis.overlay_digest.<ovk>`**
  (a mismatch is FAIL naming the tensor — this is the ONLY content check possible on a non-identity
  overlay); **the overlay carries no model KV** (V4.3 clause 5); G3 passes against the base.
- **G8.3** overlay↔base direction: bytes EQUAL to the base slab ⇒ REQUIRES
  `overlay_identity=true` (otherwise FAIL: a pointless overlay is a build error, and an identity
  overlay is legal *only* when it says so). Bytes DIFFER ⇒ `overlay_identity` must be absent/false.
- **G9** symmetry refusals: a BASE containing `.ov<k>` ⇒ FAIL "overlay tensors in a base"; an OVERLAY
  containing `.g<digits>` ⇒ FAIL "grouping declaration in an overlay"; a grouped base whose group
  boundary cuts `variant_band` ⇒ FAIL naming the boundary and the band.

### V4.6 Tool obligations (not yet implemented — do not brief any of this as existing)

- `tools/check_chain.py` — G8, G8.3, G9 above; G5 skipped for overlays.
- `tools/gguf_reader_probe.cpp` — the probe must refuse an overlay as a BASE with a named failure
  ("overlay file, not a base") **before `GGUFModel::load()` is called**. Two reasons, both MEASURED
  from the tool: `load()` CHECKs the fixed 579-tensor surface, so an overlay's few tensors fail as a
  missing-tensor message that names nothing useful; and if it ever got past `load()`, the coverage
  CHECK would fail too, because **`<logical>.ov<k>` CONTAINS the substring `ffn_gate_up_exps`, so
  `is_expert_logical()` matches it** and then requires segments covering `[0, 128)`. This is a
  requirement of the probe, not of the reader.
- `tools/group_experts.py` — refuses an already-grouped source (§8.3); must also refuse an OVERLAY
  source, and must never emit `.ov<k>`.
- `tools/stamp_gguf.py` — writes the V4.4 names; `overlay_identity` is REQUIRED on an identity
  overlay and forbidden otherwise.

### V4.7 What V4 does NOT decide, and the measurement that should decide it

**Whether a band should exist at all.** V4.2 makes the *contiguous-band* design expressible; it does
not endorse it. The competing design (`../aubier/docs/expert-swap.md` §4) puts the override list in the
manifest and needs no band, and it is the only one that supports a *scattered* expert set.

The deciding measurement is cheap and uses instruments already in this repository: **is there a
stable contiguous cold band?** `t22a`/`t22b` already measure per-expert routing; ask of several
sessions whether ~4 experts are consistently cold on a general sample AND on the target corpus
(aubier's dual criterion). Stable band ⇒ V4.2 is the better path. No stable band ⇒ use the
manifest row-range design and leave `variant_band` unimplemented (an absent key costs nothing, which
is the point of putting it in this namespace rather than in the engine).

This mirrors §5's discipline: the prior is small, and a null result is a legitimate and useful
outcome.
