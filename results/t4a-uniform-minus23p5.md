# T4-a — the uniform −23.5 % expert-byte CONTROL

The artifact aubier's engine needs to compare the hotness-*allocated* treatment against:
one flat file, built from the F16 in a single `llama-quantize` pass, in which **every one
of the 128 experts in all 36 layers** loses the same 23.5 % of its bytes and **nothing
else changes**.

| | |
|---|---|
| artifact | `/var/lib/taillis/models/t4a-uniform-minus23p5.gguf` |
| sha256 (final, stamped) | `db4c82070fad41a8b099f9b7ab4321a1698b66988b4b3d909617025f36d2f783` |
| size | 27 206 669 120 B = 25.34 GiB |
| tensors | 579 (flat; **no** grouping, no `taillis.expert_groups`) |
| built | 2026-10-18 18:45→19:15 UTC, 1 774 983.88 ms wall (MEASURED, `logs/t4a-build.log`) |
| GPU | none used: no `gpu.lock`, no `-ngl`, nothing beyond the CPU `llama-cli` load smoke |

Everything below is measured on this host unless marked otherwise. **A gate is the exit
code**, and each gate's exit code is quoted with it.

---

## 1. The recipe (verbatim)

Uniform across all 128 experts and all 36 layers:

```
blk.N.ffn_gate_up_exps.weight   ->  iq1_s     (from iq2_xxs)
blk.N.ffn_down_exps.weight      ->  iq2_xxs   (from iq2_s on 32 layers, iq3_s on 4)
EVERY OTHER TENSOR              ->  unchanged, at the type the shipped flat file uses
```

Source is the **F16**, never a requantise of a 2-bit file (`--allow-requantize` is not
used anywhere; findings O4/T1's rule). IQ types require an importance matrix and that
matrix is the workload one, `imatrix-q8-250c.gguf`.

MEASURED, the imatrix is sufficient for this recipe and no more: 432 entries; **all 72
expert tensors are covered**; the 147 tensors without an entry are exactly the ones the
shipped file keeps at F32 plus `token_embd`/`output` (which never take imatrix data).

## 2. The `--tensor-type-file`, derived from the shipped artifact

The non-expert types are not retyped from a document — they are read out of the shipped
flat file's own tensor table (`…UD-IQ2_XXS.gguf`), which carries 19 logical types over
579 tensors. The two expert logicals are then overridden per §1. The file has **no
comments** (the parser rejects `#`), every pattern is **anchored**, and the last line is a
**catch-all sentinel** that would make any unlisted tensor F16 — i.e. a ~222 GiB output
rather than a silent mis-type.

```
^blk\.[0-9]+\.attn_k_b\.weight$=Q8_0
^blk\.[0-9]+\.attn_kv_a_mqa\.weight$=Q8_0
^blk\.[0-9]+\.attn_kv_a_norm\.weight$=F32
^blk\.[0-9]+\.attn_norm\.weight$=F32
^blk\.[0-9]+\.attn_output\.weight$=Q5_K
^blk\.[0-9]+\.attn_q_a\.weight$=Q5_K
^blk\.[0-9]+\.attn_q_a_norm\.weight$=F32
^blk\.[0-9]+\.attn_q_b\.weight$=Q8_0
^blk\.[0-9]+\.attn_v_b\.weight$=Q8_0
^blk\.[0-9]+\.ffn_down_exps\.weight$=IQ2_XXS
^blk\.[0-9]+\.ffn_down_shexp\.weight$=Q6_K
^blk\.[0-9]+\.ffn_gate_inp\.weight$=F32
^blk\.[0-9]+\.ffn_gate_shexp\.weight$=Q5_K
^blk\.[0-9]+\.ffn_gate_up_exps\.weight$=IQ1_S
^blk\.[0-9]+\.ffn_norm\.weight$=F32
^blk\.[0-9]+\.ffn_up_shexp\.weight$=Q5_K
^output\.weight$=Q4_K
^output_norm\.weight$=F32
^token_embd\.weight$=Q4_K
^.*$=F16
```

sha256 of that file (stamped as `taillis.provenance.type_file_sha256`):
`1ed8310287222d7cfa06d58e78ae9398ec73f6ad20e51ca41afdad38ccad9961`.

Two parser facts the file depends on, both read out of the source rather than assumed
(INTERPRETATION from `src/llama-quant.cpp`, MEASURED against the dry run):

- the file's unit is a whitespace-separated **`name=type`** token — not `name type`.
  The first attempt used a space and the dry run refused it
  (`parse_tensor_type: malformed tensor type '…'`). **The dry run is what caught this,
  before any hour of CPU.**
- patterns are `std::regex_search`, **first match wins**, and overrides are consulted
  **only when the positional ftype is a quantised type** (`ggml_is_quantized(default_type)`,
  `src/llama-quant.cpp:716`). Hence the positional type is `IQ2_XXS` (a real quant type)
  and the fall-through guard is the trailing `^.*$` sentinel — **not** a positional of
  `f16`, which would have silently disabled every override instead.

Generator (reproduces the file byte-for-byte from the two inputs; run with the `gguf`
venv):

```python
#!/usr/bin/env python3
import re
from collections import Counter
from pathlib import Path
from gguf import GGUFReader

F16  = "/var/lib/taillis/models/Mistral-Small-4-119B-2603-f16.gguf"
SHIP = "<aubier>/models/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf"   # read-only
OVERRIDE = {"ffn_gate_up_exps.weight": "IQ1_S", "ffn_down_exps.weight": "IQ2_XXS"}
SENTINEL = "F16"

f16, ship = GGUFReader(F16), GGUFReader(SHIP)
names = [t.name for t in f16.tensors]
ship_types = {t.name: t.tensor_type.name for t in ship.tensors}
assert len(names) == 579 and set(names) == set(ship_types)

collapse = lambda n: re.sub(r"^blk\.\d+\.", "blk.N.", n)
suffix   = lambda n: re.sub(r"^blk\.\d+\.", "", n)

logical = {}
for n in names:                       # shipped type per logical; uniform except ffn_down_exps
    key, t = collapse(n), ship_types[n]
    if suffix(n) in OVERRIDE:         logical[key] = OVERRIDE[suffix(n)]
    elif key in logical:              assert logical[key] == t
    else:                             logical[key] = t
assert len(logical) == 19

expected = {n: (OVERRIDE.get(suffix(n)) or ship_types[n]) for n in names}
pattern  = lambda k: "^" + (r"blk\.[0-9]+\." + re.escape(k[6:]) if k.startswith("blk.N.")
                            else re.escape(k)) + "$"
entries  = sorted((pattern(k), t) for k, t in logical.items()) + [("^.*$", SENTINEL)]

def plan(name):                       # same semantics as the C++: first match wins
    for pat, t in entries:
        if re.search(pat, name): return t, pat
    raise AssertionError(name)

assert all(plan(n)[0] == expected[n] for n in names)          # no mis-type
assert not [n for n in names if plan(n)[1] == "^.*$"]         # no fall-through
print("\n".join(f"{p}={t}" for p, t in entries))
```

## 3. The command (verbatim)

Dry run first (exit 0, 245 ms):

```
LD_LIBRARY_PATH=$HOME/futaie/aubier/builds/cpu/bin \
  ~/futaie/aubier/builds/cpu/bin/llama-quantize \
    --imatrix ~/futaie/taillis/imatrix-q8-250c.gguf \
    --tensor-type-file /tmp/t4a/t4a-types.txt \
    --dry-run \
    /var/lib/taillis/models/Mistral-Small-4-119B-2603-f16.gguf \
    /var/lib/taillis/models/t4a-uniform-minus23p5.gguf \
    IQ2_XXS 32
```

The real run is the same command **minus `--dry-run`**, run as a **tracked background
task** (pid verified alive after launch, not a detached job):

```
LD_LIBRARY_PATH=$HOME/futaie/aubier/builds/cpu/bin \
  ~/futaie/aubier/builds/cpu/bin/llama-quantize \
    --imatrix ~/futaie/taillis/imatrix-q8-250c.gguf \
    --tensor-type-file /tmp/t4a/t4a-types.txt \
    /var/lib/taillis/models/Mistral-Small-4-119B-2603-f16.gguf \
    /var/lib/taillis/models/t4a-uniform-minus23p5.gguf \
    IQ2_XXS 32
```

Exit 0; `quant size = 25938.75 MiB (1.83 BPW)`; 579/579 tensors processed. **No
`--allow-requantize`**, no `--pure`, no `--include/--exclude-weights`, no
`--output-tensor-type`, no `--token-embedding-type` — the type file alone decides.

The dry-run's plan was then diffed against the expected table: **0 differing rows** over
579 tensors, and its total (25 938.75 MiB = 27 198 754 816 B) equalled the predicted
expert + non-expert payload byte-for-byte before the real run started.

## 4. Byte accounting (MEASURED vs PREDICTED)

| quantity | predicted | measured | delta |
|---|---:|---:|---:|
| expert bytes (72 tensors) | 25 065 160 704 | **25 065 160 704** | **0** |
| every other tensor (507) | 2 133 594 112 | **2 133 594 112** | **0** |
| payload total | 27 198 754 816 | 27 198 754 816 | 0 |
| file (payload + header/KV/padding) | — | 27 206 669 120 | +7 914 304 |

```
experts   32 782 680 064 B (30.5312 GiB)  ->  25 065 160 704 B (23.3438 GiB)   -23.54 %
per expert 7 114 296.9 B (6.7847 MiB)     ->  5 439 488.0 B (5.1875 MiB)       x 0.764585
other tensors 2 133 594 112 B (1.9871 GiB), unchanged
whole file    34 924 187 712 B (arm A)    ->  27 206 669 120 B                 -22.10 %
```

**It is −23.54 %, not −25.0 %.** No uniform step lands on 25 %; the number is not
rounded in this write-up and should not be rounded downstream. The prediction is exact
because the two new types are exact for these shapes and **no `tensor_type_fallback`
fired** (0 fallback warnings; both `ne[0]`=4096 and 2048 are divisible by the 256-block).

## 5. Gates — each with its exit code

**(a) type read-back from the built file — PASS (exit 0).**
579 tensors. `ffn_gate_up_exps` × 36 → `IQ1_S` (single distinct type); `ffn_down_exps` ×
36 → `IQ2_XXS`; **exactly 72 tensors differ from the shipped file's types and they are
exactly the 72 expert tensors**; expected-vs-got diff rows: **0**.

**(b) non-expert isolation vs ARM A — PASS (exit 0).** Payload sha256 per tensor, all 579:

```
non-expert tensors: 507   identical=507   DIFFERENT=0
expert     tensors:  72   identical=0     differ=72
```

This is the gate that defines the control: **every non-expert tensor is byte-identical to
arm A** (same F16 source, same type, same tool, same imatrix), so this file differs from
arm A in exactly the two expert types. The converse assertion is in the same gate — all
72 expert payloads **must** differ, so the comparison cannot be vacuous. Because a stamp
rewrites the file and shifts the tensor-data offsets, the gate was re-run on the bytes
each rewrite produced — after the provenance stamp, and again after the `format_version`
re-stamp to `0` (§8.1): **still 507/507 identical and 72/72 differing, exit 0**
(`logs/t4a-gates-abc-post-stamp.log`, `logs/t4a-gates-abc-post-v0.log`).

**(c) byte accounting — PASS (exit 0).** Deltas 0 and 0, table in §4.

**(d) stock load — PASS (exit 0).** The file is FLAT, so stock llama.cpp accepts it:

```
LD_LIBRARY_PATH=$HOME/futaie/aubier/builds/cpu/bin \
  ~/futaie/aubier/builds/cpu/bin/llama-cli \
    -m /var/lib/taillis/models/t4a-uniform-minus23p5.gguf \
    -p "The capital of France is" -n 4 -c 1024 --no-warmup -st --no-display-prompt
```

Exit 0, CPU only, no `-ngl`; a real forward pass, not just a header parse:
`Prompt: 19.0 t/s | Generation: 2.5 t/s`, generation `The capital of France`. Run against
the **final stamped artifact** (`format_version = 0`), so this too is measured on the
bytes `db4c8207…f783` names.

*This gate could have passed falsely and was caught doing it:* the first invocations used
`-c 64` and `-c 512`; both **exit-0'd** while the chat template's ~545-token request
overflowed the context, so **no token was generated** (`Error: request (545 tokens)
exceeds the available context size`). Exit 0 alone was not evidence; the recorded run
generates text. This is the "a gate is the exit code, not the tail of stdout" rule biting
in the *other* direction — here the exit code was clean and the **work had not happened**,
so the gate needs a second, positive witness (a generated token) and not just a status.
(A coherence check on four tokens is a sanity signal only — no quality claim is made
here.)

**(e) the numbers aubier needs** — see §6.

## 6. What aubier can plug in

MEASURED (this artifact):

- bytes/expert **5 439 488 B = 5.1875 MiB** (gate_up `IQ1_S` 3.125 MiB + down `IQ2_XXS`
  2.0625 MiB), against 6.7847 MiB today; ratio **0.764585**.
- total expert bytes **25 065 160 704 B = 23.3438 GiB** for 36×128; non-expert
  2 133 594 112 B (unchanged); whole file 25.34 GiB.
- per layer: `ffn_gate_up_exps` 400.00 MiB (was 528.00), `ffn_down_exps` 264.00 MiB
  (was 328.00 on the 32 `IQ2_S` layers and 440.00 on the four `IQ3_S` layers: L4, L28,
  L32, L35 — those four now match the other 32, so the *uniform* cut is also uniform in
  the engine's eyes).

PROJECTION (linear in bytes, aubier's own derivative; **not measured on the engine**):

- misses/token: 306 MiB → **234.0 MiB** (−72.0 MiB), at the same expert count/token.
- residency: 1728 slot-layers → **≈ 2260** in the same VRAM (1728 / 0.764585).
- at aubier's 17–34 MiB per +1 tok/s, that is ≈ **+2.1 … +4.2 tok/s** on B1.
- on aubier's own basis (6.6875 MiB/expert, 30.1 GiB): **5.113 MiB/expert, 23.01 GiB**.
  The two bases differ only in whether the four `IQ3_S` down layers are counted; the
  measured numbers above are the shipped-file-truthful ones (arm A's expert set =
  30.5312 GiB).

## 7. Provenance (stamped, then read back with `--show`)

`tools/stamp_gguf.py`, in place and atomic; `--show` on the final file prints (it elides
the long `quantize_flags` value on the display line — the full value is quoted here and
was read back from the KV block directly):

```
taillis.format_version: UINT32 0
taillis.parent_sha256: STRING '5a11cfc5a5c49759f48e5842c549ef2f1bddfcb15ba83577d5f909a514653d2c'
taillis.provenance.input_sha256: STRING '83b58fbcc27456bb5cb334ade30ad92aa6a01cd922f00bcc471dc60501a9af3b'
taillis.provenance.quantize_tool: STRING 'llama-quantize@50e3e3e480c80b9ee0ea067e475602d540dc62c8'
taillis.provenance.quantize_flags: STRING '--imatrix ~/futaie/taillis/imatrix-q8-250c.gguf --tensor-type-file /tmp/t4a/t4a-types.txt IQ2_XXS 32'
taillis.provenance.imatrix_sha256: STRING 'dd02a7e6533fbe17938367401e18aad8f5d986f3c607b1e9e50472de8fbe1d59'
taillis.provenance.type_file_sha256: STRING '1ed8310287222d7cfa06d58e78ae9398ec73f6ad20e51ca41afdad38ccad9961'
```

- `parent_sha256` names **ARM A** (`t14-armA-imatrix.gguf`,
  `5a11cfc5…653d2c`), **not** the F16. That is §9.3.1's declared-comparator rule: for a
  region-quantised artifact the comparator is the flat quant from the same ≥16-bit
  source, same tool and flags, at the uniform type — a sibling of the immediate input.
  The F16 (`83b58fbc…af3b`) is `provenance.input_sha256`.
- **KV inheritance checked, not assumed:** the F16 carries no `taillis.*` key, and the
  quantize output before stamping carried **0** `taillis.*` keys (55 KV) — it inherits
  `quantize.imatrix.*` from the tool itself and nothing stale. After stamping, 62 KV,
  7 `taillis.*`, and the payload gates were re-run (§5).
- **The `format_version` correction is a second atomic rewrite**, and `--show` after it
  reports 62 KV / 7 `taillis.*` with **one** value changed (`1` → `0`); the file size is
  unchanged (27 206 669 120 B, the u32 is the same width) while the sha256 moved, which
  is why the isolation proof and the stock-load smoke were both re-made on the final
  bytes rather than inherited from before it.
- `quantize_tool` is the commit the **binary self-reports** (`version: 0.6.0-dev
  (build 11534, commit 50e3e3e48)`, first line of its own output — MEASURED).
- Nothing was requantised from a quantised file, so no `--allow-requantize` appears in
  the flags and no `+dirty` marker is claimed.

## 8. Caveats — what the evidence does and does not support

1. **`format_version = 0`, the FLAT case.** The first stamp of this artifact wrote `1`
   and it was wrong: `layout.md` §8.1 defines `1` as a claim about a grouping *derived
   from member tensor names*, and this file has no members — it is simply FLAT. §8.1's
   amended v0 branch names exactly this case ("neither the array nor any member-shaped
   name is present, and the file is simply FLAT"), so the artifact was **re-stamped to
   `0`** before anyone scored it and every gate was re-run on the rewritten bytes (§5).
   It was inert either way — the reader derives no members and reads the file flat
   (`engine/gguf_model.h read_expert_groups()`: `if (members.empty()) continue;`) — but a
   provenance record claiming a derived grouping over a file that has none is a lie, and
   that is the thing this namespace exists to prevent. This file and arm A (flat, key
   absent ⇒ 0) now agree on the semantics as well as on the tensors.
2. **The tool's source tree is not fully verifiable.** The binary's build directory
   (`/var/data/workspace/mistral4-rx9070/src/llama.cpp`) no longer exists on this host
   (MEASURED: absent), so the tree that produced it cannot be re-inspected. aubier's
   local tree at commit `50e3e3e4` has `tools/quantize/quantize.cpp` **unmodified**, but
   carries unrelated `ggml` modifications (`ggml.c`, `ggml-cpu.c`, `ggml.h`,
   `ggml-rpc.h`, CUDA) that the provenance string does not name. The string identifies
   the upstream commit the tool declares, not a verifiable tree.
3. **The type file lived in `/tmp`** (a 27 GiB GGUF output does not fit the repo's write
   scope, and the type file is a build input, not an artifact). Its sha256 is stamped and
   its full contents plus generator are reproduced in §§2–3, so it is reconstructible;
   `quantize_flags` therefore quotes a `/tmp` path.
4. **`-23.54 %`, not `-25.0 %`.** Do not round it downstream.
5. **Log-parse trap, recorded so nobody re-derives a false alarm:** for
   `output.weight` and `token_embd.weight`, `llama-quantize`'s own log line is split by
   the imatrix-load messages, so a line-based parse of the build log appears to show
   them left at `f16`. The built file's read-back is authoritative and both are `Q4_K`
   (288.00 MiB each) — confirmed by gates (a) and (c), and by the exact byte match.
6. **No quality number exists.** This unit builds the control; perplexity/eval on it is
   a separate unit, and no quality claim is made or implied by the coherent four-token
   generation in gate (d).
7. **Inputs untouched:** the F16 (2026-10-12), arm A (2026-10-13) and the shipped file
   (2026-10-09) keep their mtimes and sizes; nothing under `~/futaie/aubier` was written
   (read-only), and no GPU was taken.

## 9. Reproduce checklist

1. Generate the type file (§2) from the F16 and the shipped file; `sha256sum` must equal
   `1ed8310…9961`.
2. `--dry-run` (§3); diff its per-tensor plan against the shipped types with the two
   overrides applied — **0 rows** — and check its total is 25 938.75 MiB.
3. Real run (§3) as a tracked background job; exit 0; `quant size = 25938.75 MiB`.
4. Gates (a)–(c) on the built file; stamp (§7) with `format_version = 0` (§8.1's flat
   case — **not** `1`); re-run (a)–(c) and (d) on the rewritten bytes. `--show` must
   list exactly the seven keys in §7, and `sha256sum` must equal `db4c8207…f783`.
5. Logs: `logs/t4a-dryrun.log`, `logs/t4a-build.log`, `logs/t4a-gates-abc.log`,
   `logs/t4a-stamp.log`, `logs/t4a-gates-abc-post-stamp.log`,
   `logs/t4a-restamp-v0.log`, `logs/t4a-gates-abc-post-v0.log`,
   `logs/t4a-stock-load.log`, `logs/t4a-stock-load-post-v0.log` (gitignored).
