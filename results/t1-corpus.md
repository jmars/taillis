# results/t1-corpus.md — T1 corpus + scaffolding unit report

Date: 2026-10-11. Unit: TAILLIS T1-CORPUS (host-only; no model load, no GPU,
no gpu.lock taken — the aubier profiling unit owns the GPU today).

## What was built

| file | what |
|---|---|
| `tools/build_corpus.py` | builds the calibration corpus from this machine's hax agent transcripts |
| `eval/test_corpus.py` | self-test that CAN fail (fixture-based; gate = exit code) |
| `recipes/example-t1.txt` | `--tensor-type-file` format doc + first T1 recipe, PROPOSAL — NOT RUN |
| `corpus/agent_sessions.txt` | the corpus (PRIVATE — gitignored, verified) |
| `corpus/manifest.json` | stats + sha256 (gitignored, verified) |
| `venvs/corpus/` | venv with jinja2 3.1.6 (gitignored) |

Nothing outside the write scope was edited. `git status` additionally shows a
pre-existing modification to `tools/dl_source.sh` that this unit did NOT make
(it was already in the tree when this unit started).

## Transcript schema — two corrections to the brief (MEASURED, all 1288 files)

1. The `session` record keys on **`type`**, not `kind`; every other record
   keys on `kind`. The brief said "a `kind` field (`session`, ...)".
2. Reasoning text lives in **`reasoning_text`**, not `text` (the brief listed
   reasoning under `text`). `tool_call` carries `arguments` (a JSON string),
   `tool_result` carries `output`.

Kinds and counts across the tree: session 1288, user 5188, assistant 37666,
reasoning 73844, tool_call/tool_result 147130 each, turn_boundary 134097,
turn_usage 133992; `hax_version` v0.5.0 everywhere; 0 parse failures over
1288 files. Consequence: a hax "turn" is one model call, not one user
message; the corpus defines a turn as one user message (2,418 of them) —
stated here so the number is not mis-read.

## Marker convention (MEASURED from ds4's source, not from memory)

`antirez/ds4` `gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py` writes
`\n\n===== DS4_IMATRIX_PROMPT {rid} {category} {mode} {source} =====\n` before
each rendered prompt. This corpus uses the same shape with `TAILLIS_IMATRIX_PROMPT`.
One subtlety measured on the built corpus: a transcript that quotes this unit's
own brief contains the bare string `TAILLIS_IMATRIX_PROMPT` inside a
conversation body (line 11239), so a splitter must anchor on the full
**line-start** form `^===== TAILLIS_IMATRIX_PROMPT ` — that matches exactly
397 = the conversation count.

## Chat template status

`models/Mistral-Small-4-119B-2603/` does not exist yet (the 225.3 GiB download
is in flight), so **the corpus was produced by the plain-text fallback
renderer** (`render_mode: plain-text-fallback` in the manifest). The builder
auto-detects `chat_template.jinja` at the model path (or `--template PATH`)
and will render through it with jinja2 on the next build; the corpus should be
REBUILT with the template before the imatrix run if it lands in time — a
prompt-shape change is exactly what the routed experts are sensitive to
(ds4's own README says the same).

## Corpus stats (MEASURED, manifest sha256 e20a2b7c1c341fd844e30eb80cb40ba1e7c0934ff06b788c88ebbfdad9a8e20e)

- conversations: 397 · turns (user messages): 2,418
- bytes: 8,379,810 · coarse tokens (bytes/4): 2,094,952 — inside the 1–3M target
- sources: 49 session dirs, round-robin so no project exceeds 6.7% of bytes
  (first build with sorted order put 74% of bytes into one project and never
  reached the largest source tree; fixed before shipping)
- redactions applied: 356
- build command: `python3 tools/build_corpus.py` (defaults; ~50 s wall)
- budget: 8 MiB default, `budget_exhausted: true` (deterministic stop point)

## Privacy verification (MEASURED)

- `git check-ignore -v corpus/agent_sessions.txt corpus/manifest.json` → both
  ignored (`.gitignore:13 corpus/**/*.txt`, `.gitignore:11 corpus/*.json`).
  The write scope already covers what the unit writes; `.gitignore` was NOT edited.
- Residual secret-shape scan over the built corpus, all zero matches:
  `sk-…{20,}`, `ghp_…`, `AKIA…`, `hf_…`, `gsk_…`, `xox[bpars]-`, JWT-ish
  `ey….`…`, URL-embedded credentials `://user:pass@`, PEM `BEGIN … PRIVATE KEY`.
- The builder never prints corpus text (counts and sizes only).
- The redaction asymmetry proof:
  - `python3 eval/test_corpus.py --no-redact` → **exit 1** (4 checks FAIL,
    including the planted fake credential surviving in the render)
  - `python3 eval/test_corpus.py` → **exit 0** (15 checks, ALL PASS)

## Test gate (MEASURED)

`python3 eval/test_corpus.py` → exit 0. Checks: turn count, event count,
system preamble + turn/assistant/tool text present in the render, marker
count == conversation count, manifest bytes AND sha256 match the file on
disk, planted `api_key = …` redacted, planted URL credential redacted,
redaction counted, bare token format redacted, prose not over-redacted.
Two real bugs were caught and fixed by this test before the real build:
(1) `\g<1>` backreferences in replacement strings were emitted literally
(the `sub` callback path does not expand them — fixed with `m.expand`);
(2) the redaction counter KeyError'd on a plain dict.

## Recipe format (MEASURED from the pristine baseline)

`--tensor-type-file` entries are `<regex>=<type>`: the name part is compiled
with `std::regex` (`src/llama.cpp/src/llama-quant.cpp`), entries are
whitespace-split, so a regex must not contain spaces; entries are lowercased
first (`quantize.cpp:335`). See `recipes/example-t1.txt` (PROPOSAL — NOT RUN,
with its one known simplification stated in the file).

## Could NOT verify (honest list)

1. **The rendered prompts against Mistral Small 4's real chat template** —
   the template file does not exist yet; the fallback renderer's shape is a
   guess about what the engine will actually feed the model. REBUILD when the
   template lands.
2. **First-match-wins semantics of `--tensor-type-file`** — INTERPRETATION
   from reading `parse_tensor_type_file`; verify with `--dry-run` on the first
   real quantize.
3. **That the corpus actually helps the imatrix** — no imatrix has been run;
   that is the next unit and needs the GPU lock.
4. **Token estimate accuracy** — bytes/4 is the coarse convention; no real
   tokenizer pass was made.
5. **Full-coverage secret scan** — the residual scan covers known secret
   SHAPES; a secret that matches none of them (e.g. a custom token format
   without a credential-shaped key name) could survive. The self-test's
   redaction check is exercised, but absence-of-evidence on the real corpus
   is bounded by the pattern list.

## T1 next step

DISPATCH-READY: the imatrix run itself (`llama-imatrix -f
corpus/agent_sessions.txt -m <bf16-src-gguf> -o imatrix.dat`) needs (a) the
BF16 source GGUF from the in-flight download (taillis/models/, gitignored,
currently absent), and (b) the GPU lock. If the chat template lands first,
rebuild the corpus with it (same command; sha256 will change).

---

## Rebuild through the chat template — unit T1.1b (2026-10-11T16:19–16:37Z)

The chat template has landed (`models/Mistral-Small-4-119B-2603/
chat_template.jinja`, 15168 bytes; the builder's `DEFAULT_TEMPLATE` resolves
to it), so the corpus was rebuilt through it. **`render_mode` is now
`chat-template`** with a non-null `template_path` in `corpus/manifest.json`.

### Pre-state backup (asymmetry record, gitignored — verified with `git check-ignore`)

`corpus/agent_sessions-plaintext.txt` + `corpus/manifest-plaintext.json` hold
the fallback build byte-for-byte (8,379,810 bytes, sha256 `e20a2b7c…e20e`).

### Before / after (MEASURED)

| | fallback (pre-state) | `--no-template` control | **template rebuild** |
|---|---|---|---|
| render_mode | plain-text-fallback | plain-text-fallback | **chat-template** |
| conversations | 397 | 395 | **12** |
| turns (user msgs) | 2,418 | 2,440 | **162** |
| bytes | 8,379,810 | 8,384,790 | **5,130,457** |
| coarse tokens (bytes/4) | 2,094,952 | 2,096,197 | **1,282,614** |
| redactions (counted) | 356 | 358 | 237 |
| parse_errors | 0 | 0 | 0 |
| sha256 | `e20a2b7c…e20e` | `95cce474…5469` | **`4ee0aa5b…b04c`** |
| budget_exhausted | false | false | false |

Control vs pre-state differs slightly (−2 conv, +22 turns, +4,980 bytes): the
source tree keeps growing (this session included), so the round-robin order
shifted. That drift is ~0.06% of bytes — it cannot explain the template arm.

### Asymmetry proof — the template engaged (MEASURED)

Two builds of the same source set, different renderers, different outputs:

- template build → `corpus/agent_sessions.txt`, sha256 `4ee0aa5b…b04c`,
  5,130,457 bytes, 12 markers (= conversation count, line-start anchored)
- `--no-template` build → `corpus/selftest-fallback.txt/.json`, sha256
  `95cce474…5469`, 8,384,790 bytes

**Different sha256 AND different byte counts** → the template changed the
bytes; a byte-identical pair would have meant the template did not engage.
The manifest arms agree: `chat-template` + non-null path vs
`plain-text-fallback` + null.

### Why the token estimate fell 39% (2,094,952 → 1,282,614)

NOT the template markup: rendered through the template, one conversation's
output is the sum of its capped event texts **plus 2–4%** of template syntax
(measured on two conversations: capped-event sums 104,168 / 2,477,898 chars →
renders 106,911 / 2,527,929 bytes).

The cause is a **missing truncation**: `render_template`
(`tools/build_corpus.py:205-228`) never applies `render_plain`'s
`CONVERSATION_CAP` of 24,000 chars (`tools/build_corpus.py:200-202`), so each
conversation keeps its full body. Chunk sizes: template median 107,076 bytes,
p90 1,579,869, max 2,528,091 — versus fallback median 24,348, max 26,521.
The 8 MiB budget therefore trips after **12 conversations** (13 processed, the
13th rejected: it renders to 22,136,483 bytes — 2.6× the entire budget), and
total bytes drop 39%. This contradicts the builder's own stated design
("budget spent on breadth over depth"). **The fix is a one-line cap in
`render_template` — outside this unit's write scope (reported, not applied);
rebuild after it lands.**

Budget note: `budget_exhausted` is `false` in all three manifests (the flag is
`total_bytes >= budget_bytes`, and the loop stops *before* crossing), but the
budget does cap both arms — the fallback arm stops 3,818 bytes short, the
template arm at 61% of budget. **Correction:** the earlier claim in this file
(`budget_exhausted: true`) is contradicted by the pre-state manifest, which
records `false`.

### Redaction counts are substitutions applied, not surviving spans (MEASURED)

- template arm: 237 counted = 23 in written text + **214 from the rejected
  13th conversation** (rendered, redacted, then dropped by the budget break)
- fallback arm: 358 counted vs 11 surviving markers — `render_plain` redacts
  all events, then cuts the joined body at 24,000 chars, so markers past the
  cut never reach the file
- per-conversation check: counted == surviving (15 == 15) when nothing is cut

Pre-existing in both arms; the reported count is the builder's substitution
counter, not a corpus property. Residual secret-shape scan over the NEW
corpus, all zero: `sk-…{20,}`, `ghp_…`, `AKIA…`, `hf_…`, `gsk_…`,
`xox[bpars]-`, JWT-ish, `://user:pass@`, PEM blocks (23 `[REDACTED` markers
present).

### Gates (all exit 0, MEASURED)

- `python3 eval/test_corpus.py` → 0 (15 checks, ALL PASS)
- `sha256sum corpus/agent_sessions.txt` == manifest `sha256` →
  `4ee0aa5bca5ed3a7fea85b22b5ea0da0e60e440995dcbc1c1771b7f6ddbdb04c`
- manifest `utf8_bytes` == disk size → 5,130,457 == 5,130,457
- `git check-ignore` covers all six corpus outputs (build, backups, control)
- `git status --porcelain` → only the orchestrator's pre-existing
  `tools/build_corpus.py` edit; nothing committed
- rebuild command: `venvs/corpus/bin/python3 tools/build_corpus.py` (system
  python3 has no jinja2; the venv has 3.1.6), exit 0, 12.4 s; control arm
  `--no-template --out corpus/selftest-fallback.txt --manifest
  corpus/selftest-fallback.json`, exit 0, 48.4 s

### Could NOT verify (new, honest list)

1. **Whether the template render matches what the runtime feeds the model** —
   this run renders via jinja2 with explicit `bos_token`/`eos_token` strings
   and `add_generation_prompt=False`; no runtime chat-processing invocation
   was compared against it.
2. **Whether 12 conversations suffice for imatrix calibration** — the breadth
   collapse above is a regression against the design intent; the cap fix +
   rebuild is the orchestrator's call (out of scope here).
3. **Token estimate accuracy** — still bytes/4; no tokenizer pass.
4. **Full-coverage secret detection** — bounded by the pattern list, as before.

## t11c — the template cap landed, corpus rebuilt, breadth restored (2026-10-11)

Unit `taillis-t11c`. The cap the t11b section above asked for is now in:
`render_template` (`tools/build_corpus.py:205-274`) caps the rendered **message
list** (never the rendered string — a byte slice can cut a special token),
dropping whole trailing messages until the render fits `CONVERSATION_CAP`, and
the corpus has been rebuilt through it.

### The cap cuts on ROLE boundaries only, and that is measured, not stylistic

Rendering `messages[0:k]` is a prefix of rendering `messages[0:n]` **only when
the cut falls on a role boundary**. This template merges consecutive same-role
messages into one `[INST]...[/INST]` block (tool_call and tool_result both map
to role "user"), so a cut inside such a run renders the block CLOSED early: on
a synthetic fixture, `render(messages[:2])` ends `[/INST]` where the full render
continues with the next message's text (MEASURED, synthetic text only). The
implementation therefore considers only role-boundary prefixes and binary
searches the largest one that fits (O(log n) renders, cached, instead of a
re-render per dropped message).

Second measured template property, recorded in the docstring: every render ends
with a constant newline run emitted after the template's loop
(`keep_trailing_newline`), so "the truncated render is a prefix of the uncapped
render" holds once that constant tail is stripped from both. The test asserts
exactly that, and additionally that the body **equals** the render of the
message prefix its surviving messages name — which is what a byte slice cannot
satisfy.

### Before / after (MEASURED, `corpus/manifest*.json`)

| arm | conv | sources | turns | bytes | tok est | sha256 (12) |
|---|---|---|---|---|---|---|
| template, uncapped (t11b, 16:19) | 12 | 12 | 162 | 5,130,457 | 1,282,614 | `4ee0aa5b...` |
| template, capped (**this unit, rebuilt**) | **452** | **51** | **2,801** | **8,379,965** | **2,094,991** | `cb37d019...` |
| fallback (`--no-template`, same tree, this unit) | 393 | 51 | 2,231 | 8,384,506 | 2,096,126 | `b0e028b4...` |
| fallback (shipped pre-state, 16:19) | 397 | 47 | 2,418 | 8,379,810 | 2,094,952 | `e20a2b7c...` |

Breadth is back to the fallback's order of magnitude and the token estimate
lands at the fallback's (2.0950M vs 2.0961M) — the two expectations the t11c
brief set. The single-conversation dominance is gone: the largest chunk is
25,800 bytes with its marker (0.31% of the corpus) where t11b had one
conversation at 2,528,091 B (49%). Cap compliance: largest rendered body
**23,999 chars** vs the 24,000 cap, 0 bodies over it, 17 truncated at/near it.
Per-source top-5 by bytes (exact `per_source` keys):
`var-data-workspace-fixpoint-linux-fx-harness.7a169835ce5e9615` 26 conv /
584,792 B, `home-jaye-enlighten.ec95736e7e4c123a` 26 / 569,582,
`var-data-workspace-neoplatonic-library.80c2a87ea5116a6f` 25 / 567,773,
`var-data-workspace-mistral4-rx9070.91d007921538b1b0` 25 / 567,334,
`var-data-workspace-fixpoint-linux-datalog-dafsa.359c854b04b966fe` 26 /
558,089. Redactions 358,
parse_errors 0, `budget_exhausted` false, sha256 == disk and manifest bytes ==
disk (8,379,965). Rebuild: `venvs/corpus/bin/python3 tools/build_corpus.py`,
exit 0, 84.7 s (up from t11b's 12.4 s — the binary search renders over-cap
conversations; still ~1.5 min).

### Character cap, not a byte cap (stated, not hidden)

The cap matches `render_plain`'s convention: `len(body)` on a `str`, i.e.
**characters**. 32 of 452 bodies therefore exceed 24,000 **bytes** (max 25,638
B = 23,999 chars). The invariant "largest conversation <= the cap" holds in the
unit the code names; if it is wanted in bytes, that is a one-constant decision
for the orchestrator, and it would apply to `render_plain` too.

### The new test, and its asymmetry (MEASURED)

`eval/test_corpus.py` grew part 7 (8 checks) — the surface that had no test,
which is why the defect shipped. It builds a synthetic transcript longer than
the cap, renders through the real template, and asserts: uncapped render exceeds
the cap (else the part is vacuous), capped render fits it, capped render is a
prefix of the uncapped one, ends on a complete block, dropped whole trailing
messages contiguously, **equals** the render of the surviving message prefix,
is deterministic, and carries no injected `[TRUNCATED]` marker. `--builder PATH`
loads an alternate module; when jinja2 is absent (system python3) the part is
delegated to `venvs/corpus/bin/python3` (`--template-cap-only`) and the parent
gates on the child's exit code.

- POST-FIX `venvs/corpus/bin/python3 eval/test_corpus.py --template-cap-only` →
  **exit 0, 8/8 PASS**
- PRE-FIX (`git show HEAD:tools/build_corpus.py`) same command → **exit 1**,
  FAIL "capped render fits CONVERSATION_CAP" (97,410 vs 24,000) and FAIL
  "dropped whole trailing messages" (48/48 kept)
- byte-slice MUTANT (`full[:CAP]`, built in /tmp) → **exit 1**, FAIL "ends on a
  complete message block" and FAIL "equals the render of the surviving message
  prefix" (24,000 vs 24,472) — the mutant passes the prefix check by
  construction, which is why the equality check exists
- `python3 eval/test_corpus.py` → 0; `--builder <pre-fix>` → 1;
  `--no-redact` → 1 (redaction arm unchanged)

### The t11b control-arm tripwire could not be reproduced, and why (MEASURED)

The t11c brief required the `--no-template` arm to reproduce the t11b control
`95cce474...5469 / 8,384,790 B` exactly, and to treat a move as proof of shared
code being touched. **It does not reproduce: today's same-command run gives
393 conv / 2,231 turns / 8,384,506 B / `b0e028b4...`.** That is not the cap
fix. Two measurements settle it:

1. **Same input, both revisions, byte-identical.** A reflink snapshot of the
   session tree (`cp -a --reflink=auto`, /tmp, deleted after the run) run
   through HEAD's builder and through the fixed builder gives the SAME bytes:
   `30a2e5ec...` / 393 conv / 2,229 turns / 8,384,483 B, and the fixed module
   run twice gives the same sha again. `git diff` touches only
   `render_template` (+ the orchestrator's `DEFAULT_TEMPLATE` constant).
2. **The input moved, not the code.** 14 `*.jsonl` under the sessions tree have
   mtimes after 16:24, and the fallback arm's own numbers moved with them
   (turns 2,440 → 2,231/2,229, bytes 8,384,790 → 8,384,483/8,384,506). The
   fallback arm is exact only for a frozen tree; the t11b control value was a
   timestamp-dependent measurement, not a code fingerprint. **The tripwire
   should be re-based on a snapshot run (as above), not on a stored sha.**

### Could NOT verify (new, honest list)

1. **Reproducibility of the shipped corpus.** It is built from a live session
   tree that this session itself writes to; the numbers above are from the
   rebuild timestamp. Only the snapshot runs are reproducible.
2. **Whether the template render matches what the runtime feeds the model** —
   unchanged from t11b: jinja2 with explicit bos/eos and
   `add_generation_prompt=False`, no runtime comparison.
3. **Group-granular truncation slack.** The role-boundary rule under-fills: the
   median body is 21,532 chars against the 24,000 cap. How much that costs in
   imatrix quality is unmeasured.
4. **Token estimate accuracy** (still bytes/4) and **full-coverage secret
   detection** (still bounded by the pattern list) — residual scan over the new
   corpus, all zero: `sk-`, `ghp_`, `AKIA`, `hf_`, `gsk_`, `xox[bpars]-`,
   JWT-ish, `://user:pass@`, PEM, bare `bearer`.

## SUPERSEDED — appended 2026-10-11 (orchestrator), not a rewrite

Every corpus number above describes a corpus that no longer exists. This
document is the historical record of unit `taillis-t1c`; the values are kept as
measured then and are **not** the current corpus. The supersession is unit
`taillis-t13corpus` (report on `handoff-taillis-t13corpus-result`), which
re-set the builder's two limits — `CONVERSATION_CAP` 24,000 → 1,048,576 and
`DEFAULT_BUDGET_BYTES` 8,388,608 → 1,073,741,824 (now a runaway guard, not a
token target) — and rebuilt from the whole session tree. Unit `taillis-t13sample`
then made a second, narrower change: the within-source emission order was
chronological, so a `--chunks N` PREFIX of the corpus drew each source's OLDEST
session; it is now a sha256-of-relative-path order. That change is a **pure
permutation** — proven by building one frozen snapshot with both builders:
**identical byte count, different sha256**. The corpus as it now stands:

**1368 conversations / 52 sources / 5230 turns / 337,447,903 B (~322 MiB) /
~84M tokens / 0 parse errors**, sha256 `813dbc12…9cde`. Supersedes the
334,751,311 B / `1733902e…2511` (1360-conv) build, which in turn superseded
`cb37d019…c606` / 452 conversations / 8,379,965 B. First 500 chunks now cover 40
conversations from 40 sources at mean within-source rank 0.525 (was 36/36 at
rank 0.000).

The 24,000-char cap, not the byte budget, was the dominant limiter (it cut
373.4 MiB of available record text to 26.3 MiB). The `t1-f16`-era references
elsewhere to `cb37d019…c606` / 452 conversations / 2,094,991 tokens are the
superseded build. The per-record `CAPS` were deliberately **not** re-tuned, so
the composition argument in §"Per-record caps" above still describes the
current build's shaping; only the two limits and the order moved.
