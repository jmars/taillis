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
