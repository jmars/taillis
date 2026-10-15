#!/usr/bin/env bash
# T1.2: FP8 (E4M3) Mistral Small 4 checkpoint -> F16 GGUF, reproducible wrapper.
#
#   tools/convert_f16.sh [--gate-only]
#
# Records, and re-runs, exactly what produced
#   models/Mistral-Small-4-119B-2603-f16.gguf
# See results/t1-f16.md for the verification of that artifact.
#
# Pinned dependency (aubier tree is READ-ONLY to us: this script reads it, never patches it):
#   converter : ~/futaie/aubier/src/llama.cpp/convert_hf_to_gguf.py
#               llama.cpp commit 50e3e3e480c80b9ee0ea067e475602d540dc62c8
#               ("CUDA: Remove redundant CUDA copies after SSM_SCAN (#29807)", 2026-10-09)
#   venv      : venvs/convert  (see logs/convert-venv-freeze.txt)
#
#   MEASURED environment deviation from llama.cpp's own pin (llama.cpp pins
#   transformers==4.57.6): this checkpoint's tokenizer_config.json declares
#   tokenizer_class "TokenizersBackend", which only exists in transformers 5.x, so
#   BOTH vocab paths (ModelBase.get_vocab_base and DeepseekV2Model.set_vocab) fail
#   under 4.57.6 with "Tokenizer class TokenizersBackend does not exist". Installed
#   transformers==5.3.0 (the checkpoint's own declared transformers_version is
#   5.3.0.dev0). The weight path does not use transformers at all (it reads the
#   safetensors shards directly via gguf.utility.SafetensorsLocal), so this affects
#   vocab/chat-template handling only.
set -euo pipefail

TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TREE"

LLAMACPP="$HOME/futaie/aubier/src/llama.cpp"
CONVERTER="$LLAMACPP/convert_hf_to_gguf.py"
PINNED_COMMIT=50e3e3e480c80b9ee0ea067e475602d540dc62c8
SRC=models/Mistral-Small-4-119B-2603
OUT=models/Mistral-Small-4-119B-2603-f16.gguf
PY=venvs/convert/bin/python3
# host has an unrelated engine unit live: bound CPU and priority, never take the GPU
LAUNCH=(nice -n 10 taskset -c 0-7 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8)

die() { echo "FATAL: $*" >&2; exit 1; }

[ -x "$PY" ] || die "$PY missing; see results/t1-f16.md for the venv recipe"
[ -f "$CONVERTER" ] || die "$CONVERTER missing"
have=$(git -C "$LLAMACPP" rev-parse HEAD)
[ "$have" = "$PINNED_COMMIT" ] || die "converter tree is at $have, not the pinned $PINNED_COMMIT"

echo "== converter : $CONVERTER"
echo "== commit    : $have"
echo "== venv      :"
"$PY" -m pip freeze | sed 's/^/     /'

# The converter picks its model class from the top-level 'architectures' entry
# (Mistral3ForConditionalGeneration -> Mistral3Model), whose text_config model_type
# 'mistral4' makes it delegate to Mistral4Model -> gguf arch 'mistral4'.
# NEVER pass --mistral-format: that path selects MistralMoeModel, arch deepseek2.
CMD=("$PY" "$CONVERTER" "$SRC" --outtype f16 --outfile "$OUT")

echo "== STEP 1: arch gate (--dry-run; reads all shard headers, writes nothing)"
# nanosecond mtime + size: %Y alone is 1-second resolution, so a same-second
# rewrite of a pre-existing artifact could pass the unchanged-check
OUT_BEFORE=$(stat -Lc '%.9Y %s' "$OUT" 2>/dev/null || true)
"${LAUNCH[@]}" "${CMD[@]}" --dry-run 2>&1 | tee logs/t12-dryrun.log | tail -5
grep -q 'n_tensors = 579' logs/t12-dryrun.log || die "unexpected dry-run plan"
# the gate is: the dry-run WROTE NOTHING.  A pre-existing artifact from an
# earlier build must not trip it — absence alone was the wrong predicate
# (it made every --gate-only run after T1.2 die here).
if [ -n "$OUT_BEFORE" ]; then
  [ "$(stat -Lc '%.9Y %s' "$OUT")" = "$OUT_BEFORE" ] || die "--dry-run modified $OUT"
else
  [ -e "$OUT" ] && die "--dry-run created $OUT"
fi
echo "   gate: 579 tensors planned, no file written."

# STEP 3 (R1, layout.md §7): stamp the provenance keys onto the fresh artifact.
# FUTURE builds only — the on-disk F16 is never rewritten to add keys (layout.md
# §7.3: the existing artifact is documented, not modified).  taillis.parent_sha256
# is NOT stamped here: a flat converted file has no grouped parent; group_experts.py
# stamps it when the file is later regrouped.  dl_source.sh snapshot_download()ed
# the repo without a revision pin, so the revision recorded is the branch.
SOURCE_CHECKPOINT="mistralai/Mistral-Small-4-119B-2603@main"
# format_version = 0, NOT 1: under layout.md §8.1 the value 1 means "the grouping
# is derived from the member tensor names" — a flat converted file is v0 (absent
# would mean the same; 0 says so explicitly).  group_experts.py stamps the 1 when
# the file is later regrouped with --format v1.
STAMP_CMD=(venvs/convert/bin/python3 tools/stamp_gguf.py --file "$OUT"
           --set-u32 taillis.format_version=0
           --set-str "taillis.provenance.source_checkpoint=$SOURCE_CHECKPOINT"
           --set-str "taillis.provenance.converter=convert_hf_to_gguf.py@$PINNED_COMMIT")

if [ "${1:-}" = "--gate-only" ]; then
  echo "== --gate-only: stopping before the write; the STEP 3 stamp would run:"
  echo "   CMD: ${STAMP_CMD[*]}"
  exit 0
fi

echo "== STEP 2: conversion (~221.6 GiB; tens of minutes to a few hours)"
printf 'CMD: %s\nSTART: %s\n' "${CMD[*]}" "$(date -Is)" > logs/t12-convert.log
"${LAUNCH[@]}" "${CMD[@]}" >> logs/t12-convert.log 2>&1
echo "END: $(date -Is)" >> logs/t12-convert.log
echo "== done: $(ls -l "$OUT")"

echo "== STEP 3: stamp R1 provenance (layout.md §7) onto the fresh artifact"
"${LAUNCH[@]}" "${STAMP_CMD[@]}"
