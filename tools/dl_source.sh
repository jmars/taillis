#!/usr/bin/env bash
# Fetch the BF16 source weights (the entry cost of every recipe here: requantizing an existing
# 2-bit GGUF is explicitly lossy -- llama-quantize's own --allow-requantize warning).
# Resumable: huggingface_hub skips completed files and resumes partial ones.
# Gate = exit code of the python fetch; then tools/verify_source.sh checks byte-exact sizes.
set -u -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="mistralai/Mistral-Small-4-119B-2603"
DEST="$HERE/models/Mistral-Small-4-119B-2603"
mkdir -p "$DEST" "$HERE/logs"

exec "$HERE/venvs/hf/bin/python" - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"])
print("downloaded to", path)
PY
