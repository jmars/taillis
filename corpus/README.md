# corpus/

Calibration text for the T1 workload imatrix.

**Built from private material — never commit the outputs.** The corpora are derived from this
machine's own agent session transcripts (`~/.local/state/hax/sessions/**/*.jsonl`), which are private.
`.gitignore` excludes `corpus/*.txt` and `corpus/*.json`; only the builder and this README are tracked.

Build with `tools/build_corpus.py`. The consumed artifact is a single text file with one rendered
conversation per `TAILLIS_IMATRIX_PROMPT` marker, which is what `llama-imatrix -f` reads.
