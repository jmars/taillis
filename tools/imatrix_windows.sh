#!/usr/bin/env bash
# T1.3 — workload imatrix runner harness (host-side; the harness itself runs nothing).
#
#   tools/imatrix_windows.sh [options]
#
# WHY THIS EXISTS — MEASURED by the orchestrator 2026-10-11 unless marked PROJECTION:
#   * The F16 GGUF is 222 GiB and this host has 125 GB RAM, so a run is a page-fault
#     stream over the weights and never resident. The model was moved off the ZFS HDD
#     pool (measured 198 MB/s, O_DIRECT) onto the NVMe (measured 3.4 GB/s read) at
#     /var/lib/taillis/models/Mistral-Small-4-119B-2603-f16.gguf, with a symlink at
#     models/. This harness resolves the REAL path and prints it together with its
#     filesystem, so the operator can see which pool the run will fault from.
#   * `--chunks N` is a PREFIX of the corpus, not a window: tools/imatrix/imatrix.cpp:1466
#     computes n_chunk = min(n_chunks, n_chunk_max) and the loop at :1491 starts at i=0;
#     only `--chunk/--from-chunk` (:1447) moves the window. So a single bounded run
#     samples the corpus's beginning — which is why a real run is a set of WINDOWS over
#     the corpus merged with `--in-file` (tools/imatrix/README.md, "combine existing
#     imatrices"), and why the run is RESUMABLE across the engine track's GPU windows
#     instead of one long lock-hold.
#   * Throughput levers (tools/imatrix/README.md + the code): `--no-ppl` ("useful if you
#     want to speed up the processing and do not care about perplexity"); a batch larger
#     than n_ctx gives n_seq = n_batch/n_ctx chunks in flight (imatrix.cpp:1475, :1491
#     steps by n_seq), which pipelines the fault stream. Default geometry below is
#     `-c 4096 -b 8192 -t 32 --no-ppl`, each overridable.
#   * Cost model: MEASURED 2026-10-12 (partial run, below) — one 4096-token chunk routes
#     ~128 selections per expert (4096 * top-4 / 128), so one chunk touches essentially the
#     whole ~220 GB expert set. The earlier "30-65 s/chunk on NVMe" was a PROJECTION and it
#     was WRONG by ~20-40x: it used the SEQUENTIAL bandwidth (3.2 GB/s O_DIRECT), but an
#     mmap'd model faults in via RANDOM 4 KB reads, which measured ~100 MB/s. Observed:
#     343 GB read in 57 min (~100 MB/s) against a 220 GB/chunk working set => roughly
#     25-50 min/chunk, i.e. a bounded 750-chunk pass is WEEKS, not hours. `--smoke` exists
#     to give the real number, but on this host the answer is already decisive: the run
#     does not fit in RAM (222 GiB vs 125 GB) and no chunk budget makes it viable.
#     See docs/cloud-imatrix-runbook.md.
#   * Default budget `--chunks 750` (~3M tokens at n_ctx 4096) is likewise a PROJECTION
#     pending the smoke; it sits inside the plan's original 1-3M-token band, which was the
#     right size for a RUN even though it was the wrong size for a CORPUS.
#
# THE GPU LOCK: every run holds ~/futaie/aubier/scripts/gpu.lock through flock(1) for the
#   WHOLE script lifetime — one hold across all windows and the merge. The lock is not only
#   about VRAM: aubier's own record (todo.md §6) measured an unlocked peer CPU run
#   inflating another arm's timing by 26%. `--no-lock` is the escape hatch for a window the
#   engine track has explicitly granted, and it says so loudly.
#
# BINARY PROVENANCE: the plan requires T1 to work with UNMODIFIED llama.cpp, and some
#   builds/ trees are aubier's modified fork. The default is builds/hip (GGML_HIP=ON)
#   because its CMakeCache CMAKE_HOME_DIRECTORY is .../src/llama.cpp — the stock tree —
#   while every other GPU-capable build points at src/llama-fork, src/m3f, src/llama-route,
#   ... Provenance is re-checked and printed on every run, and a non-stock binary is
#   REFUSED unless `--allow-nonstock` is passed explicitly.
#
# FAILURE IS PROPAGATED: an imatrix invocation that exits nonzero aborts the run with THAT
#   status, and an invocation that exits 0 without writing its output file is a failure too.
#   Neither is ever reported as a success-shaped RESULT (`s/chunk 0.0`, `output ... (? B)`).
#   This is also why the argv is built in the CURRENT shell — see build_window_cmd.
#
# PRIVACY: counts, sizes and paths only — never corpus body text.
set -euo pipefail

# ---------------------------------------------------------------- defaults
AUBIER="${TAILLIS_AUBIER:-$HOME/futaie/aubier}"
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="$TREE/models/Mistral-Small-4-119B-2603-f16.gguf"
CORPUS="$TREE/corpus/agent_sessions.txt"
IMATRIX_BIN="$AUBIER/builds/hip/bin/llama-imatrix"
LOCK="$AUBIER/scripts/gpu.lock"

CTX=4096
BATCH=8192
THREADS=32
# The binary's own -ngl default is `auto`, which would silently offload and change the
# s/chunk this harness prints; pin the geometry (0 = CPU) and opt in with --ngl.
NGL=0
PPL=0                    # 1 -> --ppl, 0 -> --no-ppl (default --no-ppl, per the README's speed note)
CHUNKS=750               # PROJECTION pending the smoke; see the header.
CHUNK_OFF=0
SMOKE_CHUNKS=8
WINDOW_SIZE=0
WINDOWS=0
OUTDIR="$TREE"
OUTPUT="imatrix.gguf"
OUTPUT_FORMAT=""
SCRATCH="${TMPDIR:-/tmp}/taillis-imatrix-smoke.gguf"
FORCE=0
DRYRUN=0
SMOKE=0
NO_LOCK=0
LOCK_TIMEOUT=0           # 0 = block until the lock is free
ALLOW_NONSTOCK=0
EXTRA=()

usage() {
    cat <<'EOF'
tools/imatrix_windows.sh — T1.3 imatrix runner harness (bounded, windowed, resumable).

MODES
  --smoke                 one short window (default 8 chunks) at offset 0, --no-ppl, into
                          --scratch; prints wall time, chunks, tokens, MEASURED s/chunk and
                          tokens/s. This is the measurement that sets the real budget.
  --window-size S --windows K
                          K windows of S chunks each, window i at offset (i-1)*S, written to
                          <outdir>/imatrix-w<02i>.gguf, then merged with --in-file into
                          --output. A window whose file already exists is skipped unless
                          --force (resumable: an interrupted run never redoes a done window).
  (neither)               one single run: --chunk OFFSET --chunks N -> --output.
  --window-size S without --windows derives K = ceil(--chunks / S).

GEOMETRY / IO
  -m, --model PATH        default models/Mistral-Small-4-119B-2603-f16.gguf (real path printed)
  -f, --corpus PATH       default corpus/agent_sessions.txt
      --imatrix-bin PATH  default $AUBIER/builds/hip/bin/llama-imatrix (stock HIP; provenance
                          re-checked and printed on every run)
  -c, --ctx N             default 4096
  -b, --batch N           default 8192 (> n_ctx gives n_seq chunks in flight)
  -t, --threads N         default 32
      --ngl N             layers to offload, default 0 (CPU). The binary's own default is
                          `auto`, which would silently change the geometry printed here.
      --ppl               compute perplexity (default: --no-ppl, per the README's speed note)
      --chunk N           start chunk for the single-run mode (default 0)
      --chunks N          chunks for the single-run mode / the window-budget derivation
                          (default 750, a PROJECTION pending --smoke)
      --smoke-chunks N    chunks for --smoke (default 8)
      --extra ARG         repeatable; appended verbatim to every model invocation
  -o, --output NAME       merged/final output name, default imatrix.gguf
      --output-format F   gguf (default when unset) or dat -> passed through to the binary
      --outdir DIR        where window files and the output go, default the repo root
      --scratch PATH      --smoke output, default $TMPDIR/taillis-imatrix-smoke.gguf
      --force             redo windows/output that already exist
      --no-lock           do NOT hold gpu.lock: the operator asserts the engine track granted
                          this window. Prints a loud warning.
      --lock-timeout N    seconds to wait for gpu.lock, 0 = wait indefinitely (default 0)
      --allow-nonstock    permit a fork / unidentifiable binary (LOUD warning; the plan wants
                          unmodified llama.cpp, so the default refuses)
      --dry-run           print every command line and exit 0 without loading anything
  -h, --help              this text; exit 0

--opt=value is accepted as well as --opt value. The lock is
~/futaie/aubier/scripts/gpu.lock, held with flock(1) for the whole run.
EOF
}

say()  { printf '%s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- args
# Accept --opt=value as well as --opt value (the value may itself contain '=').
argv=()
for a in "$@"; do
    case "$a" in
        --*=*) argv+=("${a%%=*}" "${a#*=}") ;;
        *)     argv+=("$a") ;;
    esac
done
set -- ${argv[@]+"${argv[@]}"}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        -m|--model)       MODEL="${2:?--model needs a path}"; shift 2 ;;
        -f|--corpus)      CORPUS="${2:?--corpus needs a path}"; shift 2 ;;
        --imatrix-bin)    IMATRIX_BIN="${2:?--imatrix-bin needs a path}"; shift 2 ;;
        -c|--ctx)         CTX="${2:?--ctx needs a number}"; shift 2 ;;
        -b|--batch)       BATCH="${2:?--batch needs a number}"; shift 2 ;;
        -t|--threads)     THREADS="${2:?--threads needs a number}"; shift 2 ;;
        --ngl)            NGL="${2:?--ngl needs a number}"; shift 2 ;;
        --ppl)            PPL=1; shift ;;
        --chunk)          CHUNK_OFF="${2:?--chunk needs a number}"; shift 2 ;;
        --chunks)         CHUNKS="${2:?--chunks needs a number}"; shift 2 ;;
        --smoke-chunks)   SMOKE_CHUNKS="${2:?--smoke-chunks needs a number}"; shift 2 ;;
        --window-size)    WINDOW_SIZE="${2:?--window-size needs a number}"; shift 2 ;;
        --windows)        WINDOWS="${2:?--windows needs a number}"; shift 2 ;;
        -o|--output)      OUTPUT="${2:?--output needs a name}"; shift 2 ;;
        --output-format)  OUTPUT_FORMAT="${2:?--output-format needs a value}"; shift 2 ;;
        --outdir)         OUTDIR="${2:?--outdir needs a path}"; shift 2 ;;
        --scratch)        SCRATCH="${2:?--scratch needs a path}"; shift 2 ;;
        --force)          FORCE=1; shift ;;
        --no-lock)        NO_LOCK=1; shift ;;
        --lock-timeout)   LOCK_TIMEOUT="${2:?--lock-timeout needs a number}"; shift 2 ;;
        --allow-nonstock) ALLOW_NONSTOCK=1; shift ;;
        --dry-run)        DRYRUN=1; shift ;;
        --smoke)          SMOKE=1; shift ;;
        --extra)          EXTRA+=("${2:?--extra needs an argument}"); shift 2 ;;
        *)                die "unknown option: $1 (try --help)" ;;
    esac
done

for v in "ctx:$CTX" "batch:$BATCH" "threads:$THREADS" "chunk:$CHUNK_OFF" "chunks:$CHUNKS" \
         "smoke-chunks:$SMOKE_CHUNKS" "window-size:$WINDOW_SIZE" "windows:$WINDOWS" \
         "lock-timeout:$LOCK_TIMEOUT"; do
    k="${v%%:*}"; n="${v#*:}"
    [[ "$n" =~ ^[0-9]+$ ]] || die "--$k must be a non-negative integer, got '$n'"
done
case "$NGL" in all|auto) ;; *) [[ "$NGL" =~ ^[0-9]+$ ]] || die "--ngl must be a number, 'all' or 'auto'";; esac
[ "$CTX" -gt 0 ] || die "--ctx must be > 0"
[ "$BATCH" -gt 0 ] || die "--batch must be > 0"
[ "$THREADS" -gt 0 ] || die "--threads must be > 0"
case "$OUTPUT_FORMAT" in ""|gguf|dat) ;; *) die "--output-format must be gguf or dat" ;; esac
if [ "$SMOKE" = 1 ] && [ "$WINDOW_SIZE" != 0 ]; then
    die "--smoke and --window-size are mutually exclusive"
fi
if [ "$WINDOW_SIZE" != 0 ]; then
    [ "$WINDOW_SIZE" -gt 0 ] || die "--window-size must be > 0"
    if [ "$WINDOWS" = 0 ]; then
        WINDOWS=$(( (CHUNKS + WINDOW_SIZE - 1) / WINDOW_SIZE ))
        [ "$WINDOWS" -gt 0 ] || WINDOWS=1
    fi
elif [ "$WINDOWS" != 0 ]; then
    die "--windows needs --window-size"
fi

# ---------------------------------------------------------------- resolve inputs
[ -e "$MODEL" ]      || die "model not found: $MODEL"
[ -f "$CORPUS" ]     || die "corpus not found: $CORPUS"
[ -x "$IMATRIX_BIN" ] || die "imatrix binary not found or not executable: $IMATRIX_BIN"
MODEL_REAL="$(realpath -e -- "$MODEL")"
CORPUS_REAL="$(realpath -e -- "$CORPUS")"
BIN_REAL="$(realpath -e -- "$IMATRIX_BIN")"
if [ "$DRYRUN" = 1 ]; then
    # dry-run writes NOTHING, not even the output directory
    OUTDIR_REAL="$(realpath -m -- "$OUTDIR")"
else
    mkdir -p -- "$OUTDIR"
    OUTDIR_REAL="$(realpath -e -- "$OUTDIR")"
    [ -d "$OUTDIR_REAL" ] || die "--outdir is not a directory: $OUTDIR"
fi

# ---------------------------------------------------------------- binary provenance
# Stock trees live at <aubier>/src/llama.cpp; the build caches still name the dead
# /var/data/workspace/mistral4-rx9070 prefix (aubier todo.md §0), which maps to $AUBIER.
BIN_DIR="$(dirname "$BIN_REAL")"
CACHE="$BIN_DIR/../CMakeCache.txt"
DEAD_PREFIX="/var/data/workspace/mistral4-rx9070"
CACHE_HOME=""
HIP_FLAG=""
if [ -f "$CACHE" ]; then
    CACHE_HOME="$(grep -m1 '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$CACHE" | cut -d= -f2- || true)"
    HIP_FLAG="$(grep -m1 '^GGML_HIP:BOOL=' "$CACHE" | cut -d= -f2- || true)"
fi
SRC_TREE="${CACHE_HOME/#$DEAD_PREFIX/$AUBIER}"
STOCKNESS="UNKNOWN"
if [ -n "$CACHE_HOME" ]; then
    case "$(basename -- "$SRC_TREE")" in
        llama.cpp) STOCKNESS="STOCK" ;;
        *)         STOCKNESS="FORK" ;;
    esac
fi

# The binaries carry an absolute RUNPATH into the dead path, so they need an explicit
# LD_LIBRARY_PATH (aubier todo.md §0: "<rocm lib>:$BUILDS/<arm>/bin"). Derive both ends
# rather than hardcode: the rocm lib dir sits under the venv's site-packages.
ROCM_LIB="$(ls -d "$AUBIER"/venvs/rocm/lib/python3*/site-packages/_rocm_sdk_devel/lib 2>/dev/null | head -1 || true)"
LD_LIBRARY_PATH="${ROCM_LIB:+$ROCM_LIB:}$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

LDD_OUT="$(ldd "$BIN_REAL" 2>&1 || true)"
N_MISSING="$(printf '%s\n' "$LDD_OUT" | grep -c 'not found' || true)"
RUNPATH="$(readelf -d "$BIN_REAL" 2>/dev/null | grep -m1 -i 'runpath\|rpath' | sed 's/.*Library runpath: \[//; s/\].*//' || true)"

say "== binary    : $BIN_REAL"
if [ -n "$CACHE_HOME" ]; then
    say "== provenance: $STOCKNESS — CMakeCache CMAKE_HOME_DIRECTORY=$CACHE_HOME"
else
    say "== provenance: UNKNOWN — no CMakeCache.txt at $CACHE"
fi
say "             : resolved source tree = ${SRC_TREE:-unidentifiable}${HIP_FLAG:+ (GGML_HIP=$HIP_FLAG)}"
if [ -n "$SRC_TREE" ] && [ -d "$SRC_TREE/.git" ]; then
    SRC_HEAD="$(git -C "$SRC_TREE" rev-parse HEAD 2>/dev/null || printf '?')"
    SRC_DIRTY="$(git -C "$SRC_TREE" status --porcelain 2>/dev/null | wc -l || printf '?')"
    say "             : git HEAD=$SRC_HEAD uncommitted-path-count=$SRC_DIRTY"
fi
say "             : RUNPATH=${RUNPATH:-none} -> overridden via LD_LIBRARY_PATH=${BIN_DIR}${ROCM_LIB:+,$ROCM_LIB}"
say "             : ldd unresolved objects = $N_MISSING"
# T1 is pinned to unmodified llama.cpp, so an unidentifiable/fork binary is refused here.
if [ "$STOCKNESS" != "STOCK" ] && [ "$ALLOW_NONSTOCK" != 1 ]; then
    die "$BIN_REAL is $STOCKNESS (source tree ${SRC_TREE:-unidentifiable}), not a stock llama.cpp build.
       T1 requires UNMODIFIED llama.cpp. The GPU-capable stock candidates on this host are:
         $AUBIER/builds/hip/bin/llama-imatrix       (GGML_HIP=ON,    src/llama.cpp)
         $AUBIER/builds/vulkan/bin/llama-imatrix    (GGML_VULKAN=ON, src/llama.cpp)
       Pass --imatrix-bin <stock binary>, or --allow-nonstock to override deliberately."
fi
if [ "$STOCKNESS" != "STOCK" ]; then
    say "WARNING: $STOCKNESS binary in use under --allow-nonstock — the result is NOT attributable"
    say "         to unmodified llama.cpp."
fi
if [ "$N_MISSING" != 0 ]; then
    if [ "$DRYRUN" = 1 ]; then
        say "WARNING: $N_MISSING unresolved object(s) with LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    else
        die "$N_MISSING library/ies unresolved with LD_LIBRARY_PATH=$LD_LIBRARY_PATH (aubier todo.md §0)"
    fi
fi

# ---------------------------------------------------------------- resolved model / corpus
# Say WHICH copy of the weights the run will fault from: symlink vs regular file, and the
# filesystem behind it (the HDD pool and the NVMe are ~17x apart in measured read speed).
MODEL_SIZE="$(stat -c %s -- "$MODEL_REAL" 2>/dev/null || printf '?')"
if [ -L "$MODEL" ]; then
    say "== model     : $MODEL -> $MODEL_REAL (symlink resolved; $(( MODEL_SIZE / 1073741824 )) GiB)"
else
    say "== model     : $MODEL_REAL (regular file, no symlink; $(( MODEL_SIZE / 1073741824 )) GiB)"
fi
say "             : backing filesystem = $(df -h --output=source,fstype,avail -- "$MODEL_REAL" | tail -1 | tr -s ' ')"
CORPUS_SIZE="$(stat -c %s -- "$CORPUS_REAL")"
say "== corpus    : $CORPUS_REAL ($CORPUS_SIZE B; ~$(( CORPUS_SIZE / 4 )) tokens at 4 B/token — PROJECTION)"
say "             : ~$(( CORPUS_SIZE / 4 / CTX )) chunks available at n_ctx=$CTX (PROJECTION; counts only, never body text)"
say "== geometry  : n_ctx=$CTX n_batch=$BATCH n_seq=$(( BATCH / CTX > 0 ? BATCH / CTX : 1 )) threads=$THREADS ngl=$NGL ppl=$PPL"
say "== output    : outdir=$OUTDIR_REAL output=$OUTPUT format=${OUTPUT_FORMAT:-gguf} $( [ "$FORCE" = 1 ] && printf '(force)' || printf '(skip-if-exists)' )"

# ---------------------------------------------------------------- command construction
COMMON=("$BIN_REAL" -m "$MODEL_REAL" -f "$CORPUS_REAL" -c "$CTX" -b "$BATCH" -t "$THREADS" -ngl "$NGL")
if [ "$PPL" = 1 ]; then COMMON+=(--ppl); else COMMON+=(--no-ppl); fi
if [ -n "$OUTPUT_FORMAT" ]; then COMMON+=(--output-format "$OUTPUT_FORMAT"); fi
if [ "${#EXTRA[@]}" -gt 0 ]; then COMMON+=("${EXTRA[@]}"); fi

# The argv MUST be built in the CALLING shell. These builders used to also print their
# result, so callers wrote `note "CMD: $(window_cmd ...)"` — and a command substitution runs
# in a SUBSHELL, so the `WIN_ARGV=(...)` assignment was discarded. `"${WIN_ARGV[@]}"` then
# expanded to ZERO WORDS: bash ran an empty simple command, which SUCCEEDS, and the runner
# printed a full RESULT block (`wall time 0.0 s`, `s/chunk 0.0`, `output ... (? B)`) having
# executed nothing. Building and printing are now separate concerns: the builder sets the
# array plus CMD_STR, and the caller prints "$CMD_STR".
WIN_ARGV=()
CMD_STR=""
build_window_cmd() {
    WIN_ARGV=("${COMMON[@]}" --chunk "$1" --chunks "$2" -o "$3")
    CMD_STR="$(printf '%q ' "${WIN_ARGV[@]}")"
}
# build_merge_cmd OUTFILE FILE... -> as above.
MERGE_ARGV=()
build_merge_cmd() {
    MERGE_ARGV=("$BIN_REAL")
    local f
    for f in "${@:2}"; do MERGE_ARGV+=(--in-file "$f"); done
    MERGE_ARGV+=(-o "$1")
    if [ -n "$OUTPUT_FORMAT" ]; then MERGE_ARGV+=(--output-format "$OUTPUT_FORMAT"); fi
    CMD_STR="$(printf '%q ' "${MERGE_ARGV[@]}")"
}

# run_argv LABEL ARGV... -> run the command; a nonzero status aborts with THAT status, so a
# failing binary can never be followed by a success-shaped report.
run_argv() {
    local label="$1"; shift
    # An EMPTY argv is the defect's symptom: bash runs an empty simple command, which
    # succeeds. Refuse it explicitly rather than reporting it as a run.
    if [ $# -eq 0 ]; then
        say "FAILED: $label has an EMPTY argv — refusing (an empty command succeeds)"
        exit 1
    fi
    local st=0
    "$@" || st=$?
    if [ "$st" -ne 0 ]; then
        say "FAILED: $label exited $st — aborting; no RESULT is reported"
        exit "$st"
    fi
}
# A command that exits 0 but wrote no output file is a failure too, not `s/chunk 0.0`.
require_output() {
    if [ ! -f "$1" ]; then
        say "FAILED: $2 exited 0 but wrote no output file at $1 — aborting"
        exit 1
    fi
}

elapsed()  { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.1f", b-a}'; }
rate()     { awk -v s="$1" -v c="$2" 'BEGIN{ if (c+0>0) printf "%.1f", s/c; else printf "n/a" }'; }
tokrate()  { awk -v s="$1" -v t="$2" 'BEGIN{ if (s>0) printf "%.0f", t/s; else printf "n/a" }'; }

# Running measured rate. Scoped to THIS invocation: skipped windows are not counted, so the
# figure is never padded with windows whose duration we did not observe.
MEAS_ELAPSED=0
MEAS_CHUNKS=0
running_rate() { rate "$MEAS_ELAPSED" "$MEAS_CHUNKS"; }

# ---------------------------------------------------------------- dry run
if [ "$DRYRUN" = 1 ]; then
    say "== DRY RUN — nothing is loaded, no lock is taken, nothing is written"
    if [ "$SMOKE" = 1 ]; then
        say "== smoke window: offset 0, chunks $SMOKE_CHUNKS -> $SCRATCH"
        build_window_cmd 0 "$SMOKE_CHUNKS" "$SCRATCH"
        note "CMD: $CMD_STR"
    elif [ "$WINDOW_SIZE" != 0 ]; then
        wfiles=()
        for ((i = 1; i <= WINDOWS; i++)); do
            off=$(( (i - 1) * WINDOW_SIZE ))
            wf="$(printf '%s/imatrix-w%02d.gguf' "$OUTDIR_REAL" "$i")"
            wfiles+=("$wf")
            say "== window $i/$WINDOWS: offset $off chunks $WINDOW_SIZE -> $wf"
            if [ -f "$wf" ] && [ "$FORCE" != 1 ]; then
                note "SKIP: $wf exists (use --force to redo) — resumable"
            else
                build_window_cmd "$off" "$WINDOW_SIZE" "$wf"
                note "CMD: $CMD_STR"
            fi
        done
        say "== merge ${#wfiles[@]} window file(s) -> $OUTDIR_REAL/$OUTPUT"
        if [ -f "$OUTDIR_REAL/$OUTPUT" ] && [ "$FORCE" != 1 ]; then
            note "SKIP: $OUTDIR_REAL/$OUTPUT exists (use --force to redo)"
        else
            build_merge_cmd "$OUTDIR_REAL/$OUTPUT" "${wfiles[@]}"
            note "CMD: $CMD_STR"
        fi
    else
        out="$OUTDIR_REAL/$OUTPUT"
        say "== single run: offset $CHUNK_OFF chunks $CHUNKS -> $out"
        if [ -f "$out" ] && [ "$FORCE" != 1 ]; then
            note "SKIP: $out exists (use --force to redo)"
        else
            build_window_cmd "$CHUNK_OFF" "$CHUNKS" "$out"
            note "CMD: $CMD_STR"
        fi
    fi
    say "== dry run complete: exit 0"
    exit 0
fi

# ---------------------------------------------------------------- lock
if [ "$NO_LOCK" = 1 ]; then
    say "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    say "!! --no-lock: gpu.lock is NOT held. The operator asserts the engine track has"
    say "!! explicitly granted this window. An unlocked run contends with peer lanes and"
    say "!! has measured +26% inflation on a peer arm (aubier todo.md §6)."
    say "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
else
    [ -f "$LOCK" ] || die "GPU lock file missing: $LOCK (aubier scaffold); refusing to run unlocked"
    command -v flock >/dev/null 2>&1 || die "flock(1) not found; refusing to run unlocked"
    exec {LOCK_FD}>>"$LOCK"
    if [ "$LOCK_TIMEOUT" -gt 0 ]; then
        if ! flock -x -w "$LOCK_TIMEOUT" "$LOCK_FD"; then
            die "gpu.lock busy after ${LOCK_TIMEOUT}s; holder: $(fuser -v -- "$LOCK" 2>&1 | tail -2 | tr -s ' ')"
        fi
    else
        say "== lock      : waiting for $LOCK (flock -x, no timeout)"
        flock -x "$LOCK_FD"
    fi
    say "== lock      : acquired $LOCK (fd $LOCK_FD) — one hold for the whole run (windows + merge)"
fi

# ---------------------------------------------------------------- smoke
if [ "$SMOKE" = 1 ]; then
    say "== smoke     : offset 0, chunks $SMOKE_CHUNKS, n_ctx=$CTX, out $SCRATCH"
    build_window_cmd 0 "$SMOKE_CHUNKS" "$SCRATCH"
    note "CMD: $CMD_STR"
    rm -f -- "$SCRATCH"
    T0="$(date +%s.%N)"
    run_argv "smoke run" "${WIN_ARGV[@]}"
    T1="$(date +%s.%N)"
    EL="$(elapsed "$T0" "$T1")"
    TOK=$(( SMOKE_CHUNKS * CTX ))
    require_output "$SCRATCH" "smoke run"
    say "== smoke RESULT (MEASURED, this host, this invocation):"
    say "     wall time : ${EL} s"
    say "     chunks    : $SMOKE_CHUNKS"
    say "     tokens    : $TOK (= chunks * n_ctx)"
    say "     s/chunk   : $(rate "$EL" "$SMOKE_CHUNKS")"
    say "     tokens/s  : $(tokrate "$EL" "$TOK")"
    say "     output    : $SCRATCH ($(stat -c %s -- "$SCRATCH" 2>/dev/null || printf '?') B)"
    say "   Budget follows from s/chunk: --windows K --window-size S with K*S chunks inside the"
    say "   window the engine track grants. NOTE: an earlier 30-65 s/chunk figure in the header"
    say "   was a PROJECTION that used SEQUENTIAL bandwidth; mmap faults are RANDOM 4 KB reads"
    exit 0
fi

# ---------------------------------------------------------------- windows or single run
if [ "$WINDOW_SIZE" != 0 ]; then
    wfiles=()
    for ((i = 1; i <= WINDOWS; i++)); do
        off=$(( (i - 1) * WINDOW_SIZE ))
        wf="$(printf '%s/imatrix-w%02d.gguf' "$OUTDIR_REAL" "$i")"
        wfiles+=("$wf")
        say "== window $i/$WINDOWS: offset $off chunks $WINDOW_SIZE -> $wf"
        say "   model=$MODEL_REAL corpus=$CORPUS_REAL n_ctx=$CTX n_batch=$BATCH threads=$THREADS ngl=$NGL"
        say "   window offsets so far: 0..$off; measured s/chunk (this run) = $(running_rate) over $MEAS_CHUNKS chunk(s)"
        if [ -f "$wf" ] && [ "$FORCE" != 1 ]; then
            note "SKIP: exists (use --force to redo) — resumable"
            continue
        fi
        build_window_cmd "$off" "$WINDOW_SIZE" "$wf"
        note "CMD: $CMD_STR"
        T0="$(date +%s.%N)"
        run_argv "window $i" "${WIN_ARGV[@]}"
        T1="$(date +%s.%N)"
        EL="$(elapsed "$T0" "$T1")"
        require_output "$wf" "window $i"
        MEAS_ELAPSED="$(awk -v a="$MEAS_ELAPSED" -v b="$EL" 'BEGIN{printf "%.1f", a+b}')"
        MEAS_CHUNKS=$(( MEAS_CHUNKS + WINDOW_SIZE ))
        say "   window $i done in ${EL} s = $(rate "$EL" "$WINDOW_SIZE") s/chunk, $(tokrate "$EL" $(( WINDOW_SIZE * CTX ))) tokens/s"
        say "   running measured s/chunk (this run) = $(running_rate) over $MEAS_CHUNKS chunk(s)"
    done

    say "== merge     : ${#wfiles[@]} window file(s) -> $OUTDIR_REAL/$OUTPUT"
    if [ -f "$OUTDIR_REAL/$OUTPUT" ] && [ "$FORCE" != 1 ]; then
        note "SKIP: $OUTDIR_REAL/$OUTPUT exists (use --force to redo)"
    else
        build_merge_cmd "$OUTDIR_REAL/$OUTPUT" "${wfiles[@]}"
        note "CMD: $CMD_STR"
        T0="$(date +%s.%N)"
        run_argv "merge" "${MERGE_ARGV[@]}"
        T1="$(date +%s.%N)"
        require_output "$OUTDIR_REAL/$OUTPUT" "merge"
        say "   merge done in $(elapsed "$T0" "$T1") s"
    fi
    say "== done      : $OUTDIR_REAL/$OUTPUT ($(stat -c %s -- "$OUTDIR_REAL/$OUTPUT" 2>/dev/null || printf '?') B)"
    say "   measured this run: $(running_rate) s/chunk over $MEAS_CHUNKS chunk(s); skipped windows are not counted"
else
    out="$OUTDIR_REAL/$OUTPUT"
    say "== single run: offset $CHUNK_OFF chunks $CHUNKS -> $out"
    say "   model=$MODEL_REAL corpus=$CORPUS_REAL n_ctx=$CTX n_batch=$BATCH threads=$THREADS ngl=$NGL"
    if [ -f "$out" ] && [ "$FORCE" != 1 ]; then
        note "SKIP: exists (use --force to redo)"
        exit 0
    fi
    build_window_cmd "$CHUNK_OFF" "$CHUNKS" "$out"
    note "CMD: $CMD_STR"
    T0="$(date +%s.%N)"
    run_argv "run" "${WIN_ARGV[@]}"
    T1="$(date +%s.%N)"
    EL="$(elapsed "$T0" "$T1")"
    require_output "$out" "run"
    say "== done      : $out ($(stat -c %s -- "$out" 2>/dev/null || printf '?') B)"
    say "   wall time ${EL} s = $(rate "$EL" "$CHUNKS") s/chunk MEASURED, $(tokrate "$EL" $(( CHUNKS * CTX ))) tokens/s"
fi
