#!/usr/bin/env bash
# Self-test for tools/imatrix_windows.sh — a test that CAN fail.
#
# It exercises the INTEGRATION, not the printer: a stub executable is put at
# --imatrix-bin and its argv is recorded by the stub itself. The stub sleeps a
# known interval and writes the -o output path, so the test can assert that the
# binary was really invoked, that the reported wall time covers it, and that the
# reported output size is the stub's real file size rather than `?`.
#
# Why this test exists: the shipped runner built its argv inside a command
# substitution (`note "CMD: $(window_cmd ...)"`), so the `WIN_ARGV=(...)`
# assignment landed in a subshell and was discarded; the execution site
# `"${WIN_ARGV[@]}"` then expanded to zero words — an empty command that
# SUCCEEDS. The runner printed a full RESULT block (`wall time 0.0 s`,
# `s/chunk 0.0`, `output ... (? B)`) having executed nothing and exited 0. The
# previous gate tested `--dry-run`, which exercises only the printing half, so it
# could not fail. This test asserts on the stub's marker file instead.
#
# No GPU work and no model load: the stub ignores the model and the corpus
# (placeholders in a temp dir), and every case runs with --no-lock so the test
# never contends for gpu.lock.
#
# Gate: this process's exit code. 0 = pass, nonzero = fail.
#
# The failing-before / passing-after asymmetry (MEASURED 2026-10-12 — take the
# pre-fix arm from a saved copy, NOT from git: tools/imatrix_windows.sh was
# untracked when the defect was fixed):
#   cp tools/imatrix_windows.sh /tmp/broken/tools/imatrix_windows.sh   # pre-fix copy
#   bash eval/test_imatrix_windows.sh --script /tmp/broken/tools/imatrix_windows.sh
#                                                    # must FAIL (exit != 0), failing
#                                                    # exactly the "invoked" assertions
#   bash eval/test_imatrix_windows.sh                # must PASS (exit 0)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${IMATRIX_SCRIPT:-$REPO/tools/imatrix_windows.sh}"

while [ $# -gt 0 ]; do
    case "$1" in
        --script) SCRIPT="${2:?--script needs a path}"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[ -f "$SCRIPT" ] || { printf 'FAIL: no such runner script: %s\n' "$SCRIPT" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/taillis-imatrix-test.XXXXXX")"
WORK="$(realpath -e -- "$WORK")"
trap 'rm -rf -- "$WORK"' EXIT

MODEL="$WORK/model.gguf"          # placeholder: the stub never reads it
CORPUS="$WORK/corpus.txt"         # placeholder: counts/paths only, no body text
: > "$MODEL"
printf 'placeholder\n' > "$CORPUS"

mkdir -p -- "$WORK/bin" "$WORK/smoke" "$WORK/wins"
STUB="$WORK/bin/llama-imatrix"
cat > "$STUB" <<'STUB'
#!/usr/bin/env bash
# Test stub for --imatrix-bin: record argv, sleep, create the -o output, exit.
set -u
printf '%s\n' "$0 $*" >> "$STUB_MARKER"   # $0: the script must exec the RESOLVED binary
sleep "${STUB_SLEEP:-1}"
rc="${STUB_EXIT:-0}"
if [ "$rc" = 0 ]; then
    out=""; prev=""
    for a in "$@"; do
        [ "$prev" = "-o" ] && out="$a"
        prev="$a"
    done
    [ -n "$out" ] || { printf 'stub: no -o argument\n' >&2; exit 99; }
    mkdir -p -- "$(dirname -- "$out")"
    head -c "${STUB_BYTES:-4096}" /dev/zero > "$out"
fi
exit "$rc"
STUB
chmod +x -- "$STUB"
STUB_REAL="$(realpath -e -- "$STUB")"
MODEL_REAL="$(realpath -e -- "$MODEL")"
CORPUS_REAL="$(realpath -e -- "$CORPUS")"

FAILS=0
ok()  { printf 'ok   %s\n' "$*"; }
bad() { printf 'FAIL %s\n' "$*"; FAILS=$(( FAILS + 1 )); }
check() { # check <description> <test-command...>
    local d="$1"; shift
    if "$@"; then ok "$d"; else bad "$d"; fi
}

run_script() { # run_script <logfile> [args...] -> sets RC and LOG
    LOG="$1"; shift
    bash "$SCRIPT" "$@" >"$LOG" 2>&1
    RC=$?
}

# 4096-byte output, 2 s sleep: the timer must cover the sleep, the size must be
# the stub's file, not `?`.
STUB_MARKER="$WORK/smoke.marker"
STUB_SLEEP=2
STUB_BYTES=4096
STUB_EXIT=0
export STUB_MARKER STUB_SLEEP STUB_BYTES STUB_EXIT

printf '== case 1: --smoke with a stub that records argv, sleeps 2 s, writes 4096 B\n'
SCRATCH="$WORK/smoke/out.gguf"
run_script "$WORK/case1.log" --smoke --no-lock --allow-nonstock \
    --imatrix-bin "$STUB" --model "$MODEL" --corpus "$CORPUS" \
    --outdir "$WORK/smoke" --scratch "$SCRATCH"
check "case 1 exits 0"                test "$RC" -eq 0
check "case 1 the stub was INVOKED"   test -s "$STUB_MARKER"

EXPECT1="$STUB_REAL -m $MODEL_REAL -f $CORPUS_REAL -c 4096 -b 8192 -t 32 -ngl 0 --no-ppl --chunk 0 --chunks 8 -o $SCRATCH"
GOT1="$(head -1 -- "$STUB_MARKER" 2>/dev/null || true)"
check "case 1 the stub was invoked ONCE" test "$(grep -c . -- "$STUB_MARKER" 2>/dev/null || printf 0)" = 1
if [ "$GOT1" = "$EXPECT1" ]; then ok "case 1 argv is exactly the T1.3 invocation"
else bad "case 1 argv mismatch
     expected: $EXPECT1
     actual  : ${GOT1:-<no marker file>}"; fi

WALL="$(grep -m1 'wall time' "$WORK/case1.log" | sed 's/.*: *//; s/ *s$//')"
SIZE="$(grep -m1 '^     output    :' "$WORK/case1.log" | sed 's/.*(\(.*\) B)$/\1/')"
check "case 1 wall time >= stub sleep (got ${WALL:-<absent>} s, sleep 2 s)" \
    awk -v w="${WALL:-0}" -v s="$STUB_SLEEP" 'BEGIN{exit !(w + 0 >= s + 0)}'
check "case 1 reported output size is the stub's real size (got ${SIZE:-<absent>} B)" \
    test "${SIZE:-x}" = "$STUB_BYTES"
check "case 1 the output file exists on disk" test -s "$SCRATCH"

printf '== case 2: a stub that exits 7 must abort the run and print no success RESULT\n'
STUB_MARKER="$WORK/fail.marker"
STUB_SLEEP=0.5
STUB_EXIT=7
export STUB_MARKER STUB_SLEEP STUB_EXIT
run_script "$WORK/case2.log" --smoke --no-lock --allow-nonstock \
    --imatrix-bin "$STUB" --model "$MODEL" --corpus "$CORPUS" \
    --outdir "$WORK/smoke" --scratch "$WORK/smoke/fail.gguf"
check "case 2 exits NONZERO (got $RC)"  test "$RC" -ne 0
check "case 2 propagates the stub's status 7 (got $RC)" test "$RC" -eq 7
check "case 2 the stub was invoked"     test -s "$STUB_MARKER"
check "case 2 prints no success RESULT block" \
    bash -c '! grep -q "^== smoke RESULT" "$1"' _ "$WORK/case2.log"
check "case 2 does not report an output size" \
    bash -c '! grep -q "^     output    :" "$1"' _ "$WORK/case2.log"

printf '== case 3: --windows 2 --window-size 4 must invoke the stub per window, then merge\n'
STUB_MARKER="$WORK/wins.marker"
STUB_SLEEP=1
STUB_BYTES=4096
STUB_EXIT=0
export STUB_MARKER STUB_SLEEP STUB_BYTES STUB_EXIT
WOUT="$WORK/wins"
run_script "$WORK/case3.log" --no-lock --allow-nonstock --window-size 4 --windows 2 \
    --imatrix-bin "$STUB" --model "$MODEL" --corpus "$CORPUS" --outdir "$WOUT"
check "case 3 exits 0" test "$RC" -eq 0
check "case 3 the stub was invoked exactly 3 times (2 windows + 1 merge)" \
    test "$(grep -c . -- "$STUB_MARKER" 2>/dev/null || printf 0)" = 3

EXPECT3A="$STUB_REAL -m $MODEL_REAL -f $CORPUS_REAL -c 4096 -b 8192 -t 32 -ngl 0 --no-ppl --chunk 0 --chunks 4 -o $WOUT/imatrix-w01.gguf"
EXPECT3B="$STUB_REAL -m $MODEL_REAL -f $CORPUS_REAL -c 4096 -b 8192 -t 32 -ngl 0 --no-ppl --chunk 4 --chunks 4 -o $WOUT/imatrix-w02.gguf"
EXPECT3C="$STUB_REAL --in-file $WOUT/imatrix-w01.gguf --in-file $WOUT/imatrix-w02.gguf -o $WOUT/imatrix.gguf"
GOT3A="$(sed -n 1p -- "$STUB_MARKER" 2>/dev/null || true)"
GOT3B="$(sed -n 2p -- "$STUB_MARKER" 2>/dev/null || true)"
GOT3C="$(sed -n 3p -- "$STUB_MARKER" 2>/dev/null || true)"
if [ "$GOT3A" = "$EXPECT3A" ]; then ok "case 3 window 1 invoked at --chunk 0"
else bad "case 3 window 1 argv mismatch
     expected: $EXPECT3A
     actual  : ${GOT3A:-<no marker file>}"; fi
if [ "$GOT3B" = "$EXPECT3B" ]; then ok "case 3 window 2 invoked at --chunk 4"
else bad "case 3 window 2 argv mismatch
     expected: $EXPECT3B
     actual  : ${GOT3B:-<no marker file>}"; fi
if [ "$GOT3C" = "$EXPECT3C" ]; then ok "case 3 merge invoked with both --in-file windows"
else bad "case 3 merge argv mismatch
     expected: $EXPECT3C
     actual  : ${GOT3C:-<no marker file>}"; fi
check "case 3 the merged output exists on disk" test -s "$WOUT/imatrix.gguf"

printf '== case 4: --dry-run must print the command and execute NOTHING\n'
STUB_MARKER="$WORK/dry.marker"
export STUB_MARKER
rm -f -- "$STUB_MARKER"
run_script "$WORK/case4.log" --dry-run --smoke --no-lock --allow-nonstock \
    --imatrix-bin "$STUB" --model "$MODEL" --corpus "$CORPUS" \
    --outdir "$WORK/smoke" --scratch "$WORK/smoke/dry.gguf"
check "case 4 exits 0"                test "$RC" -eq 0
check "case 4 prints the command"     bash -c 'grep -q "CMD: $2" "$1"' _ "$WORK/case4.log" "$STUB_REAL"
check "case 4 executed nothing"       test ! -e "$STUB_MARKER"

printf '\n'
if [ "$FAILS" = 0 ]; then
    printf 'ALL PASS (%s) — the runner really invokes the binary; no GPU, no model loaded\n' "$SCRIPT"
    exit 0
fi
printf '%d CHECK(S) FAILED (%s)\n' "$FAILS" "$SCRIPT"
exit 1
