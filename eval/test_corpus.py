#!/usr/bin/env python3
"""Self-test for tools/build_corpus.py — a test that CAN fail.

Builds a tiny synthetic transcript fixture in a temp dir (no private data is
read), runs the builder over it, and asserts:

  1. turn extraction yields the expected user-turn count
  2. the render contains the system preamble and each turn's text
  3. marker count == conversation count
  4. a PLANTED fake credential is redacted   (fails with --no-redact)
  5. the manifest's byte count matches the corpus file on disk
  6. no secret token FORMAT survives unredacted in the render
  7. the TEMPLATE arm is capped at CONVERSATION_CAP on message boundaries:
     the truncated render fits the cap, is a prefix of the uncapped render of
     the same conversation, and dropped whole trailing messages with no partial
     block at the end. This is the part that would have caught the shipped
     defect (render_template had no cap at all while render_plain did).

Part 7 needs jinja2 and runs in-process when jinja2 is importable; the system
python3 does not have it, so otherwise the part is delegated to
venvs/corpus/bin/python3 and this process gates on the child's exit code
(hence --template-cap-only, the child mode that runs only part 7).

Gate: this process's exit code. 0 = pass, nonzero = fail.

The failing-before / passing-after asymmetry:

  redaction check:
    python3 eval/test_corpus.py --no-redact   # must FAIL (exit != 0)
    python3 eval/test_corpus.py               # must PASS (exit 0)

  template-cap check (--builder loads an alternate build_corpus.py; HEAD's copy
  is the pre-fix renderer, which has no cap):
    git show HEAD:tools/build_corpus.py > /tmp/prefix_build_corpus.py
    venvs/corpus/bin/python3 eval/test_corpus.py --template-cap-only \\
        --builder /tmp/prefix_build_corpus.py   # must FAIL (exit != 0)
    venvs/corpus/bin/python3 eval/test_corpus.py --template-cap-only  # PASS (0)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "models" / "Mistral-Small-4-119B-2603" / "chat_template.jinja"
VENV_PYTHON = HERE.parent / "venvs" / "corpus" / "bin" / "python3"


def _builder_path() -> Path:
    argv = sys.argv[1:]
    if "--builder" in argv:
        return Path(argv[argv.index("--builder") + 1]).resolve()
    return HERE.parent / "tools" / "build_corpus.py"


BUILDER = _builder_path()

spec = importlib.util.spec_from_file_location("build_corpus", BUILDER)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

if "--no-redact" in sys.argv[1:]:
    bc.REDACTORS = []  # the failing-before arm

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


def write_fixture(root: Path) -> Path:
    """A synthetic session: 2 user turns, assistant, reasoning, tool pair,
    and two planted fake credentials (obviously fake, still credential-shaped)."""
    sess = root / "proj-session.abc123"
    sess.mkdir()
    records = [
        {"type": "session", "hax_version": "v0.5.0", "id": "test", "cwd": "/tmp/proj", "model": "test-model"},
        {"kind": "user", "text": "First user turn: fix the flaky test in parser.py"},
        {"kind": "reasoning", "reasoning_text": "The flaky test waits on a race in the queue."},
        {"kind": "tool_call", "call_id": "c1", "tool_name": "bash", "arguments": "{\"command\": \"python3 -m pytest parser_test.py\"}"},
        {"kind": "tool_result", "call_id": "c1", "output": "1 passed in 0.02s"},
        {"kind": "assistant", "text": "Fixed: the queue now drains before teardown."},
        {"kind": "turn_boundary"},
        {"kind": "user", "text": "Second turn: commit it"},
        {"kind": "assistant", "text": "Done. Note the deploy key is api_key = FAKEKEY-notarealsecret99 and the db is postgres://ci_user:FAKEPASS-notreal@db.internal/proj"},
        {"kind": "turn_usage", "provider": "t", "model": "t", "usage": {}},
    ]
    f = sess / "2026-01-01T00-00-00Z_test.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return f


def write_long_fixture(root: Path) -> Path:
    """A synthetic conversation far longer than CONVERSATION_CAP, with
    consecutive same-role records (tool_call and tool_result both render as
    role "user") so a cut that is not on a role boundary is caught by the
    prefix assertion. Every record carries a unique marker so the test can see
    exactly which whole messages survived."""
    sess = root / "long-session.def456"
    sess.mkdir()
    records = [{"type": "session", "hax_version": "v0.5.0", "id": "long",
                "cwd": "/tmp/proj", "model": "test-model"}]
    for i in range(12):
        records.append({"kind": "user", "text": f"MK-{i:02d}-USER " + "u" * 3000})
        records.append({"kind": "tool_call", "call_id": f"c{i}", "tool_name": "bash",
                        "arguments": '{"command": "echo MK-%02d-CALL"}' % i})
        records.append({"kind": "tool_result", "call_id": f"c{i}",
                        "output": f"MK-{i:02d}-RESULT " + "r" * 3000})
        records.append({"kind": "assistant", "text": f"MK-{i:02d}-ASSIST " + "a" * 3000})
    f = sess / "2026-01-01T00-00-01Z_long.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return f


def run_template_cap_part() -> None:
    """Part 7: message-granular capping on the template arm.

    Runs against bc.render_template. On the pre-fix module (no cap at all) the
    length assertion fails; on a byte-slicing implementation the prefix
    assertion fails.
    """
    if not TEMPLATE.exists():
        check("template-arm cap: template present", False, f"missing {TEMPLATE}")
        return

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture = write_long_fixture(root)
        meta, events, _ = bc.extract_conversation(fixture)

        capped = bc.render_template(meta, events, str(TEMPLATE), {})

        # The uncapped render of the SAME conversation: render_template reads the
        # module-level cap at call time, so raising it (on either revision of the
        # module) yields the untruncated render.
        saved = bc.CONVERSATION_CAP
        bc.CONVERSATION_CAP = 10**9
        try:
            uncapped = bc.render_template(meta, events, str(TEMPLATE), {})
        finally:
            bc.CONVERSATION_CAP = saved

        check("template-arm uncapped render exceeds the cap (fixture is long enough)",
              len(uncapped) > bc.CONVERSATION_CAP,
              f"uncapped={len(uncapped)} cap={bc.CONVERSATION_CAP}")
        check("template-arm capped render fits CONVERSATION_CAP",
              len(capped) <= bc.CONVERSATION_CAP,
              f"capped={len(capped)} cap={bc.CONVERSATION_CAP}")

        # The render carries a constant newline run the template emits after its
        # loop, so the prefix relation is asserted on the renders with that
        # constant tail stripped (see render_template's docstring).
        body = capped.rstrip("\n")
        check("template-arm capped render is a prefix of the uncapped render",
              uncapped.startswith(body) and len(body) > 0,
              f"capped_body={len(body)} uncapped={len(uncapped)}")
        check("template-arm capped render ends on a complete message block",
              body.endswith("[/INST]") or body.endswith("</s>"),
              f"tail={body[-8:]!r}")

        markers = [f"MK-{i:02d}-{part}" for i in range(12)
                   for part in ("USER", "CALL", "RESULT", "ASSIST")]
        kept = [m for m in markers if m in capped]
        check("template-arm dropped whole trailing messages (contiguous prefix, no partial)",
              0 < len(kept) < len(markers) and kept == markers[:len(kept)],
              f"kept={len(kept)}/{len(markers)}")

        # Independent form of the same property: one event renders to exactly one
        # message, so the surviving markers name the message prefix the body must
        # BE -- not merely start like. This is what rules out a byte slice of the
        # rendered string, which passes every prefix assertion by construction.
        j = len(kept)
        saved = bc.CONVERSATION_CAP
        bc.CONVERSATION_CAP = 10**9
        try:
            prefix = bc.render_template(meta, events[:j], str(TEMPLATE), {})
        finally:
            bc.CONVERSATION_CAP = saved
        check("template-arm capped render equals the render of the surviving message prefix",
              body == prefix.rstrip("\n"),
              f"capped_body={len(body)} prefix_render={len(prefix.rstrip(chr(10)))}")

        check("template-arm capped render is deterministic",
              capped == bc.render_template(meta, events, str(TEMPLATE), {}))
        check("template-arm inserted no [TRUNCATED] marker",
              "[TRUNCATED]" not in capped)


def run_all() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture = write_fixture(root)
        out_txt = root / "corpus.txt"
        out_man = root / "manifest.json"

        counts = {}
        meta, events, n_turns = bc.extract_conversation(fixture)
        check("turns == 2", n_turns == 2, f"got {n_turns}")
        check("events == 7", len(events) == 7, f"got {len(events)}")

        body = bc.render_plain(meta, events, counts)
        check("render has system preamble", "hax agent session" in body and "/tmp/proj" in body)
        check("render has turn 1 text", "fix the flaky test" in body)
        check("render has turn 2 text", "Second turn: commit it" in body)
        check("render has assistant text", "drains before teardown" in body)
        check("render has tool result", "1 passed in 0.02s" in body)

        manifest = bc.build(root, out_txt, out_man, template_path=None, budget_bytes=1_000_000)
        text = out_txt.read_text(encoding="utf-8")

        check("marker count == conversation count", text.count(bc.MARKER) == manifest["conversation_count"] == 1,
              f"markers={text.count(bc.MARKER)} conv={manifest['conversation_count']}")
        check("manifest bytes match disk", manifest["utf8_bytes"] == out_txt.stat().st_size,
              f"manifest={manifest['utf8_bytes']} disk={out_txt.stat().st_size}")
        check("manifest sha256 matches disk", manifest["sha256"] == __import__("hashlib").sha256(out_txt.read_bytes()).hexdigest())

        # The redaction assertions — the point of the test.
        check("api_key assignment redacted",
              "FAKEKEY-notarealsecret99" not in body and "api_key" in body and bc.REDACTED in body)
        check("URL credential redacted",
              "FAKEPASS-notreal" not in body and "postgres://" in body and bc.REDACTED in body)
        check("redaction counted", counts.get("redactions", 0) >= 2, f"counts={dict(counts)}")

        # A bare token format must not survive even without a key name.
        bare = bc.redact("the key is sk-FAKE0123456789abcdefghijklmnop and done", {})
        check("bare token format redacted", "sk-FAKE0123456789abcdefghijklmnop" not in bare)
        clean = bc.redact("plain prose mentions the word token and password policy documents, none of it a secret", {})
        check("prose not mangled", "plain prose mentions the word token" in clean, f"got {clean!r}")

        # Part 7: the template arm's message-granular cap. jinja2 is not in the
        # system python3 (MEASURED 2026-10-11), so where it is missing the part
        # runs under the repo venv and this process gates on the child's exit.
        if _has_jinja2():
            run_template_cap_part()
        elif VENV_PYTHON.exists():
            cp = subprocess.run(
                [str(VENV_PYTHON), str(Path(__file__).resolve()),
                 "--template-cap-only", "--builder", str(BUILDER)],
                capture_output=True, text=True,
            )
            sys.stdout.write(cp.stdout)
            if cp.stderr:
                sys.stderr.write(cp.stderr)
            check("template-arm cap (delegated to venvs/corpus)", cp.returncode == 0,
                  f"child exit {cp.returncode}")
        else:
            check("template-arm cap", False,
                  f"jinja2 absent and {VENV_PYTHON} not found")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("\nALL PASS")
    return 0


def _has_jinja2() -> bool:
    try:
        import jinja2  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> int:
    if "--template-cap-only" in sys.argv[1:]:  # child mode: part 7 only
        run_template_cap_part()
        if FAILURES:
            print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
            return 1
        print("\nALL PASS")
        return 0
    return run_all()


if __name__ == "__main__":
    sys.exit(main())
