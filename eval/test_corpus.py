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

Gate: this process's exit code. 0 = pass, nonzero = fail.

The failing-before / passing-after asymmetry for the redaction check:
    python3 eval/test_corpus.py --no-redact   # must FAIL (exit != 0)
    python3 eval/test_corpus.py               # must PASS (exit 0)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parent / "tools" / "build_corpus.py"

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


def main() -> int:
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

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
