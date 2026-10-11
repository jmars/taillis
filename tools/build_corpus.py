#!/usr/bin/env python3
"""Build the taillis T1 calibration corpus for `llama-imatrix`.

Source: this machine's own hax agent session transcripts
(`~/.local/state/hax/sessions/**/*.jsonl`). One rendered conversation per
visible `TAILLIS_IMATRIX_PROMPT` marker, mimicking ds4's
`DS4_IMATRIX_PROMPT` convention (verified against
antirez/ds4 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py:
"\\n\\n===== DS4_IMATRIX_PROMPT {rid} {category} {mode} {source} =====\\n").

Transcript schema (MEASURED 2026-10-11, hax_version v0.5.0 across all 1288
files): the `session` record keys on `type` (not `kind`); every other record
keys on `kind` in {user, assistant, reasoning, tool_call, tool_result,
turn_boundary, turn_usage}. Text lives in `text` (user/assistant),
`reasoning_text` (reasoning), `arguments` (tool_call, a JSON string) and
`output` (tool_result).

PRIVACY: the transcripts are private and may contain credentials. Every
credential-shaped string is redacted before rendering (see REDACTORS); the
builder prints only counts and sizes, never corpus text.

If the Mistral Small 4 chat template (chat_template.jinja) is available, pass
--template and the conversations are rendered through it with jinja2.
Otherwise the plain-text fallback below is used (that is what produced the
corpus built on 2026-10-11; the template was still absent then).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_SESSIONS = "~/.local/state/hax/sessions"
DEFAULT_TEMPLATE = str(Path(__file__).resolve().parent.parent
                       / "models/Mistral-Small-4-119B-2603/chat_template.jinja")
MARKER = "TAILLIS_IMATRIX_PROMPT"
DEFAULT_BUDGET_BYTES = 8_388_608  # ~2M tokens at the coarse bytes/4 estimate

# Per-record character caps. Tool outputs dominate raw transcript bytes (the
# median tool_result is small but the tail is huge); the budget is spent on
# breadth (many conversations) over depth (long tool dumps).
CAPS = {
    "user": 6000,
    "assistant": 8000,
    "reasoning": 4000,
    "tool_call": 4000,
    "tool_result": 2000,
}
CONVERSATION_CAP = 24_000  # chars of rendered body per conversation

# ---------------------------------------------------------------------------
# Redaction. Ordered list of (compiled regex, replacement). Each match is
# counted; the secret portion is replaced, the surrounding shape is kept so
# the rendered text still looks like real agent traffic.
# ---------------------------------------------------------------------------

CRED_KEYS = (
    r"api[_-]?key|apikey|secret(?:[_-]?key|[_-]?token)?|access[_-]?token|"
    r"auth[_-]?token|refresh[_-]?token|password|passwd|pwd|passphrase|"
    r"client[_-]?secret|private[_-]?key|authorization|bearer|"
    r"aws[_-]?secret[_-]?access[_-]?key|aws[_-]?access[_-]?key[_-]?id|"
    r"hf[_-]?token|anthropic[_-]?api[_-]?key|openai[_-]?api[_-]?key|"
    r"deepseek[_-]?api[_-]?key|jina[_-]?api[_-]?key|gitlab[_-]?token"
)

REDACTED = "[REDACTED]"

# Known secret token formats (values only; independent of any key name).
_TOKEN_PREFIXES = (
    r"sk-ant-[A-Za-z0-9_\-]{16,}",
    r"sk-[A-Za-z0-9_\-]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"glpat-[A-Za-z0-9_\-]{15,}",
    r"xox[bparsa]-[A-Za-z0-9\-]{10,}",
    r"gsk_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"ASIA[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z_\-]{30,}",
    r"hf_[A-Za-z0-9]{30,}",
    r"ey[A-Za-z0-9_\-]{30,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT-ish
)

REDACTORS = [
    # URL-embedded credentials: scheme://user:pass@host
    (re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s/@:]+:[^\s/@]+@"), lambda m: m.group(0).split("//")[0] + "//" + REDACTED + "@"),
    # Authorization headers / bearer tokens
    (re.compile(r"(?i)\b(?:bearer|authorization(?:\s*:\s*bearer)?)\s*[:=]?\s*([A-Za-z0-9._\-]{12,})"), "BEARER " + REDACTED),
    # key: value / key = value / "key": "value" with a credential-shaped key
    (re.compile(r"(?i)(\"?(?:" + CRED_KEYS + r")\"?\s*(?:\"|')?\s*[:=]\s*(?:\"|')?)([^\s\"'}>,]{8,})"), r"\g<1>" + REDACTED),
    # shell/env style: export FOO_TOKEN=..., FOO_TOKEN=...
    (re.compile(r"(?i)\b((?:export\s+)?[A-Z0-9_]*(?:" + CRED_KEYS + r")[A-Z0-9_]*\s*=\s*)([^\s\"']{8,})"), r"\g<1>" + REDACTED),
    # bare well-known token formats
    (re.compile("(" + "|".join(_TOKEN_PREFIXES) + ")"), REDACTED),
    # PEM private-key blocks (whole block, DOTALL)
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED-PEM-KEY]"),
]

REDACT_COUNT_KEY = "__redactions__"


def redact(text: str, counts: Counter | None = None) -> str:
    """Redact credential-shaped strings. Returns the text; increments counts."""
    for rx, repl in REDACTORS:
        def _sub(m, repl=repl):
            if counts is not None:
                counts["redactions"] = counts.get("redactions", 0) + 1
            if callable(repl):
                return repl(m)
            return m.expand(repl)
        text = rx.sub(_sub, text)
    return text


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_jsonl(path: Path):
    """Yield (kind, record). The session record keys on `type`; the rest on `kind`."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                counts = globals().setdefault(REDACT_COUNT_KEY, Counter())
                counts["parse_errors"] += 1
                continue
            yield (r.get("kind") or r.get("type"), r)


def extract_conversation(path: Path):
    """Parse one transcript file into (session_meta, events, n_turns).

    A "turn" here is one user message (the loop step the agent works through);
    hax's own turn_boundary/turn_usage records mark model calls, which vastly
    outnumber user messages (134k vs 5.2k across this tree) and are not the
    unit the corpus budget is shaped around.
    """
    meta = {}
    events = []
    for kind, r in parse_jsonl(path):
        if kind == "session":
            meta = r
        elif kind in ("user", "assistant"):
            t = r.get("text") or ""
            if t.strip():
                events.append((kind, t))
        elif kind == "reasoning":
            t = r.get("reasoning_text") or ""
            if t.strip():
                events.append(("reasoning", t))
        elif kind == "tool_call":
            a = r.get("arguments") or ""
            if isinstance(a, (dict, list)):
                a = json.dumps(a, ensure_ascii=False)
            events.append(("tool_call", f"{r.get('tool_name')} {a}"))
        elif kind == "tool_result":
            o = r.get("output") or ""
            if isinstance(o, (dict, list)):
                o = json.dumps(o, ensure_ascii=False)
            events.append(("tool_result", o))
    n_turns = sum(1 for k, _ in events if k == "user")
    return meta, events, n_turns


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_plain(meta: dict, events, redact_counts: Counter) -> str:
    """Plain-text fallback renderer (used when no chat template is available)."""
    parts = []
    cwd = meta.get("cwd") or ""
    model = meta.get("model") or ""
    parts.append(f"SYSTEM: hax agent session, model {model}, working directory {cwd}.")
    for kind, text in events:
        text = text[: CAPS[kind]]
        text = redact(text, redact_counts)
        if kind == "user":
            parts.append("USER: " + text)
        elif kind == "assistant":
            parts.append("ASSISTANT: " + text)
        elif kind == "reasoning":
            parts.append("THINKING: " + text)
        elif kind == "tool_call":
            parts.append("TOOL_CALL: " + text)
        elif kind == "tool_result":
            parts.append("TOOL_RESULT: " + text)
    body = "\n".join(parts) + "\n"
    if len(body) > CONVERSATION_CAP:
        body = body[:CONVERSATION_CAP] + "\n[TRUNCATED]\n"
    return body


def render_template(meta: dict, events, template_path: str, redact_counts: Counter):
    """Render through a chat template jinja file (llama.cpp-style args).

    The rendered body is capped at CONVERSATION_CAP exactly as render_plain's
    is, but the cap is applied to the MESSAGE LIST, never to the rendered
    string: a byte slice can cut through a special token and produce a
    malformed sequence. Two properties are load-bearing:

      (1) the cap drops WHOLE trailing messages, and
      (2) the truncated body is a valid render of a message PREFIX, and a
          prefix of the uncapped render of the same conversation -- nothing
          inside a kept message is altered. One measured caveat belongs to the
          template, not to us: every render ends with a constant newline run
          emitted after its loop (keep_trailing_newline), so the prefix
          relation holds once that constant tail is stripped from both.

    The cut is restricted to ROLE boundaries. This template merges consecutive
    same-role messages into one [INST]...[/INST] block (tool_call and
    tool_result both map to role "user"), so a cut inside such a run renders
    the block CLOSED early and the result is no longer a prefix of the uncapped
    render (MEASURED 2026-10-11 on synthetic fixtures). Keeping only every
    role-boundary prefix keeps property (2) true; it costs some granularity,
    which is bounded by the run length (a run is a user turn or a run of
    assistant/reasoning messages, not the whole conversation).
    """
    import jinja2  # deferred: only needed on the template path

    tmpl_src = Path(template_path).read_text(encoding="utf-8")
    tmpl = jinja2.Environment(
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
        autoescape=False,
    ).from_string(tmpl_src)

    messages = [{"role": "system", "content": f"hax agent session, model {meta.get('model')}, working directory {meta.get('cwd')}."}]
    for kind, text in events:
        text = redact(text[: CAPS[kind]], redact_counts)
        if kind == "user":
            messages.append({"role": "user", "content": text})
        elif kind in ("assistant", "reasoning"):
            messages.append({"role": "assistant", "content": text})
        elif kind == "tool_call":
            messages.append({"role": "user", "content": "TOOL_CALL: " + text})
        elif kind == "tool_result":
            messages.append({"role": "user", "content": "TOOL_RESULT: " + text})

    def at(k: int) -> str:
        """Render the first k messages (cached; one render per distinct k)."""
        if k not in cache:
            cache[k] = tmpl.render(
                messages=messages[:k], add_generation_prompt=False,
                bos_token="<s>", eos_token="</s>",
            ) + "\n"
        return cache[k]

    cache: dict[int, str] = {}
    full = at(len(messages))
    if len(full) <= CONVERSATION_CAP:
        return full

    # Largest role-boundary prefix that fits the cap. Rendered length is
    # monotone in the number of messages kept (each kept message contributes a
    # whole block), so binary search finds it in O(log n) renders instead of
    # re-rendering once per dropped message.
    cuts = [k for k in range(1, len(messages))
            if messages[k]["role"] != messages[k - 1]["role"]]
    if not cuts:  # a lone message: keeping it or dropping it are the only options
        return full
    lo, hi = 0, len(cuts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(at(cuts[mid])) <= CONVERSATION_CAP:
            lo = mid + 1
        else:
            hi = mid - 1
    # hi < 0 means even the shortest prefix (the system preamble alone) is over
    # the cap; return that prefix anyway rather than slicing the render.
    return at(cuts[hi] if hi >= 0 else cuts[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(sessions_dir: Path, out_txt: Path, out_manifest: Path,
          template_path: str | None, budget_bytes: int) -> dict:
    redact_counts: Counter = Counter()
    files = sorted(sessions_dir.rglob("*.jsonl"))
    if not files:
        print(f"ERROR: no *.jsonl under {sessions_dir}", file=sys.stderr)
        sys.exit(2)

    # Round-robin across source directories (deterministic): plain sorted
    # order let one large project consume the whole budget before the others
    # were reached (measured: 74% of bytes from a single project). One
    # conversation per source per rotation keeps the mix balanced.
    by_source: dict[str, list[Path]] = {}
    for p in files:
        by_source.setdefault(str(p.parent.relative_to(sessions_dir)), []).append(p)
    for lst in by_source.values():
        lst.sort()
    queues = [by_source[k] for k in sorted(by_source)]
    order = []
    while queues:
        nxt = []
        for q in queues:
            if q:
                order.append(q.pop(0))
                if q:
                    nxt.append(q)
        queues = nxt

    chunks = []
    per_source = {}
    total_bytes = 0
    n_conv = 0
    for p in order:
        meta, events, n_turns = extract_conversation(p)
        if not events:
            continue
        rel = str(p.relative_to(sessions_dir))
        rid = hashlib.sha256(rel.encode()).hexdigest()[:12]
        body = render_plain(meta, events, redact_counts) if not template_path \
            else render_template(meta, events, template_path, redact_counts)
        marker = f"\n\n===== {MARKER} {rid} agent chat {rel} =====\n"
        chunk = marker + body
        chunk_bytes = len(chunk.encode("utf-8"))
        if total_bytes + chunk_bytes > budget_bytes and n_conv > 0:
            break
        chunks.append(chunk)
        total_bytes += chunk_bytes
        n_conv += 1
        src = rel.split("/")[0]
        s = per_source.setdefault(src, {"conversations": 0, "turns": 0, "bytes": 0})
        s["conversations"] += 1
        s["turns"] += n_turns
        s["bytes"] += chunk_bytes

    text = "".join(chunks)
    data = text.encode("utf-8")
    out_txt.write_bytes(data)

    manifest = {
        "version": 1,
        "purpose": "taillis T1 workload imatrix calibration corpus (Mistral Small 4)",
        "sessions_root": str(sessions_dir),
        "template_path": template_path or None,
        "render_mode": "chat-template" if template_path else "plain-text-fallback",
        "conversation_count": n_conv,
        "turn_count": sum(s["turns"] for s in per_source.values()),
        "utf8_bytes": len(data),
        "rough_token_estimate_bytes_div_4": len(data) // 4,
        "budget_bytes": budget_bytes,
        "budget_exhausted": total_bytes >= budget_bytes,
        "per_source": per_source,
        "redactions": redact_counts.get("redactions", 0),
        "parse_errors": redact_counts.get("parse_errors", 0),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", default=DEFAULT_SESSIONS)
    ap.add_argument("--out", default="corpus/agent_sessions.txt")
    ap.add_argument("--manifest", default="corpus/manifest.json")
    ap.add_argument("--template", default=None,
                    help=f"chat_template.jinja to render with (default: use it if present at {DEFAULT_TEMPLATE})")
    ap.add_argument("--no-template", action="store_true", help="force the plain-text fallback renderer")
    ap.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET_BYTES)
    ap.add_argument("--no-redact", action="store_true",
                    help="disable redaction (ONLY for the failing-before half of the self-test; never for a real corpus)")
    args = ap.parse_args()

    if args.no_redact:
        # Test-only escape hatch: neutralise every redactor.
        global REDACTORS
        REDACTORS = []

    sessions_dir = Path(args.sessions).expanduser().resolve()
    template_path = None
    if not args.no_template:
        cand = args.template or DEFAULT_TEMPLATE
        if cand and Path(cand).exists():
            template_path = cand

    manifest = build(sessions_dir, Path(args.out), Path(args.manifest), template_path, args.budget_bytes)

    # Counts and sizes only -- never corpus text.
    print(f"render_mode: {manifest['render_mode']}")
    print(f"conversations: {manifest['conversation_count']}")
    print(f"turns: {manifest['turn_count']}")
    print(f"bytes: {manifest['utf8_bytes']}")
    print(f"rough tokens (bytes/4): {manifest['rough_token_estimate_bytes_div_4']}")
    print(f"redactions applied: {manifest['redactions']}")
    print(f"parse_errors: {manifest['parse_errors']}")
    print(f"sha256: {manifest['sha256']}")


if __name__ == "__main__":
    main()
