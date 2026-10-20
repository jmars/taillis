#!/usr/bin/env python3
"""Shared constants and helpers for the LORA gates (REBUILT after a machine reboot wiped
/tmp mid-unit; the work dirs live under /var so tmpfs pressure cannot fake a failure).

NOTHING here re-implements the merge: every number a gate asserts is computed from the
ADAPTER BYTES and the F16 BYTES, independently of tools/repack_overlay.py.  Gate (c) is
the only script that imports the tool, because its whole job is to compare OUR arithmetic
against llama.cpp's."""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

TREE = Path("/home/jaye/futaie/taillis")
PY = TREE / "venvs/convert/bin/python3"
TOOL = TREE / "tools/repack_overlay.py"
BASE = Path("/var/lib/taillis/models/t14-armA-imatrix.gguf")
F16 = TREE / "models/Mistral-Small-4-119B-2603-f16.gguf"
F16_SHA = "83b58fbcc27456bb5cb334ade30ad92aa6a01cd922f00bcc471dc60501a9af3b"
IMATRIX = TREE / "imatrix-q8-250c.gguf"
TYPES = TREE / "recipes/example-t1.entries.txt"
QUANTIZE = Path("/home/jaye/futaie/aubier/builds/cpu/bin/llama-quantize")
EXPORT_LORA = Path("/home/jaye/futaie/aubier/builds/cpu/bin/llama-export-lora")
LLAMA_ENV = {"LD_LIBRARY_PATH": "/home/jaye/futaie/aubier/builds/cpu/bin",
             "PATH": "/usr/bin:/bin:/usr/sbin"}
OUTDIR = Path("/var/lib/taillis/models/lora-gate")
GATEDIR = Path("/tmp/lora-gate")
ADAPTERS = GATEDIR / "adapters"
WORK = OUTDIR / "work"          # /var, NOT tmpfs


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 23)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_adapter(name: str, *args: str) -> Path:
    """Build (or reuse) a synthetic PEFT adapter; return its safetensors path.

    The cache is keyed on the ARGUMENTS as well as the name: two gates asking for
    different expert sets under one name must not silently share one file (that bug made
    gate (b)'s "uncovered pairs" check pass vacuously once, on a cached 0,1 adapter).
    """
    out = ADAPTERS / f"{name}.d"
    sw = out / "adapter_model.safetensors"
    stamp = out / "ARGS"
    want = " ".join(args)
    if sw.exists() and stamp.exists() and stamp.read_text() == want:
        return sw
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(PY), str(GATEDIR / "make_adapter.py"), "--out", str(out),
                        *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"make_adapter failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    stamp.write_text(want)
    return sw


def build(out: Path, *, adapter: Path | None = None, config: Path | None = None,
          overrides=("20:ffn_gate_up_exps:0,1", "20:ffn_down_exps:0,1"),
          workdir: Path | None = None, extra=(), quant_type="IQ2_XXS",
          requantise=True, imatrix=True, f16=True, types=True, quantize=True) -> tuple:
    """Run the tool.  Returns (CompletedProcess, out_path)."""
    if out.exists():
        out.unlink()
    cmd = [str(PY), str(TOOL), "--base", str(BASE), "--out", str(out)]
    if requantise:
        cmd += ["--requantise"]
    if f16:
        cmd += ["--f16", str(F16)]
    for o in overrides:
        cmd += ["--override", o]
    if imatrix:
        cmd += ["--imatrix", str(IMATRIX)]
    if types:
        cmd += ["--type-file", str(TYPES)]
    if quantize:
        cmd += ["--llama-quantize", str(QUANTIZE)]
    cmd += ["--quant-type", quant_type, "--input-sha256", F16_SHA]
    wd = workdir or (WORK / out.stem)
    cmd += ["--workdir", str(wd)]
    if adapter is not None:
        cmd += ["--lora", str(adapter)]
    if config is not None:
        cmd += ["--lora-config", str(config)]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True), out


def gguf_tensors(path) -> tuple:
    """name -> (numpy payload, ggml shape) for every tensor, READ FROM THE FILE."""
    import gguf
    r = gguf.GGUFReader(str(path), "r")
    return {t.name: (np.array(t.data, copy=True), tuple(int(x) for x in t.shape))
            for t in r.tensors}, r


def gguf_fields(path) -> dict:
    import gguf
    r = gguf.GGUFReader(str(path), "r")
    return {k: f.contents() for k, f in r.fields.items()}


def raw_tensor(path, name: str) -> np.ndarray:
    """A >=16-bit tensor's FULL data as numpy (ne reversed), float32."""
    import gguf
    r = gguf.GGUFReader(str(path), "r")
    for t in r.tensors:
        if t.name == name:
            return np.array(t.data, dtype=np.float32)
    raise SystemExit(f"{name} not in {path}")


class Slabs:
    """ONE expert slab of a >=16-bit GGUF, by RAW offset — never the whole tensor
    (blk.20.ffn_gate_up_exps is 4.3 GiB; one expert is 32 MiB)."""

    def __init__(self, path):
        import gguf
        self.path = path
        self.r = gguf.GGUFReader(str(path), "r")
        self.by_name = {t.name: t for t in self.r.tensors}

    def slab(self, name: str, expert: int) -> np.ndarray:
        t = self.by_name[name]
        ne = tuple(int(x) for x in t.shape)
        stride = t.n_bytes // ne[2]
        with open(self.path, "rb") as f:
            f.seek(t.data_offset + expert * stride)
            buf = f.read(stride)
        return np.frombuffer(buf, dtype=np.float16).reshape(ne[1], ne[0]).astype(np.float32)


def read_slice_f16(path) -> np.ndarray:
    """The tool's merged F16 slice readback: numpy (n_ov, ne1, ne0), float32."""
    import gguf
    r = gguf.GGUFReader(str(path), "r")
    assert len(r.tensors) == 1, (path, [t.name for t in r.tensors])
    return np.array(r.tensors[0].data, dtype=np.float32)


HF = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}   # tool name -> HF module


def adapter_pairs(sw: Path, layer: int, expert: int) -> dict:
    """module -> (A float32, B float32) read straight from the safetensors, prefix-blind."""
    from safetensors.numpy import load_file
    T = load_file(str(sw))
    out = {}
    for k, v in T.items():
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            continue
        if f"layers.{layer}." not in k or f".experts.{expert}." not in k:
            continue
        mod = k.split(".")[-3]
        slot = "A" if k.endswith(".lora_A.weight") else "B"
        out.setdefault(mod, {})[slot] = v.astype(np.float32)
    return {m: (d["A"], d["B"]) for m, d in out.items() if "A" in d and "B" in d}


def adapter_scale(sw: Path) -> tuple:
    """(alpha, rank, scale) from the config + the tensors: alpha/rank, or 1.0 if absent."""
    import json
    cfg = json.loads((Path(sw).parent / "adapter_config.json").read_text())
    from safetensors.numpy import load_file
    T = load_file(str(sw))
    rank = next((v.shape[1] if k.endswith(".lora_B.weight") else v.shape[0])
                for k, v in T.items() if k.endswith(".lora_B.weight"))
    alpha = cfg.get("lora_alpha")
    return alpha, rank, (alpha / rank if alpha is not None else 1.0)


def ulp_stats(a, b) -> tuple:
    """(count differing, max |f16 index distance|) between two F16 arrays."""
    ia = np.asarray(a, dtype=np.float16).view(np.int16).astype(np.int32)
    ib = np.asarray(b, dtype=np.float16).view(np.int16).astype(np.int32)
    d = np.abs(ia - ib)
    return int(np.count_nonzero(d)), int(d.max())


FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def finish(name: str) -> None:
    print()
    if FAILS:
        print(f"RESULT: {name}: {len(FAILS)} FAILED -> {FAILS}")
        sys.exit(1)
    print(f"RESULT: {name}: ALL OK")
    sys.exit(0)
