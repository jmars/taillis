#!/usr/bin/env python3
"""Hand-built MINIMAL llama-arch GGUF: 1 layer, n_embd=64, vocab=32, F16.

This is the xcheck unit's instrument, RE-WRITTEN here because /tmp was wiped by a machine
reboot mid-unit (the original /tmp/xcheck/make_tiny_base.py no longer exists).  It is the
same shape and the same construction: written directly with gguf-py so llama-export-lora
gets a base it loads in milliseconds and the vendored converter (which cannot run) is
never involved.  Deterministic seed, so a rebuilt base is the same base.
"""
import sys

import gguf
import numpy as np

OUT = sys.argv[1]
N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_LAYER, N_VOCAB = 64, 2, 2, 128, 1, 32
rng = np.random.default_rng(1234)


def ne(*dims):
    return tuple(reversed(dims))


T = {}


def add(name, ne_dims, std):
    T[name] = (rng.standard_normal(ne(*ne_dims)) * std).astype(np.float16)


def add1(name, ne_dims):
    T[name] = (np.ones(ne(*ne_dims), dtype=np.float32) * 0.9).astype(np.float16)


add("token_embd.weight", (N_EMBD, N_VOCAB), 0.05)
add("output.weight", (N_EMBD, N_VOCAB), 0.05)
add1("output_norm.weight", (N_EMBD,))
add1("blk.0.attn_norm.weight", (N_EMBD,))
add1("blk.0.ffn_norm.weight", (N_EMBD,))
add("blk.0.attn_q.weight", (N_EMBD, N_EMBD), 0.02)
add("blk.0.attn_k.weight", (N_EMBD, N_EMBD), 0.02)
add("blk.0.attn_v.weight", (N_EMBD, N_EMBD), 0.02)
add("blk.0.attn_output.weight", (N_EMBD, N_EMBD), 0.02)
add("blk.0.ffn_gate.weight", (N_EMBD, N_FF), 0.02)
add("blk.0.ffn_up.weight", (N_EMBD, N_FF), 0.02)
add("blk.0.ffn_down.weight", (N_FF, N_EMBD), 0.02)

w = gguf.GGUFWriter(OUT, arch="llama", use_temp_file=False)
w.add_name("lora-gate-tiny-llama")
w.add_context_length(128)
w.add_embedding_length(N_EMBD)
w.add_block_count(N_LAYER)
w.add_feed_forward_length(N_FF)
w.add_head_count(N_HEAD)
w.add_head_count_kv(N_HEAD_KV)
w.add_rope_dimension_count(N_EMBD // N_HEAD)
w.add_layer_norm_rms_eps(1e-5)
w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(N_VOCAB)])
w.add_array("tokenizer.ggml.scores", [0.0] * N_VOCAB)
w.add_array("tokenizer.ggml.token_type", [1] * N_VOCAB)
w.add_string("tokenizer.ggml.model", "llama")
for name, arr in T.items():
    w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {OUT}: {len(T)} tensors")
