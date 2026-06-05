# GGUF Backend Implementation Design

This document details the design, functionality, and constraints of the GGUF backend (`LlamaCppBackend`) within the `gemma4-mlx-speculative` project. This backend provides the same core capabilities as the MLX backend for Prefix Acceleration, Exact Rollback, and Template Draft, while using GGUF-specific implementation mechanisms.

## 1. Rationale for Low-Level `eval` / `sample`
Instead of using `llama-cpp-python`'s high-level `create_completion` or `__call__` methods, the GGUF backend utilizes low-level `eval` and `sample` loops.
- **State Reset Avoidance:** `create_completion` implicitly forces a full state reset on branch transitions, which is destructive for recurrent/hybrid models.
- **Granular Control:** Token-exact prefix caching, precise rollback mechanisms, and speculative candidate verification inherently require cycle-accurate evaluation and sampling, bypassing high-level wrappers.

## 2. Prefix Acceleration Strategy
- **Session Caching:** During `create_session`, the entire prefix is tokenized and evaluated (chunked to avoid out-of-bounds `llama_decode` errors), and the initial internal `llama_context` memory maintains this prefix evaluation.
- **Reusability:** During `generate`, if a session state exists, the backend begins suffix evaluation seamlessly from the maintained state, bypassing prefix prefill entirely.
- **Conceptual Parity:** While MLX achieves this via Python-level immutable array snapshotting, `llama-cpp` relies on the underlying C++ context naturally retaining the computed state.

## 3. Template Draft & Rollback Mechanism
The GGUF backend avoids the massive overhead of complete state duplication for its Template Draft verification loop.
- **Rollback Tracking:** Before evaluating a speculative candidate block, the current sequence length (`self.llm.n_tokens`) is recorded.
- **Mismatch Resolution:** If a prediction mismatch occurs during candidate validation, the system rolls back `self.llm.n_tokens` to its pre-block length and invokes `self.llm._ctx.kv_cache_seq_rm(-1, rollback_n_tokens, -1)` to explicitly delete the discarded branch from the C++ KV Cache.
- **Performance Gain:** Using `save_state()` / `load_state()` per block resulted in heavy slowdowns (`decode_speedup <= 1.0x`) due to the continuous serialization and memory copying of the multi-gigabyte Llama context. Utilizing `kv_cache_seq_rm` achieves decode speedups of >2.0x.

## 4. Limitations
- **Greedy Only:** The current Template Draft loop and custom sampling logic strictly assume a greedy sampling strategy. Non-greedy sampling (`temperature`, `top-p`, etc.) is out of scope and explicitly not supported.
- **Rollback Dependency:** Rollback operations absolutely depend on `kv_cache_seq_rm` and `n_tokens` resetting.
- **API Choice:** `create_completion` is bypassed entirely in favor of the low-level `eval` and `sample` loop to ensure exact token-level rollback and caching guarantees.
- **Isolated Candidates:** GGUF/Qwen-specific drafting candidates must be strictly isolated to `experiments/template_candidates_gguf_qwen.json` to prevent corrupting the generalized MLX ruleset. Other GGUF models will require their own dedicated candidate JSON files and verification benchmarks.

## 5. Quick Checks

For short, localized verification of the GGUF backend, the following commands can be executed. These check prefix acceleration, exact rollback, and template drafting without running the full comprehensive test suite.

```bash
# 1. Basic Backend Benchmark
PYTHONPATH=. .venv/bin/python experiments/benchmark_llama_cpp_backend.py --json

# 2. State Restore Benchmark
PYTHONPATH=. .venv/bin/python experiments/benchmark_llama_cpp_state_restore.py --json

# 3. Template Draft Benchmark
PYTHONPATH=. .venv/bin/python experiments/benchmark_llama_cpp_template_draft.py \
  --model "$HOME/Documents/Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf" \
  --n-ctx 2048 \
  --n-gpu-layers -1 \
  --repeat-lines 20 \
  --max-tokens 128 \
  --draft-block-size 8 \
  --template-min-tokens 1 \
  --runs 1 \
  --json | python -c 'import sys,json; d=json.load(sys.stdin); r=d["runs"][0]; print("ok:", d["ok"], "| token_match:", d["token_match"], "| draft_enabled:", d["template_draft_enabled"], "| accepted:", r["C_accepted"], "| drafted:", r["C_drafted"], "| rejected:", r["C_rejected"], "| mismatch_match:", r["mismatch_token_match"], "| decode_speedup:", r["C_vs_B_decode_speedup"])'
```
