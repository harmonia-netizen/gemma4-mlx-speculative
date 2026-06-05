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

## 4. Constraints and Usage
- **Greedy Only:** The current Template Draft loop and custom sampling logic strictly assume a greedy sampling strategy. Supporting `temperature` or `top-p` requires additional rollback implementations for exact match verification.
- **Isolated Candidates:** GGUF/Qwen-specific drafting candidates must be strictly isolated to `experiments/template_candidates_gguf_qwen.json` to prevent corrupting the generalized MLX ruleset.
- **Entrypoint:** For practical app usage, initialize the backend as follows:
  ```python
  from gemma4_mlx_runtime import SessionCacheAPI
  
  api = SessionCacheAPI.load(
      model_path="/path/to/gguf/model.gguf",
      candidate_json_path="experiments/template_candidates_gguf_qwen.json",
      backend="llama_cpp"
  )
  ```
