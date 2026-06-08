# local-speculative-runtime

This is an experimental **MLX/GGUF dual backend runtime** designed to test prompt manipulation, memory profiling, session cache limits, and eviction strategies safely in a multi-turn environment. It provides a common API to evaluate different models, with capabilities clearly defined per backend.

**Status:** Experimental runtime (Not a production package)

## Package Entrypoint & Backends

You can import the runtime components via the experimental package entrypoint:

```python
from local_speculative_runtime import SessionCacheAPI, get_memory_stats

# MLX Backend (Default: Template Draft + Prefix/Session cache + snapshot/restore)
# By default, this uses the shared experiments/template_candidates.json
api_mlx = SessionCacheAPI.load("mlx-community/gemma-4-26b-a4b-it-8bit", backend="mlx")

# GGUF / llama.cpp Backend
# NOTE: GGUF requires an explicit model-specific candidate JSON.
# Built-in examples currently include a Qwen-oriented candidate set, but this backend can run any GGUF.
api_gguf = SessionCacheAPI.load(
    model_path="/path/to/model.gguf",
    candidate_json_path="experiments/template_candidates_gguf_qwen.json",  # Example for Qwen
    backend="llama_cpp"
)

stats = get_memory_stats()
```

## Supported Backends

### 1. MLX Backend
- Template Draft fast-path
- Exact KV snapshot/restore
- Long Input Guard
- Verified primarily with Gemma 4 MLX (e.g. `mlx-community/gemma-4-26b-a4b-it-8bit`)

### 2. GGUF Backend
- Powered by `llama-cpp-python`
- Load, session management, generation, and multi-turn workflows are verified to work generically.
- **Prefix Acceleration:** Fully supported via low-level eval/sample.
- **Template Draft & Exact Rollback:** Fully functional and performant via lightweight `kv_cache_seq_rm`.
- See [GGUF Backend Design](docs/gguf_backend_design.md) for architectural details and implementation notes.

## Core Features (MLX Backend)

This project focuses on three main architectural directions:

1. **Template Candidate + Target Verification**: Instead of using a smaller draft model, this runtime uses predefined "template candidates". The engine verifies these candidates against the target model in a single batch, providing significant speedups for highly predictable outputs.
2. **Prefix / Session Cache Reuse**: Evaluates and caches long shared context prefixes (KV cache). The engine can restore this snapshot for subsequent multi-turn requests, reducing repeated prefill work.
3. **Long Input Guard**: Built-in capacity guard that monitors and restricts prompt lengths to prevent out-of-memory (OOM) crashes on Apple Silicon.

## CLI Usage

The runtime provides a unified CLI for both MLX and GGUF backends.

**MLX:**
```bash
python -m local_speculative_runtime.cli \
  --backend mlx \
  --model mlx-community/gemma-4-26b-a4b-it-8bit \
  --prompt "Return exactly: OK" \
  --max-tokens 16 \
  --json
```
*Note: For MLX, do not specify `--model-type` or `--candidate-json`.*

**GGUF:**
```bash
python -m local_speculative_runtime.cli \
  --backend llama_cpp \
  --model "$HOME/Documents/model.gguf" \
  --model-type qwen \
  --prompt "Return exactly: OK" \
  --max-tokens 16 \
  --json
```
*Note: For GGUF, either `--model-type` or `--candidate-json` is strictly required. The built-in preset is currently `qwen` only; other models require specifying their specific candidate rules via `--candidate-json`.*

## OpenAI-Compatible API

The runtime includes an experimental OpenAI-compatible API server. Currently, it supports `/v1/models` and non-streaming `/v1/chat/completions`.
Internally, chat messages are split into a reusable prefix (system prompt and history) and a latest-turn suffix to maximize the benefits of `SessionCacheAPI`'s prefix reuse.

**Start the Server:**
```bash
# MLX Backend
LSR_BACKEND=mlx LSR_MODEL="mlx-community/gemma-4-26b-a4b-it-8bit" python -m local_speculative_runtime.openai_api

# GGUF Backend
LSR_BACKEND=llama_cpp LSR_MODEL="/path/to/model.gguf" LSR_MODEL_TYPE="qwen" python -m local_speculative_runtime.openai_api
```

**Smoke Test (Optional):**
You can use the built-in smoke test scripts to verify that the server endpoints (`/v1/models` and `/v1/chat/completions`) work correctly with your configured model.
If `LSR_MODEL` is not set, the scripts will simply skip testing and exit.
```bash
# HTTP Smoke Test
PYTHONPATH=. LSR_BACKEND=gguf LSR_MODEL="$HOME/Documents/model.gguf" LSR_MODEL_TYPE=qwen \
  .venv/bin/python experiments/smoke_openai_api.py

# OpenAI Python SDK Smoke Test (Requires `openai` package)
PYTHONPATH=. LSR_BACKEND=gguf LSR_MODEL="$HOME/Documents/model.gguf" LSR_MODEL_TYPE=qwen \
  .venv/bin/python experiments/smoke_openai_sdk.py
```

**cURL Example:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Return exactly: OK"}],
    "max_tokens": 16
  }'
```
*Note: Streaming is not implemented yet. However, if `stream: true` is requested, the server will process it as a non-streaming fallback and return a normal JSON response with the `X-LSR-Warning: stream=true is not supported; returned non-streaming response` header. For GGUF backends, `LSR_MODEL_TYPE` or `LSR_CANDIDATE_JSON` must be specified. Some GGUF models can use `--generation-mode high-level` for quality-first fallback when low-level speculative generation is not compatible.*

## Quick Start & Verification


**1. Run lightweight completion checks and tests:**
```bash
python experiments/run_completion_checks.py
```

**2. Run full model-based verification checks:**
```bash
python experiments/run_model_checks.py
```
*Note: `run_model_checks.py` requires actual models. It may trigger large HF Hub downloads (e.g., Gemma 4 26b 8-bit) or load large local GGUF models.*

**3. Run the Template Draft & Session Cache benchmark (MLX):**
```bash
python experiments/benchmark_template_draft_runtime.py \
  --repeat-lines 500 \
  --runs 1 \
  --max-tokens 128 \
  --draft-block-size 8 \
  --template-min-tokens 1
```

**4. Run the memory stats benchmark:**
```bash
python experiments/benchmark_memory_stats.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
