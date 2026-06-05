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

## Quick Start & Verification


**1. Run full completion checks and tests:**
```bash
python experiments/run_completion_checks.py
```

**2. Run the Template Draft & Session Cache benchmark (MLX):**
```bash
python experiments/benchmark_template_draft_runtime.py \
  --repeat-lines 500 \
  --runs 1 \
  --max-tokens 128 \
  --draft-block-size 8 \
  --template-min-tokens 1
```

**3. Run the memory stats benchmark:**
```bash
python experiments/benchmark_memory_stats.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
