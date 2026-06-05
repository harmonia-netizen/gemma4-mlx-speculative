# gemma4-mlx-speculative: Gemma 4 MLX Template Draft & Session Cache Runtime

This repository contains an experimental runtime for local inference acceleration on Apple Silicon / MLX with Gemma 4 models. It explores an alternative approach to speculative decoding using deterministic templates and prefix caching.

**Status:** Experimental runtime (Not a production package)

## Package Entrypoint

You can import the runtime components via the experimental package entrypoint:

```python
from gemma4_mlx_runtime import SessionCacheAPI, get_memory_stats

api = SessionCacheAPI()
stats = get_memory_stats()
```

## Core Features

This project focuses on three main architectural directions:

1. **Template Candidate + Target Verification**: Instead of using a smaller draft model (which incurred high overhead), this runtime uses predefined "template candidates" (e.g., fixed bash commands). The engine verifies these candidates against the target model in a single batch, providing significant speedups for highly predictable outputs.
2. **Prefix / Session Cache Reuse**: Evaluates and caches long shared context prefixes (KV cache). The engine can restore this snapshot for subsequent multi-turn requests, reducing repeated prefill work.
3. **Long Input Guard**: Built-in capacity guard that monitors and restricts prompt lengths to prevent out-of-memory (OOM) crashes on Apple Silicon.

*Note: MTP (Multi-Token Prediction) and small-model lightweight drafts were explored but are not adopted in the current iteration.*

## Quick Start & Verification

**1. Run full completion checks and tests:**
```bash
python experiments/run_completion_checks.py
```

**2. Run the Template Draft & Session Cache benchmark:**
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
