# Speculative Decoding Experiments Summary

## Overview
This document summarizes the progression of experiments aimed at accelerating local Gemma 4 inference on Apple Silicon / MLX.

## Baseline
- **Target Greedy**: Standard token-by-token generation using the primary model.

## Failed / Less Useful Attempts
- **Small-Model Draft (v1-v4)**: Utilizing a 4B model as a draft for the larger target model. While correctness was preserved, the accept rate for command generation and the draft overhead made it slower than the baseline.
- **Full Cache Clone Verification (v11)**: Attempted to clone the cache before verifying templates to prevent state corruption. Safe, but too slow due to clone overhead.

## Successful Path
The successful approach shifted away from dual-model setups to a single-model template verification system. 

- **v12 Fast Path**: Implemented a direct `forward_many` fast path. Achieved ~2.2x speedup on exact template matches.
- **v13 Restore Fallback**: Implemented safe mismatch recovery via `full_snapshot` and `restore_full`, retaining fast-path speed while ensuring mismatch safety.
- **v14 Candidate Provider**: Separated template generation into a provider, avoiding known bad candidates (e.g. `medium_pytest_plan`).
- **v15 Candidate Confidence Gating**: Introduced the `Candidate` dataclass and a practical confidence-based selection mechanism.
- **template_draft_engine**: Consolidated the successful logic into a reusable experiment runner, which is the current recommended path.

## Results Table

| Version | Purpose | Result | Decision |
|---|---|---|---|
| v5/v6 | Correctness baseline | Token match OK, too slow | Abandoned for speed |
| v7 | Partial cache transaction | Correctness OK, still not speed path | Abandoned for simplicity |
| v8/v9/v10 | Template draft exploration | Safe but not enough | Continued iteration |
| v11 | Cloned cache verification | Safe but slow | Abandoned clone approach |
| v12 | Direct forward_many fast path | About ~2.2x speedup on exact match | Adopted as core speed path |
| v13 | Fast path + restore fallback | Speed retained and mismatch safe | Adopted for fallback safety |
| v14 | Candidate provider | Avoids known bad candidates | Adopted for modularity |
| v15 | Candidate confidence gating | Practical selection mechanism | Adopted |
| engine | Reusable experiment runner | Fast, safe, reusable | **Current recommended path** |

## Current Recommended Command
```bash
.venv/bin/python experiments/benchmark_template_draft_engine.py --case exact_pytest_plan --warmup-runs 2 --runs 10 --draft-block-size 8 --template-min-tokens 1
```

## Current Limitations
- Speedup depends heavily on candidate quality and exactness.
- Only deterministic/template candidates are explored currently.
- Candidate registry is still hardcoded in the engine.
- No package/API is available yet.
- Prompt coverage is relatively small.

## Next Steps
- External candidate registry implementation.
- Expand to more real agent output templates.
- Confidence scoring via an automated prompt classifier.
- Package/API cleanup and abstraction.
- Benchmark on more diverse prompts and longer outputs.
