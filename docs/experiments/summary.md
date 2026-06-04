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
| engine | Reusable experiment runner | Fast, safe, reusable | Baseline decode engine |
| long_context | Prefill bottleneck analysis | Prefill takes 98% of elapsed time | Identified need for Prefix Cache |
| prefix_reuse | Prefix Cache snapshot/restore | ~2.9x amortized speedup | Adopted for long contexts |
| integrated_runtime | Template Draft + Prefix Cache | Combined speedups achieved safely | **Current final architecture prototype** |

## Further Reading
- [Template Draft Engine Detailed Docs](template-draft-engine.md)
- [Prefix Cache Reuse Docs](prefix-cache-reuse.md)
- [Integrated Runtime Prototype & Final Summary](final-summary.md)

## Current Recommended Command
```bash
.venv/bin/python experiments/benchmark_template_draft_runtime.py --repeat-lines 500 --runs 1 --max-tokens 128 --draft-block-size 8 --template-min-tokens 1
```

## Current Limitations
- Speedup depends heavily on candidate quality and exactness.
- Cache memory management (LRU, Session scopes) is still unimplemented.
- Candidate registry is still hardcoded.
- Prototype code is not yet packaged into an importable library.

## Next Steps
- Implement `PrefixCacheManager` LRU and Session state management.
- Externalize `CandidateRegistry` to configuration files (JSON/YAML).
- Package the prototype into an easy-to-use API.
- Multi-turn benchmark integration for continuous agent flows.
