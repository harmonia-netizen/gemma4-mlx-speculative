# gemma4-mlx-speculative: Gemma 4 MLX Template Draft Experiments

Experimental repository for local inference acceleration on Apple Silicon / MLX with Gemma 4 models.

## What this is

This project explores an alternative approach to speculative decoding for MLX / Gemma 4. Instead of using a smaller draft model (which incurred high overhead), this project verifies deterministic "template candidates" against the target model in a single batch. When the output exactly matches the candidate template, the engine adopts the fast path, resulting in significant speedups for highly predictable outputs (like fixed bash commands).

## Current Result

For workflows where deterministic templates match the target greedy output (e.g. `exact_pytest_plan`):
- Target greedy `decode_sec`: ~0.42s
- Template draft engine `decode_sec`: ~0.19s
- **Speedup**: ~2.2x
- **Accepted**: 100%

For workflows where candidates are uncertain or mismatch (e.g. `medium_pytest_plan`):
- The `candidate gating` system filters out bad candidates.
- The engine defaults back to greedy generation, incurring negligible overhead (greedy-equivalent speed).
- Token match remains 100% accurate.

## Current Architecture

The project has evolved into a combined **Runtime Prototype** composed of:
1. **CandidateRegistry**: Manages predictable prompt patterns and template candidates.
2. **PrefixCacheManager**: Hashes, caches, and instantly restores the KV cache for long shared context prefixes.
3. **TemplateDraftRuntime**: Orchestrates the prefix restore, suffix prefill, candidate proposal, target verification, and fallback lifecycle.

## Main Results

- **Exact Output Decode**: ~**2.2x** speedup (Template Draft Engine).
- **Long Context Prefix Reuse**: ~**2.9x** overall amortized speedup for ~15k token sequences across multiple generated suffixes.
- **Integrated Runtime Prototype**: Successfully unified Prefix Cache Reuse and Template Draft, maintaining 100% token match while delivering combined prefill and decode accelerations.

## Why not small-model draft?

Initial experiments with a 4B draft model successfully maintained correctness. However, the overhead of loading and running the draft model for short command generation outweighed its benefits. 

The deterministic/template candidate + target verification approach proved far more promising for this specific local-agent use case, avoiding draft model overhead entirely while accelerating highly predictable text.

## How Template Draft works

1. **Candidate provider** proposes deterministic template candidates based on the prompt.
2. **Confidence / min_tokens gating** filters out low-probability candidates.
3. The engine uses the target model's `forward_many` to verify the candidate sequence in one pass.
4. **If exactly matched**, the engine retains the advanced cache state and skips token-by-token generation.
5. **If mismatched**, the engine triggers a snapshot `restore_full` to roll back the cache and falls back to greedy generation.
6. The resulting token sequence is guaranteed to perfectly match the target greedy output.

## Key Files
- `experiments/template_draft_runtime.py`: The final integrated runtime prototype combining Prefix Reuse and Template Draft.
- `experiments/benchmark_template_draft_runtime.py`: The benchmark suite for the runtime.
- `docs/experiments/final-summary.md`: The final architectural overview and results.
- `experiments/template_draft_engine.py`: The underlying speculative engine implementation.
- `docs/experiments/summary.md`: Historical progression of the experiments.

## Quick Start

Run the combined Prefix Reuse + Template Draft Runtime benchmark (15k context):
```bash
cd ~/mlx
.venv/bin/python experiments/benchmark_template_draft_runtime.py --repeat-lines 500 --runs 1 --max-tokens 128 --draft-block-size 8 --template-min-tokens 1
```

Run the standalone fast-path template draft benchmark:
```bash
.venv/bin/python experiments/benchmark_template_draft_engine.py --case exact_pytest_plan --warmup-runs 2 --runs 10 --draft-block-size 8 --template-min-tokens 1
```

## Status
Research prototype / experiment. Not yet a packaged library.

## License

Apache License 2.0. See [LICENSE](LICENSE).
