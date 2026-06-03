# gemma4-mlx-speculative

Experimental repository for local inference acceleration on Apple Silicon / MLX with Gemma 4 models.

This project explores:

- draft speculative decoding with Gemma 4 target and smaller draft models
- KV cache transaction design for MLX / RotatingKVCache
- full and partial cache restore correctness checks
- prompt cache and long-context inference behavior
- local agent inference constraints, including safe executor separation

## Current status

The repository currently contains correctness-first experiments for custom speculative decoding.

Confirmed so far:

- target greedy and speculative outputs can be matched
- offset/_idx-only rollback is insufficient
- full cache restore and partial RotatingKVCache restore can preserve token-decision behavior
- 4B model draft is currently slower than target-only decoding on short command outputs

## Goal

The goal is to build a reproducible experimental foundation for safe and efficient local LLM agent inference, especially for long-context and command-producing workflows.
