# Gemma 4 speculative decoding v5

## Status

v5 is a correctness-first implementation.

## Result

Target greedy:
- tokens: 7
- decode_sec: 0.148
- decode_tok_s: 47.341

Speculative v5:
- tokens: 7
- decode_sec: 0.466
- decode_tok_s: 15.018
- accepted: 4 / 8 = 50.0%
- rejected: 3
- output matched target greedy

## Decision

- Use v5 as correctness baseline.
- Do not adopt v5 for speed.
- Position rollback using offset/_idx is rejected because sequence rollback/replay changed logits.
- Next step is RotatingKVCache-specific truncate or MLX cache API support.
