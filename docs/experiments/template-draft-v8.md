# Template draft v8

## Status

v8 replaces the small-model draft with a deterministic template drafter.

## Result

Short pytest prompt:

Target greedy:
- output: `pytest --tb=short`
- decode_tok_s: about 47 tok/s

Small-model draft v7:
- decode_tok_s: about 14 tok/s
- accepted: 4 / 11

Template draft v8:
- decode_tok_s: about 42 tok/s
- accepted: 6 / 6
- output matched target greedy

## Decision

Template draft is a better default than 4B small-model draft for short command-like agent outputs.

4B draft remains useful as a correctness experiment, but not as the default speed path.
