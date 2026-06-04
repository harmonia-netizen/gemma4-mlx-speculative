# Template Draft v12 Fast Path Evaluation

## v12 Fast Path Specification
In v12, we implemented an aggressively optimized "fast path" specifically targeted at templates that exactly match the target model's output:
- **No Cache Cloning**: We bypass `clone_cache_objects` entirely.
- **Direct Forward**: The target cache is directly fed into `forward_many(target_lm, proposal, target_cache)`.
- **FAIL FAST on Mismatch**: If any token inside the proposal mismatches the target, we throw `SystemExit(3)` (FAIL FAST) instead of attempting to fallback. Proposals containing stop tokens immediately disable the template.
- **Zero Copy Adoption**: When the full block matches perfectly, `target_next_logits` is simply the last slice `verify_logits[:, -1, :]` and `target_cache` is already up-to-date, requiring no replay or commit steps.

## Speed Comparison with v11
- **Target Case**: `exact_pytest_plan` (medium text, ~18 tokens)
- **v11 (Cloned Verify)**: `decode_sec_speedup = ~0.50x` (Slower than greedy)
- **v12 (Fast Path)**: `decode_sec_speedup = ~2.21x` (Over 2x faster than greedy)

### Results (exact_pytest_plan)
- `greedy_decode_sec`: ~0.464s
- `template_decode_sec`: ~0.209s
- `accepted`: 170 / 170 (100.0%)

## Success / Failure
**Success!**
By eliminating Python-level cache loops, deep cloning, and explicit `commit_tokens_to_cache` operations, the speculative draft mechanism achieves significant performance gains (`> 2.0x` speedup) for fully-matched candidate blocks.

## Next Required Fallback Design
Since v12 currently uses a strict `SystemExit(3)` when a mismatch occurs inside the block, it is unsafe for general use where candidates might differ from the target model's output. 
To make v12 robust enough for production, the fallback mechanism must be redesigned:
1. **Cache Snapshots**: We need to leverage partial KV cache snapshots before `forward_many`.
2. **Rollback**: If a mismatch occurs at index `i`, we must restore the cache state using the snapshot to cleanly rollback the mispredicted tokens, allowing the generation loop to fallback to standard greedy generation.
3. **Resilience**: This snapshot-based rollback will guarantee mathematical correctness and exact token match while maintaining the massive speed benefits of `forward_many`.
