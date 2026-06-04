# Template Draft v11 Speed Benchmark Results

## short_pytest_command (short text)
- draft-block-size 8:
  - median_decode_sec: greedy 0.163s, template 0.323s
  - decode_sec_speedup: 0.504x
  - accepted: 60/60 (100.0%)
- draft-block-size 1:
  - median_decode_sec: greedy 0.156s, template 0.316s
  - decode_sec_speedup: 0.495x
  - accepted: 60/60 (100.0%)

## exact_pytest_plan (medium text)
- draft-block-size 8:
  - median_decode_sec: greedy 0.448s, template 0.895s
  - decode_sec_speedup: 0.501x
  - accepted: 170/170 (100.0%)

## Conclusion & Next Steps
- **Cloned Cache Safety**: Using `clone_cache_objects` to isolate the target cache during verification is safe and guarantees 100% correctness without poisoning the main cache. Reject fallback perfectly recovers to target greedy.
- **Overhead**: Creating cloned cache objects and running `forward_one` on the cloned cache introduces significant overhead in Python. As a result, even when 100% of tokens are accepted, the decode speed is about half of the standard greedy decoding (`~0.5x` speedup) and overhead dominates.
- **Current Judgment**: For both short and medium sentences, the current implementation of template verification over a cloned cache is too slow due to the overhead of cloning and sequential evaluation.
- **Future Direction**: The main target for speed evaluation remains medium/long sentences where `accepted > 0` is expected. However, to actually achieve a speedup > 1.0x, the verification process needs fundamental optimization.
