import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_speculative_runtime import (
    get_memory_stats,
    format_memory_stats,
    memory_stats_to_dict,
    maybe_reset_mlx_peak_memory,
    maybe_clear_mlx_cache
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--reset-peak", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        maybe_clear_mlx_cache()
    if args.reset_peak:
        maybe_reset_mlx_peak_memory()

    stats = get_memory_stats("benchmark")

    if args.json:
        d = memory_stats_to_dict(stats)
        print(json.dumps(d, indent=2))
    else:
        print("Memory Stats Dictionary:")
        for k, v in memory_stats_to_dict(stats).items():
            print(f"  {k}: {v}")
        print("\nFormatted String:")
        print(format_memory_stats(stats))
        print(f"\nAvailable MLX Memory APIs: {stats.mlx_available} ({stats.mlx_source})")

    print("\nOK: memory stats benchmark completed")

if __name__ == "__main__":
    main()
