import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gemma4_mlx_runtime import (
    get_memory_stats,
    memory_stats_to_dict,
    format_memory_stats,
    maybe_reset_mlx_peak_memory,
    maybe_clear_mlx_cache,
    __version__
)

def test_memory_stats_api():
    print("test_memory_stats_api...")
    assert isinstance(__version__, str)
    
    # Check execution without exceptions
    maybe_reset_mlx_peak_memory()
    maybe_clear_mlx_cache()
    
    stats = get_memory_stats(label="test")
    assert stats is not None
    
    # Format works
    fmt = format_memory_stats(stats)
    assert isinstance(fmt, str)
    assert len(fmt) > 0
    
    # Dict conversion works and is JSON serializable
    d = memory_stats_to_dict(stats)
    json_str = json.dumps(d)
    assert "test" in json_str or "unavailable" in json_str
    
    print("  OK")

if __name__ == "__main__":
    test_memory_stats_api()
    print("All tests passed.")
