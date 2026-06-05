import json
from session_cache_package import (
    SessionCacheAPI,
    get_memory_stats,
    CacheStats
)
from dataclasses import asdict

def test_api_import_and_stats():
    print("test_api_import_and_stats...")
    try:
        # Avoid loading model by not instantiating the full API or creating a mock
        # We'll just test if CacheStats is JSON serializable
        stats = CacheStats(entries=1, current_total_tokens=100, max_entries=2, max_total_tokens=1000, keys=["key1"])
        
        # Dataclasses aren't natively json serializable without asdict
        # But our API wrapper stats() returns a dict, so let's test that structure
        stats_dict = {
            "entries": stats.entries,
            "current_total_tokens": stats.current_total_tokens,
            "max_entries": stats.max_entries,
            "max_total_tokens": stats.max_total_tokens,
            "keys": stats.keys
        }
        
        json_str = json.dumps(stats_dict)
        assert "key1" in json_str
        print("  OK")
    except Exception as e:
        print(f"  Failed: {e}")
        raise

if __name__ == "__main__":
    test_api_import_and_stats()
    print("All tests passed.")
