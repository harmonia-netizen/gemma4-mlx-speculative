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

import unittest.mock as mock

def test_clear_session_behavior():
    print("test_clear_session_behavior...")
    
    # Mock model loading to avoid loading actual model
    mock_model = mock.Mock()
    mock_processor = mock.Mock()
    mock_processor.tokenizer = mock.Mock()
    mock_processor.tokenizer.encode = lambda x: [1, 2, 3] # fake tokens
    
    # Patch d.load in session_cache_api
    with mock.patch("session_cache_api.d.load", return_value=(mock_model, mock_processor)):
        api = SessionCacheAPI(model_path="dummy", candidate_json_path="experiments/template_candidates.json")
        
        # We need to mock PrefixCacheManager.get_or_create to return a dummy entry
        from template_draft_runtime import PrefixCacheEntry
        import time
        dummy_entry = PrefixCacheEntry(
            text_hash="hash1",
            token_ids=[1, 2, 3],
            snapshot=None,
            cache=None,
            prefill_sec=0.1,
            created_at=time.time(),
            last_used_at=time.time(),
            hit_count=0
        )
        api.runtime.prefix_manager.get_or_create = mock.Mock(return_value=dummy_entry)
        
        # Test create_session
        res = api.create_session("sess1", "hello")
        assert res["ok"]
        
        # Explicitly add entry to manager since get_or_create is mocked
        api.runtime.prefix_manager.entries["hash1"] = dummy_entry
        api.runtime.prefix_manager.current_total_tokens += 3
        
        # Test clear_session(drop_cache=False)
        clear_res1 = api.clear_session("sess1", drop_cache=False)
        assert clear_res1["ok"]
        assert clear_res1["dropped_cache"] is False
        assert "sess1" not in api.runtime.sessions
        assert "hash1" in api.runtime.prefix_manager.entries # Cache remains
        
        # Test clear_session(drop_cache=True)
        res = api.create_session("sess2", "hello")
        api.runtime.prefix_manager.entries["hash1"] = dummy_entry
        
        clear_res2 = api.clear_session("sess2", drop_cache=True)
        assert clear_res2["ok"]
        assert clear_res2["dropped_cache"] is True
        assert "hash1" not in api.runtime.prefix_manager.entries # Cache dropped
        
        # Test evicted session generate
        res = api.create_session("sess3", "hello")
        api.runtime.prefix_manager.entries["hash1"] = dummy_entry
        # Evict manually
        api.runtime.prefix_manager.entries.pop("hash1")
        
        gen_res = api.generate("sess3", "suffix")
        assert not gen_res["ok"]
        assert "evicted" in gen_res["error"].lower()
        
    print("  OK")

if __name__ == "__main__":
    test_api_import_and_stats()
    test_clear_session_behavior()
    print("All tests passed.")
