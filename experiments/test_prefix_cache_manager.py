import time
from template_draft_runtime import PrefixCacheManager, PrefixCacheEntry

def add_entry_for_test(manager: PrefixCacheManager, key: str, token_count: int, last_used_at: float):
    # Mock entry for testing
    entry = PrefixCacheEntry(
        text_hash=key,
        token_ids=[0] * token_count,
        snapshot=None,
        cache=None,
        prefill_sec=0.0,
        created_at=time.time(),
        last_used_at=last_used_at,
        hit_count=0,
        evicted_keys=None
    )
    manager.entries[key] = entry
    manager.current_total_tokens += token_count

def test_eviction_max_entries():
    print("test_eviction_max_entries...")
    manager = PrefixCacheManager(max_entries=2, max_total_tokens=1000)
    
    add_entry_for_test(manager, "key1", 10, time.time() - 10)
    add_entry_for_test(manager, "key2", 10, time.time() - 5)
    
    # Adding 3rd entry should evict key1
    evicted = manager.evict_if_needed(additional_tokens=10)
    assert "key1" in evicted, f"Expected key1 in evicted, got {evicted}"
    assert "key1" not in manager.entries
    assert "key2" in manager.entries
    print("  OK")

def test_eviction_max_total_tokens():
    print("test_eviction_max_total_tokens...")
    manager = PrefixCacheManager(max_entries=10, max_total_tokens=100)
    
    add_entry_for_test(manager, "key1", 60, time.time() - 10)
    
    # Adding 60 more should evict key1
    evicted = manager.evict_if_needed(additional_tokens=60)
    assert "key1" in evicted, f"Expected key1 in evicted, got {evicted}"
    assert "key1" not in manager.entries
    assert manager.current_total_tokens == 0
    print("  OK")

def test_single_prefix_exceeds_limit():
    print("test_single_prefix_exceeds_limit...")
    manager = PrefixCacheManager(max_entries=10, max_total_tokens=100)
    
    try:
        # get_or_create raises ValueError if prefix_ids exceeds max_total_tokens
        manager.get_or_create("huge_text", [0]*150, target_model=None, max_kv_size=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds max_total_tokens" in str(e)
    print("  OK")

def test_remove_key():
    print("test_remove_key...")
    manager = PrefixCacheManager(max_entries=10, max_total_tokens=100)
    add_entry_for_test(manager, "key1", 60, time.time() - 10)
    assert manager.current_total_tokens == 60
    
    res = manager.remove("key1")
    assert res is True
    assert "key1" not in manager.entries
    assert manager.current_total_tokens == 0
    
    res_unknown = manager.remove("unknown")
    assert res_unknown is False
    assert manager.current_total_tokens == 0
    print("  OK")

if __name__ == "__main__":
    test_eviction_max_entries()
    test_eviction_max_total_tokens()
    test_single_prefix_exceeds_limit()
    test_remove_key()
    print("All tests passed.")
