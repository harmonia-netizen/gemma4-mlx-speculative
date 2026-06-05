import json
from agent_mock_runtime import AgentMockRuntime

def test_mock_runtime():
    print("test_mock_runtime...")
    # Using small dummy model string, actually we won't instantiate real model here 
    # to avoid heavy loading in tests. The underlying target_model="mlx-community/gemma-4-26b-a4b-it-8bit" 
    # will be loaded in __init__, so this test might take a few seconds if model needs to be mapped.
    # To keep it lightweight, we just check instantiation and properties.
    
    # We will test using actual initialization to ensure it works, but we won't process large text.
    runtime = AgentMockRuntime()
    
    # Check if stats are JSON serializable
    stats_dict = runtime.stats()
    json_str = json.dumps(stats_dict)
    assert "entries" in json_str
    
    # Create a small session
    init_res = runtime.initialize_context("test_sess", "hello world")
    assert init_res["ok"]
    
    # Check if clear works
    clear_res = runtime.clear("test_sess")
    assert clear_res["ok"]
    
    # Verify candidate exclusion
    res = runtime.handle_task("non_existent_sess", "rm -rf /")
    assert not res["ok"]
    assert "not found" in res["error"].lower()
    
    print("  OK")

if __name__ == "__main__":
    test_mock_runtime()
    print("All tests passed.")
