import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_speculative_runtime.backends import BackendCapabilities, GenerationResult

def test_backend_capabilities_serialization():
    print("test_backend_capabilities_serialization...")
    cap = BackendCapabilities(
        name="test",
        supports_prefix_cache=True,
        supports_template_verify=False,
        supports_snapshot_restore=False,
        supports_token_logprobs=False,
        notes=["Note 1", "Note 2"]
    )
    
    # Needs to be dict convertible for stats()
    from dataclasses import asdict
    d = asdict(cap)
    j = json.dumps(d)
    assert "Note 1" in j
    print("  OK")

def test_generation_result_serialization():
    print("test_generation_result_serialization...")
    res = GenerationResult(
        ok=True,
        text="hello",
        token_ids=[1, 2, 3],
        elapsed_sec=1.0,
        prompt_tokens=10,
        completion_tokens=3,
        error=None,
        backend="test",
        metadata={"a": 1}
    )
    from dataclasses import asdict
    d = asdict(res)
    j = json.dumps(d)
    assert "hello" in j
    print("  OK")

def test_import_backends():
    print("test_import_backends...")
    # Import MLX
    from local_speculative_runtime.mlx_backend import MLXBackend
    assert MLXBackend is not None
    
    # Import LlamaCpp
    from local_speculative_runtime.llama_cpp_backend import LlamaCppBackend
    assert LlamaCppBackend is not None
    print("  OK")

if __name__ == "__main__":
    test_backend_capabilities_serialization()
    test_generation_result_serialization()
    test_import_backends()
    print("All tests passed.")
