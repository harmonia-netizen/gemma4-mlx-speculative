import sys
import json
from local_speculative_runtime.llama_cpp_backend import LlamaCppBackend

def test_llama_cpp_light():
    print("test_llama_cpp_light...")
    
    try:
        import llama_cpp
    except ImportError:
        print("  SKIP: llama-cpp-python is not installed")
        sys.exit(0)

    # Initialize without loading model
    backend = LlamaCppBackend(model_path="dummy.gguf", auto_load=False)
    assert backend.backend_name == "llama_cpp"
    assert backend.available is True
    assert backend.llm is None
    
    caps = backend.capabilities()
    assert caps.supports_template_verify is True
    assert caps.supports_snapshot_restore is True
    assert caps.state_restore_status == "supported"

    stats = backend.stats()
    assert "capabilities" in stats
    assert stats["loaded"] is False

    # Should be JSON serializable
    json.dumps(stats)
    
    # Generate without model loaded
    res = backend.generate(None, "test")
    assert res.ok is False
    assert "Model not loaded" in res.error

    print("  OK")

if __name__ == "__main__":
    test_llama_cpp_light()
